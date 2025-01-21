"""Contains the Voxelizer class, which is used to convert molecular
structures into voxel tensors.
"""


import ctypes
import itertools
from dataclasses import dataclass
from enum import Enum
import math
from typing import TYPE_CHECKING, List, Any, Optional, Tuple

import torch  # type: ignore
import numba  # type: ignore
import numba.cuda  # type: ignore
import numpy as np  # type: ignore
from collagen.core.voxelization import gen_grid_gpu

from ..molecules.atom_featurizer import AtomicNumFeaturizer, DeepFragReceptorFeaturizer
from functools import lru_cache

if TYPE_CHECKING:
    import collagen.core.molecules.atom_featurizer

# There are max 1024 threads in each block. Found ** 6 ** to be optimal after
# trial and error. 8 is what Harrison had. Note can't be greater than 10 because
# 11 * 11 * 11 > 1024.
GPU_DIM = 6  # Cubic root of threads per block


@dataclass
class VoxelParams(object):

    """A VoxelParams object describes how a molecular structure is converted
    into a voxel tensor.

    Attributes:
        resolution (float): The distance in Angstroms between neighboring grid
            points. A smaller number means more zoomed-in.
        width (int): The number of gridpoints in each spatial dimension.
        atom_scale (float): A multiplier applied to atomic radii.
        atom_shape: (VoxelParams.AtomShapeType): Describes the atomic density
            sampling function.
        acc_type: (VoxelParams.AccType): Describes how overlapping atomic
            densities are handled.
        atom_featurizer (AtomFeaturizer): An atom featurizer that describes how
            to assign atoms to layers.
    """

    class AtomShapeType(Enum):

        """Describes the atomic density sampling function."""

        EXP = 0  # simple exponential sphere fill
        SPHERE = 1  # fixed sphere fill
        CUBE = 2  # fixed cube fill
        GAUSSIAN = 3  # continous piecewise expenential fill
        LJ = 4
        DISCRETE = 5

    class AccType(Enum):

        """Describes how overlapping atomic densities are handled."""

        SUM = 0
        MAX = 1

    resolution: float = 1.0
    width: int = 24
    atom_scale: float = 1
    atom_shape: AtomShapeType = AtomShapeType.EXP
    acc_type: AccType = AccType.SUM
    atom_featurizer: Optional[
        "collagen.core.molecules.atom_featurizer.AtomFeaturizer"
    ] = None
    calc_voxels: bool = True
    calc_fps: bool = True

    def validate(self):
        """Validate the VoxelParams object."""
        assert self.resolution > 0, f"resolution must be >0 (got {self.resolution})"
        assert self.atom_featurizer is not None, "atom_featurizer must not be None"

    def tensor_size(self, batch=1, feature_mult=1) -> Tuple[int, int, int, int, int]:
        """Compute the required tensor size given the voxel parameters.

        Args:
            batch (int, optional): Number of molecules in the target tensor
                (default: 1).
            feature_mult (int, optional): Optional multiplier for the channel
                size.

        Returns:
            Tuple[int, int, int, int, int]: The tensor size.
        """
        assert self.atom_featurizer is not None, "atom_featurizer must not be None"

        N = self.atom_featurizer.size() * feature_mult
        W = self.width
        return (batch, N, W, W, W)


class VoxelParamsDefault(object):

    """A default set of VoxelParams. Same params as used in DeepFrag paper."""

    DeepFrag = VoxelParams(
        resolution=0.75,
        width=24,
        atom_scale=1.75,
        atom_shape=VoxelParams.AtomShapeType.EXP,
        acc_type=VoxelParams.AccType.SUM,
        atom_featurizer=DeepFragReceptorFeaturizer([6, 8, 7, 16]),
    )


def numba_ptr(tensor: "torch.Tensor", cpu: bool = False) -> Any:
    """Convert a PyTorch tensor to a Numba pointer.

    Args:
        tensor (torch.Tensor): The tensor to convert.
        cpu (bool, optional): If True, the tensor is copied to the CPU before
            conversion (default: False).

    Returns:
        Any: The Numba pointer.
    """
    if cpu:
        return tensor.numpy()

    # Get Cuda context.
    ctx = numba.cuda.cudadrv.driver.driver.get_active_context()

    memory = numba.cuda.cudadrv.driver.MemoryPointer(
        ctx, ctypes.c_ulong(tensor.data_ptr()), tensor.numel() * 4
    )
    return numba.cuda.cudadrv.devicearray.DeviceNDArray(
        tensor.size(),
        [i * 4 for i in tensor.stride()],
        np.dtype("float32"),
        gpu_data=memory,
        stream=torch.cuda.current_stream().cuda_stream,
    )


# @numba.cuda.jit
# def gpu_gridify(
#     grid: "torch.Tensor",
#     atom_num: "torch.Tensor",
#     atom_coords,
#     atom_mask,
#     atom_radii,
#     layer_offset,
#     batch_idx,
#     width,
#     res,
#     center,
#     rot,
#     atom_scale,
#     atom_shape,
#     acc_type,
# ):
#     """Add atoms to the grid in a GPU kernel.

#     This kernel converts atom coordinate information to 3D voxel information.
#     Each GPU thread is responsible for one specific grid point. This function
#     receives a list of atomic coordinates and atom layers and simply iterates
#     over the list to find nearby atoms and add their effect.

#     Voxel information is stored in a 5D tensor of type: BxTxNxNxN where:
#         B = Batch size
#         N = Number of atom layers
#         W = Grid width (in gridpoints)

#     Each invocation of this function will write information to a specific batch
#     index specified by batch_idx. Additionally, the layer_offset parameter can
#     be set to specify a fixed offset to add to each atom_layer item.

#     How it works:
#     1. Each GPU thread controls a single gridpoint. This gridpoint coordinate
#         is translated to a "real world" coordinate by applying rotation and
#         translation vectors.
#     2. Each thread iterates over the list of atoms and checks for atoms within
#         a threshold to add to the grid.

#     Args:
#         grid: DeviceNDArray tensor where grid information is stored.
#         atom_num: Number of atoms.
#         atom_coords: Array containing (x,y,z) atom coordinates.
#         atom_mask: A uint32 array of size atom_num containing a destination
#             layer bitmask (i.e. if bit k is set, write atom to index k).
#         atom_radii: A float32 array of size atom_num containing invidiual
#             atomic radius values.
#         layer_offset: A fixed offset added to each atom layer index.
#         batch_idx: Index specifiying where to write information.
#         width: Number of grid points in each dimension.
#         res: Distance between neighboring grid points in angstroms.
#             (1 == gridpoint every angstrom)
#             (0.5 == gridpoint every half angstrom, e.g. tighter grid)
#         center: (x,y,z) coordinate of grid center.
#         rot: (x,y,z,y) rotation quaternion.
#         atom_scale: Scale factor applied to each atom radius.
#         atom_shape: Atom shape.
#         acc_type: Accumulation type.
#     """
#     x, y, z = numba.cuda.grid(3)

#     # center grid points around origin. "width" is number of grid points in each
#     # dimension.
#     tx = x - (width / 2)
#     ty = y - (width / 2)
#     tz = z - (width / 2)

#     # scale by resolution
#     tx = tx * res
#     ty = ty * res
#     tz = tz * res

#     # apply rotation vector
#     aw = rot[0]
#     ax = rot[1]
#     ay = rot[2]
#     az = rot[3]

#     bw = 0
#     bx = tx
#     by = ty
#     bz = tz

#     # multiply by rotation vector
#     cw = (aw * bw) - (ax * bx) - (ay * by) - (az * bz)
#     cx = (aw * bx) + (ax * bw) + (ay * bz) - (az * by)
#     cy = (aw * by) + (ay * bw) + (az * bx) - (ax * bz)
#     cz = (aw * bz) + (az * bw) + (ax * by) - (ay * bx)

#     # multiply by conjugate
#     # dw = (cw * aw) - (cx * (-ax)) - (cy * (-ay)) - (cz * (-az))
#     dx = (cw * (-ax)) + (cx * aw) + (cy * (-az)) - (cz * (-ay))
#     dy = (cw * (-ay)) + (cy * aw) + (cz * (-ax)) - (cx * (-az))
#     dz = (cw * (-az)) + (cz * aw) + (cx * (-ay)) - (cy * (-ax))

#     # apply translation vector
#     tx = dx + center[0]
#     ty = dy + center[1]
#     tz = dz + center[2]

#     i = 0
#     while i < atom_num:
#         # fetch atom
#         fx, fy, fz = atom_coords[i]
#         mask = atom_mask[i]

#         r = atom_radii[i] * atom_scale
#         r2 = r * r

#         i += 1

#         # invisible atoms
#         if mask == 0:
#             continue

#         # quick cube bounds check to accelerate calculations
#         if abs(fx - tx) > r2 or abs(fy - ty) > r2 or abs(fz - tz) > r2:
#             continue

#         # value to add to this gridpoint
#         val = 0

#         if atom_shape == 0:  # AtomShapeType.EXP
#             # exponential sphere fill
#             # compute squared distance to atom
#             d2 = (fx - tx) ** 2 + (fy - ty) ** 2 + (fz - tz) ** 2
#             if d2 > r2:
#                 continue

#             # compute effect
#             val = math.exp((-2 * d2) / r2)
#         elif atom_shape == 1:  # AtomShapeType.SPHERE
#             # solid sphere fill
#             # compute squared distance to atom
#             d2 = (fx - tx) ** 2 + (fy - ty) ** 2 + (fz - tz) ** 2
#             if d2 > r2:
#                 continue

#             val = 1
#         elif atom_shape == 2:  # AtomShapeType.CUBE
#             # solid cube fill
#             val = 1
#         elif atom_shape == 3:  # AtomShapeType.GAUSSIAN
#             # (Ragoza, 2016)
#             #
#             # piecewise gaussian sphere fill
#             # compute squared distance to atom
#             d2 = (fx - tx) ** 2 + (fy - ty) ** 2 + (fz - tz) ** 2
#             d = math.sqrt(d2)

#             if d > r * 1.5:
#                 continue
#             elif d > r:
#                 val = math.exp(-2.0) * ((4 * d2 / r2) - (12 * d / r) + 9)
#             else:
#                 val = math.exp((-2 * d2) / r2)
#         elif atom_shape == 4:  # AtomShapeType.LJ
#             # (Jimenez, 2017) - DeepSite
#             #
#             # LJ potential
#             # compute squared distance to atom
#             d2 = (fx - tx) ** 2 + (fy - ty) ** 2 + (fz - tz) ** 2
#             d = math.sqrt(d2)

#             if d > r * 1.5:
#                 continue
#             else:
#                 val = 1 - math.exp(-((r / d) ** 12))
#         elif atom_shape == 5:  # AtomShapeType.DISCRETE
#             # nearest-gridpoint
#             # L1 distance
#             if (
#                 abs(fx - tx) < (res / 2)
#                 and abs(fy - ty) < (res / 2)
#                 and abs(fz - tz) < (res / 2)
#             ):
#                 val = 1

#         # add value to layers
#         for k in range(32):
#             if (mask >> k) & 1:
#                 idx = (batch_idx, layer_offset + k, x, y, z)
#                 if acc_type == 0:  # AccType.SUM
#                     numba.cuda.atomic.add(grid, idx, val)
#                 elif acc_type == 1:  # AccType.MAX
#                     numba.cuda.atomic.max(grid, idx, val)


@numba.jit(nopython=True)
def cpu_gridify(
    grid: np.ndarray,
    atom_num: int,
    atom_coords: List[Tuple[float, float, float]],
    atom_mask: List[int],
    atom_radii: List[float],
    layer_offset: int,
    batch_idx: int,
    width: int,
    res: float,
    center: Tuple[float, float, float],
    rot: Tuple[float, float, float, float],
    atom_scale: float,
    atom_shape: int,
    acc_type: int,
):
    """Add atoms to the grid on the CPU. See gpu_gridify() for argument details.

    Args:
        grid (np.ndarray): The grid to write to.
        atom_num (int): The atom number.
        atom_coords (List[Tuple[float, float, float]]): Array containing
            (x,y,z) atom coordinates.
        atom_mask (List[int]): A uint32 array of size atom_num containing a
            destination layer bitmask (i.e. if bit k is set, write atom to
            index k).
        atom_radii (List[float]): A float32 array of size atom_num containing individual
            atomic radius values.
        layer_offset (int): A fixed offset added to each atom layer index.
        batch_idx (int): Index specifiying where to write information.
        width (int): Number of grid points in each dimension.
        res (float): Distance between neighboring grid points in angstroms.
            (1 == gridpoint every angstrom)
            (0.5 == gridpoint every half angstrom, e.g. tighter grid)
        center (List[float]): (x,y,z) coordinate of grid center.
        rot (List[float]): (x,y,z,y) rotation quaternion.
        atom_scale (float): A float32 value specifying the scale of the atoms.
        atom_shape (int): An AtomShapeType specifying the shape of the atoms.
        acc_type (int): An AccType specifying the accumulation type.
    """
    for x in range(width):
        for y in range(width):
            for z in range(width):

                # center around origin
                tx = x - (width / 2)
                ty = y - (width / 2)
                tz = z - (width / 2)

                # scale by resolution
                tx = tx * res
                ty = ty * res
                tz = tz * res

                # apply rotation vector
                aw = rot[0]
                ax = rot[1]
                ay = rot[2]
                az = rot[3]

                bw = 0
                bx = tx
                by = ty
                bz = tz

                # multiply by rotation vector
                cw = (aw * bw) - (ax * bx) - (ay * by) - (az * bz)
                cx = (aw * bx) + (ax * bw) + (ay * bz) - (az * by)
                cy = (aw * by) + (ay * bw) + (az * bx) - (ax * bz)
                cz = (aw * bz) + (az * bw) + (ax * by) - (ay * bx)

                # multiply by conjugate
                # dw = (cw * aw) - (cx * (-ax)) - (cy * (-ay)) - (cz * (-az))
                dx = (cw * (-ax)) + (cx * aw) + (cy * (-az)) - (cz * (-ay))
                dy = (cw * (-ay)) + (cy * aw) + (cz * (-ax)) - (cx * (-az))
                dz = (cw * (-az)) + (cz * aw) + (cx * (-ay)) - (cy * (-ax))

                # apply translation vector
                tx = dx + center[0]
                ty = dy + center[1]
                tz = dz + center[2]

                i = 0
                while i < atom_num:
                    # fetch atom
                    fx, fy, fz = atom_coords[i]
                    mask = atom_mask[i]

                    r = atom_radii[i] * atom_scale
                    r2 = r * r

                    i += 1

                    # invisible atoms
                    if mask == 0:
                        continue

                    # quick cube bounds check
                    if abs(fx - tx) > r2 or abs(fy - ty) > r2 or abs(fz - tz) > r2:
                        continue

                    # value to add to this gridpoint
                    val = 0

                    if atom_shape == 0:  # AtomShapeType.EXP
                        # exponential sphere fill
                        # compute squared distance to atom
                        d2 = (fx - tx) ** 2 + (fy - ty) ** 2 + (fz - tz) ** 2
                        if d2 > r2:
                            continue

                        # compute effect
                        val = math.exp((-2 * d2) / r2)
                    elif atom_shape == 1:  # AtomShapeType.SPHERE
                        # solid sphere fill
                        # compute squared distance to atom
                        d2 = (fx - tx) ** 2 + (fy - ty) ** 2 + (fz - tz) ** 2
                        if d2 > r2:
                            continue

                        val = 1
                    elif atom_shape == 2:  # AtomShapeType.CUBE
                        # solid cube fill
                        val = 1
                    elif atom_shape == 3:  # AtomShapeType.GAUSSIAN
                        # (Ragoza, 2016)
                        #
                        # piecewise gaussian sphere fill
                        # compute squared distance to atom
                        d2 = (fx - tx) ** 2 + (fy - ty) ** 2 + (fz - tz) ** 2
                        d = math.sqrt(d2)

                        if d > r * 1.5:
                            continue
                        elif d > r:
                            val = math.exp(-2.0) * ((4 * d2 / r2) - (12 * d / r) + 9)
                        else:
                            val = math.exp((-2 * d2) / r2)
                    elif atom_shape == 4:  # AtomShapeType.LJ
                        # (Jimenez, 2017) - DeepSite
                        #
                        # LJ potential
                        # compute squared distance to atom
                        d2 = (fx - tx) ** 2 + (fy - ty) ** 2 + (fz - tz) ** 2
                        d = math.sqrt(d2)

                        if d > r * 1.5:
                            continue
                        else:
                            val = 1 - math.exp(-((r / d) ** 12))
                    elif atom_shape == 5:  # AtomShapeType.DISCRETE
                        # nearest-gridpoint
                        # L1 distance
                        if (
                            abs(fx - tx) < (res / 2)
                            and abs(fy - ty) < (res / 2)
                            and abs(fz - tz) < (res / 2)
                        ):
                            val = 1

                    # add value to layers
                    for k in range(32):
                        if (mask >> k) & 1:
                            idx = (batch_idx, layer_offset + k, x, y, z)
                            if acc_type == 0:  # AccType.SUM
                                grid[idx] += val
                            elif acc_type == 1:  # AccType.MAX
                                grid[idx] = max(grid[idx], val)


@lru_cache(maxsize=None)
def _get_num_blocks_and_threads(num_points: int) -> Tuple[int, int]:
    # Following the helpful guide here:
    # http://selkie.macalester.edu/csinparallel/modules/CUDAArchitecture/build/html/2-Findings/Findings.html#cuda-best-practices
    # NOTE: Ended up not using this. Important that block size and threads per
    # block be tuples.

    gpu = numba.cuda.get_current_device()
    # print("MAX_THREADS_PER_BLOCK", gpu.MAX_THREADS_PER_BLOCK)  # 1024

    # Try to make the number of threads per block a multiple of 32.
    threads_per_block = [32 * i for i in range(1, 1 + gpu.MAX_THREADS_PER_BLOCK // 32)]

    # Keep the number of threads per block and the number of blocks as close to equal as you can without violating the first tip.
    nums_blocks = [math.ceil(num_points / t) for t in threads_per_block]
    diffs = [math.fabs(t - n) for t, n in zip(threads_per_block, nums_blocks)]
    idx = np.argmin(diffs)

    thread_per_block = threads_per_block[idx]
    num_blocks = nums_blocks[idx]

    return num_blocks, thread_per_block


def _debug_grid_to_xyz(grid):
    grid_cpu = grid.copy_to_host()
    merged = np.sum(grid_cpu[0], axis=0)
    txt = "".join(
        "X\t" + str(x) + "\t" + str(y) + "\t" + str(z) + "\n"
        for x, y, z in itertools.product(range(24), range(24), range(24))
        if merged[x, y, z] > 0
    )
    count = len(txt.split("\n")) - 1
    with open("tmp.xyz", "w") as f:
        f.write(str(count) + "\n")
        f.write("TITLE\n")
        f.write(txt)

    print(count)

    import pdb

    pdb.set_trace()


# time_debug = 0
# cnt = 0


def mol_gridify(
    grid: np.ndarray,
    atom_coords: List[Tuple[float, float, float]],
    atom_mask: List[int],
    atom_radii: List[float],
    layer_offset: int,
    batch_idx: int,
    width: int,
    res: float,
    center: Tuple[float, float, float],
    rot: Tuple[float, float, float, float],
    atom_scale: float,
    atom_shape: int,
    acc_type: int,
    cpu: bool = False,
):
    """Provide a wrapper around cpu_gridify()/gpu_gridify().

    Args:
        grid (np.ndarray): The grid to write to.
        atom_coords (List[Tuple[float, float, float]]): Array containing
            (x,y,z) atom coordinates.
        atom_mask (List[int]): A uint32 array of size atom_num containing a
            destination layer bitmask (i.e. if bit k is set, write atom to
            index k).
        atom_radii (List[float]): A float32 array of size atom_num containing individual
            atomic radius values.
        layer_offset (int): A fixed offset added to each atom layer index.
        batch_idx (int): Index specifiying where to write information.
        width (int): Number of grid points in each dimension.
        res (float): Distance between neighboring grid points in angstroms.
            (1 == gridpoint every angstrom)
            (0.5 == gridpoint every half angstrom, e.g. tighter grid)
        center (List[float]): (x,y,z) coordinate of grid center.
        rot (List[float]): (x,y,z,y) rotation quaternion.
        atom_scale (float): A float32 value specifying the scale of the atoms.
        atom_shape (int): An AtomShapeType specifying the shape of the atoms.
        acc_type (int): An AccType specifying the accumulation type.
        cpu (bool, optional): Whether to use the CPU or GPU implementation.
    """
    # global GPU_DIM
    # global time_debug, cnt

    if cpu:
        cpu_gridify(
            grid,
            len(atom_coords),
            atom_coords,
            atom_mask,
            atom_radii,
            layer_offset,
            batch_idx,
            width,  # num grid points in each direction
            res,
            center,
            rot,
            atom_scale,
            atom_shape,
            acc_type,
        )
    else:
        # import time
        # t1 = time.time()
        # for t in range(50):
        gpu_func = None
        if acc_type == 0:  # AccType.SUM
            if atom_shape == 0:  # AtomShapeType.EXP
                gpu_func = gen_grid_gpu.gpu_gridify_exp_sum
            elif atom_shape == 1:  # AtomShapeType.SPHERE
                gpu_func = gen_grid_gpu.gpu_gridify_sphere_sum
            elif atom_shape == 2:  # AtomShapeType.CUBE
                gpu_func = gen_grid_gpu.gpu_gridify_cube_sum
            elif atom_shape == 3:  # AtomShapeType.GAUSSIAN
                gpu_func = gen_grid_gpu.gpu_gridify_gaussian_sum
            elif atom_shape == 4:  # AtomShapeType.LJ
                gpu_func = gen_grid_gpu.gpu_gridify_lj_sum
            elif atom_shape == 5:  # AtomShapeType.DISCRETE
                gpu_func = gen_grid_gpu.gpu_gridify_discrete_sum
        elif acc_type == 1:  # AccType.MAX
            if atom_shape == 0:  # AtomShapeType.EXP
                gpu_func = gen_grid_gpu.gpu_gridify_exp_max
            elif atom_shape == 1:  # AtomShapeType.SPHERE
                gpu_func = gen_grid_gpu.gpu_gridify_sphere_max
            elif atom_shape == 2:  # AtomShapeType.CUBE
                gpu_func = gen_grid_gpu.gpu_gridify_cube_max
            elif atom_shape == 3:  # AtomShapeType.GAUSSIAN
                gpu_func = gen_grid_gpu.gpu_gridify_gaussian_max
            elif atom_shape == 4:  # AtomShapeType.LJ
                gpu_func = gen_grid_gpu.gpu_gridify_lj_max
            elif atom_shape == 5:  # AtomShapeType.DISCRETE
                gpu_func = gen_grid_gpu.gpu_gridify_discrete_max

        assert gpu_func is not None, "Invalid atom_shape or acc_type"

        dw = ((width - 1) // GPU_DIM) + 1

        # num_blocks, thread_per_block = _get_num_blocks_and_threads(grid.shape[-3] * grid.shape[-2] * grid.shape[-1])

        # gpu_gridify[(dw, dw, dw), (GPU_DIM, GPU_DIM, GPU_DIM)](
        gpu_func[(dw, dw, dw), (GPU_DIM, GPU_DIM, GPU_DIM)](
            # gpu_func[num_blocks, thread_per_block](
            grid,
            len(atom_coords),
            atom_coords,
            atom_mask,
            atom_radii,
            layer_offset,
            batch_idx,
            width,
            res,
            center,
            rot,
            atom_scale,
            # atom_shape,  # commented out for JDD system
            # acc_type,  # commented out for JDD system
        )

        # cnt = cnt + 1
        # if cnt < 100:
        #     time_debug = time_debug + time.time() - t1
        #     print(cnt, time_debug)
