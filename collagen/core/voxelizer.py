import ctypes
from dataclasses import dataclass
from enum import Enum
import math
from typing import List, Any

import torch
import numba
import numba.cuda
import numpy as np
import rdkit

from .featurizer import AtomicNumFeaturizer

GPU_DIM = 8


@dataclass
class VoxelParams(object):
    """
    A VoxelParams object describes how a molecular structure is converted into a voxel tensor.

    Attributes:
        resolution (float): The distance in Angstroms between neighboring grid points. A smaller number means more zoomed-in.
        width (int): The number of gridpoints in each spatial dimension.
        atom_scale (float): A multiplier applied to atomic radii.
        atom_shape: (VoxelParams.AtomShapeType): Describes the atomic density sampling function.
        acc_type: (VoxelParams.AccType): Describes how overlapping atomic densities are handled.
        atom_featurizer (AtomFeaturizer): An atom featurizer that describes how to assign atoms to layers.
    """

    class AtomShapeType(Enum):
        EXP = 0  # simple exponential sphere fill
        SPHERE = 1  # fixed sphere fill
        CUBE = 2  # fixed cube fill
        GAUSSIAN = 3  # continous piecewise expenential fill
        LJ = 4
        DISCRETE = 5

    class AccType(Enum):
        SUM = 0
        MAX = 1

    resolution: float = 1.0
    width: int = 24
    atom_scale: float = 1
    atom_shape: AtomShapeType = AtomShapeType.EXP
    acc_type: AccType = AccType.SUM
    atom_featurizer: "collagen.core.featurizer.AtomFeaturizer" = None

    def validate(self):
        assert self.resolution > 0, f"resolution must be >0 (got {self.resolution})"
        assert self.atom_featurizer is not None, f"atom_featurizer must not be None"

    def tensor_size(self, batch=1, feature_mult=1):
        """
        Compute the required tensor size given the voxel parameters.

        Args:
            batch (int, optional): Number of molecules in the target tensor (default: 1).
            feature_mult (int, optional): Optional multiplier for the channel size.
        """
        N = self.atom_featurizer.size() * feature_mult
        W = self.width
        return (batch, N, W, W, W)


class VoxelParamsDefault(object):
    DeepFrag = VoxelParams(
        resolution=0.75,
        width=24,
        atom_scale=1.75,
        atom_shape=VoxelParams.AtomShapeType.EXP,
        acc_type=VoxelParams.AccType.SUM,
        atom_featurizer=AtomicNumFeaturizer([1, 6, 7, 8, 16]),
    )


def numba_ptr(tensor: "torch.Tensor", cpu: bool = False):
    if cpu:
        return tensor.numpy()

    # Get Cuda context.
    ctx = numba.cuda.cudadrv.driver.driver.get_active_context()

    memory = numba.cuda.cudadrv.driver.MemoryPointer(
        ctx, ctypes.c_ulong(tensor.data_ptr()), tensor.numel() * 4
    )
    cuda_arr = numba.cuda.cudadrv.devicearray.DeviceNDArray(
        tensor.size(),
        [i * 4 for i in tensor.stride()],
        np.dtype("float32"),
        gpu_data=memory,
        stream=torch.cuda.current_stream().cuda_stream,
    )

    return cuda_arr


@numba.cuda.jit
def gpu_gridify(
    grid,
    atom_num,
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
    atom_shape,
    acc_type,
):
    """
    Adds atoms to the grid in a GPU kernel.

    This kernel converts atom coordinate information to 3D voxel information.
    Each GPU thread is responsible for one specific grid point. This function
    receives a list of atomic coordinates and atom layers and simply iterates
    over the list to find nearby atoms and add their effect.

    Voxel information is stored in a 5D tensor of type: BxTxNxNxN where:
        B = Batch size
        N = Number of atom layers
        W = Grid width (in gridpoints)

    Each invocation of this function will write information to a specific batch
    index specified by batch_idx. Additionally, the layer_offset parameter can
    be set to specify a fixed offset to add to each atom_layer item.

    How it works:
    1. Each GPU thread controls a single gridpoint. This gridpoint coordinate
        is translated to a "real world" coordinate by applying rotation and
        translation vectors.
    2. Each thread iterates over the list of atoms and checks for atoms within
        a threshold to add to the grid.

    Args:
        grid: DeviceNDArray tensor where grid information is stored.
        atom_num: Number of atoms.
        atom_coords: Array containing (x,y,z) atom coordinates.
        atom_mask: A uint32 array of size atom_num containing a destination
            layer bitmask (i.e. if bit k is set, write atom to index k).
        atom_radii: A float32 array of size atom_num containing invidiual
            atomic radius values.
        layer_offset: A fixed ofset added to each atom layer index.
        batch_idx: Index specifiying where to write information.
        width: Number of grid points in each dimension.
        res: Distance between neighboring grid points in angstroms.
            (1 == gridpoint every angstrom)
            (0.5 == gridpoint every half angstrom, e.g. tighter grid)
        center: (x,y,z) coordinate of grid center.
        rot: (x,y,z,y) rotation quaternion.
    """
    x, y, z = numba.cuda.grid(3)

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
                    numba.cuda.atomic.add(grid, idx, val)
                elif acc_type == 1:  # AccType.MAX
                    numba.cuda.atomic.max(grid, idx, val)


@numba.jit(nopython=True)
def cpu_gridify(
    grid,
    atom_num,
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
    atom_shape,
    acc_type,
):
    """
    Adds atoms to the grid on the CPU.
    See gpu_gridify() for argument details.
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


def mol_gridify(
    grid,
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
    atom_shape,
    acc_type,
    cpu=False,
):
    """Wrapper around cpu_gridify()/gpu_gridify()."""
    if cpu:
        cpu_gridify(
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
            atom_shape,
            acc_type,
        )
    else:
        dw = ((width - 1) // GPU_DIM) + 1
        gpu_gridify[(dw, dw, dw), (GPU_DIM, GPU_DIM, GPU_DIM)](
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
            atom_shape,
            acc_type,
        )
