
import ctypes
from dataclasses import dataclass
from enum import Enum
import math

import torch
import numba
import numba.cuda
import numpy as np


GPU_DIM = 8


@dataclass
class VoxelParams(object):

    class PointType(Enum):
        EXP = 0       # simple exponential sphere fill
        SPHERE = 1    # fixed sphere fill
        CUBE = 2      # fixed cube fill
        GAUSSIAN = 3  # continous piecewise expenential fill
        LJ = 4
        DISCRETE = 5

    class AccType(Enum):
        SUM = 0
        MAX = 1
    
    # Distance in Angstroms between neighboring grid points. A smaller number
    # means a higher resolution.
    resolution: float = 1.0

    # Total number of gridpoints per dimension.
    width: int = 24

    # Effective atomic radius of atoms. (in Angstroms)
    point_radius: float = 1.75

    # The point type describes how individual atoms are represented.
    point_type: PointType = PointType.EXP

    # The accumulation type describes how overlapping atomic densities are handled. 
    acc_type: AccType = AccType.SUM

    # Specifies how atoms should be represented in tensor layers.
    atom_featurizer: 'AtomFeaturizer' = None

    def validate(self):
        assert self.resolution > 0, f'resolution must be >0 (got {self.resolution})'
        assert self.atom_featurizer is not None, f'atom_featurizer must not be None'

    def tensor_size(self, batch=1):
        N = self.atom_featurizer.size()
        W = self.width
        return (batch,N,W,W,W)


def numba_ptr(tensor: 'torch.Tensor', cpu: bool = False):
    if cpu:
        return tensor.numpy()

    # Get Cuda context.
    ctx = numba.cuda.cudadrv.driver.driver.get_active_context()

    memory = numba.cuda.cudadrv.driver.MemoryPointer(
        ctx, 
        ctypes.c_ulong(tensor.data_ptr()), tensor.numel() * 4)
    cuda_arr = numba.cuda.cudadrv.devicearray.DeviceNDArray(
        tensor.size(),
        [i*4 for i in tensor.stride()],
        np.dtype('float32'),
        gpu_data=memory,
        stream=torch.cuda.current_stream().cuda_stream
    )

    return cuda_arr

# def shared_tensor(shape, cpu: bool = False):
#     """Creates a PyTorch Tensor and Numba array with shared GPU memory backing.

#     Args:
#         shape: The shape of the array.
#         cpu: If True, just instantiate and return a numpy array.

#     Returns:
#         (torch_arr, cuda_arr)
#     """

#     if cpu:
#         t = np.empty(shape=shape, dtype=np.float32)
#         return (t,t)

#     # Get Cuda context.
#     ctx = numba.cuda.cudadrv.driver.driver.get_active_context()

#     # Set up tensor on GPU.
#     t = torch.zeros(size=shape, dtype=torch.float32).cuda()

#     memory = numba.cuda.cudadrv.driver.MemoryPointer(ctx, ctypes.c_ulong(t.data_ptr()), t.numel() * 4)
#     cuda_arr = numba.cuda.cudadrv.devicearray.DeviceNDArray(
#         t.size(),
#         [i*4 for i in t.stride()],
#         np.dtype('float32'),
#         gpu_data=memory,
#         stream=torch.cuda.current_stream().cuda_stream
#     )

#     return (t, cuda_arr)


@numba.cuda.jit
def gpu_gridify(grid, atom_num, atom_coords, atom_mask, layer_offset,
                batch_idx, width, res, center, rot,
                point_radius, point_type, acc_type
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
        layer_offset: A fixed ofset added to each atom layer index.
        batch_idx: Index specifiying where to write information.
        width: Number of grid points in each dimension.
        res: Distance between neighboring grid points in angstroms.
            (1 == gridpoint every angstrom)
            (0.5 == gridpoint every half angstrom, e.g. tighter grid)
        center: (x,y,z) coordinate of grid center.
        rot: (x,y,z,y) rotation quaternion.
    """
    x,y,z = numba.cuda.grid(3)

    # center around origin
    tx = x - (width/2)
    ty = y - (width/2)
    tz = z - (width/2)

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
        i += 1

        # invisible atoms
        if mask == 0:
            continue

        # point radius squared
        r = point_radius
        r2 = point_radius * point_radius

        # quick cube bounds check
        if abs(fx-tx) > r2 or abs(fy-ty) > r2 or abs(fz-tz) > r2:
            continue

        # value to add to this gridpoint
        val = 0

        if point_type == 0: # PointType.EXP
            # exponential sphere fill
            # compute squared distance to atom
            d2 = (fx-tx)**2 + (fy-ty)**2 + (fz-tz)**2
            if d2 > r2:
                continue

            # compute effect
            val = math.exp((-2 * d2) / r2)
        elif point_type == 1: # PointType.SPHERE
            # solid sphere fill
            # compute squared distance to atom
            d2 = (fx-tx)**2 + (fy-ty)**2 + (fz-tz)**2
            if d2 > r2:
                continue

            val = 1
        elif point_type == 2: # PointType.CUBE
            # solid cube fill
            val = 1
        elif point_type == 3: # PointType.GAUSSIAN
            # (Ragoza, 2016)
            #
            # piecewise gaussian sphere fill
            # compute squared distance to atom
            d2 = (fx-tx)**2 + (fy-ty)**2 + (fz-tz)**2
            d = math.sqrt(d2)

            if d > r * 1.5:
                continue
            elif d > r:
                val = math.exp(-2.0) * ( (4*d2/r2) - (12*d/r) + 9 )
            else:
                val = math.exp((-2 * d2) / r2)
        elif point_type == 4: # PointType.LJ
            # (Jimenez, 2017) - DeepSite
            #
            # LJ potential
            # compute squared distance to atom
            d2 = (fx-tx)**2 + (fy-ty)**2 + (fz-tz)**2
            d = math.sqrt(d2)

            if d > r * 1.5:
                continue
            else:
                val = 1 - math.exp(-((r/d)**12))
        elif point_type == 5: # PointType.DISCRETE
            # nearest-gridpoint
            # L1 distance
            if abs(fx-tx) < (res/2) and abs(fy-ty) < (res/2) and abs(fz-tz) < (res/2):
                val = 1

        # add value to layers
        for k in range(32):
            if (mask >> k) & 1:
                idx = (batch_idx, layer_offset+k, x, y, z)
                if acc_type == 0: # AccType.SUM
                    numba.cuda.atomic.add(grid, idx, val)
                elif acc_type == 1: # AccType.MAX
                    numba.cuda.atomic.max(grid, idx, val)


@numba.jit(nopython=True)
def cpu_gridify(grid, atom_num, atom_coords, atom_mask, layer_offset,
                batch_idx, width, res, center, rot,
                point_radius, point_type, acc_type
                ):
    """
    Adds atoms to the grid on the CPU.
    See gpu_gridify() for argument details.
    """
    for x in range(width):
        for y in range(width):
            for z in range(width):

                # center around origin
                tx = x - (width/2)
                ty = y - (width/2)
                tz = z - (width/2)

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
                    i += 1

                    # invisible atoms
                    if mask == 0:
                        continue

                    # point radius squared
                    r = point_radius
                    r2 = point_radius * point_radius

                    # quick cube bounds check
                    if abs(fx-tx) > r2 or abs(fy-ty) > r2 or abs(fz-tz) > r2:
                        continue

                    # value to add to this gridpoint
                    val = 0

                    if point_type == 0: # PointType.EXP
                        # exponential sphere fill
                        # compute squared distance to atom
                        d2 = (fx-tx)**2 + (fy-ty)**2 + (fz-tz)**2
                        if d2 > r2:
                            continue

                        # compute effect
                        val = math.exp((-2 * d2) / r2)
                    elif point_type == 1: # PointType.SPHERE
                        # solid sphere fill
                        # compute squared distance to atom
                        d2 = (fx-tx)**2 + (fy-ty)**2 + (fz-tz)**2
                        if d2 > r2:
                            continue

                        val = 1
                    elif point_type == 2: # PointType.CUBE
                        # solid cube fill
                        val = 1
                    elif point_type == 3: # PointType.GAUSSIAN
                        # (Ragoza, 2016)
                        #
                        # piecewise gaussian sphere fill
                        # compute squared distance to atom
                        d2 = (fx-tx)**2 + (fy-ty)**2 + (fz-tz)**2
                        d = math.sqrt(d2)

                        if d > r * 1.5:
                            continue
                        elif d > r:
                            val = math.exp(-2.0) * ( (4*d2/r2) - (12*d/r) + 9 )
                        else:
                            val = math.exp((-2 * d2) / r2)
                    elif point_type == 4: # PointType.LJ
                        # (Jimenez, 2017) - DeepSite
                        #
                        # LJ potential
                        # compute squared distance to atom
                        d2 = (fx-tx)**2 + (fy-ty)**2 + (fz-tz)**2
                        d = math.sqrt(d2)

                        if d > r * 1.5:
                            continue
                        else:
                            val = 1 - math.exp(-((r/d)**12))
                    elif point_type == 5: # PointType.DISCRETE
                        # nearest-gridpoint
                        # L1 distance
                        if abs(fx-tx) < (res/2) and abs(fy-ty) < (res/2) and abs(fz-tz) < (res/2):
                            val = 1

                    # add value to layers
                    for k in range(32):
                        if (mask >> k) & 1:
                            idx = (batch_idx, layer_offset+k, x, y, z)
                            if acc_type == 0: # AccType.SUM
                                grid[idx] += val
                            elif acc_type == 1: # AccType.MAX
                                grid[idx] = max(grid[idx], val)


def mol_gridify(grid, atom_coords, atom_mask, layer_offset, batch_idx,
                width, res, center, rot, point_radius, point_type, acc_type,
                cpu=False):
    """Wrapper around cpu_gridify()/gpu_gridify()."""
    if cpu:
        cpu_gridify(
            grid, len(atom_coords), atom_coords, atom_mask, layer_offset,
            batch_idx, width, res, center, rot, point_radius, point_type, acc_type
        )
    else:
        dw = ((width - 1) // GPU_DIM) + 1
        gpu_gridify[(dw,dw,dw), (GPU_DIM,GPU_DIM,GPU_DIM)](
            grid, len(atom_coords), atom_coords, atom_mask, layer_offset,
            batch_idx, width, res, center, rot, point_radius, point_type, acc_type
        )




def rand_rot():
    """Returns a random uniform quaternion rotation."""
    q = np.random.normal(size=4) # sample quaternion from normal distribution
    q = q / np.sqrt(np.sum(q**2)) # normalize
    return q


# def get_batch(data, batch_size=16, batch_set=None, width=48, res=0.5,
#               ignore_receptor=False, ignore_parent=False, fixed_rot=None,
#               point_radius=2, point_type=POINT_TYPE.EXP,
#               acc_type=ACC_TYPE.SUM):
#     """Builds a batch grid from a FragmentDataset.

#     Args:
#         data: a FragmentDataset object
#         rec_channels: number of receptor channels
#         parent_channels: number of parent channels
#         batch_size: size of the batch
#         batch_set: if not None, specify a list of data indexes to use for each
#             item in the batch
#         width: grid width
#         res: grid resolution
#         ignore_receptor: if True, ignore receptor atoms
#         ignore_parent: if True, ignore parent atoms

#     Returns: (torch_grid, batch_set)
#         torch_grid: pytorch Tensor with voxel information
#         examples: list of examples used
#     """
#     assert (not (ignore_receptor and ignore_parent)), "Can't ignore parent and receptor!"

#     batch_size = int(batch_size)
#     width = int(width)

#     rec_channels = data.rec_layers()
#     lig_channels = data.lig_layers()

#     dim = 0
#     if not ignore_receptor:
#         dim += rec_channels
#     if not ignore_parent:
#         dim += lig_channels

#     # create a tensor with shared memory on the gpu
#     torch_grid, cuda_grid = make_tensor((batch_size, dim, width, width, width))

#     if batch_set is None:
#         batch_set = np.random.choice(len(data), size=batch_size, replace=False)

#     examples = [data[idx] for idx in batch_set]

#     for i in range(len(examples)):
#         example = examples[i]
#         rot = fixed_rot
#         if rot is None:
#             rot = rand_rot()

#         if ignore_receptor:
#             mol_gridify(
#                 cuda_grid,
#                 example['p_coords'],
#                 example['p_types'],
#                 layer_offset=0,
#                 batch_idx=i,
#                 width=width,
#                 res=res,
#                 center=example['conn'],
#                 rot=rot,
#                 point_radius=point_radius,
#                 point_type=point_type,
#                 acc_type=acc_type
#             )
#         elif ignore_parent:
#             mol_gridify(
#                 cuda_grid,
#                 example['r_coords'],
#                 example['r_types'],
#                 layer_offset=0,
#                 batch_idx=i,
#                 width=width,
#                 res=res,
#                 center=example['conn'],
#                 rot=rot,
#                 point_radius=point_radius,
#                 point_type=point_type,
#                 acc_type=acc_type
#             )
#         else:
#             mol_gridify(
#                 cuda_grid,
#                 example['p_coords'],
#                 example['p_types'],
#                 layer_offset=0,
#                 batch_idx=i,
#                 width=width,
#                 res=res,
#                 center=example['conn'],
#                 rot=rot,
#                 point_radius=point_radius,
#                 point_type=point_type,
#                 acc_type=acc_type
#             )
#             mol_gridify(
#                 cuda_grid,
#                 example['r_coords'],
#                 example['r_types'],
#                 layer_offset=lig_channels,
#                 batch_idx=i,
#                 width=width,
#                 res=res,
#                 center=example['conn'],
#                 rot=rot,
#                 point_radius=point_radius,
#                 point_type=point_type,
#                 acc_type=acc_type
#             )

#     return torch_grid, examples


# def get_raw_batch(r_coords, r_types, p_coords, p_types, rec_typer, lig_typer,
#                   conn, num_samples=32, width=24, res=1, fixed_rot=None,
#                   point_radius=1.5, point_type=0, acc_type=0, cpu=False):
#     """Sample a raw batch with provided atom coordinates.

#     Args:
#         r_coords: receptor coordinates
#         r_types: receptor types (layers)
#         p_coords: parent coordinates
#         p_types: parent types (layers)
#         conn: (x,y,z) connection point
#         num_samples: number of rotations to sample
#         width: grid width
#         res: grid resolution
#         fixed_rot: None or a fixed 4-element rotation vector
#         point_radius: atom radius in Angstroms
#         point_type: shape of the atom densities
#         acc_type: atom density accumulation type
#         cpu: if True, generate batches with cpu_gridify
#     """
#     B = num_samples
#     T = rec_typer.size() + lig_typer.size()
#     N = width

#     if cpu:
#         t = np.zeros((B,T,N,N,N))
#         torch_grid = t
#         cuda_grid = t
#     else:
#         torch_grid, cuda_grid = make_tensor((B,T,N,N,N))

#     r_mask = np.zeros(len(r_types), dtype=np.uint32)
#     p_mask = np.zeros(len(p_types), dtype=np.uint32)

#     for i in range(len(r_types)):
#         r_mask[i] = rec_typer.apply(*r_types[i])

#     for i in range(len(p_types)):
#         p_mask[i] = lig_typer.apply(*p_types[i])

#     for i in range(num_samples):
#         rot = fixed_rot
#         if rot is None:
#             rot = rand_rot()

#         mol_gridify(
#             cuda_grid,
#             p_coords,
#             p_mask,
#             layer_offset=0,
#             batch_idx=i,
#             width=width,
#             res=res,
#             center=conn,
#             rot=rot,
#             point_radius=point_radius,
#             point_type=point_type,
#             acc_type=acc_type,
#             cpu=cpu
#         )

#         mol_gridify(
#             cuda_grid,
#             r_coords,
#             r_mask,
#             layer_offset=lig_typer.size(),
#             batch_idx=i,
#             width=width,
#             res=res,
#             center=conn,
#             rot=rot,
#             point_radius=point_radius,
#             point_type=point_type,
#             acc_type=acc_type,
#             cpu=cpu
#         )

#     return torch_grid
