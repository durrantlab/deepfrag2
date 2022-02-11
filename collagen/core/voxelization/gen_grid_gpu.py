import numba
import numba.cuda
import math

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
    layer_offset: A fixed offset added to each atom layer index.
    batch_idx: Index specifiying where to write information.
    width: Number of grid points in each dimension.
    res: Distance between neighboring grid points in angstroms.
        (1 == gridpoint every angstrom)
        (0.5 == gridpoint every half angstrom, e.g. tighter grid)
    center: (x,y,z) coordinate of grid center.
    rot: (x,y,z,y) rotation quaternion.
"""


@numba.cuda.jit(device=True, inline=True)
def prepare_grid(width, res, rot, center):
    x, y, z = numba.cuda.grid(3)

    # center grid points around origin. "width" is number of grid points in each
    # dimension.
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
    # NOTE: Surprizingly, simplifying the below given that bw == 0 does not improve speed benchmarks.
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

    return x, y, z, tx, ty, tz


@numba.cuda.jit(device=True, inline=True)
def get_atom(atom_coords, atom_mask, atom_radii, atom_scale, i, tx, ty, tz):
    # fetch atom
    fx, fy, fz = atom_coords[i]
    mask = atom_mask[i]

    r = atom_radii[i] * atom_scale
    r2 = r * r

    # invisible atoms
    if mask == 0:
        return fx, fy, fz, r2, r, mask, False

    # quick cube bounds check to accelerate calculations
    if abs(fx - tx) > r2 or abs(fy - ty) > r2 or abs(fz - tz) > r2:
        return fx, fy, fz, r2, r, mask, False

    return fx, fy, fz, r2, r, mask, True


@numba.cuda.jit(device=True, inline=True)
def add_value_to_layers(mask, batch_idx, layer_offset, acc_type, grid, val, x, y, z):
    # add value to layers
    # ORIG VERISON:
    # for k in range(32):
    #     if (mask >> k) & 1:
    #         idx = (batch_idx, layer_offset + k, x, y, z)
    #         if acc_type == 0:  # AccType.SUM
    #             numba.cuda.atomic.add(grid, idx, val)
    #         elif acc_type == 1:  # AccType.MAX
    #             numba.cuda.atomic.max(grid, idx, val)

    if acc_type == 0:  # AccType.SUM
        for k in range(32):
            if (mask >> k) & 1:
                idx = (batch_idx, layer_offset + k, x, y, z)
                numba.cuda.atomic.add(grid, idx, val)
    elif acc_type == 1:  # AccType.MAX
        for k in range(32):
            if (mask >> k) & 1:
                idx = (batch_idx, layer_offset + k, x, y, z)
                numba.cuda.atomic.max(grid, idx, val)


@numba.cuda.jit()
def gpu_gridify_cube(
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
    acc_type,
):
    x, y, z, tx, ty, tz = prepare_grid(width, res, rot, center)

    i = 0
    while i < atom_num:
        fx, fy, fz, r2, r, mask, valid = get_atom(
            atom_coords, atom_mask, atom_radii, atom_scale, i, tx, ty, tz
        )
        i += 1
        if not valid:
            continue

        # solid cube fill
        val = 1

        # add value to layers
        add_value_to_layers(mask, batch_idx, layer_offset, acc_type, grid, val, x, y, z)


@numba.cuda.jit()
def gpu_gridify_discrete(
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
    acc_type,
):
    x, y, z, tx, ty, tz = prepare_grid(width, res, rot, center)

    i = 0
    while i < atom_num:
        fx, fy, fz, r2, r, mask, valid = get_atom(
            atom_coords, atom_mask, atom_radii, atom_scale, i, tx, ty, tz
        )
        i += 1
        if not valid:
            continue

        # value to add to this gridpoint
        val = 0
        # nearest-gridpoint
        # L1 distance
        if (
            abs(fx - tx) < (res / 2)
            and abs(fy - ty) < (res / 2)
            and abs(fz - tz) < (res / 2)
        ):
            val = 1

        # add value to layers
        add_value_to_layers(mask, batch_idx, layer_offset, acc_type, grid, val, x, y, z)


@numba.cuda.jit()
def gpu_gridify_exp(
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
    acc_type,
):
    x, y, z, tx, ty, tz = prepare_grid(width, res, rot, center)

    i = 0
    while i < atom_num:
        fx, fy, fz, r2, r, mask, valid = get_atom(
            atom_coords, atom_mask, atom_radii, atom_scale, i, tx, ty, tz
        )
        i += 1
        if not valid:
            continue

        # value to add to this gridpoint
        # val = 0
        # exponential sphere fill
        # compute squared distance to atom
        d2 = (fx - tx) ** 2 + (fy - ty) ** 2 + (fz - tz) ** 2
        if d2 > r2:
            continue

        # compute effect
        val = math.exp((-2 * d2) / r2)

        # add value to layers
        add_value_to_layers(mask, batch_idx, layer_offset, acc_type, grid, val, x, y, z)


@numba.cuda.jit()
def gpu_gridify_gaussian(
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
    acc_type,
):
    x, y, z, tx, ty, tz = prepare_grid(width, res, rot, center)

    i = 0
    while i < atom_num:
        fx, fy, fz, r2, r, mask, valid = get_atom(
            atom_coords, atom_mask, atom_radii, atom_scale, i, tx, ty, tz
        )
        i += 1
        if not valid:
            continue

        # value to add to this gridpoint
        val = 0
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

        # add value to layers
        add_value_to_layers(mask, batch_idx, layer_offset, acc_type, grid, val, x, y, z)


@numba.cuda.jit()
def gpu_gridify_lj(
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
    acc_type,
):
    x, y, z, tx, ty, tz = prepare_grid(width, res, rot, center)

    i = 0
    while i < atom_num:
        fx, fy, fz, r2, r, mask, valid = get_atom(
            atom_coords, atom_mask, atom_radii, atom_scale, i, tx, ty, tz
        )
        i += 1
        if not valid:
            continue

        # value to add to this gridpoint
        val = 0
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

        # add value to layers
        add_value_to_layers(mask, batch_idx, layer_offset, acc_type, grid, val, x, y, z)


@numba.cuda.jit()
def gpu_gridify_sphere(
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
    acc_type,
):
    x, y, z, tx, ty, tz = prepare_grid(width, res, rot, center)

    i = 0
    while i < atom_num:
        fx, fy, fz, r2, r, mask, valid = get_atom(
            atom_coords, atom_mask, atom_radii, atom_scale, i, tx, ty, tz
        )
        i += 1
        if not valid:
            continue

        # value to add to this gridpoint
        val = 0
        # solid sphere fill
        # compute squared distance to atom
        d2 = (fx - tx) ** 2 + (fy - ty) ** 2 + (fz - tz) ** 2
        if d2 > r2:
            continue

        val = 1

        # add value to layers
        add_value_to_layers(mask, batch_idx, layer_offset, acc_type, grid, val, x, y, z)

