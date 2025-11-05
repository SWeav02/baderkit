# -*- coding: utf-8 -*-
import numpy as np
from numba import njit, prange

from baderkit.core.utilities.basic import coords_to_flat, wrap_point, wrap_point_w_shift
from baderkit.core.utilities.union_find import (
    find_root_no_compression,
    find_root_with_shift,
    union_w_roots,
    union_with_shift,
)

###############################################################################
# Ring/Cage Identification
###############################################################################


@njit(cache=True)
def get_pt_ring_cage(frac_coords, tol=1e-3):
    """
    Checks if a set of points is similar to a point, ring, or cage. This is a very
    rough estimate made only from the maxima. A more rigorous method might incorporate
    voxels slightly below these values.
    """

    # if we only have 1 or 2 points we cannot make a ring or cage
    if len(frac_coords) < 3:
        return 0
    # if we have exactly 3 we will always have a ring
    elif len(frac_coords) == 3:
        return 1

    # reference coord used for unwrapping
    ref0 = 0.0
    ref1 = 0.0
    ref2 = 0.0
    ref_set = False

    # create an array to store unwrapped coords
    unwrapped_frac_coords = np.empty_like(frac_coords, dtype=np.float64)

    # scan all maxima and pick those that belong to this target_group
    for idx, (c0, c1, c2) in enumerate(frac_coords):

        # first seen -> set reference for unwrapping
        if not ref_set:
            ref0, ref1, ref2 = c0, c1, c2
            ref_set = True

        # unwrap coordinate relative to reference: unwrapped = coord - round(coord - ref)
        # Using np.round via float -> use built-in round for numba compatibility
        # but call round(x) (returns float)
        unwrapped_frac_coords[idx, 0] = c0 - round(c0 - ref0)
        unwrapped_frac_coords[idx, 1] = c1 - round(c1 - ref1)
        unwrapped_frac_coords[idx, 2] = c2 - round(c2 - ref2)

    # get average of points
    avg_pt = np.zeros(3, dtype=np.float64)
    for frac_coord in unwrapped_frac_coords:
        avg_pt += frac_coord
    avg_pt /= len(unwrapped_frac_coords)

    # Center points
    pts = unwrapped_frac_coords - avg_pt
    # Singular value decomposition (variance across each direction)
    _, s, vh = np.linalg.svd(pts, full_matrices=False)
    # Smallest singular value gives out-of-plane deviation
    deviation = s[-1] / s[0]
    if deviation < tol:
        # we are fairly planar and consider this to be a ring
        return 1
    else:
        # we are not planar and consider this a cage
        return 2


###############################################################################
# Saddle Point Identification
###############################################################################


@njit(cache=True)
def trans_to_idx(i, j, k, size):
    val_range = size * 2 + 1
    return (i + size) * (val_range**2) + (j + size) * val_range + (k + size)


@njit(cache=True, inline="always")
def shift_to_index(cx, cy, cz):
    # each value can range from -2 to 2, essentially giving us a base 5 system
    index = (cx + 2) * 25 + (cy + 2) * 5 + (cz + 2)
    return index


@njit(cache=True, inline="always")
def index_to_shift(index):
    cx = index // 25 - 2
    cy = (index % 25) // 5 - 2
    cz = index % 5 - 2
    return cx, cy, cz


@njit(cache=True, inline="always")
def get_connections_in_box(size):
    shifts = []
    lower_connection = []
    upper_connection = []
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            for k in range(-size, size + 1):
                shifts.append((i, j, k))
                idx = trans_to_idx(i, j, k, size)
                for ni in range(-1, 2):
                    for nj in range(-1, 2):
                        for nk in range(-1, 2):
                            # skip center
                            if ni == 0 and nj == 0 and nk == 0:
                                continue
                            # get shifted point
                            si = i + ni
                            sj = j + nj
                            sk = k + nk
                            # skip values outside range
                            if abs(si) > size or abs(sj) > size or abs(sk) > size:
                                continue
                            # get index
                            neigh_idx = trans_to_idx(si, sj, sk, size)
                            # skip previous indices to avoid repeat connections
                            if neigh_idx < idx:
                                continue
                            lower_connection.append(idx)
                            upper_connection.append(neigh_idx)

    lower_connection = np.array(lower_connection, dtype=np.uint16)
    upper_connection = np.array(upper_connection, dtype=np.uint16)
    connections = np.column_stack((lower_connection, upper_connection))
    return np.array(shifts, dtype=np.int8), connections


@njit(cache=True)
def check_if_possible_saddle(
    i,
    j,
    k,
    value,
    data,
    shifts,
    shift_connections,
    greater,
):
    """
    Checks if a point could be a saddle point using the nearest neighboring points
    (3x3x3).
    """
    # check if there are at least 2 groups in the immediate neighborhood if we
    # were to allow values including the central one
    nx, ny, nz = data.shape

    # create trackers for neighbors
    value_mask = np.zeros(27, dtype=np.bool_)
    connections = np.arange(27, dtype=np.uint8)
    root_mask = np.zeros(27, dtype=np.bool_)

    # mark mask
    no_same = True
    for shift_idx, (si, sj, sk) in enumerate(shifts):
        # skip center so that it is never in mask
        if shift_idx == 13:
            continue
        ni, nj, nk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
        neigh_value = data[ni, nj, nk]
        if (greater and neigh_value > value) or (not greater and neigh_value < value):
            # mark in mask and instantiate as a root
            value_mask[shift_idx] = True
            root_mask[shift_idx] = True
        elif neigh_value == value:
            no_same = False
            break

    # if any values are the same as our center, we need a larger neighborhood
    # to be sure if this region is a bif
    if not no_same:
        return True
    # iterate over connections and make unions
    for connection_idx, (shift1, shift2) in enumerate(shift_connections):
        # skip if one is not in mask
        if not value_mask[shift1] or not value_mask[shift2]:
            continue
        # mark union and reduce roots
        union_w_roots(connections, shift1, shift2, root_mask)

    # if there is more than one root, this could be a bifurcation
    maybe_bif = np.count_nonzero(root_mask) > 1

    return maybe_bif


@njit(cache=True)
def check_if_saddle(
    i,
    j,
    k,
    value,
    data,
    shifts,
    shift_connections,
    greater,
):
    """
    Checks if a point is a saddle by grouping connected surrounding points (5x5x5) with
    and without the central points value. If the number of groups change this
    is a potential bifurcation.
    """
    nx, ny, nz = data.shape

    # create trackers for neighbors
    value_mask = np.zeros(125, dtype=np.uint8)
    connections = np.arange(125, dtype=np.uint8)
    root_mask = np.zeros(125, dtype=np.bool_)

    # mark mask
    for shift_idx, (si, sj, sk) in enumerate(shifts):
        ni, nj, nk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
        neigh_value = data[ni, nj, nk]
        if (greater and neigh_value > value) or (not greater and neigh_value < value):
            value_mask[shift_idx] = 2
            root_mask[shift_idx] = True

        elif neigh_value == value:
            value_mask[shift_idx] = 1

    include_indices = []
    # iterate over connections and make unions
    for connection_idx, (shift1, shift2) in enumerate(shift_connections):
        connection_type = min(value_mask[shift1], value_mask[shift2])
        if connection_type == 0:
            continue
        elif connection_type == 2:
            union_w_roots(connections, shift1, shift2, root_mask)
        elif connection_type == 1:
            include_indices.append(connection_idx)

    n_exclude_groups = np.count_nonzero(root_mask)

    # make unions for exact vals
    for connection_idx in include_indices:
        shift1, shift2 = shift_connections[connection_idx]
        union_w_roots(connections, shift1, shift2, root_mask)
    n_include_groups = np.count_nonzero(root_mask)

    is_bif = n_exclude_groups != n_include_groups

    return is_bif


@njit(parallel=True, cache=True)
def find_potential_saddle_points(data, edge_mask, greater=False):
    """
    Finds all points in the grid that might be saddle points. Generally overestimates
    the actual number of points.
    """
    nx, ny, nz = data.shape

    bif_mask = np.zeros_like(data, dtype=np.bool_)

    # we want to find points that connect domains or voids.
    # Imagine creating an isosurface and constructing solids from the values
    # above (below) it. A bifurcation occurs at a point where two solids
    # change from being connected to disconnected if the value increases (decreases)

    # For speed, we map out possible connections for each neighbor. We get
    # neighbor connections for first and second neighbors (3x3x3, 5x5x5)
    trans3, trans_connections3 = get_connections_in_box(1)
    trans5, trans_connections5 = get_connections_in_box(2)

    # loop over each edge point and find bifurcation voxels
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):

                # skip anything not on the edge
                if not edge_mask[i, j, k]:
                    continue

                value = data[i, j, k]

                # do a first check with the nearest neighbors to see if this
                # has the potential to be a bifurcation
                if not check_if_possible_saddle(
                    i, j, k, value, data, trans3, trans_connections3, greater
                ):
                    continue

                # check if point is a potential bifurcation
                is_bif = check_if_saddle(
                    i, j, k, value, data, trans5, trans_connections5, greater
                )

                if is_bif:
                    bif_mask[i, j, k] = True

    return bif_mask


###############################################################################
# Labeling and Dimensionality
###############################################################################


@njit(parallel=False, cache=True)
def find_periodic_cycles(
    solid,
    previous_solid,
    old_cycles,
    parent,
    offset_x,
    offset_y,
    offset_z,
    old_roots,
    root_mask,
    neighbors,
):
    """
    Finds the cycles a labeled solid makes through a periodic cell. Allows for
    continuation from a previous smaller solid for speed.
    """
    nx, ny, nz = solid.shape
    ny_nz = ny * nz

    # get the current roots
    new_roots = np.nonzero(root_mask)[0]
    n_roots = len(new_roots)

    # create a new list for cycles in the current mask
    new_cycles = []
    for i in range(n_roots):
        cycle_list = [np.array((-1, -1, -1), dtype=np.float64)]
        cycle_list = cycle_list[1:]
        new_cycles.append(cycle_list)

    # add cycles from the previous round
    for old_root, old_cycle in zip(old_roots, old_cycles):
        if len(old_cycle) == 0:
            continue
        # get the new root
        new_root = find_root_no_compression(parent, old_root)
        new_root_idx = np.searchsorted(new_roots, new_root)
        # update to include previous values
        for cycle in old_cycle:
            new_cycles[new_root_idx].append(cycle)

    # now iterate over new points and find new cycles
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not solid[i, j, k] or previous_solid[i, j, k]:
                    continue
                idx = coords_to_flat(i, j, k, ny_nz, nz)

                # find root
                ra, ox, oy, oz = find_root_with_shift(
                    parent, offset_x, offset_y, offset_z, idx
                )
                root_idx = np.searchsorted(new_roots, ra)
                cycles = new_cycles[root_idx]
                if len(cycles) == 3:
                    # This root is already 3d and we can continue
                    continue

                for di, dj, dk in neighbors:
                    ni, nj, nk, si, sj, sk = wrap_point_w_shift(
                        i + di, j + dj, k + dk, nx, ny, nz
                    )
                    if not solid[ni, nj, nk]:
                        continue
                    neigh_idx = coords_to_flat(ni, nj, nk, ny_nz, nz)
                    rb, ox1, oy1, oz1 = find_root_with_shift(
                        parent, offset_x, offset_y, offset_z, neigh_idx
                    )

                    # if not the same root, no cycle can be formed
                    if ra != rb:
                        continue

                    # calculate total cycle offset
                    cx = ox - ox1 - si
                    cy = oy - oy1 - sj
                    cz = oz - oz1 - sk

                    if cx == 0 and cy == 0 and cz == 0:
                        continue

                    # Check if this is a new linearly independent cycle
                    # Project onto existing basis
                    for q1, q2, q3 in cycles:
                        proj = cx * q1 + cy * q2 + cz * q3
                        cx = cx - proj * q1
                        cy = cy - proj * q2
                        cz = cz - proj * q3

                    # calculate norm
                    norm = (cx**2 + cy**2 + cz**2) ** 0.5
                    if norm > 1e-12:  # independent
                        v = np.array((cx, cy, cz), dtype=np.float64)
                        # Normalize and add to orthogonal basis
                        v = v / norm
                        cycles.append(v)

    return new_cycles, new_roots


@njit(cache=True)
def get_domain_dimensionality(
    parent,
    offset_x,
    offset_y,
    offset_z,
    roots,
    dims,
    domain_point,
    ny_nz,
    nz,
):
    """
    Gets the dimensionality of a domain given a point lying in it
    """
    x, y, z = domain_point
    idx = coords_to_flat(x, y, z, ny_nz, nz)
    # get root
    root_idx, _, _, _ = find_root_with_shift(parent, offset_x, offset_y, offset_z, idx)
    # check root dimensionality
    final_dim = 0
    for root, dim in zip(roots, dims):
        if root == root_idx:
            final_dim = dim
            break
    return final_dim


@njit(cache=True)
def get_connected_groups(
    solid,
    previous_solid,
    root_mask,
    roots,
    cycles,
    parent,
    offset_x,
    offset_y,
    offset_z,
    size,
    neighbors,
):
    """
    Finds unions and periodic offsets for points in a periodic solid. Slower than
    scipy's implementation, but allows for iterative updates with increasingly
    large solids.
    """
    nx, ny, nz = solid.shape
    ny_nz = ny * nz

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):

                # NOTE: Doing a check like this has such a small time cost
                # that I didn't see a difference between doing a mock lookup/continue
                # for a 30^3 cube and 400^3 cube.
                if not solid[i, j, k] or previous_solid[i, j, k]:
                    continue

                idx = coords_to_flat(i, j, k, ny_nz, nz)

                found_neigh = False
                for di, dj, dk in neighbors:
                    # get wrapped neighbor and get any shift across a periodic
                    # boundary
                    ni, nj, nk, si, sj, sk = wrap_point_w_shift(
                        i + di, j + dj, k + dk, nx, ny, nz
                    )

                    if not solid[ni, nj, nk]:
                        continue
                    found_neigh = True
                    # get neighbors flattened index
                    neigh_idx = coords_to_flat(ni, nj, nk, ny_nz, nz)

                    # accumulate offset/shift and check for cycle
                    union_with_shift(
                        root_mask,
                        parent,
                        offset_x,
                        offset_y,
                        offset_z,
                        size,
                        idx,
                        neigh_idx,
                        si,
                        sj,
                        sk,
                    )

                # if we have no neighbors, we are a root with ourself
                if not found_neigh:
                    root_mask[idx] = True

    # Now we find the unique cycles that occur for each root. Each cycle can
    # only take the form (cx, cy, cz) where each value is in -2, -1, 0, 1, 2.
    # This is essentially a base 5 system and we can convert any cycle to one
    # of 125 possible values. Thus we need an array of shape len(roots)x125
    cycles, roots = find_periodic_cycles(
        solid=solid,
        previous_solid=previous_solid,
        old_cycles=cycles,
        parent=parent,
        offset_x=offset_x,
        offset_y=offset_y,
        offset_z=offset_z,
        old_roots=roots,
        root_mask=root_mask,
        neighbors=neighbors,
    )

    # Now get dimensionalities of each root
    # dimensionalities, cycles = get_root_dims(cycles)
    dimensionalities = np.zeros(len(cycles), dtype=np.uint8)
    for idx, cycle in enumerate(cycles):
        dimensionalities[idx] = len(cycle)

    return (
        root_mask,
        parent,
        offset_x,
        offset_y,
        offset_z,
        roots,
        cycles,
        dimensionalities,
    )
