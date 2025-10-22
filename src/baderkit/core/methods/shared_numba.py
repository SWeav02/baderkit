# -*- coding: utf-8 -*-

import math

import numpy as np
from numba import njit, prange  # , types
from numpy.typing import NDArray

from baderkit.core.toolkit.grid_numba import linear_slice

###############################################################################
# General methods
###############################################################################


@njit(fastmath=True, cache=True, inline="always")
def flat_to_coords(idx, ny_nz, nz):
    i = idx // (ny_nz)
    j = (idx % (ny_nz)) // nz
    k = idx % nz
    return i, j, k


@njit(fastmath=True, cache=True, inline="always")
def coords_to_flat(i, j, k, ny_nz, nz):
    return i * (ny_nz) + j * nz + k


@njit(cache=True, inline="always")
def get_best_neighbor(
    data: NDArray[np.float64],
    i: np.int64,
    j: np.int64,
    k: np.int64,
    neighbor_transforms: NDArray[np.int64],
    neighbor_dists: NDArray[np.int64],
):
    """
    For a given coordinate (i,j,k) in a grid (data), finds the neighbor with
    the largest gradient.

    Parameters
    ----------
    data : NDArray[np.float64]
        The data for each voxel.
    i : np.int64
        First coordinate
    j : np.int64
        Second coordinate
    k : np.int64
        Third coordinate
    neighbor_transforms : NDArray[np.int64]
        Transformations to apply to get to the voxels neighbors
    neighbor_dists : NDArray[np.int64]
        The distance to each voxels neighbor

    Returns
    -------
    best_transform : NDArray[np.int64]
        The transformation to the best neighbor
    best_neigh : NDArray[np.int64]
        The coordinates of the best neigbhor

    """
    nx, ny, nz = data.shape
    # get the value at this point
    base = data[i, j, k]
    # create a tracker for the best increase in value
    best = 0.0
    # create initial best transform. Default to this point
    bti = 0
    btj = 0
    btk = 0
    # create initial best neighbor
    bni = i
    bnj = j
    bnk = k
    # For each neighbor get the difference in value and if its better
    # than any previous, replace the current best
    for (si, sj, sk), dist in zip(neighbor_transforms, neighbor_dists):
        # loop
        ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
        # calculate the difference in value taking into account distance
        diff = (data[ii, jj, kk] - base) / dist
        # if better than the current best, note the best and the
        # current label
        if diff > best:
            best = diff
            bti, btj, btk = (si, sj, sk)
            bni, bnj, bnk = (ii, jj, kk)

    # return the best shift and neighbor
    return (
        np.array((bti, btj, btk), dtype=np.int64),
        np.array((bni, bnj, bnk), dtype=np.int64),
    )


@njit(parallel=True, cache=True)
def get_edges(
    labeled_array: NDArray[np.int64],
    neighbor_transforms: NDArray[np.int64],
    vacuum_mask: NDArray[np.bool_],
):
    """
    In a 3D array of labeled voxels, finds the voxels that neighbor at
    least one voxel with a different label.

    Parameters
    ----------
    labeled_array : NDArray[np.int64]
        A 3D array where each entry represents the basin label of the point.
    neighbor_transforms : NDArray[np.int64]
        The transformations from each voxel to its neighbors.
    vacuum_mask : NDArray[np.bool_]
        A 3D array representing the location of the vacuum

    Returns
    -------
    edges : NDArray[np.bool_]
        A mask with the same shape as the input grid that is True at points
        on basin edges.

    """
    nx, ny, nz = labeled_array.shape
    # create 3D array to store edges
    edges = np.zeros_like(labeled_array, dtype=np.bool_)
    # loop over each voxel in parallel
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # if this voxel is part of the vacuum, continue
                if vacuum_mask[i, j, k]:
                    continue
                # get this voxels label
                label = labeled_array[i, j, k]
                # iterate over the neighboring voxels
                for si, sj, sk in neighbor_transforms:
                    # wrap points
                    ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
                    # get neighbors label
                    neigh_label = labeled_array[ii, jj, kk]
                    # if any label is different, the current voxel is an edge.
                    # Note this in our edge array and break
                    # NOTE: we also check that the neighbor is not part of the
                    # vacuum
                    if neigh_label != label and not vacuum_mask[ii, jj, kk]:
                        edges[i, j, k] = True
                        break
    return edges


@njit(parallel=True, cache=True)
def get_maxima(
    data: NDArray[np.float64],
    neighbor_transforms: NDArray[np.int64],
    vacuum_mask: NDArray[np.bool_],
    use_minima: bool = False,
):
    """
    For a 3D array of data, return a mask that is True at local maxima.

    Parameters
    ----------
    data : NDArray[np.float64]
        A 3D array of data.
    neighbor_transforms : NDArray[np.int64]
        The transformations from each voxel to its neighbors.
    vacuum_mask : NDArray[np.bool_]
        A 3D array representing the location of the vacuum
    use_minima : bool, optional
        Whether or not to search for minima instead of maxima.

    Returns
    -------
    maxima : NDArray[np.bool_]
        A mask with the same shape as the input grid that is True at points
        that are local maxima.

    """
    nx, ny, nz = data.shape
    # create 3D array to store maxima
    maxima = np.zeros_like(data, dtype=np.bool_)
    # loop over each voxel in parallel
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # if this voxel is part of the vacuum, continue
                if vacuum_mask[i, j, k]:
                    continue
                # get this voxels value
                value = data[i, j, k]
                is_max = True
                # iterate over the neighboring voxels
                for si, sj, sk in neighbor_transforms:
                    # wrap points
                    ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
                    if not use_minima:
                        if data[ii, jj, kk] > value:
                            is_max = False
                            break
                    else:
                        if data[ii, jj, kk] < value:
                            is_max = False
                            break
                if is_max:
                    maxima[i, j, k] = True
    return maxima


@njit(fastmath=True, cache=True)
def get_basin_charges_and_volumes(
    data: NDArray[np.float64],
    labels: NDArray[np.int64],
    cell_volume: np.float64,
    maxima_num: np.int64,
):
    nx, ny, nz = data.shape
    total_points = nx * ny * nz
    # create variables to store charges/volumes
    charges = np.zeros(maxima_num, dtype=np.float64)
    volumes = np.zeros(maxima_num, dtype=np.float64)
    vacuum_charge = 0.0
    vacuum_volume = 0.0
    # iterate in parallel over each voxel
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                charge = data[i, j, k]
                label = labels[i, j, k]
                if label < 0:
                    vacuum_charge += charge
                    vacuum_volume += 1
                else:
                    charges[label] += charge
                    volumes[label] += 1.0
    # calculate charge and volume
    volumes = volumes * cell_volume / total_points
    charges = charges / total_points
    vacuum_volume = vacuum_volume * cell_volume / total_points
    vacuum_charge = vacuum_charge / total_points
    return charges, volumes, vacuum_charge, vacuum_volume


@njit(cache=True, inline="always")
def wrap_point(
    i: np.int64, j: np.int64, k: np.int64, nx: np.int64, ny: np.int64, nz: np.int64
) -> tuple[np.int64, np.int64, np.int64]:
    """
    Wraps a 3D point (i, j, k) into the periodic bounds defined by the grid dimensions (nx, ny, nz).

    If any of the input coordinates are outside the bounds [0, nx), [0, ny), or [0, nz),
    they are wrapped around using periodic boundary conditions.

    Parameters
    ----------
    i : np.int64
        x-index of the point.
    j : np.int64
        y-index of the point.
    k : np.int64
        z-index of the point.
    nx : np.int64
        Number of grid points along x-direction.
    ny : np.int64
        Number of grid points along y-direction.
    nz : np.int64
        Number of grid points along z-direction.

    Returns
    -------
    tuple[np.int64, np.int64, np.int64]
        The wrapped (i, j, k) indices within the bounds.
    """
    if i >= nx:
        i -= nx
    elif i < 0:
        i += nx
    if j >= ny:
        j -= ny
    elif j < 0:
        j += ny
    if k >= nz:
        k -= nz
    elif k < 0:
        k += nz
    return i, j, k


@njit(cache=True, inline="always")
def get_gradient_simple(
    data: NDArray[np.float64],
    voxel_coord: NDArray[np.int64],
    # car2lat: NDArray[np.float64],
    dir2lat: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.int64], np.bool_]:
    """
    Peforms a neargrid step from the provided voxel coordinate.

    Parameters
    ----------
    data : NDArray[np.float64]
        A 3D grid of values for each point.
    voxel_coord : NDArray[np.int64]
        The point to make the step from.
    car2lat : NDArray[np.float64]
        A matrix that converts a coordinate in cartesian space to fractional
        space.

    Returns
    -------
    charge_grad_frac : NDArray[np.float64]
        The gradient in direct space at this voxel coord

    """
    nx, ny, nz = data.shape
    i, j, k = voxel_coord
    # calculate the gradient at this point in voxel coords
    charge000 = data[i, j, k]
    charge001 = data[i, j, (k + 1) % nz]
    charge010 = data[i, (j + 1) % ny, k]
    charge100 = data[(i + 1) % nx, j, k]
    charge00_1 = data[i, j, (k - 1) % nz]
    charge0_10 = data[i, (j - 1) % ny, k]
    charge_100 = data[(i - 1) % nx, j, k]

    gi = (charge100 - charge_100) / 2.0
    gj = (charge010 - charge0_10) / 2.0
    gk = (charge001 - charge00_1) / 2.0

    if charge100 <= charge000 and charge_100 <= charge000:
        gi = 0.0
    if charge010 <= charge000 and charge0_10 <= charge000:
        gj = 0.0
    if charge001 <= charge000 and charge00_1 <= charge000:
        gk = 0.0

    # convert to direct
    # NOTE: Doing this rather than the original car2lat with two np.dot operations
    # saves about half the time.
    r0 = dir2lat[0, 0] * gi + dir2lat[0, 1] * gj + dir2lat[0, 2] * gk
    r1 = dir2lat[1, 0] * gi + dir2lat[1, 1] * gj + dir2lat[1, 2] * gk
    r2 = dir2lat[2, 0] * gi + dir2lat[2, 1] * gj + dir2lat[2, 2] * gk
    return r0, r1, r2


# NOTE
# This is an alternative method for calculating the gradient that uses all of
# the neighbors for each grid point to get an overdetermined system with improved
# sampling. I didn't find it made a big difference.
@njit(cache=True, inline="always")
def get_gradient_overdetermined(
    data,
    i,
    j,
    k,
    vox_transforms,
    transform_dists,
    car2lat,
    inv_norm_cart_trans,
):
    nx, ny, nz = data.shape
    # Value at the central point
    point_value = data[i, j, k]
    # Number of neighbor displacements/transforms
    num_transforms = len(vox_transforms)

    # Array to hold finite‐difference estimates along each transform direction
    diffs = np.zeros(num_transforms)
    # Loop over each neighbor transform
    for trans_idx in range(num_transforms):
        # Displacement vector in voxel (grid) coordinates
        x, y, z = vox_transforms[trans_idx]
        # Compute “upper” neighbor index, wrapped by periodic boundaries
        ui, uj, uk = wrap_point(i + x, j + y, k + z, nx, ny, nz)
        # Compute “lower” neighbor index (opposite direction), also wrapped
        li, lj, lk = wrap_point(i - x, j - y, k - z, nx, ny, nz)
        # Values at the neighboring points
        upper_value = data[ui, uj, uk]
        lower_value = data[li, lj, lk]

        # If both neighbors are below or equal to the center, zero out this direction
        # (prevents spurious negative slopes if data dips on both sides)
        if lower_value <= point_value and upper_value <= point_value:
            diffs[trans_idx] = 0.0
        else:
            # Standard central‐difference estimate: (f(i+Δ) – f(i–Δ)) / (2Δ)
            diffs[trans_idx] = (upper_value - lower_value) / (
                2.0 * transform_dists[trans_idx]
            )

    # Solve the overdetermined system to get the Cartesian gradient:
    #   norm_cart_transforms.T @ cart_grad ≈ diffs
    # Use the pseudoinverse to handle more directions than dimensions
    # inv_norm_cart_trans = np.linalg.pinv(norm_cart_transforms) where
    # norm_cart_transforms is an N, 3 shaped array pointing to 13 unique neighbors
    ti, tj, tk = inv_norm_cart_trans @ diffs
    # Convert Cartesian gradient to fractional (lattice) coordinates
    ti_new = car2lat[0, 0] * ti + car2lat[0, 1] * tj + car2lat[0, 2] * tk
    tj_new = car2lat[1, 0] * ti + car2lat[1, 1] * tj + car2lat[1, 2] * tk
    tk_new = car2lat[2, 0] * ti + car2lat[2, 1] * tj + car2lat[2, 2] * tk

    ti, tj, tk = ti_new, tj_new, tk_new
    return ti, tj, tk


@njit(fastmath=True, cache=True)
def merge_frac_coords(
    frac_coords,
):

    # We'll accumulate (unwrapped) coordinates into total
    total0 = 0.0
    total1 = 0.0
    total2 = 0.0
    count = 0

    # reference coord used for unwrapping
    ref0 = 0.0
    ref1 = 0.0
    ref2 = 0.0
    ref_set = False

    # scan all maxima and pick those that belong to this target_group
    for c0, c1, c2 in frac_coords:

        # first seen -> set reference for unwrapping
        if not ref_set:
            ref0, ref1, ref2 = c0, c1, c2
            ref_set = True

        # unwrap coordinate relative to reference: unwrapped = coord - round(coord - ref)
        # Using np.round via float -> use built-in round for numba compatibility
        # but call round(x) (returns float)
        un0 = c0 - round(c0 - ref0)
        un1 = c1 - round(c1 - ref1)
        un2 = c2 - round(c2 - ref2)

        # add to total
        total0 += un0
        total1 += un1
        total2 += un2
        count += 1

    if count == 1:
        # return original point wrapped to [0,1)
        return np.array((ref0 % 1.0, ref1 % 1.0, ref2 % 1.0), dtype=np.float64)

    else:
        # return average of points
        avg0 = (total0 / count) % 1.0
        avg1 = (total1 / count) % 1.0
        avg2 = (total2 / count) % 1.0
        return np.array((avg0, avg1, avg2), dtype=np.float64)


@njit(fastmath=True, cache=True)
def merge_frac_coords_weighted(
    frac_coords,
    values,
):
    # normalize values
    values /= values.sum()

    # We'll accumulate (unwrapped) coordinates into total
    total0 = 0.0
    total1 = 0.0
    total2 = 0.0

    # reference coord used for unwrapping
    ref0 = 0.0
    ref1 = 0.0
    ref2 = 0.0
    ref_set = False

    # scan all maxima and pick those that belong to this target_group
    for (c0, c1, c2), weight in zip(frac_coords, values):

        # first seen -> set reference for unwrapping
        if not ref_set:
            ref0, ref1, ref2 = c0, c1, c2
            ref_set = True

        # unwrap coordinate relative to reference: unwrapped = coord - round(coord - ref)
        # Using np.round via float -> use built-in round for numba compatibility
        # but call round(x) (returns float)
        un0 = c0 - round(c0 - ref0)
        un1 = c1 - round(c1 - ref1)
        un2 = c2 - round(c2 - ref2)

        # add to total
        total0 += un0 * weight
        total1 += un1 * weight
        total2 += un2 * weight

    return np.array((total0 % 1.0, total1 % 1.0, total2 % 1.0), dtype=np.float64)


@njit(cache=True)
def find_root(parent, x):
    """Find root with partial path compression"""
    while x != parent[x]:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@njit(cache=True)
def find_root_no_compression(parent, x):
    while x != parent[x]:
        x = parent[x]
    return x


@njit(cache=True, inline="always")
def union(parents, x, y):
    rx = find_root(parents, x)
    ry = find_root(parents, y)

    parents[ry] = rx


@njit(cache=True, parallel=True)
def initialize_labels_from_maxima(
    data,
    spline_coeffs,
    maxima_vox,
    max_vox_offset=5,
):
    nx, ny, nz = data.shape
    ny_nz = ny * nz

    # create an array to store values at each maximum
    maxima_values = np.empty(len(maxima_vox), dtype=np.float64)

    # get the fractional representation of each maximum
    maxima_frac = maxima_vox / np.array(data.shape, dtype=np.int64)

    # create a flat array of labels. These will initially all be -1
    labels = np.full(nx * ny * nz, -1, dtype=np.int64)

    # Now we initialize the maxima
    maxima_labels = []
    for max_idx, (i, j, k) in enumerate(maxima_vox):
        # get value at maximum
        maxima_values[max_idx] = data[i, j, k]
        # set as initial group root
        max_idx = coords_to_flat(i, j, k, ny_nz, nz)
        labels[max_idx] = max_idx
        maxima_labels.append(max_idx)

    # We check each maximum to see if it borders another maximum. This can happen
    # either when two neighboring grid points have the exact same value, or when
    # two points slighly further straddle an off grid maximum such that the points
    # around them are still lower. The latter does not have an obvious cutoff
    # distance, so we instead scan through our maxima finding the closest neighbor
    # that hasn't been checked yet. We assume grid points are relatively evenly
    # spaced

    # first, we find unchecked lowest neighbors
    maxima_neighs = np.empty(len(maxima_vox) - 1, dtype=np.int64)
    dists = np.empty(len(maxima_vox) - 1, dtype=np.int64)
    for max_idx in prange(len(maxima_vox) - 1):
        max_frac = maxima_frac[max_idx]
        max_vox = maxima_vox[max_idx]
        # iterate over maxima after this point and find the closest
        best_dist = max_vox_offset  # set to max reasonable number of voxels away
        best_neigh = -1
        for neigh_max_idx in range(max_idx + 1, len(maxima_vox)):
            neigh_frac = maxima_frac[neigh_max_idx]
            # unwrap relative to central
            fi, fj, fk = neigh_frac - np.round(neigh_frac - max_frac)
            # convert to voxel
            fi = round(fi * nx)
            fj = round(fj * ny)
            fk = round(fk * nz)
            # get offset
            fi = fi - max_vox[0]  # get offset from point
            fj = fj - max_vox[1]
            fk = fk - max_vox[2]
            # if the highest offset is 1, this is an adjacent maximum and
            # we immediately give it a distance of 1 to indicate this
            if max(abs(fi), abs(fj), abs(fk)) == 1:
                best_dist = 1
                best_neigh = neigh_max_idx
                break

            # otherwise we calculate the distance in grid coordinates
            dist = (fi**2 + fj**2 + fk**2) ** (1 / 2)
            if dist < best_dist:
                best_dist = dist
                best_neigh = neigh_max_idx
        maxima_neighs[max_idx] = best_neigh
        dists[max_idx] = best_dist

    # Now, for each maximum we check its nearest neighbor. If its within a
    # single voxel we immediately combine, and if its further, we use a spline
    # interpolation to determine if there is at least some minima between them
    for max_idx, (maxima_neigh, dist) in enumerate(zip(maxima_neighs, dists)):
        if maxima_neigh == -1:
            # there were no neighs within our cutoff so this point has no
            # neighs
            continue

        # get the labels for each
        max_label = maxima_labels[max_idx]
        neigh_label = maxima_labels[maxima_neigh]
        # get the roots in case these maxima have been unioned previously
        max_root = find_root(labels, max_label)
        neigh_root = find_root(labels, neigh_label)
        if max_root == neigh_root:
            # we've already combined these neighbors indirectly. continue
            continue

        # get root maxima indices
        max_root_idx = np.searchsorted(maxima_labels, max_root)
        neigh_root_idx = np.searchsorted(maxima_labels, neigh_root)
        to_union = False
        if dist == 1:
            # we are within a single voxel of each other and must be the same
            # maximum (at least at this resolution).
            to_union = True
        else:
            # we want to interpolate between our points. If there is at least
            # one local minimum, these are separete maxima
            # BUGFIX: We want to interpolate to the highest maximum in the group
            # to avoid combining two real maxima because there is a fake maxima
            # split between them.
            max_frac = maxima_frac[max_root_idx]
            neigh_frac = maxima_frac[neigh_root_idx]
            # unwrap the neighbor to the closest point to the current max
            neigh_frac = neigh_frac - np.round(neigh_frac - max_frac)

            # set number of interpolation points to roughly the number of voxels
            # between the points * 5
            n_points = math.ceil(dist) * 5
            values = linear_slice(
                spline_coeffs, max_frac, neigh_frac, n=n_points, is_frac=True
            )
            # check for a local minimum
            left = values[1:-1] < values[:-2]
            right = values[1:-1] < values[2:]
            if np.any(left & right):
                continue
            # if there is no min, these belong to the same maximum. union
            to_union = True
        if to_union:
            # set the higher value as the root
            value1 = maxima_values[max_root_idx]
            value2 = maxima_values[neigh_root_idx]
            higher = max(value1, value2)
            # union the roots (not their maxima indices)
            if value1 == higher:
                union(labels, max_root, neigh_root)
            else:
                union(labels, neigh_root, max_root)

    # get the roots of each maximum
    maxima_roots = []
    for max_idx in maxima_labels:
        maxima_roots.append(find_root_no_compression(labels, max_idx))
    maxima_roots = np.array(maxima_roots, dtype=np.int64)

    # Now we want to calculate the new frac coords for each group
    # find unique labels and their count
    unique_roots = np.unique(maxima_roots)
    n_unique = len(unique_roots)

    # Prepare result array to store frac coords and grid coords
    all_frac_coords = np.zeros((n_unique, 3), dtype=np.float64)
    all_grid_coords = np.zeros((n_unique, 3), dtype=np.int64)

    # Parallel loop: for each unique label, scan new_labels and get average
    # frac coords
    for u_idx in prange(n_unique):
        target_root = unique_roots[u_idx]
        frac_coords = []
        maxima_w_root_labels = []
        maxima_w_root_values = []
        # track highest value and position
        highest_value = -1e300
        highest_idx = -1
        i = -1
        j = -1
        k = -1
        for max_idx, root in enumerate(maxima_roots):
            if root == target_root:
                # note this maximum has this root
                maxima_w_root_labels.append(maxima_labels[max_idx])
                frac_coords.append(maxima_frac[max_idx])
                maxima_w_root_values.append(maxima_values[max_idx])
                if maxima_values[max_idx] > highest_value:
                    # update our highest point and value
                    highest_value = maxima_values[max_idx]
                    i, j, k = maxima_vox[max_idx]
                    highest_idx = max_idx

        # combine frac coords
        merged_coord = merge_frac_coords_weighted(
            frac_coords, np.array(maxima_w_root_values, dtype=np.float64)
        )

        # We want to get the point closest to the merged coord. This point must
        # also be the ongrid maximum for this area.
        ci = round(merged_coord[0] * nx) % nx
        cj = round(merged_coord[1] * ny) % ny
        ck = round(merged_coord[2] * nz) % nz
        best_max_label = coords_to_flat(ci, cj, ck, ny_nz, nz)
        center_value = data[ci, cj, ck]
        # make sure this point is one of our maxima and is has as high a value
        if not best_max_label in maxima_w_root_labels or center_value < highest_value:
            # default back to the max point we found in our loop
            ci = i
            cj = j
            ck = k
            best_max_label = coords_to_flat(ci, cj, ck, ny_nz, nz)
            merged_coord = maxima_frac[highest_idx]

        # add the new frac and grid coords
        all_frac_coords[u_idx] = merged_coord
        all_grid_coords[u_idx] = (ci, cj, ck)
        # relabel all maxima to point to the highest maximum in the group
        for max_label in maxima_w_root_labels:
            labels[max_label] = best_max_label

    return labels, all_frac_coords, all_grid_coords


@njit(cache=True, fastmath=True)
def get_min_avg_surface_dists(
    labels,
    frac_coords,
    edge_mask,
    matrix,
    max_value,
):
    nx, ny, nz = labels.shape
    # create array to store best dists, sums, and counts
    dists = np.full(len(frac_coords), max_value, dtype=np.float64)
    dist_sums = np.zeros(len(frac_coords), dtype=np.float64)
    edge_totals = np.zeros(len(frac_coords), dtype=np.uint32)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # skip outside edges
                if not edge_mask[i, j, k]:
                    continue
                # get label at edge
                label = labels[i, j, k]
                # add to our count
                edge_totals[label] += 1
                # convert from voxel indices to frac
                fi = i / nx
                fj = j / ny
                fk = k / nz
                # calculate the distance to the appropriate frac coord
                ni, nj, nk = frac_coords[label]
                # get differences between each index
                di = ni - fi
                dj = nj - fj
                dk = nk - fk
                # wrap at edges to be as close as possible
                di -= round(di)
                dj -= round(dj)
                dk -= round(dk)
                # convert to cartesian coordinates
                ci = di * matrix[0, 0] + dj * matrix[1, 0] + dk * matrix[2, 0]
                cj = di * matrix[0, 1] + dj * matrix[1, 1] + dk * matrix[2, 1]
                ck = di * matrix[0, 2] + dj * matrix[1, 2] + dk * matrix[2, 2]
                # calculate distance
                dist = np.linalg.norm(np.array((ci, cj, ck), dtype=np.float64))
                # add to our total
                dist_sums[label] += dist
                # if this is the lowest distance, update radius
                if dist < dists[label]:
                    dists[label] = dist
    # get average dists
    average_dists = dist_sums / edge_totals
    return dists, average_dists
