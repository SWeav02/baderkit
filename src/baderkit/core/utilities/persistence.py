# -*- coding: utf-8 -*-

import math

import numpy as np
from numba import njit, prange

from baderkit.core.utilities.basic import (
    compute_wrap_offset,
    coords_to_flat,
    dist,
    wrap_point,
)
from baderkit.core.utilities.basins import (
    get_best_neighbor,
    reorder_labels,
)
from baderkit.core.utilities.critical_points import refine_critical_points
from baderkit.core.utilities.interpolation import linear_slice
from baderkit.core.utilities.transforms import (
    ALL_NEIGHBOR_TRANSFORMS,
    INT_TO_IMAGE,
    get_transform_dists,
)
from baderkit.core.utilities.union_find import (
    find_root_with_shift1,
    union,
    union_with_shift1,
)

# Predefine types
# key_type = UniTuple(int64, 3)
# value_type = int64

# @njit(cache=True)
# def unique_and_inverse_axis0(arr):
#     n = arr.shape[0]
#     arr64 = arr.astype(np.int64)

#     # Use the predeclared Numba types
#     d = Dict.empty(key_type=key_type, value_type=value_type)

#     inverse = np.empty(n, dtype=np.int64)
#     unique = List.empty_list(key_type)

#     next_idx = 0

#     for i in range(n):
#         key = (arr64[i, 0], arr64[i, 1], arr64[i, 2])

#         if key in d:
#             inverse[i] = d[key]
#         else:
#             d[key] = next_idx
#             inverse[i] = next_idx
#             unique.append(key)
#             next_idx += 1

#     unique_arr = np.empty((next_idx, 3), dtype=arr.dtype)

#     for i in range(next_idx):
#         row = unique[i]
#         unique_arr[i, 0] = row[0]
#         unique_arr[i, 1] = row[1]
#         unique_arr[i, 2] = row[2]

#     return unique_arr, inverse


@njit(cache=True)
def get_persistence_value(value1, value2, conn_value, p1, p2, p_conn):
    eps = 1e-12
    # get larger and smaller diffs
    diff1 = abs(value1 - conn_value)
    diff2 = abs(value2 - conn_value)

    # get distance between the points
    p1_dist = dist(p1, p_conn)
    p2_dist = dist(p2, p_conn)
    distance = p1_dist + p2_dist
    if distance == 0:
        return 0.0

    # get approximate average value between extrema
    avg = (
        p1_dist * (value1 + conn_value) / 2 + p2_dist * (value2 + conn_value) / 2
    ) / distance

    # get persistence score:
    #   p = smaller_diff*dist / (average_value)
    persistence_score = min(diff1, diff2) * distance / (abs(avg) + eps)

    return persistence_score


@njit(cache=True)
def grow_arrays(neighbors, values, conn_coords, neigh_coords):
    n, m = neighbors.shape
    new_m = m * 2

    new_neighbors = np.full((n, new_m), -1, neighbors.dtype)
    new_values = np.zeros((n, new_m), values.dtype)
    new_conn_coords = np.zeros((n, new_m, 3), conn_coords.dtype)
    new_neigh_coords = np.zeros((n, new_m, 3), neigh_coords.dtype)

    new_neighbors[:, :m] = neighbors
    new_values[:, :m] = values
    new_conn_coords[:, :m] = conn_coords
    new_neigh_coords[:, :m] = neigh_coords

    return new_neighbors, new_values, new_conn_coords, new_neigh_coords


@njit(cache=True)
def get_shift(coord):
    shift = np.empty(3, dtype=np.int8)
    if coord[0] < 0:
        shift[0] = -1
    elif coord[0] >= 0:
        shift[0] = 1
    if coord[1] < 0:
        shift[1] = -1
    elif coord[1] >= 0:
        shift[1] = 1
    if coord[2] < 0:
        shift[2] = -1
    elif coord[2] >= 0:
        shift[2] = 1
    return shift


@njit(cache=True)
def union_by_persistence(
    neighbors,
    values,
    conn_coords,
    neigh_coords,
    cursor,
    labels,
    matrix,
    extrema_values,
    extrema_labels,
    use_minima,
    persistence_tol,
):

    all_conn_nums = np.full(len(extrema_values), -1, np.int64)

    new_union = True

    while new_union:
        new_union = False

        for ext_idx in range(len(extrema_values)):

            num = cursor[ext_idx]

            if num == all_conn_nums[ext_idx]:
                continue

            all_conn_nums[ext_idx] = num

            if num == 0:
                continue

            if num == 1:
                neigh_idx = neighbors[ext_idx, 0]
                union(labels, extrema_labels[ext_idx], extrema_labels[neigh_idx])
                new_union = True
                continue

            for i in range(num):

                neigh_idx = neighbors[ext_idx, i]
                neigh_val = extrema_values[neigh_idx]
                # get shift
                neigh_coord = neigh_coords[ext_idx, i]
                neigh_cart = neigh_coord @ matrix

                union(labels, extrema_labels[ext_idx], extrema_labels[neigh_idx])
                new_union = True

                conn_val = values[ext_idx, i]
                conn_coord = conn_coords[ext_idx, i]

                neigh_count = cursor[neigh_idx]

                for j in range(num):

                    if j <= i:
                        continue

                    neigh_idx1 = neighbors[ext_idx, j]
                    neigh_val1 = extrema_values[neigh_idx1]

                    exists = False
                    for k in range(neigh_count):
                        if neighbors[neigh_idx, k] == neigh_idx1:
                            exists = True
                            break

                    if exists:
                        continue

                    conn_val1 = values[ext_idx, j]
                    conn_coord1 = conn_coords[ext_idx, j]
                    neigh_coord1 = neigh_coords[ext_idx, j]
                    neigh_cart1 = neigh_coord1 @ matrix

                    if (use_minima and conn_val >= conn_val1) or (
                        not use_minima and conn_val <= conn_val1
                    ):
                        val = conn_val
                        coord = conn_coord
                    else:
                        val = conn_val1
                        coord = conn_coord1

                    score = get_persistence_value(
                        extrema_values[neigh_idx],
                        extrema_values[neigh_idx1],
                        val,
                        neigh_cart,
                        neigh_cart1,
                        coord,
                    )

                    if score < persistence_tol:
                        if (
                            use_minima
                            and neigh_val1 > neigh_val
                            or not use_minima
                            and neigh_val1 < neigh_val
                            or neigh_val1 == neigh_val
                            and neigh_idx1 < neigh_idx
                        ):
                            idx = cursor[neigh_idx1]
                            # grow arrays if needed
                            if idx >= neighbors.shape[1]:
                                neighbors, values, conn_coords, neigh_coords = (
                                    grow_arrays(
                                        neighbors, values, conn_coords, neigh_coords
                                    )
                                )

                            neighbors[neigh_idx1, idx] = neigh_idx
                            values[neigh_idx1, idx] = val
                            conn_coords[neigh_idx1, idx] = coord
                            neigh_coords[neigh_idx1, idx] = neigh_coord

                            cursor[neigh_idx1] += 1
                        else:
                            idx = cursor[neigh_idx]

                            # grow arrays if needed
                            if idx >= neighbors.shape[1]:
                                neighbors, values, conn_coords, neigh_coords = (
                                    grow_arrays(
                                        neighbors, values, conn_coords, neigh_coords
                                    )
                                )

                            neighbors[neigh_idx, idx] = neigh_idx1
                            values[neigh_idx, idx] = val
                            conn_coords[neigh_idx, idx] = coord
                            neigh_coords[neigh_idx, idx] = neigh_coord1

                            cursor[neigh_idx] += 1

    return labels


@njit(cache=True)
def get_approx_saddle_val(
    p0,
    p1,
    n_points,
    data,
    use_minima,
    nx,
    ny,
    nz,
):

    # check if there is a minimum between this point and its neighbor
    values = linear_slice(
        data,
        p0,
        p1,
        n=n_points,
        is_frac=True,
        method="linear",
    )

    # get the number of extrema
    s = np.sign(np.diff(values))

    if use_minima:
        # add end points
        s = np.append(-1, s)
        s = np.append(s, 1)
        # get min indices
        min_indices = np.where((s[:-1] <= 0) & (s[1:] >= 0))[0]
        if len(min_indices) < 2:
            conn_val = values.max()
        else:
            min0 = min_indices[0]
            min1 = min_indices[-1]
            conn_val = values[min0:min1].max()

    else:
        # add end points
        s = np.append(1, s)
        s = np.append(s, -1)
        # get max indices
        max_indices = np.where((s[:-1] >= 0) & (s[1:] <= 0))[0]
        if len(max_indices) < 2:
            conn_val = values.min()
        else:
            max0 = max_indices[0]
            max1 = max_indices[-1]
            conn_val = values[max0:max1].min()

    return conn_val


@njit(parallel=True, cache=True)
def group_by_low_approx_persistence(
    data,
    labels,
    images,
    extrema_values,
    extrema_labels,
    extrema_frac,
    max_cart_offset,
    use_minima,
    persistence_tol,
    matrix,
):
    nx, ny, nz = data.shape
    extrema_cart = extrema_frac @ matrix
    num_extrema = len(extrema_values)

    best_neighs = np.arange(num_extrema)
    best_shifts = np.zeros((num_extrema, 3), dtype=np.int8)

    # loop over each point and find the best neighbor
    for ext_idx in prange(num_extrema):
        ext_frac = extrema_frac[ext_idx]
        ext_cart = extrema_cart[ext_idx]
        ext_value = extrema_values[ext_idx]

        # create a tracker for the best neighbor
        best_label = int(ext_idx)
        best_persistence = np.inf
        best_shift = np.zeros(3, dtype=np.int8)
        for neigh_ext_idx in range(num_extrema):
            # skip the original index
            if ext_idx == neigh_ext_idx:
                continue
            # skip neighbors with less extreme values
            neigh_ext_value = extrema_values[neigh_ext_idx]
            if (
                use_minima
                and neigh_ext_value > ext_value
                or not use_minima
                and neigh_ext_value < ext_value
                or neigh_ext_value == ext_value
                and neigh_ext_idx > ext_idx
            ):
                continue

            # get neighbor frac coord
            neigh_frac = extrema_frac[neigh_ext_idx]
            shift = np.round(neigh_frac - ext_frac).astype(np.int8)
            wrapped_neigh_frac = neigh_frac - shift

            # check if this neighbor is outside our cutoff and continue if so
            neigh_cart = wrapped_neigh_frac @ matrix

            offset = neigh_cart - ext_cart
            dist = np.linalg.norm(offset)
            if dist > max_cart_offset:
                continue

            # get the value these extrema connect at
            n_points = max(int(round(dist * 20)), 5)
            conn_val = get_approx_saddle_val(
                ext_frac,
                wrapped_neigh_frac,
                n_points,
                data,
                use_minima,
                nx,
                ny,
                nz,
            )

            if conn_val == np.inf:
                continue

            conn_cart = ext_cart + offset / 2

            # get persistence
            persistence_score = get_persistence_value(
                ext_value, neigh_ext_value, conn_val, ext_cart, neigh_cart, conn_cart
            )

            # if our persistence is below our tolerance and the current best,
            # update the best label
            if (
                persistence_score < persistence_tol
                and persistence_score < best_persistence
            ):
                best_label = neigh_ext_idx
                best_shift = shift
                best_persistence = persistence_score
        best_neighs[ext_idx] = best_label
        best_shifts[ext_idx] = best_shift

    # go through and union each label
    for ext_idx, (neigh_idx, neigh_image) in enumerate(zip(best_neighs, best_shifts)):
        label = extrema_labels[ext_idx]
        neigh_label = extrema_labels[neigh_idx]
        union_with_shift1(labels, images, label, neigh_label, neigh_image)

    return labels, images


@njit(parallel=True, cache=True)
def group_by_refinement(
    labels,
    images,
    data,
    matrix,
    extrema_labels,
    extrema_vox,
    extrema_values,
    extrema_frac,
    use_minima,
    shape,
):
    # get root indices
    roots = np.empty(len(extrema_labels), dtype=labels.dtype)
    for ext_idx in range(len(extrema_labels)):
        label = extrema_labels[ext_idx]
        root, shift = find_root_with_shift1(labels, images, label)
        roots[ext_idx] = root
    root_indices = np.where(roots == extrema_labels)[0]

    # refine remaining extrema
    root_vox = extrema_vox[root_indices]
    root_labels = extrema_labels[root_indices]
    target_idx = 0 if use_minima else 3
    refined_vox, _ = refine_critical_points(
        critical_coords=root_vox,
        data=data,
        matrix=matrix,
        target_index=target_idx,
    )

    # get best neighbors of each
    rounded = np.floor(refined_vox).astype(np.int64) % shape
    rounded_frac = rounded / shape

    best_neighs = np.arange(len(root_vox))
    best_shifts = np.zeros((len(root_vox), 3), dtype=np.int8)

    for idx in prange(len(root_indices)):
        ext_idx = root_indices[idx]
        ext_frac = rounded_frac[idx]
        ext_vox = rounded[idx]
        ext_val = extrema_values[ext_idx]
        ext_orig_frac = extrema_frac[ext_idx]

        best_idx = int(idx)
        best_value = ext_val
        best_shift = np.zeros(3, dtype=np.int8)
        for idx1 in range(len(root_indices)):
            neigh_idx = root_indices[idx1]
            neigh_val = extrema_values[neigh_idx]
            # skip neighbors with values lower than the current best
            if (
                use_minima
                and neigh_val > best_value
                or not use_minima
                and neigh_val < best_value
                or neigh_val == best_value
                and idx1 > best_idx
            ):
                continue

            # wrap neighbor to be as close as possible
            neigh_frac = rounded_frac[idx1]
            wrapped = neigh_frac - np.round(neigh_frac - ext_frac)
            wrapped_vox = wrapped * shape
            offset = wrapped_vox - ext_vox

            # skip points more than one voxel away
            if np.max(np.abs(offset)) > 1 + 1e-12:
                continue

            # get the shift to this new point
            neigh_orig_frac = extrema_frac[neigh_idx]
            total_shift = np.round(neigh_orig_frac - ext_orig_frac).astype(np.int8)

            best_idx = idx1
            best_shift = total_shift
        # update label and image
        best_neighs[idx] = best_idx
        best_shifts[idx] = best_shift

    # go through and label
    for idx, (idx1, shift) in enumerate(zip(best_neighs, best_shifts)):
        label = root_labels[idx]
        neigh_label = root_labels[idx1]
        union_with_shift1(labels, images, label, neigh_label, shift)

    return labels, images, refined_vox, root_labels


@njit(cache=True)
def init_by_approx_persistence(
    data,
    labels,
    extrema_mask,
    extrema_vox,
    persistence_tol,
    matrix,
    max_cart_offset,
    use_minima=False,
):
    shape = np.array(data.shape, dtype=np.int64)
    nx, ny, nz = shape
    ny_nz = ny * nz

    extrema_frac = extrema_vox / shape

    # create an array to store values at each maximum
    extrema_values = np.empty(len(extrema_vox), dtype=np.float64)
    extrema_labels = np.empty(len(extrema_vox), dtype=np.int64)

    # create a flat array of shifts for tracking wrapping around edges. These
    # will initially all be (0,0,0)
    images = np.zeros((nx * ny * nz, 3), dtype=np.int8)

    # Now we initialize the extrema
    for ext_idx, (i, j, k) in enumerate(extrema_vox):
        # get value at extremum
        extrema_values[ext_idx] = data[i, j, k]
        # set as initial group root
        flat_ext_idx = coords_to_flat(i, j, k, ny_nz, nz)
        extrema_labels[ext_idx] = flat_ext_idx
        labels[flat_ext_idx] = flat_ext_idx

    ###########################################################################
    # 1. Combine low-persistence extrema
    ###########################################################################
    # With the right shape (e.g. highly anisotropic) a maximum/minimum may lay offgrid
    # and cause two ongrid points to appear to be higher than the region around
    # them. We merge these by taking a linear slice between each point using
    # cubic interpolation and combining those that have no minimum/maximum between
    # them.
    labels, images = group_by_low_approx_persistence(
        data,
        labels,
        images,
        extrema_values,
        extrema_labels,
        extrema_frac,
        max_cart_offset,
        use_minima,
        persistence_tol,
        matrix,
    )

    ###########################################################################
    # 2. Remove by refinement
    ###########################################################################

    labels, images, refined_vox, root_labels = group_by_refinement(
        labels,
        images,
        data,
        matrix,
        extrema_labels,
        extrema_vox,
        extrema_values,
        extrema_frac,
        use_minima,
        shape,
    )

    ###########################################################################
    # Root finding
    ###########################################################################

    # get our final roots
    roots = np.empty(len(extrema_labels), dtype=labels.dtype)
    for ext_idx in range(len(extrema_labels)):
        label = extrema_labels[ext_idx]
        root, shift = find_root_with_shift1(labels, images, label)
        roots[ext_idx] = root
        labels[label] = root
        images[label] = shift
    root_indices = np.where(roots == extrema_labels)[0]

    final_root_labels = extrema_labels[root_indices]
    final_root_vox = extrema_vox[root_indices]

    refined_frac = np.round(refined_vox / shape, 6)
    final_refined_frac = np.empty_like(final_root_vox, dtype=np.float64)
    for root_idx, label in enumerate(final_root_labels):
        prev_root_idx = np.searchsorted(root_labels, label)
        final_refined_frac[root_idx] = refined_frac[prev_root_idx]

    return (
        labels,
        images,
        final_root_vox,
        final_refined_frac,
    )


@njit(cache=True)
def group_by_persistence(
    data,
    extrema_vox,
    extrema_frac,
    basin_connections,
    saddle_values,
    saddle_carts,
    persistence_tol,
    matrix,
    use_minima=False,
):
    num_extrema = len(extrema_frac)

    # get values at extrema points
    extrema_values = np.empty(len(extrema_frac), dtype=np.float64)
    for idx, (i, j, k) in enumerate(extrema_vox):
        extrema_values[idx] = data[i, j, k]

    # get max neighbor count
    counts = np.zeros(num_extrema, dtype=np.int64)
    for ext1, ext2, image1, image2 in basin_connections:
        counts[ext1] += 1
    max_count = counts.max()

    # create arrays to track connections
    max_degree = max_count
    all_conn_neighs = np.full((num_extrema, max_degree), -1, np.int64)
    all_conn_vals = np.zeros((num_extrema, max_degree), np.float64)
    all_conn_coords = np.zeros((num_extrema, max_degree, 3), np.float64)
    all_neigh_coords = np.zeros((num_extrema, max_degree, 3), np.float64)

    cursor = np.zeros(num_extrema, dtype=np.int64)

    for ext_idx in prange(num_extrema):
        for i, (ext1, ext2, image1, image2) in enumerate(basin_connections):
            if ext1 != ext_idx:
                continue
            neigh_count = cursor[ext1]

            saddle_val = saddle_values[i]
            saddle_cart = saddle_carts[i]

            c1 = (extrema_frac[ext1] + INT_TO_IMAGE[image1]) @ matrix
            c2 = (extrema_frac[ext2] + INT_TO_IMAGE[image2]) @ matrix

            score = get_persistence_value(
                extrema_values[ext1],
                extrema_values[ext2],
                saddle_val,
                c1,
                c2,
                saddle_cart,
            )
            if score < persistence_tol:
                all_conn_neighs[ext_idx, neigh_count] = ext2
                all_conn_vals[ext_idx, neigh_count] = saddle_val
                all_conn_coords[ext_idx, neigh_count] = saddle_cart
                all_neigh_coords[ext_idx, neigh_count] = c2

                cursor[ext_idx] += 1

    # create array to track unions between basins
    unions = np.arange(num_extrema)

    if max_count > 0:
        unions = union_by_persistence(
            all_conn_neighs,
            all_conn_vals,
            all_conn_coords,
            all_neigh_coords,
            cursor,
            labels=unions,
            matrix=matrix,
            extrema_values=extrema_values,
            extrema_labels=np.arange(len(extrema_values)),
            use_minima=use_minima,
            persistence_tol=persistence_tol,
        )

    # update labels to the highest valued point
    unions, _ = reorder_labels(
        unions,
        data,
        extrema_labels=np.arange(num_extrema),
        extrema_values=extrema_values,
        use_minima=use_minima,
    )

    # Find the images each grouped maximum must cross to reach its parent
    root_transforms = np.empty((len(unions), 3), dtype=np.int8)
    for ext_idx, root in enumerate(unions):
        # get fractional coordinates
        ext_frac = extrema_frac[ext_idx]
        root_frac = extrema_frac[root]
        # get best image to wrap the maxima to its root
        root_transforms[ext_idx] = compute_wrap_offset(ext_frac, root_frac)

    return unions, root_transforms


@njit(cache=True, parallel=True)
def get_persistence_cutoffs(data, groups, use_minima, group_vals, max_dist=5):
    persistence_cutoffs = np.full(len(groups), np.inf, dtype=np.float64)
    shape = np.array(data.shape)
    for group_idx in prange(len(groups)):
        # get group
        group = groups[group_idx]
        # wrap all to the same region
        group_frac = group / shape
        ref = group_frac[0]
        group_frac = group_frac - np.round(group_frac - ref)
        # convert back to vox
        group = group_frac * shape
        # get distances between them
        neighs = np.empty((len(group)), dtype=np.int32)
        dists = np.empty((len(group)), dtype=np.float64)
        for i in range(len(group)):
            best_dist = np.inf
            best_neigh = -1
            ci, cj, ck = group[i]
            for j in range(len(group)):
                if i == j:
                    continue
                ci1, cj1, ck1 = group[j]
                dist = ((ci - ci1) ** 2 + ((cj - cj1) ** 2) + ((ck - ck1) ** 2)) ** (
                    1 / 2
                )
                if dist < best_dist:
                    best_dist = dist
                    best_neigh = j
            neighs[i] = best_neigh
            dists[i] = best_dist

        # get the lowest/highest connecting value between maxima/minima
        best_val = group_vals[group_idx]
        for j, (i, dist) in enumerate(zip(neighs, dists)):
            # skip dists above our cutoff
            if dist > max_dist:
                continue
            p1 = group[i]
            p2 = group[j]
            n = math.ceil(dist * 3)
            # otherwise get values between
            values = linear_slice(
                data=data, p1=p1, p2=p2, n=n, is_frac=False, method="linear"
            )

            if use_minima:
                val = values.max()
                best_val = max(val, best_val)
            else:
                val = values.min()
                best_val = min(val, best_val)
        persistence_cutoffs[group_idx] = best_val
    return persistence_cutoffs


@njit(cache=True)
def get_persistence_groups(
    labels,
    data,
    persistence_cutoffs,
    extrema_vox,
    use_minima,
):
    nx, ny, nz = data.shape
    max_val = len(extrema_vox)

    persistence_groups = []
    for vox in extrema_vox:
        temp = [vox]
        temp = temp[1:]
        persistence_groups.append(temp)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # get label
                label = labels[i, j, k]
                # skip unlabeled points
                if label >= max_val:
                    continue
                # get value and cutoff
                value = data[i, j, k]
                cutoff = persistence_cutoffs[label]
                # if value is above the cutoff, add to the group
                if not use_minima and value >= cutoff or use_minima and value <= cutoff:
                    point = np.array((i, j, k), dtype=np.int16)
                    persistence_groups[label].append(point)

    # convert to arrays
    array_groups = []
    for i in persistence_groups:
        new_group = np.empty((len(i), 3), dtype=np.int16)
        for idx, val in enumerate(i):
            new_group[idx] = val
        array_groups.append(new_group)
    return array_groups
