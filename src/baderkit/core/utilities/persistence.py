# -*- coding: utf-8 -*-

import math

import numpy as np
from numba import njit, prange

from baderkit.core.utilities.basic import (
    dist,
    wrap_point,
    coords_to_flat,
    compute_wrap_offset,
)
from baderkit.core.utilities.interpolation import linear_slice
from baderkit.core.utilities.union_find import union
from baderkit.core.utilities.basins import (
    get_best_neighbor,
    reorder_labels,
    )
from baderkit.core.utilities.critical_points import refine_critical_points
from baderkit.core.utilities.transforms import (
    INT_TO_IMAGE,
    ALL_NEIGHBOR_TRANSFORMS,
    get_transform_dists
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
        p1_dist * (value1 + conn_value) / 2
        + p2_dist * (value2 + conn_value) / 2
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
def union_by_persistence(
    neighbors,
    values,
    conn_coords,
    neigh_coords,
    cursor,
    labels,
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
                neigh = neighbors[ext_idx, 0]
                union(labels, extrema_labels[ext_idx], extrema_labels[neigh])
                new_union = True
                continue

            for i in range(num):

                neigh_idx = neighbors[ext_idx, i]

                union(labels, extrema_labels[ext_idx], extrema_labels[neigh_idx])
                new_union = True

                conn_val = values[ext_idx, i]
                conn_coord = conn_coords[ext_idx, i]
                neigh_coord = neigh_coords[ext_idx, i]

                neigh_count = cursor[neigh_idx]

                for j in range(num):

                    if j <= i:
                        continue

                    neigh_idx1 = neighbors[ext_idx, j]

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
                        neigh_coord,
                        neigh_coord1,
                        coord,
                    )

                    if score < persistence_tol:

                        idx = cursor[neigh_idx]

                        # grow arrays if needed
                        if idx >= neighbors.shape[1]:
                            neighbors, values, conn_coords, neigh_coords = grow_arrays(
                                neighbors, values, conn_coords, neigh_coords
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
def get_all_approx_saddle_vals(
    data,
    labels,
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

    # get the number of neighbors for each point
    counts = np.zeros(len(extrema_frac), dtype=np.int64)
    num_extrema = len(extrema_values)

    for ext_idx in prange(num_extrema):
        ext_frac = extrema_frac[ext_idx]
        ext_cart = extrema_cart[ext_idx]
        ext_value = extrema_values[ext_idx]
        for neigh_ext_idx in range(num_extrema):
            if ext_idx == neigh_ext_idx:
                continue
            neigh_ext_value = extrema_values[neigh_ext_idx]
            if (
                use_minima and neigh_ext_value > ext_value
                or not use_minima and neigh_ext_value < ext_value
                or neigh_ext_value == ext_value and neigh_ext_idx < ext_idx
                    ):
                continue

            # get neighbor frac coord
            neigh_frac = extrema_frac[neigh_ext_idx]

            wrapped_neigh_frac = neigh_frac - np.round(neigh_frac - ext_frac)

            neigh_cart = wrapped_neigh_frac @ matrix

            dist = np.linalg.norm(neigh_cart-ext_cart)
            if dist > max_cart_offset:
                continue
            counts[ext_idx] += 1
    max_count = counts.max()

    # create arrays to track connections
    max_degree = max_count
    all_conn_neighs = np.full((num_extrema, max_degree), -1, np.int64)
    all_conn_vals = np.zeros((num_extrema, max_degree), np.float64)
    all_conn_coords = np.zeros((num_extrema, max_degree, 3), np.float64)
    neigh_cart_coords = np.zeros((num_extrema, max_degree, 3), np.float64)

    cursor = np.zeros(num_extrema, dtype=np.int64)

    # get connection values
    for ext_idx in prange(num_extrema):
        ext_cart = extrema_cart[ext_idx]
        ext_frac = extrema_frac[ext_idx]
        ext_value = extrema_values[ext_idx]
        for neigh_ext_idx in range(num_extrema):
            if ext_idx == neigh_ext_idx:
                continue
            neigh_ext_value = extrema_values[neigh_ext_idx]
            if (
                use_minima and neigh_ext_value > ext_value
                or not use_minima and neigh_ext_value < ext_value
                or neigh_ext_value == ext_value and neigh_ext_idx < ext_idx
                    ):
                continue
            # get neighbor frac coord
            neigh_frac = extrema_frac[neigh_ext_idx]
            wrapped_neigh_frac = neigh_frac - np.round(neigh_frac - ext_frac)

            neigh_cart = wrapped_neigh_frac @ matrix
            offset = neigh_cart - ext_cart

            dist = np.linalg.norm(offset)
            if dist > max_cart_offset:
                continue

            n_points = max(int(round(dist*20)), 5)

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

            conn_cart = ext_cart + offset/2

            # get persistence
            persistence_score = get_persistence_value(
                ext_value,
                neigh_ext_value,
                conn_val,
                ext_cart,
                neigh_cart,
                conn_cart
            )

            # add low persistence to our list
            if persistence_score < persistence_tol:
                neigh_count = cursor[ext_idx]
                all_conn_neighs[ext_idx, neigh_count] = neigh_ext_idx
                all_conn_vals[ext_idx, neigh_count] = conn_val
                all_conn_coords[ext_idx, neigh_count] = conn_cart
                neigh_cart_coords[ext_idx, neigh_count] = neigh_cart

                cursor[ext_idx] += 1
    return (
        all_conn_neighs,
        all_conn_vals,
        all_conn_coords,
        neigh_cart_coords,
        cursor
    )

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

    # get distance to each neighbor
    neighbor_dists = get_transform_dists(
        ALL_NEIGHBOR_TRANSFORMS,
        shape=shape,
        matrix=matrix,
        )

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
    (
        all_conn_neighs,
        all_conn_vals,
        all_conn_coords,
        neigh_cart_coords,
        cursor,
    ) = get_all_approx_saddle_vals(
        data,
        labels,
        extrema_values,
        extrema_labels,
        extrema_frac,
        max_cart_offset,
        use_minima,
        persistence_tol,
        matrix,
    )

    labels = union_by_persistence(
        all_conn_neighs,
        all_conn_vals,
        all_conn_coords,
        neigh_cart_coords,
        cursor,
        labels,
        extrema_values,
        extrema_labels,
        use_minima,
        persistence_tol,
    )

    ###########################################################################
    # 2. Remove Flat False Maxima
    ###########################################################################
    # If there is a particularly flat region, a point might have neighbors that
    # are the same value. This point may be mislabeled as a maximum if these
    # neighbors are not themselves extrema. This issue is typically caused by too
    # few sig figs in the data preventing the region from being properly distinguished

    # Now we look for any points that have neighbors with the same value that
    # aren't maxima. We hill climb from that point to find the corresponding
    # maximum
    # flat_false_maxima = np.zeros(len(extrema_vox), dtype=np.bool_)
    for ext_idx, ((i, j, k), value, ext_label) in enumerate(
        zip(extrema_vox, extrema_values, extrema_labels)
    ):

        for si, sj, sk in ALL_NEIGHBOR_TRANSFORMS:
            # get neighbor and wrap
            ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
            neigh_value = data[ii, jj, kk]
            # skip points that don't have the same value or that are also
            # extrema
            if neigh_value != value or extrema_mask[ii, jj, kk]:
                continue
            # If we're still here, this point is a false maximum. We follow the
            # path to the maximum and union
            # flat_false_maxima[ext_idx] = True
            while True:
                _, (ni, nj, nk) = get_best_neighbor(
                    data,
                    ii,
                    jj,
                    kk,
                    ALL_NEIGHBOR_TRANSFORMS,
                    neighbor_dists,
                    use_minima=use_minima,
                )
                # stop if we hit another maximum
                if extrema_mask[ni, nj, nk]:
                    break
                # otherwise, update to this point
                ii = ni
                jj = nj
                kk = nk

            # make a union
            best_ext = coords_to_flat(ni, nj, nk, ny_nz, nz)
            union(labels, ext_label, best_ext)

    ###########################################################################
    # 3. Remove by refinement
    ###########################################################################
    # update labels to the highest valued point
    labels, root_indices = reorder_labels(
        labels,
        data,
        extrema_labels,
        extrema_values,
        use_minima,
            )

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

    # combine any that refined to be adjacent
    rounded = np.floor(refined_vox).astype(np.int64) % shape
    rounded_frac = rounded / shape
    for idx in range(len(root_indices)):
        ext_idx = root_indices[idx]
        ext_frac = rounded_frac[idx]
        ext_vox = rounded[idx]
        ext_label = root_labels[idx]
        for idx1 in range(idx+1, len(root_indices)):
            neigh_idx = root_indices[idx1]
            neigh_frac = rounded_frac[idx1]
            wrapped = neigh_frac - np.round(neigh_frac - ext_frac)
            wrapped_vox = wrapped * shape
            offset = wrapped_vox - ext_vox
            if np.max(np.abs(offset)) < 1 + 1e-12:
                neigh_label = extrema_labels[neigh_idx]
                union(labels, ext_label, neigh_label)


    ###########################################################################
    # Root finding
    ###########################################################################

    # update labels one more time
    labels, root_indices = reorder_labels(
        labels,
        data,
        extrema_labels,
        extrema_values,
        use_minima,
            )

    final_root_labels = extrema_labels[root_indices]
    final_root_vox = extrema_vox[root_indices]

    refined_frac = np.round(refined_vox / shape, 6)
    final_refined_frac = np.empty_like(final_root_vox, dtype=np.float64)
    for root_idx, label in enumerate(final_root_labels):
        prev_root_idx = np.searchsorted(root_labels, label)
        final_refined_frac[root_idx] = refined_frac[prev_root_idx]

    # Find the images each grouped maximum must cross to reach its parent
    for ext_idx, ext_label in enumerate(extrema_labels):
        root = labels[ext_label]
        # get fractional coordinates
        ext_frac = extrema_frac[ext_idx]
        root_idx = np.searchsorted(final_root_labels, root)
        root_frac = final_refined_frac[root_idx]
        # get best image to wrap the maxima to its root
        images[extrema_labels[ext_idx]] = compute_wrap_offset(
            ext_frac, root_frac
        )

    return (
        labels,
        images,
        final_root_vox,
        final_refined_frac,
        extrema_vox # return original list as well
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
    neigh_cart_coords = np.zeros((num_extrema, max_degree, 3), np.float64)

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
                neigh_cart_coords[ext_idx, neigh_count] = c2

                cursor[ext_idx] += 1



    # create array to track unions between basins
    unions = np.arange(num_extrema)

    if max_count > 0:
        unions = union_by_persistence(
            all_conn_neighs,
            all_conn_vals,
            all_conn_coords,
            neigh_cart_coords,
            cursor,
            labels=unions,
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
                dist = (
                    (ci - ci1) ** 2 + ((cj - cj1) ** 2) + ((ck - ck1) ** 2)
                ) ** (1 / 2)
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
                if (
                    not use_minima
                    and value >= cutoff
                    or use_minima
                    and value <= cutoff
                ):
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