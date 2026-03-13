# -*- coding: utf-8 -*-

import itertools
import math

import numpy as np
from numba import njit, prange, types
from numpy.typing import NDArray
from numba.typed import Dict, List
from numba.types import UniTuple, int16, int64

from baderkit.core.utilities.basic import (
    dist,
    wrap_point,
    wrap_point_w_shift,
)
from baderkit.core.utilities.interpolation import linear_slice, spline_hess
from baderkit.core.utilities.union_find import find_root, union

IMAGE_TO_INT = np.empty([3, 3, 3], dtype=np.int64)
INT_TO_IMAGE = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
for shift_idx, (i, j, k) in enumerate(INT_TO_IMAGE):
    IMAGE_TO_INT[i, j, k] = shift_idx

FACE_TRANSFORMS = np.array(
    [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ],
    dtype=np.int64,
)


@njit(cache=True)
def compute_wrap_offset(point1, point2):
    """
    Computes wrap from point1 to point2

    """
    best_d2 = np.inf
    best_i = 0
    best_j = 0
    best_k = 0

    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                dx = (point2[0] + i) - point1[0]
                dy = (point2[1] + j) - point1[1]
                dz = (point2[2] + k) - point1[2]
                d2 = dx * dx + dy * dy + dz * dz

                if d2 < best_d2:
                    best_d2 = d2
                    best_i = i
                    best_j = j
                    best_k = k

    return best_i, best_j, best_k


# Predefine types
key_type = UniTuple(int64, 3)
value_type = int64


@njit(cache=True)
def unique_and_inverse_axis0(arr):
    n = arr.shape[0]
    arr64 = arr.astype(np.int64)

    # Use the predeclared Numba types
    d = Dict.empty(key_type=key_type, value_type=value_type)

    inverse = np.empty(n, dtype=np.int64)
    unique = List.empty_list(key_type)

    next_idx = 0

    for i in range(n):
        key = (arr64[i, 0], arr64[i, 1], arr64[i, 2])

        if key in d:
            inverse[i] = d[key]
        else:
            d[key] = next_idx
            inverse[i] = next_idx
            unique.append(key)
            next_idx += 1

    unique_arr = np.empty((next_idx, 3), dtype=arr.dtype)

    for i in range(next_idx):
        row = unique[i]
        unique_arr[i, 0] = row[0]
        unique_arr[i, 1] = row[1]
        unique_arr[i, 2] = row[2]

    return unique_arr, inverse

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
    # print(persistence_score)
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
def reduce_by_conn(
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
def get_conn_val_from_slice(
    p0,
    p1,
    n_points,
    data,
    use_minima,
    nx,
    ny,
    nz,
    method,
):

    # check if there is a minimum between this point and its neighbor
    values = linear_slice(
        data,
        p0,
        p1,
        n=n_points,
        is_frac=True,
        method=method,
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
def get_conn_vals(
    data,
    labels,
    extrema_values,
    extrema_labels,
    extrema_frac,
    max_cart_offset,
    use_minima,
    persistence_tol,
    method,
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

            conn_val = get_conn_val_from_slice(
                ext_frac,
                wrapped_neigh_frac,
                n_points,
                data,
                use_minima,
                nx,
                ny,
                nz,
                method,
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
def group_by_persistence(
    data,
    critical_vox,
    basin_connections,
    saddle_values,
    saddle_carts,
    persistence_tol,
    matrix,
    use_minima=False,
):
    num_critical = len(critical_vox)

    critical_frac = critical_vox / np.array(data.shape, dtype=np.float64)
    critical_cart = critical_frac @ matrix

    # get values at critical points
    critical_values = np.empty(len(critical_vox), dtype=np.float64)
    for idx, (i, j, k) in enumerate(critical_vox):
        critical_values[idx] = data[i, j, k]

    # get max neighbor count
    counts = np.zeros(num_critical, dtype=np.int64)
    for crit1, crit2, image in basin_connections:
        counts[crit1] += 1
    max_count = counts.max()

    # create arrays to track connections
    max_degree = max_count
    all_conn_neighs = np.full((num_critical, max_degree), -1, np.int64)
    all_conn_vals = np.zeros((num_critical, max_degree), np.float64)
    all_conn_coords = np.zeros((num_critical, max_degree, 3), np.float64)
    neigh_cart_coords = np.zeros((num_critical, max_degree, 3), np.float64)

    cursor = np.zeros(num_critical, dtype=np.int64)

    for crit_idx in prange(num_critical):
        for i, (crit1, crit2, image) in enumerate(basin_connections):
            if crit1 != crit_idx:
                continue
            neigh_count = cursor[crit1]

            saddle_val = saddle_values[i]
            saddle_cart = saddle_carts[i]

            c1 = critical_cart[crit1]
            c2 = (critical_frac[crit2] + INT_TO_IMAGE[image]) @ matrix

            score = get_persistence_value(
                critical_values[crit1],
                critical_values[crit2],
                saddle_val,
                c1,
                c2,
                saddle_cart,
            )
            if score < persistence_tol:
                all_conn_neighs[crit_idx, neigh_count] = crit2
                all_conn_vals[crit_idx, neigh_count] = saddle_val
                all_conn_coords[crit_idx, neigh_count] = saddle_cart
                neigh_cart_coords[crit_idx, neigh_count] = c2

                cursor[crit_idx] += 1



    # create array to track unions between basins
    unions = np.arange(num_critical)

    if max_count > 0:
        unions = reduce_by_conn(
            all_conn_neighs,
            all_conn_vals,
            all_conn_coords,
            neigh_cart_coords,
            cursor,
            labels=unions,
            extrema_values=critical_values,
            extrema_labels=np.arange(len(critical_values)),
            use_minima=use_minima,
            persistence_tol=persistence_tol,
        )

    # get the roots of all extrema
    roots = np.empty(num_critical, dtype=np.int64)
    for idx in range(num_critical):
        root = find_root(unions, idx)
        roots[idx] = root

    unique_roots = np.unique(roots)
    # update roots to the highest voxel in each group
    for root in unique_roots:
        best_value = critical_values[root]
        best_idx = root
        for idx, root_idx in enumerate(roots):
            if root_idx != root:
                continue
            if use_minima and critical_values[idx] < best_value:
                best_value = critical_values[idx]
                best_idx = idx
        # relabel all points to the new root
        for idx, root_idx in enumerate(roots):
            if root_idx != root:
                continue
            roots[idx] = best_idx

    root_transforms = np.empty((len(roots), 3), dtype=np.int8)
    # Get the transformations from each merged point to its parents
    for ext_idx, root_idx in enumerate(roots):
        crit_frac = critical_frac[ext_idx]
        root_frac = critical_frac[root_idx]
        root_transforms[ext_idx] = compute_wrap_offset(crit_frac, root_frac)

    return roots, root_transforms


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


@njit(inline="always", cache=True)
def get_extrema_saddle_connections(
    i,
    j,
    k,
    nx,
    ny,
    nz,
    data,
    labels,
    images,
    neighbor_transforms,
    max_val,
    use_minima,
):

    # iterate over transforms
    label = labels[i, j, k]
    image = images[i, j, k]
    value = data[i, j, k]
    im, jm, km = INT_TO_IMAGE[image]

    for trans in range(neighbor_transforms.shape[0]):
        # get shifts
        si = neighbor_transforms[trans, 0]
        sj = neighbor_transforms[trans, 1]
        sk = neighbor_transforms[trans, 2]

        # wrap around periodic edges and store shift
        ii, jj, kk, ssi, ssj, ssk = wrap_point_w_shift(
            i + si, j + sj, k + sk, nx, ny, nz
        )
        # skip points with a higher value
        if (
            use_minima
            and data[ii, jj, kk] > value
            or not use_minima
            and data[ii, jj, kk] < value
        ):
            continue

        # get the label and image of this neighbor
        neigh_label = labels[ii, jj, kk]
        neigh_image = images[ii, jj, kk]

        # update image to be relative to the current points transformation
        si1 = INT_TO_IMAGE[neigh_image, 0] + ssi
        sj1 = INT_TO_IMAGE[neigh_image, 1] + ssj
        sk1 = INT_TO_IMAGE[neigh_image, 2] + ssk
        neigh_image = IMAGE_TO_INT[si1, sj1, sk1]

        # note if this point belongs to a different basin
        if label != neigh_label or image != neigh_image:
            image = IMAGE_TO_INT[si1 - im, sj1 - jm, sk1 - km]

            return (
                label,
                neigh_label,
                image,
            )

    # if no neighbor was found, we just return a fake value
    return max_val, max_val, max_val


@njit(cache=True)
def get_single_point_saddles(
    connection_values,
    connection_indices,
    initial_indices,
    num_connections,
    use_minima=False,
):
    if use_minima:
        best_vals = np.full(num_connections, np.inf, dtype=np.float64)
    else:
        best_vals = np.full(num_connections, -np.inf, dtype=np.float64)
    best_indices = initial_indices.copy()

    for saddle_idx, (idx, connection_value) in enumerate(
        zip(connection_indices, connection_values)
    ):
        if not use_minima and connection_value > best_vals[idx]:
            best_vals[idx] = connection_value
            best_indices[idx] = saddle_idx
        elif use_minima and connection_value < best_vals[idx]:
            best_vals[idx] = connection_value
            best_indices[idx] = saddle_idx

    return best_vals, best_indices


@njit(cache=True)
def is_ongrid_saddle(
    data,
    i,
    j,
    k,
    nx,
    ny,
    nz,
    neighbor_transforms,
    edge_mask,
    use_minima,
):
    # get initial value and label
    value = data[i, j, k]

    for si, sj, sk in neighbor_transforms:
        # wrap around periodic edges
        ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)

        # skip points that aren't also on the edge
        if edge_mask[ii, jj, kk] != 1:
            continue

        neigh_value = data[ii, jj, kk]
        # check if a neighbor is a better saddle candidate
        if (
            use_minima
            and neigh_value < value
            or not use_minima
            and neigh_value > value
        ):
            return False

    return True

@njit(fastmath=True, cache=True)
def check_valid_newton_step(
    coord,
    data,
    r_voxel_cart2,
    inv_G,
    H_frac_to_cart,
):
    """
    Newton-step rejection using *full local neighborhood information*.
    Uses full 3x3x3 Hessian and a conservative spectral bound.
    """
    nx, ny, nz = data.shape
    eps = 1e-12
    i, j, k = coord
    i = int(i)
    j = int(j)
    k = int(k)

    # -----------------------------
    # Pull 3x3x3 neighborhood
    # -----------------------------
    n = np.empty((3, 3, 3), dtype=data.dtype)
    for di in range(-1, 2):
        ii = (i + di) % nx
        for dj in range(-1, 2):
            jj = (j + dj) % ny
            for dk in range(-1, 2):
                kk = (k + dk) % nz
                n[di + 1, dj + 1, dk + 1] = data[ii, jj, kk]

    f0 = n[1, 1, 1]

    # -----------------------------
    # Gradient (fractional)
    # -----------------------------
    gi = 0.5 * (n[2, 1, 1] - n[0, 1, 1])
    gj = 0.5 * (n[1, 2, 1] - n[1, 0, 1])
    gk = 0.5 * (n[1, 1, 2] - n[1, 1, 0])

    # Cartesian gradient norm²
    g_cart2 = (
        gi * (inv_G[0, 0] * gi + inv_G[0, 1] * gj + inv_G[0, 2] * gk)
        + gj * (inv_G[1, 0] * gi + inv_G[1, 1] * gj + inv_G[1, 2] * gk)
        + gk * (inv_G[2, 0] * gi + inv_G[2, 1] * gj + inv_G[2, 2] * gk)
    )

    # -----------------------------
    # Full Hessian (fractional)
    # -----------------------------
    Hxx = n[2, 1, 1] - 2.0 * f0 + n[0, 1, 1]
    Hyy = n[1, 2, 1] - 2.0 * f0 + n[1, 0, 1]
    Hzz = n[1, 1, 2] - 2.0 * f0 + n[1, 1, 0]

    Hxy = 0.25 * (n[2, 2, 1] - n[2, 0, 1] - n[0, 2, 1] + n[0, 0, 1])
    Hxz = 0.25 * (n[2, 1, 2] - n[2, 1, 0] - n[0, 1, 2] + n[0, 1, 0])
    Hyz = 0.25 * (n[1, 2, 2] - n[1, 2, 0] - n[1, 0, 2] + n[1, 0, 0])

    # -----------------------------
    # Local curvature scale (∞-norm)
    # -----------------------------
    H_frac_max = (
        max(abs(Hxx), abs(Hyy), abs(Hzz), abs(Hxy), abs(Hxz), abs(Hyz)) + eps
    )

    H_cart_max = H_frac_max * H_frac_to_cart
    H_cart_max2 = H_cart_max * H_cart_max

    # -----------------------------
    # gradient rejection
    # -----------------------------
    if g_cart2 > H_cart_max2:
        return -1

    # -----------------------------
    # Morse index via LDLᵀ
    # -----------------------------
    morse = 0

    # D1
    D1 = Hxx
    if D1 < 0.0:
        morse += 1
    D1 = D1 if abs(D1) > eps else eps

    # L21, L31
    L21 = Hxy / D1
    L31 = Hxz / D1

    # D2
    D2 = Hyy - L21 * Hxy
    if D2 < 0.0:
        morse += 1
    D2 = D2 if abs(D2) > eps else eps

    # L32
    L32 = (Hyz - L31 * Hxy) / D2

    # D3
    D3 = Hzz - L31 * Hxz - L32 * (Hyz - L31 * Hxy)
    if D3 < 0.0:
        morse += 1

    return morse

@njit(cache=True, parallel=True)
def get_canonical_saddle_connections(
    saddle_coords,
    data,
    labels: NDArray[np.int64],
    images: NDArray[np.int64],
    neighbor_transforms: NDArray[np.int64],
    use_minima,
        ):
    nx,ny,nz = labels.shape
    # create an array to track connections between these points.
    # For each entry we will have:
    # 1: the lower label index
    # 2: the higher label index
    # 3: the connection image between basins
    # 4: whether or not the connection image is lower -> higher (0) or higher -> lower (1)
    saddle_connections = np.empty((len(saddle_coords), 3), dtype=np.int16)

    max_val = np.iinfo(np.int16).max
    for idx in prange(len(saddle_coords)):
        i, j, k = saddle_coords[idx]
        lower, higher, shift = get_extrema_saddle_connections(
            i,
            j,
            k,
            nx,
            ny,
            nz,
            data,
            labels,
            images,
            neighbor_transforms,
            max_val,
            use_minima,
        )
        saddle_connections[idx, 0] = lower
        saddle_connections[idx, 1] = higher
        saddle_connections[idx, 2] = shift
    return saddle_connections

@njit(parallel=True, cache=True)
def get_saddles_from_basins(
    labels: NDArray[np.int64],
    images: NDArray[np.int64],
    data: NDArray[np.float64],
    neighbor_transforms: NDArray[np.int64],
    edge_mask: NDArray[np.uint8],
    use_minima: bool = False,
):
    nx, ny, nz = labels.shape

    # get the points that may be saddles
    saddle_coords = np.argwhere(edge_mask == 1)

    # first we check each point on the edge mask and determine if they are
    # potential saddles
    saddle_mask = np.zeros(len(saddle_coords), dtype=np.bool)
    for saddle_idx in prange(len(saddle_coords)):
        i, j, k = saddle_coords[saddle_idx]

        if is_ongrid_saddle(
            data,
            i,
            j,
            k,
            nx,
            ny,
            nz,
            neighbor_transforms,
            edge_mask,
            use_minima,
        ):
            saddle_mask[saddle_idx] = True

    # Now we reduce to the possible saddle points
    saddle_indices = np.where(saddle_mask)[0]
    saddle_coords = saddle_coords[saddle_indices]

    # get neighboring basins
    saddle_connections = get_canonical_saddle_connections(
        saddle_coords,
        data,
        labels=labels,
        images=images,
        neighbor_transforms=neighbor_transforms,
        use_minima=use_minima,
            )

    return saddle_coords, saddle_connections

@njit(parallel=True, cache=True)
def remove_false_saddles(
    saddle_coords,
    labels: NDArray[np.int64],
    images: NDArray[np.int64],
    data: NDArray[np.float64],
    neighbor_transforms: NDArray[np.int64],
    matrix,
    use_minima: bool = False,
        ):
    nx, ny, nz = data.shape

    # Metric tensors
    G = matrix @ matrix.T
    inv_G = np.linalg.inv(G)
    lam = np.linalg.eigvalsh(G)

    lam_min = lam[0]
    lam_max = lam[2]

    frac_to_cart = np.sqrt(lam_max)
    H_frac_to_cart = 1.0 / lam_min

    # Voxel radius in Cartesian
    r_voxel_cart = 0.5 * frac_to_cart
    r_voxel_cart2 = r_voxel_cart * r_voxel_cart

    saddle_mask = np.zeros(len(saddle_coords), dtype=np.bool)
    for saddle_idx in prange(len(saddle_coords)):
        i, j, k = saddle_coords[saddle_idx]
        # check if a newton step would move outside the voxel and
        # get morse index
        morse_idx = check_valid_newton_step(
            (i, j, k),
            data,
            r_voxel_cart2,
            inv_G,
            H_frac_to_cart,
        )
        if (
            use_minima and morse_idx == 1
            or not use_minima and morse_idx == 2
            ):
            # this is a saddle
            saddle_mask[saddle_idx] = True

    # remove false saddles
    saddle_indices = np.where(saddle_mask)[0]
    saddle_coords = saddle_coords[saddle_indices]

    # get neighboring basins
    saddle_connections = get_canonical_saddle_connections(
        saddle_coords,
        data,
        labels=labels,
        images=images,
        neighbor_transforms=neighbor_transforms,
        use_minima=use_minima,
            )
    # remove false saddles
    true_saddles = np.where(saddle_connections[:,0] != np.iinfo(saddle_connections.dtype).max)[0]
    saddle_indices=saddle_indices[true_saddles]
    saddle_connections=saddle_connections[true_saddles]

    return saddle_coords, saddle_connections