# -*- coding: utf-8 -*-

import itertools
import math

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from baderkit.core.utilities.basic import (
    dist,
    wrap_point_w_shift,
)
from baderkit.core.utilities.interpolation import linear_slice
from baderkit.core.utilities.union_find import find_root, union

IMAGE_TO_INT = np.empty([3, 3, 3], dtype=np.int64)
INT_TO_IMAGE = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
for shift_idx, (i, j, k) in enumerate(INT_TO_IMAGE):
    IMAGE_TO_INT[i, j, k] = shift_idx


# @njit(cache=True)
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


# @njit(cache=True)
def get_persistence_value(value1, value2, conn_value, p1, p2, p_conn):
    eps = 1e-12
    # get larger and smaller diffs
    diff1 = abs(value1 - conn_value)
    diff2 = abs(value2 - conn_value)

    # get distance between the points
    distance = dist(p1, p_conn) + dist(p2, p_conn)

    # get persistence score:
    # p = smaller_diff*dist / (larger_diff * dist)
    persistence_score = min(diff1, diff2) * distance / (conn_value + eps)
    # persistence_score = min(diff1, diff2) / (max(diff1, diff2) + eps)

    return persistence_score


# @njit(cache=True)
def get_conn_val_from_slice(
    p0,
    p1,
    data,
    max_cart_offset,
    use_minima,
    nx,
    ny,
    nz,
    method,
    matrix,
):

    # get closest p1 image to p0
    p1_w = p1 - np.round(p1 - p0)

    # get points in cartesian coordinates
    c0 = p0 @ matrix
    c1 = p1_w @ matrix
    offset = c1 - c0
    c_conn = c0 + offset / 2

    dist = np.linalg.norm(offset)

    # if above our cutoff, continue
    if dist > max_cart_offset:
        return np.inf, c0, c1, c_conn

    # check if there is a minimum between this point and its neighbor
    n_points = max(math.ceil(dist * 10), 5)
    values = linear_slice(
        data,
        p0,
        p1_w,
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

    return conn_val, c0, c1, c_conn


# @njit(cache=True)
def reduce_by_conn(
    all_conn_neighs,
    all_conn_vals,
    all_conn_coords,
    extrema_cart_coords,
    neigh_cart_coords,
    labels,
    extrema_values,
    extrema_labels,
    use_minima,
    persistence_tol,
):
    all_conn_nums = np.full(
        len(extrema_values), len(extrema_values), dtype=np.int64
    )

    new_union = True

    while new_union:
        new_union = False

        for ext_idx in range(len(extrema_values)):
            # get the persistence values for this extrema
            conn_vals = all_conn_vals[ext_idx]

            # skip if there has been no change for this point and update
            # the number of neighbors
            if len(conn_vals) == all_conn_nums[ext_idx]:
                continue
            all_conn_nums[ext_idx] = len(conn_vals)

            num = len(conn_vals)
            # skip if we have only one value
            if num == 0:
                continue

            neighs = all_conn_neighs[ext_idx]
            # union immediately if we only have one
            if num == 1:
                union(
                    labels, extrema_labels[ext_idx], extrema_labels[neighs[0]]
                )
                new_union = True
                continue

            conn_vals = all_conn_vals[ext_idx]
            conn_coords = all_conn_coords[ext_idx]
            # ext_coords = extrema_cart_coords[ext_idx]
            neigh_coords = neigh_cart_coords[ext_idx]
            # otherwise, we union each neighbor and add a new connection between
            # each of them as well
            for i, neigh_idx in enumerate(neighs):
                union(
                    labels, extrema_labels[ext_idx], extrema_labels[neigh_idx]
                )
                new_union = True
                conn_val = conn_vals[i]
                conn_coord = conn_coords[i]
                # ext_coord = ext_coords[i]
                neigh_coord = neigh_coords[i]
                # get neighbors
                neigh_neighs = all_conn_neighs[neigh_idx]
                neigh_conns = all_conn_vals[neigh_idx]
                neigh_conn_coords = all_conn_coords[neigh_idx]
                neigh_ext_coords = extrema_cart_coords[neigh_idx]
                neigh_neigh_coords = neigh_cart_coords[neigh_idx]

                for j, neigh_idx1 in enumerate(neighs):
                    # skip if this pair has already been added
                    if j <= i or neigh_idx1 in neigh_neighs:
                        continue
                    # otherwise, calculate the persistence
                    conn_val1 = conn_vals[j]
                    neigh_coord1 = neigh_coords[j]
                    conn_coord1 = conn_coords[j]
                    if (
                        use_minima
                        and conn_val >= conn_val1
                        or not use_minima
                        and conn_val <= conn_val1
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
                        neigh_neighs.append(neigh_idx1)
                        neigh_conns.append(val)
                        neigh_conn_coords.append(neigh_coord)
                        neigh_ext_coords.append(neigh_coord1)
                        neigh_neigh_coords.append(coord)

    return labels


# @njit(parallel=True, cache=True)
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
    # create lists to store neighbors and values between them
    all_conn_neighs = []
    all_conn_vals = []
    all_conn_coords = []
    extrema_cart_coords = []
    neigh_cart_coords = []
    # add arrays for each value
    scratch_coord = np.empty(3, dtype=np.float64)
    for i in range(len(extrema_values)):
        all_conn_neighs.append([0][1:])
        all_conn_vals.append([0.0][1:])
        all_conn_coords.append([scratch_coord][1:])
        extrema_cart_coords.append([scratch_coord][1:])
        neigh_cart_coords.append([scratch_coord][1:])

    # get initial connection values
    for ext_idx in prange(len(extrema_values)):

        ext_frac = extrema_frac[ext_idx]
        value = extrema_values[ext_idx]

        connected_neighs = []
        connection_vals = []
        connection_coords = []
        extrema_coords = []
        neigh_coords = []
        for neigh_ext_idx in range(len(extrema_values)):
            # skip self
            if ext_idx == neigh_ext_idx:
                continue

            # skip if neighbor has a lower value
            neigh_value = extrema_values[neigh_ext_idx]
            if neigh_value < value:
                continue

            # get connection value and coordinates
            neigh_frac = extrema_frac[neigh_ext_idx]
            conn_val, c0, c1, c_conn = get_conn_val_from_slice(
                ext_frac,
                neigh_frac,
                data,
                max_cart_offset,
                use_minima,
                nx,
                ny,
                nz,
                method,
                matrix,
            )
            if conn_val == np.inf:
                continue

            # get persistence
            persistence_score = get_persistence_value(
                value, neigh_value, conn_val, c0, c1, c_conn
            )
            # add low persistence to our list
            if persistence_score < persistence_tol:
                connected_neighs.append(neigh_ext_idx)
                connection_vals.append(conn_val)
                connection_coords.append(c_conn)
                extrema_coords.append(c0)
                neigh_coords.append(c1)

        all_conn_neighs[ext_idx] = connected_neighs
        all_conn_vals[ext_idx] = connection_vals
        all_conn_coords[ext_idx] = connection_coords
        extrema_cart_coords[ext_idx] = extrema_coords
        neigh_cart_coords[ext_idx] = neigh_coords
    return (
        all_conn_neighs,
        all_conn_vals,
        all_conn_coords,
        extrema_cart_coords,
        neigh_cart_coords,
    )


# @njit(cache=True)
def group_by_persistence(
    data,
    critical_vox,
    basin_connections,
    saddle_values,
    saddle_cart,
    persistence_tol,
    matrix,
    use_minima=False,
):
    num_critical = len(critical_vox)

    critical_frac = critical_vox / np.array(data.shape)

    # get values at critical points
    critical_values = np.empty(len(critical_vox), dtype=np.float64)
    for idx, (i, j, k) in enumerate(critical_vox):
        critical_values[idx] = data[i, j, k]

    # create lists to store neighbors and values between them
    all_conn_neighs = []
    all_conn_vals = []
    all_conn_coords = []
    extrema_cart_coords = []
    neigh_cart_coords = []
    # add arrays for each value
    scratch_coord = np.empty(3, dtype=np.float64)
    for i in range(num_critical):
        all_conn_neighs.append([0][1:])
        all_conn_vals.append([0.0][1:])
        all_conn_coords.append([scratch_coord][1:])
        extrema_cart_coords.append([scratch_coord][1:])
        neigh_cart_coords.append([scratch_coord][1:])

    # create array to track unions between basins
    unions = np.arange(num_critical)

    # condense to lists
    for pair_idx, (
        (crit1, crit2, image1, image2),
        conn_val,
        conn_cart,
    ) in enumerate(
        zip(
            basin_connections,
            saddle_values,
            saddle_cart,
        )
    ):
        c1 = (critical_frac[crit1] + INT_TO_IMAGE[image1]) @ matrix
        c2 = (critical_frac[crit2] + INT_TO_IMAGE[image2]) @ matrix
        score = get_persistence_value(
            critical_values[crit1],
            critical_values[crit2],
            conn_val,
            c1,
            c2,
            conn_cart,
        )
        if score < persistence_tol:
            all_conn_neighs[crit1].append(crit2)
            all_conn_vals[crit1].append(conn_val)
            all_conn_coords[crit1].append(conn_cart)
            extrema_cart_coords[crit1].append(c1)
            neigh_cart_coords[crit1].append(c2)

    unions = reduce_by_conn(
        all_conn_neighs,
        all_conn_vals,
        all_conn_coords,
        extrema_cart_coords,
        neigh_cart_coords,
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


# @njit(cache=True, parallel=True)
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


# @njit(cache=True)
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


# @njit(inline='always', cache=True)
def get_extrema_saddle_connections(
    i,
    j,
    k,
    nx,
    ny,
    nz,
    ny_nz,
    labels,
    images,
    data,
    neighbor_transforms,
    edge_mask,
    max_val,
    use_minima=False,
):

    # iterate over transforms
    label0 = labels[i, j, k]
    image0 = images[i, j, k]
    value0 = data[i, j, k]

    label1 = -1
    image1 = -1
    value1 = -1.0
    n = 0

    for trans in range(neighbor_transforms.shape[0]):
        # get shifts
        si = neighbor_transforms[trans, 0]
        sj = neighbor_transforms[trans, 1]
        sk = neighbor_transforms[trans, 2]

        # wrap around periodic edges and store shift
        ii, jj, kk, ssi, ssj, ssk = wrap_point_w_shift(
            i + si, j + sj, k + sk, nx, ny, nz
        )
        # skip neighbors that aren't part of the edge
        if edge_mask[ii, jj, kk] == 0:
            continue

        # get the label and image of this neighbor
        neigh_label = labels[ii, jj, kk]
        neigh_image = images[ii, jj, kk]
        neigh_value = data[ii, jj, kk]

        # update image to be relative to the current points transformation
        si1 = INT_TO_IMAGE[neigh_image, 0] + ssi
        sj1 = INT_TO_IMAGE[neigh_image, 1] + ssj
        sk1 = INT_TO_IMAGE[neigh_image, 2] + ssk
        neigh_image = IMAGE_TO_INT[si1, sj1, sk1]

        # skip neighbors in the same basin
        if label0 == neigh_label and image0 == neigh_image:
            continue

        # if we haven't already, note the correct neighboring basin
        if n == 0:
            label1 = neigh_label
            image1 = neigh_image
            value1 = neigh_value
            n = 1
            continue

        # check if this point improves the connection value
        if use_minima and neigh_value < value1:
            value1 = neigh_value
            if value1 <= value0:
                break
        elif not use_minima and neigh_value > value1:
            value1 = neigh_value
            if value1 >= value0:
                break

    # if no neighbor was found, we just return a fake value
    if label1 == -1:
        return max_val, max_val, max_val, False, 0.0

    # otherwise we get the unit cell transform across which these extrema connect
    i, j, k = INT_TO_IMAGE[image0]
    ii, jj, kk = INT_TO_IMAGE[image1]
    image = IMAGE_TO_INT[ii - i, jj - j, kk - k]
    inv_image = IMAGE_TO_INT[i - ii, j - jj, k - kk]

    # get best value
    if use_minima:
        best_value = max(value0, value1)
    else:
        best_value = min(value0, value1)

    # determine if the canonical image is reversed. We flip it if the neighboring
    # label is lower and if the neighboring image is lower.
    is_reversed = (label0 > label1) != (image > inv_image)

    return (
        min(label0, label1),
        max(label0, label1),
        min(image, inv_image),
        is_reversed,
        best_value,
    )


# @njit(parallel=True, cache=True)
def get_canonical_saddle_connections(
    labels: NDArray[np.int64],
    images: NDArray[np.int64],
    data: NDArray[np.float64],
    neighbor_transforms: NDArray[np.int64],
    edge_mask: NDArray[np.uint8],
    use_minima: bool = False,
):
    nx, ny, nz = labels.shape
    ny_nz = ny * nz

    # get the points that may be saddles
    saddle_coords = np.argwhere(edge_mask == 1)

    # create an array to track connections between these points.
    # For each entry we will have:
    # 1: the lower label index
    # 2: the higher label index
    # 3: the connection image between basins
    # 4: whether or not the connection image is lower -> higher (0) or higher -> lower (1)
    saddle_connections = np.empty((len(saddle_coords), 4), dtype=np.int16)
    connection_vals = np.empty(len(saddle_coords), dtype=np.float64)
    # create a mask to track important connections
    important = np.ones(len(saddle_coords), dtype=np.bool)
    max_val = np.iinfo(np.int16).max
    for idx in prange(len(saddle_coords)):
        i, j, k = saddle_coords[idx]
        lower, higher, shift, is_reversed, connection_value = (
            get_extrema_saddle_connections(
                i,
                j,
                k,
                nx,
                ny,
                nz,
                ny_nz,
                labels,
                images,
                data,
                neighbor_transforms,
                edge_mask,
                max_val,
                use_minima,
            )
        )
        if lower == max_val:
            # note this wasn't a true saddle
            important[idx] = False
            continue
        saddle_connections[idx, 0] = lower
        saddle_connections[idx, 1] = higher
        saddle_connections[idx, 2] = shift
        saddle_connections[idx, 3] = is_reversed
        connection_vals[idx] = connection_value

    # get only the connections that are important
    important = np.where(important)[0]
    saddle_coords = saddle_coords[important]
    saddle_connections = saddle_connections[important]
    connection_vals = connection_vals[important]

    return saddle_coords, saddle_connections, connection_vals


# @njit(cache=True)
def get_single_point_saddles(
    data,
    connection_values,
    saddle_coords,
    connection_indices,
    num_connections,
    use_minima=False,
):
    # create an array to store best points
    saddles = np.empty(num_connections, dtype=np.int32)
    if use_minima:
        best_vals = np.full(num_connections, np.inf, dtype=np.float64)
    else:
        best_vals = np.full(num_connections, -np.inf, dtype=np.float64)

    for saddle_idx, (idx, connection_value) in enumerate(
        zip(connection_indices, connection_values)
    ):
        if not use_minima and connection_value > best_vals[idx]:
            best_vals[idx] = connection_value
            saddles[idx] = saddle_idx
        elif use_minima and connection_value < best_vals[idx]:
            best_vals[idx] = connection_value
            saddles[idx] = saddle_idx

    return saddles, best_vals
