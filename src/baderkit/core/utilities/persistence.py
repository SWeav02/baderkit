# -*- coding: utf-8 -*-

import math

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from baderkit.core.utilities.basic import (
    coords_to_flat,
    flat_to_coords,
    dist,
    get_transforms_in_voxels
)
from baderkit.core.utilities.critical_points import (
    refine_critical_points,
    refine_critical_points_targeted,
    is_ongrid_newton_crit,

    )
from baderkit.core.utilities.basins import get_best_neighbor_with_shift
from baderkit.core.utilities.interpolation import linear_slice
from baderkit.core.utilities.transforms import (
    IMAGE_TO_INT,
    INT_TO_IMAGE,
)
from baderkit.core.utilities.union_find import (
    find_root_with_shift1,
    union,
    union_with_shift1,
)

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
    d2 = distance * distance
    if distance == 0:
        return 0.0

    # get approximate average value between extrema
    avg = (
        p1_dist * (value1 + conn_value) / 2 + p2_dist * (value2 + conn_value) / 2
    ) / distance

    # get persistence score:
    #   p = smaller_diff*dist^2 / (average_value)
    persistence_score = min(diff1, diff2) * d2 / (abs(avg) + eps)

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
    shape = np.array(labels.shape)
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
                neigh_vox = neigh_coord * shape

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
                    neigh_vox1 = neigh_coord1 * shape

                    if (use_minima and conn_val >= conn_val1) or (
                        not use_minima and conn_val <= conn_val1
                    ):
                        val = conn_val
                        coord = conn_coord
                    else:
                        val = conn_val1
                        coord = conn_coord1
                    coord_vox = coord * shape

                    score = get_persistence_value(
                        extrema_values[neigh_idx],
                        extrema_values[neigh_idx1],
                        val,
                        neigh_vox,
                        neigh_vox1,
                        coord_vox,
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

    values = linear_slice(
        data,
        p0,
        p1,
        n=n_points,
        is_frac=True,
        method="linear",
    )

    s = np.sign(np.diff(values))

    # default index (fallback)
    idx = 0

    if use_minima:
        s = np.append(-1, s)
        s = np.append(s, 1)

        min_indices = np.where((s[:-1] <= 0) & (s[1:] >= 0))[0]

        if len(min_indices) < 2:
            idx = np.argmax(values)
        else:
            min0 = min_indices[0]
            min1 = min_indices[-1]
            local_idx = np.argmax(values[min0:min1])
            idx = min0 + local_idx

    else:
        s = np.append(1, s)
        s = np.append(s, -1)

        max_indices = np.where((s[:-1] >= 0) & (s[1:] <= 0))[0]

        if len(max_indices) < 2:
            idx = np.argmin(values)
        else:
            max0 = max_indices[0]
            max1 = max_indices[-1]
            local_idx = np.argmin(values[max0:max1])
            idx = max0 + local_idx

    conn_val = values[idx]

    # convert index -> fractional coordinate
    if n_points > 1:
        t = idx / (n_points - 1.0)
    else:
        t = 0.0

    frac_pos = p0 + t * (p1 - p0)

    return conn_val, frac_pos

@njit(cache=True)
def group_by_low_approx_persistence(
    data,
    labels,
    images,
    extrema_values,
    extrema_labels,
    extrema_vox,
    max_cart_offset,
    use_minima,
    persistence_tol,
    matrix,
):
    shape = np.array(data.shape)
    nx, ny, nz = data.shape
    extrema_frac = extrema_vox / shape
    extrema_cart = extrema_frac @ matrix
    num_extrema = len(extrema_values)

    # get approximate saddle points and connected basins
    saddle_coords = []
    saddle_connections = []
    saddle_values = []

    for ext_idx in range(num_extrema):
        ext_frac = extrema_frac[ext_idx]
        ext_cart = extrema_cart[ext_idx]
        ext_vox = extrema_vox[ext_idx]
        ext_value = extrema_values[ext_idx]

        for neigh_ext_idx in range(num_extrema):
            # skip the original index
            if ext_idx == neigh_ext_idx:
                continue
            # skip neighbors with less extreme values
            neigh_ext_value = extrema_values[neigh_ext_idx]
            if (
                use_minima and neigh_ext_value > ext_value
                or not use_minima and neigh_ext_value < ext_value
                or neigh_ext_value == ext_value and neigh_ext_idx > ext_idx
            ):
                continue

            # get neighbor frac coord
            neigh_frac = extrema_frac[neigh_ext_idx]
            # get shift from neigh coord to extrema
            shift = np.round(ext_frac - neigh_frac).astype(np.int8)
            # wrap to be as close as possible
            wrapped_neigh_frac = neigh_frac + shift
            wrapped_neigh_vox = wrapped_neigh_frac * shape

            # check if this neighbor is outside our cutoff and continue if so
            neigh_cart = wrapped_neigh_frac @ matrix

            offset = neigh_cart - ext_cart
            dist = np.linalg.norm(offset)
            if dist > max_cart_offset:
                continue

            # get the value these extrema connect at
            n_points = max(int(round(dist * 20)), 5)
            conn_val, conn_frac = get_approx_saddle_val(
                ext_frac,
                wrapped_neigh_frac,
                n_points,
                data,
                use_minima,
                nx,
                ny,
                nz,
            )

            # get voxel coordinate of saddle point
            saddle_frac = ext_frac + ((wrapped_neigh_frac - ext_frac)*conn_frac)
            saddle_vox = saddle_frac * shape
            # get persistence
            persistence_score = get_persistence_value(
                ext_value, neigh_ext_value, conn_val, ext_vox, wrapped_neigh_vox, saddle_vox
            )

            # if our persistence is below our tolerance we add this saddle
            if persistence_score < persistence_tol:
                saddle_coords.append(saddle_vox)
                image = IMAGE_TO_INT[shift[0], shift[1], shift[2]]
                saddle_connections.append(
                    np.array((ext_idx, neigh_ext_idx, 13, image), dtype=np.int16)
                )
                saddle_values.append(conn_val)

    # convert lists to arrays
    saddle_coords_array = np.empty((len(saddle_coords), 3), dtype=np.float64)
    saddle_connections_array = np.empty((len(saddle_coords), 4), dtype=np.int16)
    for idx, (coord, conn) in enumerate(zip(saddle_coords, saddle_connections)):
        saddle_coords_array[idx] = coord
        saddle_connections_array[idx] = conn
    saddle_values = np.array(saddle_values, dtype=np.float64)

    # get unions and images
    unions, ext_images = group_by_persistence(
        data=data,
        extrema_vox=extrema_vox,
        extrema_values=extrema_values,
        saddle_connections=saddle_connections_array,
        saddle_values=saddle_values,
        saddle_vox=saddle_coords_array,
        persistence_tol=persistence_tol,
        matrix=matrix,
        use_minima=use_minima,
    )

    # update labels and images
    for ext_idx, (root_idx, image) in enumerate(zip(unions, ext_images)):
        label = extrema_labels[ext_idx]
        root_label = extrema_labels[root_idx]
        labels[label] = root_label
        images[label] = image

    # get reduced root indices
    roots = np.empty(len(extrema_labels), dtype=labels.dtype)
    for ext_idx in range(len(extrema_labels)):
        label = extrema_labels[ext_idx]
        root, shift = find_root_with_shift1(labels, images, label)
        roots[ext_idx] = root
        labels[label] = root
        images[label] = shift
    root_indices = np.where(roots == extrema_labels)[0]

    return labels, images, root_indices


# @njit(parallel=True, cache=True)
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
    nx, ny, nz = shape
    ny_nz = ny*nz

    # refine extrema
    target_idx = 0 if use_minima else 3
    refined_vox, successes, ctypes = refine_critical_points(
        critical_coords=extrema_vox,
        data=data,
        matrix=matrix,
        target_indices=(target_idx,),
        max_change=1000,
        max_iter=100,
        grad_tol=0.00001,
        h=0.5,
        eig_rel_tol=0.0001,
    )
    # combine successful indices that refined to the same point
    success_indices = np.where(successes)[0]

    # get best neighbors of each
    rounded = np.round(refined_vox).astype(np.int64) % shape
    rounded_frac = rounded / shape

    best_neighs = np.arange(len(success_indices))
    best_shifts = np.zeros((len(success_indices), 3), dtype=np.int8)

    for idx in prange(len(success_indices)):
        ext_idx = success_indices[idx]
        ext_frac = rounded_frac[ext_idx]
        ext_vox = rounded[ext_idx]
        ext_val = extrema_values[ext_idx]
        ext_orig_frac = extrema_frac[ext_idx]

        best_idx = ext_idx
        best_value = ext_val
        best_shift = np.zeros(3, dtype=np.int8)
        for idx1 in range(len(success_indices)):
            neigh_idx = success_indices[idx1]
            neigh_val = extrema_values[neigh_idx]
            # skip neighbors with values lower than the current best
            if (
                use_minima
                and neigh_val > best_value
                or not use_minima
                and neigh_val < best_value
                or neigh_val == best_value
                and neigh_idx > best_idx
            ):
                continue

            # wrap neighbor to be as close as possible
            neigh_frac = rounded_frac[neigh_idx]
            wrapped = neigh_frac - np.round(neigh_frac - ext_frac)
            wrapped_vox = wrapped * shape
            offset = wrapped_vox - ext_vox

            # skip points more than one voxel away
            if np.max(np.abs(offset)) > 1 + 1e-12:
                continue

            # get unrefined shift from neighbor to ext
            neigh_orig_frac = extrema_frac[neigh_idx]
            shift = np.round(ext_orig_frac - neigh_orig_frac).astype(np.int8)

            best_idx = neigh_idx
            best_shift = shift
        # update label and image
        best_neighs[idx] = best_idx
        best_shifts[idx] = best_shift

    # go through and make unions
    for idx, (idx1, shift) in enumerate(zip(best_neighs, best_shifts)):
        label = extrema_labels[success_indices[idx]]
        neigh_label = extrema_labels[idx1]
        union_with_shift1(labels, images, label, neigh_label, shift)

    # for unsuccessful refinements, we try to hill climb to a true extrema
    transforms1, transform_dists1 = get_transforms_in_voxels(1, nx, ny, nz, matrix)
    transforms2, transform_dists2 = get_transforms_in_voxels(2, nx, ny, nz, matrix)

    false_indices = np.where(successes==False)[0]
    for idx in prange(len(false_indices)):
        ext_idx = false_indices[idx]
        # get initial coords
        i, j, k = extrema_vox[ext_idx]
        image = np.zeros(3, dtype=np.int8)
        while True:
            # get neighbors within one voxel
            _, (x, y, z), shift = get_best_neighbor_with_shift(
                data=data,
                i=i,
                j=j,
                k=k,
                neighbor_transforms=transforms1,
                neighbor_dists=transform_dists1,
                use_minima=use_minima,
            )
            if i != x or j != y or k != z:
                # update point and continue
                i = x
                j = y
                k = z
                image += shift
                continue
            # get neighbors within two voxels
            _, (x, y, z), shift = get_best_neighbor_with_shift(
                data=data,
                i=i,
                j=j,
                k=k,
                neighbor_transforms=transforms2,
                neighbor_dists=transform_dists2,
                use_minima=use_minima,
            )
            if i != x or j != y or k != z:
                # update point and continue
                i = x
                j = y
                k = z
                image += shift
                continue
            # if were still here, we call this our final stop and break
            break
        # make a union to the final neighbor
        label = extrema_labels[ext_idx]
        neigh_label = coords_to_flat(i, j, k, ny_nz, nz)
        union_with_shift1(labels, images, label, neigh_label, image)

    # get reduced root indices
    roots = np.empty(len(extrema_labels), dtype=labels.dtype)
    for ext_idx in range(len(extrema_labels)):
        label = extrema_labels[ext_idx]
        root, shift = find_root_with_shift1(labels, images, label)
        roots[ext_idx] = root
        labels[label] = root
        images[label] = shift
    root_indices = np.where(roots == extrema_labels)[0]

    return labels, images, refined_vox[root_indices], roots, root_indices


# @njit(cache=True)
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
    # Combine low-persistence extrema
    ###########################################################################
    # With the right shape (e.g. highly anisotropic) a maximum/minimum may lay offgrid
    # and cause two ongrid points to appear to be higher than the region around
    # them. We merge these by taking a linear slice between each point using
    # cubic interpolation and combining those that have no minimum/maximum between
    # them.

    labels, images, root_indices = group_by_low_approx_persistence(
        data,
        labels,
        images,
        extrema_values,
        extrema_labels,
        extrema_vox,
        max_cart_offset,
        use_minima,
        persistence_tol,
        matrix,
    )
    root_values = extrema_values[root_indices]
    root_labels = extrema_labels[root_indices]
    root_vox = extrema_vox[root_indices]
    root_frac = extrema_frac[root_indices]
    ###########################################################################
    # Refine positions
    ###########################################################################
    labels, images, refined_vox, roots, root_indices = group_by_refinement(
        labels,
        images,
        data,
        matrix,
        root_labels,
        root_vox,
        root_values,
        root_frac,
        use_minima,
        shape,
    )
    final_root_vox = root_vox[root_indices]
    final_refined_frac = np.round(refined_vox / shape, 6)

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
    extrema_values,
    saddle_connections,
    saddle_values,
    saddle_vox,
    persistence_tol,
    matrix,
    use_minima=False,
):
    shape = np.array(data.shape)
    num_extrema = len(extrema_vox)
    num_saddles = len(saddle_vox)
    extrema_frac = extrema_vox / shape

    # create array to track unions between basins
    unions = np.arange(num_extrema)
    images = np.zeros((num_extrema, 3), dtype=np.int8)

    # loop until no new unions are added
    num_added = 1
    while num_added > 0:
        num_added = 0
        for saddle_idx in range(num_saddles):

            ext1, ext2, image1, image2 = saddle_connections[saddle_idx]
            root1, shift1 = find_root_with_shift1(unions, images, ext1)
            root2, shift2 = find_root_with_shift1(unions, images, ext2)

            # skip if this saddle does not involve the current root
            # NOTE: connections should be ordered by value ahead of time
            if root1 == root2:
                continue

            # convert images
            image1 = INT_TO_IMAGE[image1]
            image2 = INT_TO_IMAGE[image2]

            # get saddle coord/val
            saddle_coord = saddle_vox[saddle_idx]
            saddle_val = saddle_values[saddle_idx]

            # get extrema coords.
            tot_shift1 = image1 + shift1
            tot_shift2 = image2 + shift2

            c1 = (extrema_frac[root1] + tot_shift1) * shape
            c2 = (extrema_frac[root2] + tot_shift2) * shape

            # calculate persistence score
            score = get_persistence_value(
                extrema_values[root1],
                extrema_values[root2],
                saddle_val,
                c1,
                c2,
                saddle_coord,
            )

            # if our score is below the tolerance, union
            if score < persistence_tol:
                ext_val = extrema_values[root1]
                neigh_val = extrema_values[root2]
                if (
                    (not use_minima and neigh_val < ext_val)
                    or (use_minima and neigh_val > ext_val)
                    or (neigh_val == ext_val and root2 > root1)
                ):
                    upper = ext1
                    lower = ext2
                    image = image1 - image2
                else:
                    upper = ext2
                    lower = ext1
                    image = image2 - image1
                num_added += 1
                union_with_shift1(unions, images, lower, upper, image)

    # get the roots of each extrema
    for ext_idx in range(num_extrema):
        root, shift = find_root_with_shift1(
            unions,
            images,
            ext_idx,
        )
        unions[ext_idx] = root
        images[ext_idx] = shift

    return unions, images

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