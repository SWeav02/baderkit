# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from baderkit.core.utilities.basic import (
    get_norm,
    merge_frac_coords_weighted,
    wrap_point,
    wrap_point_w_shift,
)
from baderkit.core.utilities.interpolation import (
    spline_grad_and_hess,
)
from baderkit.core.utilities.transforms import (
    ALL_NEIGHBOR_TRANSFORMS,
    IMAGE_TO_INT,
    INT_TO_IMAGE,
)
from baderkit.core.utilities.union_find import find_root, union

###############################################################################
# Classifying Methods
###############################################################################


@njit(cache=True)
def classify_critical_point(H):
    eigvals = np.linalg.eigvalsh(H)

    n_pos = np.sum(eigvals > 0.0)
    n_neg = np.sum(eigvals < 0.0)

    if n_pos == 3:
        return 0  # minimum
    if n_neg == 3:
        return 3  # maximum
    if n_pos == 2 and n_neg == 1:
        return 1  # index-1 saddle
    if n_pos == 1 and n_neg == 2:
        return 2  # index-2 saddle

    return -1

@njit(cache=True, inline="always")
def compute_signature(evals, eig_rel_tol):

    max_eval = 0.0
    for i in range(3):
        v = abs(evals[i])
        if v > max_eval:
            max_eval = v

    tol = eig_rel_tol * max_eval + 1e-14

    morse = 0
    n_flat = 0

    for i in range(3):
        lam = evals[i]

        if abs(lam) < tol:
            n_flat += 1
        elif lam < 0.0:
            morse += 1

    return morse, n_flat

@njit(cache=True)
def is_ongrid_saddle(
    data,
    i,
    j,
    k,
    nx,
    ny,
    nz,
    edge_mask,
    use_minima,
):
    # get initial value and label
    value = data[i, j, k]

    for si, sj, sk in ALL_NEIGHBOR_TRANSFORMS:
        # wrap around periodic edges
        ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)

        # skip points that aren't also on the edge
        if not edge_mask[ii, jj, kk]:
            continue

        neigh_value = data[ii, jj, kk]
        # check if a neighbor is a better saddle candidate
        if use_minima:
            if neigh_value < value:
                return False
        else:
            if neigh_value > value:
                return False

    return True

@njit(fastmath=True, cache=True)
def is_ongrid_newton_crit(
    coord,
    data,
    r_voxel_cart2,
    inv_G,
    H_frac_to_cart,
):
    """
    Newton-step critical point classification with degeneracy handling.

    Returns:
        -1 : reject
         0 : minimum
         1 : index-1 saddle
         2 : index-2 saddle
         3 : maximum

        10 : rank-1 valley  (1 positive, 2 ~0)
        11 : rank-1 ridge   (1 negative, 2 ~0)

        20 : rank-2 degenerate min-like (2 positive, 1 ~0)
        21 : rank-2 degenerate saddle   (1 pos, 1 neg, 1 ~0)
        22 : rank-2 degenerate max-like (2 negative, 1 ~0)

        30 : rank-0 flat
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

    # -----------------------------
    # Hessian (fractional)
    # -----------------------------
    Hxx = n[2, 1, 1] - 2.0 * f0 + n[0, 1, 1]
    Hyy = n[1, 2, 1] - 2.0 * f0 + n[1, 0, 1]
    Hzz = n[1, 1, 2] - 2.0 * f0 + n[1, 1, 0]

    Hxy = 0.25 * (n[2, 2, 1] - n[2, 0, 1] - n[0, 2, 1] + n[0, 0, 1])
    Hxz = 0.25 * (n[2, 1, 2] - n[2, 1, 0] - n[0, 1, 2] + n[0, 1, 0])
    Hyz = 0.25 * (n[1, 2, 2] - n[1, 2, 0] - n[1, 0, 2] + n[1, 0, 0])

    # -----------------------------
    # Scale-aware tolerance
    # -----------------------------
    scale = (
        abs(Hxx) + abs(Hyy) + abs(Hzz) +
        2.0 * (abs(Hxy) + abs(Hxz) + abs(Hyz))
    )
    tol = 1e-6 * scale + 1e-12

    # -----------------------------
    # Determinant (degeneracy hint)
    # -----------------------------
    det = (
        Hxx * (Hyy * Hzz - Hyz * Hyz)
        - Hxy * (Hxy * Hzz - Hyz * Hxz)
        + Hxz * (Hxy * Hyz - Hyy * Hxz)
    )

    degenerate = abs(det) < tol

    # -----------------------------
    # LDLᵀ decomposition (inertia)
    # -----------------------------
    morse = 0
    zero_dirs = 0

    # D1
    D1 = Hxx
    if abs(D1) < tol:
        zero_dirs += 1
    elif D1 < 0.0:
        morse += 1
    D1 = D1 if abs(D1) > eps else eps

    L21 = Hxy / D1
    L31 = Hxz / D1

    # D2
    D2 = Hyy - L21 * Hxy
    if abs(D2) < tol:
        zero_dirs += 1
    elif D2 < 0.0:
        morse += 1
    D2 = D2 if abs(D2) > eps else eps

    L32 = (Hyz - L31 * Hxy) / D2

    # D3
    D3 = Hzz - L31 * Hxz - L32 * (Hyz - L31 * Hxy)
    if abs(D3) < tol:
        zero_dirs += 1
    elif D3 < 0.0:
        morse += 1

    rank = 3 - zero_dirs

    # -----------------------------
    # Degeneracy classification
    # -----------------------------
    if rank == 0:
        return 30  # completely flat

    if rank == 1:
        # one curvature direction
        if morse == 0:
            return 10  # valley (positive curvature)
        else:
            return 11  # ridge (negative curvature)

    if rank == 2:
        # one flat direction
        if morse == 0:
            return 20  # degenerate minimum-like
        elif morse == 1:
            return 21  # degenerate saddle
        else:
            return 22  # degenerate maximum-like

    # -----------------------------
    # Full-rank case: Newton step
    # -----------------------------
    if not degenerate:
        inv_det = 1.0 / det

        iHxx = (Hyy * Hzz - Hyz * Hyz) * inv_det
        iHyy = (Hxx * Hzz - Hxz * Hxz) * inv_det
        iHzz = (Hxx * Hyy - Hxy * Hxy) * inv_det
        iHxy = -(Hxy * Hzz - Hxz * Hyz) * inv_det
        iHxz = (Hxy * Hyz - Hyy * Hxz) * inv_det
        iHyz = -(Hxx * Hyz - Hxy * Hxz) * inv_det

        sx = -(iHxx * gi + iHxy * gj + iHxz * gk)
        sy = -(iHxy * gi + iHyy * gj + iHyz * gk)
        sz = -(iHxz * gi + iHyz * gj + iHzz * gk)

        if abs(sx) > 1.5 or abs(sy) > 1.5 or abs(sz) > 1.5:
            return -1

    # -----------------------------
    # Standard Morse classification
    # -----------------------------
    return morse


###############################################################################
# Critical Finding Methods
###############################################################################


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
    max_val,
    use_minima,
):

    # iterate over transforms
    label = labels[i, j, k]
    image = images[i, j, k]
    value = data[i, j, k]
    im, jm, km = INT_TO_IMAGE[image]

    for trans in range(ALL_NEIGHBOR_TRANSFORMS.shape[0]):
        # get shifts
        si = ALL_NEIGHBOR_TRANSFORMS[trans, 0]
        sj = ALL_NEIGHBOR_TRANSFORMS[trans, 1]
        sk = ALL_NEIGHBOR_TRANSFORMS[trans, 2]

        # wrap around periodic edges and store shift
        ii, jj, kk, ssi, ssj, ssk = wrap_point_w_shift(
            i + si, j + sj, k + sk, nx, ny, nz
        )
        # skip points with a higher value
        neigh_value = data[ii, jj, kk]
        neigh_label = labels[ii, jj, kk]
        if use_minima:
            if neigh_value > value:
                continue
        elif not use_minima:
            if neigh_value < value:
                continue
        else:
            if neigh_value == value and neigh_label > label:
                continue

        # get the label and image of this neighbor

        # update image to be relative to the current points transformation
        if ssi == 0 and ssj == 0 and ssk == 0:
            neigh_image = images[ii, jj, kk]
        else:
            neigh_image = images[ii, jj, kk]
            si1 = INT_TO_IMAGE[neigh_image, 0] + ssi
            sj1 = INT_TO_IMAGE[neigh_image, 1] + ssj
            sk1 = INT_TO_IMAGE[neigh_image, 2] + ssk
            neigh_image = IMAGE_TO_INT[si1, sj1, sk1]

        # note if this point belongs to a different basin
        if label != neigh_label or image != neigh_image:

            return (label, neigh_label, image, neigh_image)

    # if no neighbor was found, we just return a fake value
    return max_val, max_val, max_val, max_val


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


@njit(cache=True, parallel=True)
def get_saddle_connections(
    saddle_coords,
    data,
    labels: NDArray[np.int64],
    images: NDArray[np.int64],
    use_minima,
):
    nx, ny, nz = labels.shape
    # create an array to track connections between these points.
    # For each entry we will have:
    # 1: the lower label index
    # 2: the higher label index
    # 3: the connection image between basins
    # 4: whether or not the connection image is lower -> higher (0) or higher -> lower (1)
    saddle_connections = np.empty((len(saddle_coords), 4), dtype=np.int16)

    max_val = np.iinfo(np.int16).max
    for idx in prange(len(saddle_coords)):
        i, j, k = saddle_coords[idx]
        lower, higher, lower_shift, higher_shift = get_extrema_saddle_connections(
            i,
            j,
            k,
            nx,
            ny,
            nz,
            data,
            labels,
            images,
            max_val,
            use_minima,
        )
        saddle_connections[idx, 0] = lower
        saddle_connections[idx, 1] = higher
        saddle_connections[idx, 2] = lower_shift
        saddle_connections[idx, 3] = higher_shift
    return saddle_connections

@njit(cache=True, parallel=True)
def get_canonical_saddle_connections(
        saddle_connections
        ):
    num_saddles = len(saddle_connections)

    canon_saddles = np.empty((num_saddles,3), dtype=np.int64)
    for saddle_idx in prange(num_saddles):
        ext1, ext2, image1, image2 = saddle_connections[saddle_idx]
        if ext1 < ext2:
            lower = ext1
            upper = ext2
            lower_shift = image1
            upper_shift = image2
        else:
            lower = ext2
            upper = ext1
            lower_shift = image2
            upper_shift = image1
        lower_image = INT_TO_IMAGE[lower_shift]
        upper_image = INT_TO_IMAGE[upper_shift]
        mi, mj, mk = lower_image - upper_image
        image = IMAGE_TO_INT[mi, mj, mk]
        canon_saddles[saddle_idx, 0] = lower
        canon_saddles[saddle_idx, 1] = upper
        canon_saddles[saddle_idx, 2] = image

    return canon_saddles


@njit(parallel=True, cache=True)
def get_saddles_from_basins(
    labels: NDArray[np.int64],
    images: NDArray[np.int64],
    data: NDArray[np.float64],
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
            edge_mask,
            use_minima,
        ):
            saddle_mask[saddle_idx] = True

    # Now we reduce to the possible saddle points
    saddle_indices = np.where(saddle_mask)[0]
    saddle_coords = saddle_coords[saddle_indices]

    # get neighboring basins
    saddle_connections = get_saddle_connections(
        saddle_coords,
        data,
        labels=labels,
        images=images,
        use_minima=use_minima,
    )

    return saddle_coords, saddle_connections

# @njit(parallel=True, cache=True)
# def remove_false_saddles(
#     saddle_coords,
#     labels: NDArray[np.int64],
#     images: NDArray[np.int64],
#     data: NDArray[np.float64],
#     matrix,
#     use_minima: bool = False,
# ):
#     nx, ny, nz = data.shape

#     # Metric tensors
#     G = matrix @ matrix.T
#     inv_G = np.linalg.inv(G)
#     lam = np.linalg.eigvalsh(G)

#     lam_min = lam[0]
#     lam_max = lam[2]

#     frac_to_cart = np.sqrt(lam_max)
#     H_frac_to_cart = 1.0 / lam_min

#     # Voxel radius in Cartesian
#     r_voxel_cart = 0.5 * frac_to_cart
#     r_voxel_cart2 = r_voxel_cart * r_voxel_cart

#     saddle_mask = np.zeros(len(saddle_coords), dtype=np.bool)
#     for saddle_idx in prange(len(saddle_coords)):
#         i, j, k = saddle_coords[saddle_idx]
#         # check if a newton step would move outside the voxel and
#         # get morse index
#         morse_idx = is_ongrid_newton_crit(
#             (i, j, k),
#             data,
#             r_voxel_cart2,
#             inv_G,
#             H_frac_to_cart,
#         )

#         if use_minima and morse_idx == 1 or not use_minima and morse_idx == 2:
#             # this is a saddle
#             saddle_mask[saddle_idx] = True

#     # remove false saddles
#     saddle_indices = np.where(saddle_mask)[0]
#     saddle_coords = saddle_coords[saddle_indices]

#     # get neighboring basins
#     saddle_connections = get_saddle_connections(
#         saddle_coords,
#         data,
#         labels=labels,
#         images=images,
#         use_minima=use_minima,
#     )

#     # remove false saddles
#     true_saddles = np.where(
#         saddle_connections[:, 0] != np.iinfo(saddle_connections.dtype).max
#     )[0]
#     # saddle_indices = saddle_indices[true_saddles]
#     saddle_coords = saddle_coords[true_saddles]
#     saddle_connections = saddle_connections[true_saddles]

#     return saddle_coords, saddle_connections

@njit(parallel=True, cache=True)
def remove_false_saddles(
    saddle_coords,
    saddle_connections,
    canon_saddle_indices,
    unique_canon_saddles,
    labels: NDArray[np.int64],
    images: NDArray[np.int64],
    data: NDArray[np.float64],
    extrema_vox,
    matrix,
    use_minima: bool = False,
):
    nx, ny, nz = data.shape

    # create trackers for saddles
    saddle_mask = np.ones(len(saddle_coords), dtype=np.bool_)

    # get values at extrema points
    num_extrema = len(extrema_vox)
    extrema_values = np.empty(num_extrema, dtype=np.float64)
    for idx, (i, j, k) in enumerate(extrema_vox):
        extrema_values[idx] = data[i, j, k]

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

    if use_minima:
        saddle_morses = np.array((1, 11, 21, 22), dtype=np.int64)
    else:
        saddle_morses = np.array((2, 10, 20, 21), dtype=np.int64)

    for saddle_idx in prange(len(saddle_coords)):
        i, j, k = saddle_coords[saddle_idx]
        # check if a newton step would move outside the voxel and
        # get morse index
        morse_idx = is_ongrid_newton_crit(
            (i, j, k),
            data,
            r_voxel_cart2,
            inv_G,
            H_frac_to_cart,
        )
        if not morse_idx in saddle_morses:
            # this is not a saddle
            saddle_mask[saddle_idx] = False

    # for remaining possible saddles, we perform a full newton refinement to
    # check if they're valid
    possible_saddles = np.where(saddle_mask)[0]
    possible_coords = saddle_coords[possible_saddles]
    # try to refine for each type of saddle
    refined_vox, successes, ctypes = refine_critical_points(
        critical_coords=possible_coords,
        data=data,
        matrix=matrix,
        target_indices=saddle_morses,
        max_change=1000,
        max_iter=100,
        grad_tol=0.001,
        h=0.5,
        eig_rel_tol=0.001,
    )

    # correct_ctypes = np.isin(ctypes, saddle_morses)
    # The refinement seems to fail often in the ELF. This causes issues
    # later on, so I'm removing this for now. Eventually, I need a better
    # method as this results in far too many saddles.
    # success_indices = np.where(successes)[0]

    # success_indices = np.where(correct_ctypes)[0]
    # refined_vox = refined_vox[success_indices]

    # recalculate saddle connections
    rounded_vox = np.round(refined_vox).astype(np.int64) % np.array(data.shape, dtype=np.int64)
    saddle_connections = get_saddle_connections(
        rounded_vox,
        data,
        labels=labels,
        images=images,
        use_minima=use_minima,
    )
    succeeded = np.where(saddle_connections[:,0]!=np.iinfo(np.int16).max)[0]
    saddle_connections = saddle_connections[succeeded]
    refined_vox = refined_vox[succeeded]

    return refined_vox, saddle_connections

@njit(cache=True)
def remove_adjacent_saddles(refined_vox, shape):
    unions = np.arange(len(refined_vox))
    # combine any that refined to be adjacent
    rounded_frac = refined_vox / shape
    for ext_idx in range(len(unions)):
        ext_frac = rounded_frac[ext_idx]
        ext_vox = refined_vox[ext_idx]
        for neigh_idx in range(ext_idx + 1, len(unions)):
            neigh_frac = rounded_frac[neigh_idx]
            wrapped = neigh_frac - np.round(neigh_frac - ext_frac)
            wrapped_vox = wrapped * shape
            offset = wrapped_vox - ext_vox
            if np.max(np.abs(offset)) < 1 + 1e-12:
                union(unions, ext_idx, neigh_idx)
    # reduce to roots
    roots = unions.copy()
    for i in range(len(unions)):
        roots[i] = find_root(unions, i)
    important = np.where(roots == np.arange(len(refined_vox)))[0]

    return important

@njit(cache=True)
def get_saddle_saddle_connections(
    saddle1_coords,
    saddle2_coords,
    neighbor_transforms: NDArray[np.int64],
    edge_mask: NDArray[np.uint8],
):
    nx, ny, nz = edge_mask.shape

    # get the number of possible edges
    num_edges = len(np.where(np.isin(edge_mask, (1, 2, 6)))[0])
    num_edges += len(saddle1_coords) + len(saddle2_coords)

    # create an empty queue for storing which points are next
    queue = np.empty((num_edges, 3), dtype=np.uint32)

    # create arrays to store flood filled labels
    max_val = np.iinfo(np.int32).max
    flood_labels = np.full_like(edge_mask, max_val, dtype=np.int32)
    flood_images = np.full_like(edge_mask, 13, dtype=np.uint8)

    # seed saddles
    saddle_idx = 0
    saddle1_idx = 1
    saddle2_idx = -1
    for i, j, k in saddle1_coords:
        flood_labels[i, j, k] = saddle1_idx
        queue[saddle_idx] = (i, j, k)
        saddle1_idx += 1
        saddle_idx += 1
    for i, j, k in saddle2_coords:
        flood_labels[i, j, k] = saddle2_idx
        queue[saddle_idx] = (i, j, k)
        saddle2_idx -= 1
        saddle_idx += 1

    # create lists to store connections
    connections = []
    connection_coords = []
    queue_start = 0
    queue_end = saddle_idx

    while queue_start != queue_end:
        next_end = queue_end
        for edge_idx in range(queue_start, queue_end):
            i, j, k = queue[edge_idx]
            # get label and image
            label = flood_labels[i, j, k]
            mi, mj, mk = INT_TO_IMAGE[flood_images[i, j, k]]

            # iterate over each neighbor. if unlabeled, assign it the same label
            # if labeled, note a new connection
            for trans, (si, sj, sk) in enumerate(neighbor_transforms):
                # get the neighbor
                ii, jj, kk, ssi, ssj, ssk = wrap_point_w_shift(
                    i + si, j + sj, k + sk, nx, ny, nz
                )
                # skip points that can't be part of our connections
                if not edge_mask[ii, jj, kk] in (1, 2, 6):
                    continue

                # get the label of the neighbor
                neigh_label = flood_labels[ii, jj, kk]

                # get total image of the current path
                mi1 = mi + ssi
                mj1 = mj + ssj
                mk1 = mk + ssk

                shift = IMAGE_TO_INT[mi1, mj1, mk1]

                # if unlabeled, label immediately
                if neigh_label == max_val:
                    flood_labels[ii, jj, kk] = label
                    flood_images[ii, jj, kk] = shift
                    queue[next_end] = (ii, jj, kk)
                    next_end += 1
                # skip points that are both the same type of saddle
                elif (
                    (neigh_label < 0 and label < 0)
                    or (neigh_label > 0 and label > 0)
                    or (neigh_label == label)
                ):
                    continue

                else:
                    # this point belongs to a different saddle
                    # get this points image
                    ni, nj, nk = INT_TO_IMAGE[flood_images[ii, jj, kk]]
                    bi = mi1 - ni
                    bj = mj1 - nj
                    bk = mk1 - nk

                    # order from saddle1 to saddle2
                    if label > 0 and neigh_label < 0:
                        best_image = IMAGE_TO_INT[bi, bj, bk]
                        lower = abs(label) - 1
                        upper = abs(neigh_label) - 1

                    elif label < 0 and neigh_label > 0:
                        best_image = IMAGE_TO_INT[-bi, -bj, -bk]
                        lower = abs(neigh_label) - 1
                        upper = abs(label) - 1
                    else:
                        # we should never have the same label or two labels
                        # with the same sign
                        continue

                    connections.append(
                        (
                            lower,
                            upper,
                            best_image,
                        )
                    )
                    # add coord
                    connection_coords.append(queue[edge_idx])
        queue_start = queue_end
        queue_end = next_end

    return connections, connection_coords

###############################################################################
# Newton Refinement
###############################################################################


@njit(cache=True, fastmath=True, inline="always")
def clamp_step(dx, max_step):
    step_norm = get_norm(dx[0], dx[1], dx[2])
    if step_norm < 1e-8:
        return dx, False

    # scale = min(1.0, max_step / (step_norm + 1e-12))
    # scale = scale**0.5
    # return dx * scale, True
    dx *= min(1.0, max_step / (step_norm + 1e-12))
    return dx, True


@njit(cache=True, fastmath=True, inline="always")
def outside_voxel(coord, vmin, vmax):
    return np.any(coord < vmin) or np.any(coord > vmax)


@njit(cache=True, inline="always")
def flat_aware_newton_step(
    g,
    H,
    eig_rel_tol,
):
    evals, vecs = np.linalg.eigh(H)

    tol = eig_rel_tol * np.max(np.abs(evals))

    dx = np.zeros_like(g)

    for i in range(3):
        if abs(evals[i]) > tol:
            dx += -(vecs[:, i] @ g) / evals[i] * vecs[:, i]

    return dx

@njit(cache=True, inline="always")
def targeted_newton_step(
    g,
    H,
    target_index,
    eig_rel_tol,
):
    evals, vecs = np.linalg.eigh(H)

    # ----------------------------------
    # Scale-aware tolerance with floor
    # ----------------------------------
    max_eval = 0.0
    for i in range(3):
        v = abs(evals[i])
        if v > max_eval:
            max_eval = v

    # Absolute floor for degeneracy
    tol = max(eig_rel_tol * max_eval, 1e-12)

    dx = np.zeros(3)
    flat_dirs = 0

    # ----------------------------------
    # Build step in eigenbasis
    # ----------------------------------
    for i in range(3):

        lam = evals[i]

        vi0 = vecs[0, i]
        vi1 = vecs[1, i]
        vi2 = vecs[2, i]

        # Project gradient onto eigenvector
        gi = g[0] * vi0 + g[1] * vi1 + g[2] * vi2

        # ----------------------------------
        # Regularize eigenvalue (no skipping!)
        # ----------------------------------
        if abs(lam) < tol:
            flat_dirs += 1

            # Preserve direction but avoid singularity
            if lam >= 0.0:
                lam_eff = tol
            else:
                lam_eff = -tol
        else:
            lam_eff = lam

        # ----------------------------------
        # Enforce Morse signature
        # ----------------------------------
        if i < target_index:
            lam_target = -abs(lam_eff)
        else:
            lam_target = abs(lam_eff)

        # ----------------------------------
        # Accumulate Newton step
        # ----------------------------------
        scale = -(gi / lam_target)

        dx[0] += scale * vi0
        dx[1] += scale * vi1
        dx[2] += scale * vi2

    return dx, flat_dirs, evals

@njit(cache=True, inline="always")
def newton_step_regularized(
    g,
    H,
    eig_rel_tol,
):
    evals, vecs = np.linalg.eigh(H)

    # ----------------------------------
    # Scale-aware tolerance with floor
    # ----------------------------------
    max_eval = 0.0
    for i in range(3):
        v = abs(evals[i])
        if v > max_eval:
            max_eval = v

    # Absolute floor for degeneracy
    tol = eig_rel_tol * max_eval + 1e-14

    dx = np.zeros(3)
    flat_dirs = 0

    # ----------------------------------
    # Build step in eigenbasis
    # ----------------------------------
    for i in range(3):

        lam = evals[i]

        vi0 = vecs[0, i]
        vi1 = vecs[1, i]
        vi2 = vecs[2, i]

        # Project gradient onto eigenvector
        gi = g[0] * vi0 + g[1] * vi1 + g[2] * vi2

        # ----------------------------------
        # Prevent Movement along flat directions
        # ----------------------------------
        if abs(lam) < tol:
            flat_dirs += 1
            continue

        # ----------------------------------
        # Standard Newton step
        # ----------------------------------
        scale = -(gi / lam)

        dx[0] += scale * vi0
        dx[1] += scale * vi1
        dx[2] += scale * vi2

    return dx, flat_dirs, evals

@njit(cache=True, fastmath=True)
def newton_refine(
    coord,
    data,
    inv_G,
    max_change,
    max_iter,
    grad_rel_tol,
    h,
    eig_rel_tol,
):
    nx, ny, nz = data.shape
    max_step = 0.25 * min(nx, ny, nz)

    coord = coord.astype(np.float64)

    voxel_min = coord - max_change
    voxel_max = coord + max_change

    converged = False

    for _ in range(max_iter):

        # -----------------------------
        # Gradient + Hessian
        # -----------------------------
        g, H = spline_grad_and_hess(coord, data, h)

        gnorm = np.sqrt(g[0]*g[0] + g[1]*g[1] + g[2]*g[2])

        # -----------------------------
        # Newton step + spectrum (single eigensolve)
        # -----------------------------
        dx, flat_dirs, evals = newton_step_regularized(
            g,
            H,
            eig_rel_tol,
        )

        # -----------------------------
        # Curvature-scaled gradient tolerance
        # -----------------------------
        H_scale = np.max(np.abs(evals))
        H_scale = max(H_scale, 1e-12)

        grad_tol_scaled = grad_rel_tol * H_scale + 1e-14

        # -----------------------------
        # Gradient-based convergence
        # -----------------------------
        if gnorm < grad_tol_scaled:
            converged = True
            break

        # -----------------------------
        # Fully flat → require small gradient
        # -----------------------------
        if flat_dirs == 3:
            if gnorm < grad_tol_scaled:
                converged = True
                break

        # -----------------------------
        # Clamp step
        # -----------------------------
        dx, ok = clamp_step(dx, max_step)
        if not ok:
            break

        # -----------------------------
        # Update
        # -----------------------------
        coord[0] += dx[0]
        coord[1] += dx[1]
        coord[2] += dx[2]

        # -----------------------------
        # Stay inside voxel window
        # -----------------------------
        if outside_voxel(coord, voxel_min, voxel_max):
            break

    # -----------------------------
    # Final classification
    # -----------------------------

    morse, n_flat = compute_signature(evals, eig_rel_tol)

    # -----------------------------
    # Encode classification
    # -----------------------------
    rank = 3 - n_flat

    if rank == 0:
        ctype = 30

    elif rank == 1:
        if morse == 0:
            ctype = 10  # valley
        else:
            ctype = 11  # ridge

    elif rank == 2:
        if morse == 0:
            ctype = 20
        elif morse == 1:
            ctype = 21
        else:
            ctype = 22

    else:
        ctype = morse

    return coord, converged, ctype

@njit(cache=True, fastmath=True)
def newton_refine_targeted(
    coord,
    data,
    inv_G,
    max_change,
    max_iter,
    grad_rel_tol,
    h,
    eig_rel_tol,
    target_idx,
):
    nx, ny, nz = data.shape
    max_step = 0.25 * min(nx, ny, nz)

    coord = coord.astype(np.float64)

    voxel_min = coord - max_change
    voxel_max = coord + max_change

    converged = False

    for _ in range(max_iter):

        # -----------------------------
        # Gradient + Hessian
        # -----------------------------
        g, H = spline_grad_and_hess(coord, data, h)

        gnorm = np.sqrt(g[0]*g[0] + g[1]*g[1] + g[2]*g[2])

        # -----------------------------
        # Newton step + spectrum (single eigensolve)
        # -----------------------------
        dx, flat_dirs, evals = targeted_newton_step(
            g,
            H,
            target_idx,
            eig_rel_tol,
        )

        # -----------------------------
        # Curvature-scaled gradient tolerance
        # -----------------------------
        H_scale = np.max(np.abs(evals))
        H_scale = max(H_scale, 1e-12)

        grad_tol_scaled = grad_rel_tol * H_scale + 1e-14

        # -----------------------------
        # Gradient-based convergence
        # -----------------------------
        if gnorm < grad_tol_scaled:
            converged = True
            break

        # -----------------------------
        # Fully flat → require small gradient
        # -----------------------------
        if flat_dirs == 3:
            if gnorm < grad_tol_scaled:
                converged = True
                break

        # -----------------------------
        # Clamp step
        # -----------------------------
        dx, ok = clamp_step(dx, max_step)
        if not ok:
            break

        # -----------------------------
        # Update
        # -----------------------------
        coord[0] += dx[0]
        coord[1] += dx[1]
        coord[2] += dx[2]

        # -----------------------------
        # Stay inside voxel window
        # -----------------------------
        if outside_voxel(coord, voxel_min, voxel_max):
            break

    # -----------------------------
    # Final classification
    # -----------------------------

    morse, n_flat = compute_signature(evals, eig_rel_tol)

    # -----------------------------
    # Encode classification
    # -----------------------------
    rank = 3 - n_flat

    if rank == 0:
        ctype = 30

    elif rank == 1:
        if morse == 0:
            ctype = 10  # valley
        else:
            ctype = 11  # ridge

    elif rank == 2:
        if morse == 0:
            ctype = 20
        elif morse == 1:
            ctype = 21
        else:
            ctype = 22

    else:
        ctype = morse

    return coord, converged, ctype

# @njit(parallel=True, cache=True)
# def refine_critical_points_targeted(
#     critical_coords,
#     data,
#     matrix,
#     target_index,
#     max_change=2.0,
#     max_iter=30,
#     grad_tol=1e-2,
#     h=0.5,
#     eig_rel_tol=1e-02,
# ):

#     G = matrix @ matrix.T
#     inv_G = np.linalg.inv(G)

#     # create arrays to store partial coordinates
#     refined_coords = np.empty_like(critical_coords, dtype=np.float64)
#     successes = np.zeros(len(critical_coords), dtype=np.bool_)
#     ctypes = np.empty(len(critical_coords), dtype=np.int64)

#     for coord_idx in prange(len(critical_coords)):
#         coord = critical_coords[coord_idx]
#         # refine
#         new_coord, success, ctype = newton_refine_targeted(
#             coord,
#             data,
#             inv_G,
#             max_change,
#             max_iter,
#             grad_tol,
#             h,
#             eig_rel_tol,
#             target_index,
#         )
#         ctypes[coord_idx] = ctype
#         refined_coords[coord_idx] = new_coord
#         if success and ctype == target_index:
#             successes[coord_idx] = True

#     return refined_coords, successes, ctypes

@njit(parallel=True, cache=True)
def refine_critical_points(
    critical_coords,
    data,
    matrix,
    target_indices,
    max_change=0.5, # in Ang
    max_iter=30,
    grad_tol=1e-2,
    h=0.5,
    eig_rel_tol=1e-02,
):
    shape = np.array(data.shape, dtype=np.int64)
    nx, ny, nz = shape
    # convert max change to approximate number of voxels
    a = np.linalg.norm(matrix[0])
    b = np.linalg.norm(matrix[1])
    c = np.linalg.norm(matrix[2])
    max_vox = int(round(max((max_change/a)*nx, (max_change/b)*ny, (max_change/c)*nz)))



    G = matrix @ matrix.T
    inv_G = np.linalg.inv(G)

    # create arrays to store partial coordinates
    refined_coords = np.empty_like(critical_coords, dtype=np.float64)
    successes = np.zeros(len(critical_coords), dtype=np.bool_)
    ctypes = np.empty(len(critical_coords), dtype=np.int64)

    for coord_idx in prange(len(critical_coords)):
        coord = critical_coords[coord_idx]
        # refine
        new_coord, success, ctype = newton_refine(
            coord,
            data,
            inv_G,
            max_vox,
            max_iter,
            grad_tol,
            h,
            eig_rel_tol,
        )
        ctypes[coord_idx] = ctype
        # calculate distance
        initial_cart = (coord / shape) @ matrix
        new_cart = (new_coord / shape) @ matrix
        dist = np.linalg.norm(new_cart - initial_cart)
        if success and ctype in target_indices and dist <= max_change:
            successes[coord_idx] = True
            refined_coords[coord_idx] = new_coord
        else:
            refined_coords[coord_idx] = coord

    return refined_coords, successes, ctypes

###############################################################################
# Parabolic Fitting
###############################################################################


@njit(fastmath=True, cache=True)
def refine_frac_extrema_parabolic(grid, frac_coords, lattice, use_minima=False):
    """
    Numerically stable refinement of a local maximum or minimum on a 3D periodic grid.
    Fits a local quadratic and enforces correct curvature.

    Parameters
    ----------
    grid : 3D ndarray
        Periodic scalar field.
    frac_coords : tuple of float
        Fractional coordinates (fx, fy, fz) in [0, 1).
    lattice : ndarray, shape (3, 3)
        Lattice vectors as rows.
    use_minima : bool
        If True, refine a minimum instead of a maximum.

    Returns
    -------
    refined_frac : ndarray, shape (3,)
        Refined fractional coordinates (wrapped to [0, 1)).
    refined_value : float
        Interpolated value at the refined extremum.
    """

    # ------------------------------------------------------------------
    # Treat minima as maxima of the negated field
    # ------------------------------------------------------------------
    sign = -1.0 if use_minima else 1.0

    nx, ny, nz = grid.shape
    fx, fy, fz = frac_coords

    # --- Step 1: nearest grid point
    ix = int(round(fx * nx)) % nx
    iy = int(round(fy * ny)) % ny
    iz = int(round(fz * nz)) % nz

    # --- Step 2: extract 3×3×3 neighborhood (with sign applied)
    region = np.empty((3, 3, 3), dtype=np.float64)
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                region[dx + 1, dy + 1, dz + 1] = (
                    sign * grid[(ix + dx) % nx, (iy + dy) % ny, (iz + dz) % nz]
                )

    # --- Step 3: design matrix
    A = np.empty((27, 10), dtype=np.float64)
    b = np.empty(27, dtype=np.float64)

    inv_nx = 1.0 / nx
    inv_ny = 1.0 / ny
    inv_nz = 1.0 / nz

    row = 0
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                fxo = dx * inv_nx
                fyo = dy * inv_ny
                fzo = dz * inv_nz

                ox = lattice[0, 0] * fxo + lattice[1, 0] * fyo + lattice[2, 0] * fzo
                oy = lattice[0, 1] * fxo + lattice[1, 1] * fyo + lattice[2, 1] * fzo
                oz = lattice[0, 2] * fxo + lattice[1, 2] * fyo + lattice[2, 2] * fzo

                A[row, 0] = 1.0
                A[row, 1] = ox
                A[row, 2] = oy
                A[row, 3] = oz
                A[row, 4] = ox * ox
                A[row, 5] = oy * oy
                A[row, 6] = oz * oz
                A[row, 7] = ox * oy
                A[row, 8] = ox * oz
                A[row, 9] = oy * oz

                b[row] = region[dx + 1, dy + 1, dz + 1]
                row += 1

    # --- Step 4: regularized least squares
    ATA = np.dot(A.T, A)
    ATb = np.dot(A.T, b)
    for i in range(10):
        ATA[i, i] += 1e-10
    coeffs = np.linalg.solve(ATA, ATb)

    a0 = coeffs[0]
    ax, ay, az = coeffs[1], coeffs[2], coeffs[3]
    axx, ayy, azz = coeffs[4], coeffs[5], coeffs[6]
    axy, axz, ayz = coeffs[7], coeffs[8], coeffs[9]

    # --- Step 5: stationary point
    M = np.empty((3, 3), dtype=np.float64)
    M[0, 0] = 2.0 * axx
    M[1, 1] = 2.0 * ayy
    M[2, 2] = 2.0 * azz
    M[0, 1] = M[1, 0] = axy
    M[0, 2] = M[2, 0] = axz
    M[1, 2] = M[2, 1] = ayz

    grad = np.array([ax, ay, az])

    # Enforce concave-down quadratic (since we're always maximizing internally)
    trace_M = M[0, 0] + M[1, 1] + M[2, 2]
    if trace_M > 0.0:
        M *= -1.0
        grad *= -1.0

    # --- Step 6: solve for offset
    try:
        offset_cart = np.linalg.solve(M, -grad)
    except:
        offset_cart = np.zeros(3)

    # --- Step 7: clamp offset
    step_cart = np.sqrt(np.sum((lattice / np.array([[nx, ny, nz]])) ** 2, axis=1))
    max_step = np.max(step_cart)
    norm_offset = np.sqrt(np.sum(offset_cart**2))
    if norm_offset > max_step:
        offset_cart *= max_step / norm_offset

    # --- Step 8: refined value (still sign-flipped)
    x, y, z = offset_cart
    refined_value_signed = (
        a0
        + ax * x
        + ay * y
        + az * z
        + axx * x * x
        + ayy * y * y
        + azz * z * z
        + axy * x * y
        + axz * x * z
        + ayz * y * z
    )

    # Fallback check (maximize in signed space)
    region_max = np.max(region)
    if refined_value_signed < region_max:
        refined_value_signed = region_max
        offset_cart[:] = 0.0

    # --- Step 9: Cartesian → fractional
    frac_offset = np.linalg.solve(lattice.T, offset_cart)
    refined_frac = np.empty(3, dtype=np.float64)
    refined_frac[0] = (fx + frac_offset[0]) % 1.0
    refined_frac[1] = (fy + frac_offset[1]) % 1.0
    refined_frac[2] = (fz + frac_offset[2]) % 1.0

    # Convert value back to original sign
    refined_value = sign * refined_value_signed

    return refined_frac, refined_value


@njit(parallel=True, cache=True)
def refine_extrema_parabolic(
    extrema_coords,
    extrema_groups,
    data,
    labels,
    lattice,
    use_minima=False,
):
    shape = np.array(data.shape, dtype=np.int64)
    # for each group of extrema, we try and merge them into one. If the resulting
    # point is not part of the group or does not have the maximum value of the
    # group, we default to the highest point or lowest index in case of a tie.
    # The parabolic refinement is then applied to the resulting point.

    new_voxel_coords = np.empty_like(extrema_coords, dtype=np.uint16)
    frac_coords = np.empty_like(extrema_coords, dtype=np.float64)
    refined_values = np.empty(len(extrema_coords), dtype=np.float64)
    for group_idx in prange(len(extrema_coords)):
        group = extrema_groups[group_idx]
        values = np.empty(len(group), dtype=np.float64)
        for idx, (i, j, k) in enumerate(group):
            values[idx] = data[i, j, k]
        best_value = values.max()
        group_frac = group / shape
        # get average frac weighted by value
        average_frac = merge_frac_coords_weighted(group_frac, values)
        # get equivalent grid point
        ai, aj, ak = np.round(average_frac * shape).astype(np.int64)
        # check if this point is in the right basin and has the highest value
        label = labels[ai, aj, ak]
        value = data[ai, aj, ak]
        if label != group_idx or value != best_value:
            # default to current extrema representing this group
            ai, aj, ak = extrema_coords[group_idx]
            average_frac = extrema_coords[group_idx] / shape
        new_voxel_coords[group_idx] = (ai, aj, ak)

        refined_frac, new_value = refine_frac_extrema_parabolic(
            data, average_frac, lattice
        )
        frac_coords[group_idx] = refined_frac
        refined_values[group_idx] = new_value
    # round and wrap coords
    frac_coords = np.round(frac_coords, 6)
    frac_coords %= 1
    return new_voxel_coords, frac_coords, refined_values