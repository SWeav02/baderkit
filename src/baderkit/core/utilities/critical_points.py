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


# @njit(cache=True, fastmath=True)
# def compute_signature(H, eig_rel_tol):
#     evals, _ = np.linalg.eigh(H)

#     tol = eig_rel_tol * np.max(np.abs(evals))

#     n_neg = 0
#     n_flat = 0
#     for i in evals:
#         if i < -tol:
#             n_neg += 1
#         elif abs(i) < tol:
#             n_flat += 1


#     return n_neg, n_flat
@njit(cache=True)
def compute_signature(H, eig_rel_tol):

    evals = np.linalg.eigvalsh(H)
    tol = eig_rel_tol * np.max(np.abs(evals))

    n_neg = 0
    n_flat = 0

    for lam in evals:
        if lam < -tol:
            n_neg += 1
        elif abs(lam) <= tol:
            n_flat += 1

    return n_neg, n_flat


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
    # Solve Newton step H s = -g
    # -----------------------------

    # determinant
    det = (
        Hxx * (Hyy * Hzz - Hyz * Hyz)
        - Hxy * (Hxy * Hzz - Hyz * Hxz)
        + Hxz * (Hxy * Hyz - Hyy * Hxz)
    )

    if abs(det) < eps:
        return -1

    inv_det = 1.0 / det

    # inverse Hessian (symmetric)
    iHxx = (Hyy * Hzz - Hyz * Hyz) * inv_det
    iHyy = (Hxx * Hzz - Hxz * Hxz) * inv_det
    iHzz = (Hxx * Hyy - Hxy * Hxy) * inv_det
    iHxy = -(Hxy * Hzz - Hxz * Hyz) * inv_det
    iHxz = (Hxy * Hyz - Hyy * Hxz) * inv_det
    iHyz = -(Hxx * Hyz - Hxy * Hxz) * inv_det

    # Newton step
    sx = -(iHxx * gi + iHxy * gj + iHxz * gk)
    sy = -(iHxy * gi + iHyy * gj + iHyz * gk)
    sz = -(iHxz * gi + iHyz * gj + iHzz * gk)

    # -----------------------------
    # Reject if step > ~1 voxel
    # -----------------------------
    if abs(sx) > 1.5 or abs(sy) > 1.5 or abs(sz) > 1.5:
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
def get_canonical_saddle_connections(
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
        try:
            saddle_connections[idx, 1] = higher
        except:
            breakpoint()
        saddle_connections[idx, 2] = lower_shift
        saddle_connections[idx, 3] = higher_shift
    return saddle_connections


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
    saddle_connections = get_canonical_saddle_connections(
        saddle_coords,
        data,
        labels=labels,
        images=images,
        use_minima=use_minima,
    )

    return saddle_coords, saddle_connections


@njit(parallel=True, cache=True)
def remove_false_saddles(
    saddle_coords,
    labels: NDArray[np.int64],
    images: NDArray[np.int64],
    data: NDArray[np.float64],
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
        morse_idx = is_ongrid_newton_crit(
            (i, j, k),
            data,
            r_voxel_cart2,
            inv_G,
            H_frac_to_cart,
        )

        if use_minima and morse_idx == 1 or not use_minima and morse_idx == 2:
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
        use_minima=use_minima,
    )

    # remove false saddles
    true_saddles = np.where(
        saddle_connections[:, 0] != np.iinfo(saddle_connections.dtype).max
    )[0]
    # saddle_indices = saddle_indices[true_saddles]
    saddle_coords = saddle_coords[true_saddles]
    saddle_connections = saddle_connections[true_saddles]

    return saddle_coords, saddle_connections


@njit(cache=True)
def remove_adjacent_saddles(refined_vox, shape):
    unions = np.arange(len(refined_vox))
    # combine any that refined to be adjacent
    rounded = np.round(refined_vox).astype(np.int64) % shape
    rounded_frac = rounded / shape
    for ext_idx in range(len(unions)):
        ext_frac = rounded_frac[ext_idx]
        ext_vox = rounded[ext_idx]
        for neigh_idx in range(ext_idx + 1, len(unions)):
            neigh_frac = rounded_frac[neigh_idx]
            wrapped = neigh_frac - np.round(neigh_frac - ext_frac)
            wrapped_vox = wrapped * shape
            offset = wrapped_vox - ext_vox
            if np.max(np.abs(offset)) < 1 + 1e-12:
                union(unions, ext_idx, neigh_idx)
    # reduce to roots
    roots = unions.copy()
    for i in unions:
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

    max_eval = np.max(np.abs(evals))
    tol = eig_rel_tol * max_eval
    lam_floor = tol

    dx = np.zeros(3)

    for i in range(3):

        lam = evals[i]

        # enforce desired Morse signature
        if i < target_index:
            lam_target = -abs(lam)
        else:
            lam_target = abs(lam)

        # flat handling
        if abs(lam_target) < lam_floor:
            lam_target = np.sign(lam_target) * lam_floor

        gi = vecs[:, i] @ g

        dx += -(gi / lam_target) * vecs[:, i]

    return dx


@njit(cache=True, fastmath=True)
def newton_refine(
    coord,
    data,
    target_index,
    inv_G,
    max_iter,
    grad_tol,
    h,
    eig_rel_tol,
):
    nx, ny, nz = data.shape
    max_step = 0.25 * min(nx, ny, nz)

    # make sure our coord is a float
    coord = coord.astype(np.float64)

    converged = False

    for _ in range(max_iter):
        # get gradient and hessian at this point
        g, H = spline_grad_and_hess(coord, data, h)

        # get step
        # dx = flat_aware_newton_step(g, H, eig_rel_tol)
        dx = targeted_newton_step(
            g,
            H,
            target_index,
            eig_rel_tol,
        )

        # check for convergence
        if np.linalg.norm(dx) < grad_tol:
            converged = True
            break

        # enforce small step size
        dx, ok = clamp_step(dx, max_step)
        if not ok:
            break

        # update coord
        coord = coord + dx

    # final classification
    morse_index, n_flat = compute_signature(H, eig_rel_tol)

    return coord, converged, morse_index


@njit(cache=True, fastmath=True)
def newton_refine_in_voxel(
    coord,
    data,
    target_index,
    inv_G,
    max_change,
    max_iter,
    grad_tol,
    h,
    eig_rel_tol,
):
    nx, ny, nz = data.shape
    max_step = 0.25 * min(nx, ny, nz)

    # make sure our coord is a float
    coord = coord.astype(np.float64)

    voxel_min = coord - max_change
    voxel_max = coord + max_change

    converged = False

    for _ in range(max_iter):
        # get gradient and hessian at this point
        g, H = spline_grad_and_hess(coord, data, h)

        # get step
        # dx = flat_aware_newton_step(g, H, eig_rel_tol)
        dx = targeted_newton_step(
            g,
            H,
            target_index,
            eig_rel_tol,
        )

        # check for convergence
        if np.linalg.norm(dx) < grad_tol:
            converged = True
            break

        # enforce small step size
        dx, ok = clamp_step(dx, max_step)
        if not ok:
            break

        # update coord
        coord = coord + dx

        # check if we exit the allowed distance
        if outside_voxel(coord, voxel_min, voxel_max):
            break

    # final classification
    morse_index, n_flat = compute_signature(H, eig_rel_tol)

    return coord, converged, morse_index


@njit(parallel=True, cache=True)
def refine_critical_points(
    critical_coords,
    data,
    matrix,
    target_index,
    max_change=1.0,
    max_iter=30,
    grad_tol=1e-6,
    h=0.5,
    eig_rel_tol=1e-04,
):

    G = matrix @ matrix.T
    inv_G = np.linalg.inv(G)

    # create arrays to store partial coordinates
    refined_coords = np.empty_like(critical_coords, dtype=np.float64)
    successes = np.zeros(len(critical_coords), dtype=np.bool_)

    for coord_idx in prange(len(critical_coords)):
        coord = critical_coords[coord_idx]
        # refine
        new_coord, success, morse_index = newton_refine_in_voxel(
            coord,
            data,
            target_index,
            inv_G,
            max_change,
            max_iter,
            grad_tol,
            h,
            eig_rel_tol,
        )
        if success and morse_index == target_index:
            refined_coords[coord_idx] = new_coord
            successes[coord_idx] = True
        else:
            refined_coords[coord_idx] = coord

    return refined_coords, successes


# @njit(fastmath=True)
# def newton_refine_critical(
#     point,
#     data,
#     target_index,
#     is_frac: bool = True,
#     max_iter=30,
#     grad_tol=1e-6,
#     lambda0=1.0e-2,
#     lambda_up=10.0,
#     lambda_down=0.3,
#     eig_tol=1e-10,
# ):
#     """
#     Damped + signature-aware Newton refinement for extrema or saddles in 3D.

#     Parameters
#     ----------
#     point : array-like (3,)
#         Initial point in grid coordinates.
#     data : ndarray
#         Scalar field data (passed-through to interp_spline).
#     target_index : None or int in {0,1,2,3}
#         Desired Morse index: number of negative eigenvalues.
#           - None: preserve previous behavior using is_maximum flag.
#           - 0 => minimum (all eigenvalues > 0)
#           - 1 => index-1 saddle (one negative eigenvalue)
#           - 2 => index-2 saddle (two negative eigenvalues)
#           - 3 => maximum (all eigenvalues < 0)
#     is_frac : bool
#         Whether or not the provided coordinates are fractional rather than
#         in voxel coordinates
#     max_iter : int
#         Maximum Newton iterations.
#     grad_tol : float
#         Convergence tolerance on gradient norm (L2).
#     max_step : float
#         Maximum allowed step length per iteration (in grid units).
#     lambda0 : float
#         Initial damping parameter (used to clamp eigenvalues).
#     lambda_up, lambda_down : float
#         Multiplicative factors for increasing / decreasing damping.
#     h : float
#         Finite-difference spacing used by spline_grad/spline_hess.

#     eig_tol : float
#         Small threshold used when classifying eigenvalues as negative/positive.

#     Returns
#     -------
#     x : ndarray (3,)
#         Refined location (grid coordinates).
#     converged : bool
#         True if converged (and signature matches when require_signature=True).
#     info : dict
#         Diagnostics: {'grad_norm': ..., 'evals': ndarray of final Hessian eigenvalues}
#     """

#     nx, ny, nz = data.shape

#     max_step = 0.25 * min(nx, ny, nz)

#     i, j, k = point
#     i = float(i)
#     j = float(j)
#     k = float(k)

#     # convert fractional to voxel coordinates
#     if is_frac:
#         i = i * nx
#         j = j * ny
#         k = k * nz

#     for it in range(max_iter):
#         # gradient and its norm
#         gi, gj, gk = spline_grad(i, j, k, data)
#         g_norm = get_norm(gi, gj, gk)

#         # compute Hessian
#         H = spline_hess(i, j, k, data)

#         # check convergence by gradient norm
#         if g_norm < grad_tol:
#             # check Hessian signature
#             evals, _ = np.linalg.eigh(H)
#             n_neg = np.sum(evals < -eig_tol)
#             if n_neg == target_index:
#                 if is_frac:
#                     # convert back to fractional coords
#                     i = i / nx  # % 1.0
#                     j = j / ny  # % 1.0
#                     k = k / nz  # % 1.0
#                 return i, j, k, 0, g_norm, evals
#             # else: we have small gradient but wrong signature -> continue attempting to adjust

#         # Diagonalize symmetric Hessian
#         evals, vecs = np.linalg.eigh(H)  # evals in ascending order
#         # evals ascending: smallest first. We'll enforce first target_index to be negative.

#         # Build modified eigenvalues with sign enforcement and clamping magnitude >= lam
#         # For i < target_index -> force negative: -max(|evals[i]|, lam)
#         # For i >= target_index -> force positive: +max(|evals[i]|, lam)
#         # This ensures invertibility and desired signature.
#         evals_mod = np.empty_like(evals)
#         lam = max(lambda0, 0.05 * np.max(np.abs(evals)))
#         if g_norm < 10 * grad_tol:
#             enforce = True
#         else:
#             enforce = False
#         for idx_ev in range(3):

#             mag = max(abs(evals[idx_ev]), lam)

#             if enforce:
#                 if idx_ev < target_index:
#                     evals_mod[idx_ev] = -mag
#                 else:
#                     evals_mod[idx_ev] = +mag
#             else:
#                 s = 1.0 if evals[idx_ev] >= 0 else -1.0
#                 evals_mod[idx_ev] = s * mag

#         # Reconstruct modified Hessian in original basis: H_mod = V diag(evals_mod) V^T
#         H_mod = (
#             vecs * evals_mod[np.newaxis, :]
#         ) @ vecs.T  # efficient diag-multiply then matmul

#         # Solve linear system (H_mod) delta = -g
#         try:
#             di, dj, dk = np.linalg.solve(
#                 H_mod, -np.array((gi, gj, gk), dtype=np.float64)
#             )
#         except:
#             # increase damping and try next iter
#             lambda0 *= lambda_up
#             if lambda0 > 1e12:
#                 # avoid runaway and return original point
#                 if is_frac:
#                     # convert back to fractional coords
#                     i = i / nx  # % 1.0
#                     j = j / ny  # % 1.0
#                     k = k / nz  # % 1.0
#                 return i, j, k, 2, g_norm, evals
#             continue

#         # clamp step length
#         step_norm = get_norm(di, dj, dk)

#         if step_norm < 1e-8:
#             # escape
#             break

#         scale = min(1.0, max_step / (step_norm + 1e-12))
#         di *= scale**0.5
#         dj *= scale**0.5
#         dk *= scale**0.5

#         i_trial = i + di
#         j_trial = j + dj
#         k_trial = k + dk

#         # trial gradient
#         gi_trial, gj_trial, gk_trial = spline_grad(i_trial, j_trial, k_trial, data)
#         g_trial_norm = get_norm(gi_trial, gj_trial, gk_trial)

#         # Acceptance rule: accept if gradient norm decreases
#         pred = abs(gi * di + gj * dj + gk * dk)
#         c = 0.1 if lambda0 < 1e-2 else 0.5
#         if g_trial_norm < g_norm + c * pred:
#             # accept
#             i = i_trial
#             j = j_trial
#             k = k_trial
#             lambda0 = max(lambda0 * lambda_down, 1e-16)
#         else:
#             # reject and increase damping
#             lambda0 *= lambda_up
#             # continue to next iteration without updating x
#             continue

#     # finished loop without meeting convergence criteria
#     # return final diagnostic information and original point
#     evals_final, _ = np.linalg.eigh(spline_hess(i, j, k, data))
#     gi_final, gj_final, gk_final = spline_grad(i, j, k, data)
#     grad_norm_final = get_norm(gi_final, gj_final, gk_final)

#     n_neg_final = np.sum(evals_final < -eig_tol)
#     success = (grad_norm_final < grad_tol) and (n_neg_final == target_index)
#     if success:
#         success = 0
#     else:
#         success = 1

#     if is_frac:
#         # convert back to fractional coords
#         i = i / nx  # % 1.0
#         j = j / ny  # % 1.0
#         k = k / nz  # % 1.0

#     return i, j, k, success, grad_norm_final, evals_final


# @njit(parallel=True, cache=True)
# def refine_critical_points(
#     points,
#     data,
#     target_index,
#     is_frac: bool = True,
#     max_iter=30,
#     grad_tol=1e-6,
#     lambda0=1.0e-2,
#     lambda_up=10.0,
#     lambda_down=0.3,
#     eig_tol=1e-10,
# ):

#     refined_points = np.empty((len(points), 3), dtype=np.float64)
#     refined_status = np.empty(len(points), dtype=np.uint8)

#     for idx in prange(len(points)):
#         point = points[idx]

#         i, j, k, success, _, _ = newton_refine_critical(
#             point,
#             data,
#             target_index,
#             is_frac,
#             max_iter,
#             grad_tol,
#             lambda0,
#             lambda_up,
#             lambda_down,
#             eig_tol,
#         )
#         refined_points[idx, 0] = i
#         refined_points[idx, 1] = j
#         refined_points[idx, 2] = k
#         refined_status[idx] = success
#     return refined_points, refined_status


# @njit(parallel=True, cache=True)
# def refine_extrema(
#     extrema_coords,
#     data,
#     use_minima=False,
# ):
#     shape = np.array(data.shape, dtype=np.int64)
#     nx, ny, nz = shape
#     # for each group of extrema, we try and merge them into one. If the resulting
#     # point is not part of the group or does not have the maximum value of the
#     # group, we default to the highest point or lowest index in case of a tie.
#     # A newton refinement is then performed
#     if use_minima:
#         target_index = 0
#     else:
#         target_index = 3

#     new_voxel_coords = np.empty_like(extrema_coords, dtype=np.int16)

#     for coord_idx in prange(len(extrema_coords)):
#         ai, aj, ak = extrema_coords[coord_idx]

#         i, j, k, success, _, _ = newton_refine_critical(
#             (ai, aj, ak),
#             data,
#             target_index,
#             is_frac=False,
#         )
#         if success:
#             new_voxel_coords[coord_idx] = (i, j, k)
#         else:
#             new_voxel_coords[coord_idx] = (ai, aj, ak)

#     return new_voxel_coords

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