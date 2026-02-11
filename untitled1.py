# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 11:30:01 2026

@author: sammw
"""
import itertools
from baderkit.core import Bader
from numba import njit, prange
import numpy as np
from numpy.typing import NDArray
from baderkit.core.bader.methods.shared_numba import get_extrema
from baderkit.core.utilities.basic import wrap_point, coords_to_flat, wrap_point_w_shift, flat_to_coords, get_norm
from baderkit.core.utilities.union_find import union, find_root
from baderkit.core.utilities.interpolation import interp_linear, interp_spline
from baderkit.core.critical_points.critical_points import CriticalPoints

IMAGE_TO_INT = np.empty([3,3,3], dtype=np.int64)
INT_TO_IMAGE = np.array(list(itertools.product((-1,0,1), repeat=3)))
for shift_idx, (i,j,k) in enumerate(INT_TO_IMAGE):
    IMAGE_TO_INT[i,j,k] = shift_idx

FACE_TRANSFORMS = np.array([
    [1,0,0],
    [-1,0,0],
    [0,1,0],
    [0,-1,0],
    [0,0,1],
    [0,0,-1],
    ], dtype=np.int64)

@njit(cache=True)
def get_ongrid_gradient_cart(i, j, k, data, dir2car):
    nx, ny, nz = data.shape
    
    c000 = data[i,j,k]
    c100 = data[(i + 1) % nx,j,k]
    c_100 = data[(i - 1) % nx,j,k]
    c010 = data[i,(j + 1) % ny,k]
    c0_10 = data[i,(j - 1) % ny,k]
    c001 = data[i,j,(k + 1) % nz]
    c00_1 = data[i,j,(k - 1) % nz]

    # central differences in voxel coordinates
    gi = (c100 - c_100) / (2.0)
    gj = (c010 - c0_10) / (2.0)
    gk = (c001 - c00_1) / (2.0)

    # optional extrema clamping
    if c100 <= c000 and c_100 <= c000:
        gi = 0.0
    if c010 <= c000 and c0_10 <= c000:
        gj = 0.0
    if c001 <= c000 and c00_1 <= c000:
        gk = 0.0

    # convert to fractional-coordinate gradient
    gi *= nx
    gj *= ny
    gk *= nz

    # convert to Cartesian gradient
    gx = dir2car[0, 0] * gi + dir2car[0, 1] * gj + dir2car[0, 2] * gk
    gy = dir2car[1, 0] * gi + dir2car[1, 1] * gj + dir2car[1, 2] * gk
    gz = dir2car[2, 0] * gi + dir2car[2, 1] * gj + dir2car[2, 2] * gk

    return gx, gy, gz

@njit(parallel=True, cache=True)
def get_residuals(
    data,
    neighbor_transforms,
    dir2car,
):
    nx, ny, nz = data.shape
    # create 3D array to store edges
    residuals = np.empty_like(data, dtype=np.float64)
    
    # loop over each voxel in parallel
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                gi,gj,gk = get_ongrid_gradient_cart(i, j, k, data, dir2car)
                res = ((gi**2) + (gj**2) + (gk**2))**(1/2)
                residuals[i,j,k] = res
    return residuals

@njit(inline='always', cache=True)
def get_extrema_saddle_connections(
    i, j, k,
    nx, ny, nz,
    ny_nz,
    labels,
    images,
    data,
    neighbor_transforms,
    edge_mask,
    max_val,
    use_minima = False,
):
    if use_minima:
        best_value = np.inf
        edge_indices = (1,2,4,6)
    else:
        best_value = -np.inf
        edge_indices = (1,2,5,6)

    best_neigh_label = -1
    best_neigh_image = -1
    inv_image_idx = -1
    # iterate over transforms
    label = labels[i,j,k]
    image = images[i,j,k]
    value = data[i,j,k]

    mi, mj, mk = INT_TO_IMAGE[image]

    for trans in range(neighbor_transforms.shape[0]):
        # get shifts
        si = neighbor_transforms[trans, 0]
        sj = neighbor_transforms[trans, 1]
        sk = neighbor_transforms[trans, 2]

        # wrap around periodic edges and store shift
        ii, jj, kk, ssi, ssj, ssk = wrap_point_w_shift(
            i+si, j+sj, k+sk, nx, ny, nz
        )
        # skip neighbors that are not also part of the edge
        if not edge_mask[ii, jj, kk] in edge_indices:
            continue

        
        # get the label and image of this neighbor
        neigh_label = labels[ii, jj, kk]
        neigh_image = images[ii, jj, kk]

        # update image to be relative to the current points transformation
        si1 = INT_TO_IMAGE[neigh_image, 0] + ssi
        sj1 = INT_TO_IMAGE[neigh_image, 1] + ssj
        sk1 = INT_TO_IMAGE[neigh_image, 2] + ssk
        neigh_image = IMAGE_TO_INT[si1, sj1, sk1]

        # skip neighbors in the same basin
        if label == neigh_label and image == neigh_image:
            continue
        
        # get the value of the neighbor
        neigh_value = data[ii,jj,kk]
        
        if use_minima:
            best_val = max(value, neigh_value)
            improved = best_val <= best_value
        else:
            best_val = min(value, neigh_value)
            improved = best_val >= best_value
        
        if improved:
            best_value = best_val
            best_neigh_label = neigh_label
            # adjust image to point
            si1 -= mi
            sj1 -= mj
            sk1 -= mk
            best_neigh_image = IMAGE_TO_INT[si1, sj1, sk1]
            inv_image_idx = IMAGE_TO_INT[-si1, -sj1, -sk1]
            
            # we can't improve beyond this points value so we can break
            if best_value == value:
                break

    # if no neighbor was found, we just return a fake value
    if best_neigh_label == -1:
        return max_val, max_val, max_val, False, 0.0
    
    is_reversed = best_neigh_image > inv_image_idx
    
    return (
        min(label, best_neigh_label), 
        max(label, best_neigh_label), 
        min(best_neigh_image, inv_image_idx), 
        is_reversed,
        best_value)

@njit(parallel=True, cache=True)
def get_canonical_saddle_connections(
    labels: NDArray[np.int64],
    images: NDArray[np.int64],
    data: NDArray[np.float64],
    crit_mask,
    neighbor_transforms: NDArray[np.int64],
    edge_mask: NDArray[np.uint8],
    use_minima: bool = False,
):
    nx, ny, nz = labels.shape
    ny_nz = ny*nz
    
    # get the points that may be saddles
    if use_minima:
        saddle_coords = np.argwhere(np.isin(edge_mask,(2,4,6))&crit_mask)
    else:
        saddle_coords = np.argwhere(np.isin(edge_mask,(1,5,6))&crit_mask)
    
    # create an array to track connections between these points.
    # For each entry we will have:
        # 1: the lower label index
        # 2: the higher label index
        # 3: the connection image between basins
        # 4: whether or not the connection image is lower -> higher (0) or higher -> lower (1)
    saddle_connections = np.empty((len(saddle_coords),4),dtype=np.uint16)
    connection_vals = np.empty(len(saddle_coords), dtype=np.float64)

    # create a mask to track important connections
    important = np.ones(len(saddle_coords), dtype=np.bool)
    max_val = np.iinfo(np.uint16).max
    for idx in prange(len(saddle_coords)):
        i,j,k = saddle_coords[idx]
        
        lower, higher, shift, is_reversed, connection_value = get_extrema_saddle_connections(
            i, j, k,
            nx, ny, nz,
            ny_nz,
            labels,
            images,
            data,
            neighbor_transforms,
            edge_mask,
            max_val,
            use_minima,
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


@njit(cache=True)
def get_single_point_saddles(
    data,
    connection_values,
    saddle_coords,
    connection_indices,
    num_connections,
    use_minima = False,
):
    nx,ny,nz = data.shape
    ny_nz = ny*nz
    # create an array to store best points
    saddles = np.empty(num_connections, dtype=np.uint16)
    if use_minima:
        best_vals = np.full(num_connections, np.inf, dtype=np.float64)
    else:
        best_vals = np.full(num_connections, -np.inf, dtype=np.float64)

    for saddle_idx, (idx, connection_value) in enumerate(zip(connection_indices, connection_values)):
        if not use_minima and  connection_value > best_vals[idx]:
            best_vals[idx] = connection_value
            saddles[idx] = saddle_idx
        elif use_minima and connection_value < best_vals[idx]:
            best_vals[idx] = connection_value
            saddles[idx] = saddle_idx
        # else:
        #     i,j,k = flat_to_coords(idx, ny_nz, nz)
        #     value = data[i,j,k]
        #     if value == connection_value:
        #         best_vals[idx] = connection_value
        #         saddles[idx] = saddle_idx
    
    return saddles, best_vals

def refine_frac_extrema_parabolic(
        grid, 
        frac_coords, 
        lattice, 
        use_minima=False,
        ):
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
    # ix = int(round(fx * nx)) % nx
    # iy = int(round(fy * ny)) % ny
    # iz = int(round(fz * nz)) % nz
    ix = fx * nx % nx
    iy = fy * ny % ny
    iz = fz * nz % nz

    # --- Step 2: extract 3×3×3 neighborhood (with sign applied)
    region = np.empty((3, 3, 3), dtype=np.float64)
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                region[dx + 1, dy + 1, dz + 1] = (
                    sign * interp_linear(
                        (ix + dx) % nx,
                        (iy + dy) % ny,
                        (iz + dz) % nz,
                        data=grid,
                        is_frac=False
                        )
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
    step_cart = np.sqrt(
        np.sum((lattice / np.array([[nx, ny, nz]])) ** 2, axis=1)
    )
    max_step = np.max(step_cart)
    norm_offset = np.sqrt(np.sum(offset_cart ** 2))
    if norm_offset > max_step:
        offset_cart *= max_step / norm_offset

    # --- Step 8: refined value (still sign-flipped)
    x, y, z = offset_cart
    refined_value_signed = (
        a0
        + ax * x + ay * y + az * z
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
        # print(refined_value_signed, region_max)
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


@njit(fastmath=True)
def spline_grad(i, j, k, data, h=0.5, is_frac=False):
    """
    Gradient of spline-interpolated scalar field
    with respect to grid coordinates (i, j, k).
    """
    nx, ny, nz = data.shape

    # convert fractional to voxel coordinates
    if is_frac:
        i = i * nx
        j = j * ny
        k = k * nz

    fxp = interp_spline(i + h, j, k, data, False)
    fxm = interp_spline(i - h, j, k, data, False)

    fyp = interp_spline(i, j + h, k, data, False)
    fym = interp_spline(i, j - h, k, data, False)

    fzp = interp_spline(i, j, k + h, data, False)
    fzm = interp_spline(i, j, k - h, data, False)

    gx = (fxp - fxm) / (2.0 * h)
    gy = (fyp - fym) / (2.0 * h)
    gz = (fzp - fzm) / (2.0 * h)

    return gx, gy, gz

@njit(fastmath=True)
def spline_hess(i, j, k, data, h=0.5, is_frac=False):
    """
    Hessian of spline-interpolated scalar field
    with respect to grid coordinates (i, j, k).
    """
    nx, ny, nz = data.shape
    if is_frac:
        i = i * nx
        j = j * ny
        k = k * nz

    f0 = interp_spline(i, j, k, data, False)

    # Second derivatives
    f_xx = (
        interp_spline(i + h, j, k, data, False)
        - 2.0 * f0
        + interp_spline(i - h, j, k, data, False)
    ) / (h * h)

    f_yy = (
        interp_spline(i, j + h, k, data, False)
        - 2.0 * f0
        + interp_spline(i, j - h, k, data, False)
    ) / (h * h)

    f_zz = (
        interp_spline(i, j, k + h, data, False)
        - 2.0 * f0
        + interp_spline(i, j, k - h, data, False)
    ) / (h * h)

    # Mixed partials
    f_xy = (
        interp_spline(i + h, j + h, k, data, False)
        - interp_spline(i + h, j - h, k, data, False)
        - interp_spline(i - h, j + h, k, data, False)
        + interp_spline(i - h, j - h, k, data, False)
    ) / (4.0 * h * h)

    f_xz = (
        interp_spline(i + h, j, k + h, data, False)
        - interp_spline(i + h, j, k - h, data, False)
        - interp_spline(i - h, j, k + h, data, False)
        + interp_spline(i - h, j, k - h, data, False)
    ) / (4.0 * h * h)

    f_yz = (
        interp_spline(i, j + h, k + h, data, False)
        - interp_spline(i, j + h, k - h, data, False)
        - interp_spline(i, j - h, k + h, data, False)
        + interp_spline(i, j - h, k - h, data, False)
    ) / (4.0 * h * h)

    H = np.empty((3, 3))
    H[0, 0] = f_xx
    H[1, 1] = f_yy
    H[2, 2] = f_zz

    H[0, 1] = H[1, 0] = f_xy
    H[0, 2] = H[2, 0] = f_xz
    H[1, 2] = H[2, 1] = f_yz

    return H


@njit(fastmath=True)
def newton_refine_extremum(
    point,
    data,
    target_index,
    is_frac: bool = True,
    max_iter=30,
    grad_tol=1e-6,
    max_step=1.0,
    lambda0=1.0e-2,
    lambda_up=10.0,
    lambda_down=0.3,
    h=0.25,
    eig_tol=1e-10,
):
    """
    Damped + signature-aware Newton refinement for extrema or saddles in 3D.

    Parameters
    ----------
    point : array-like (3,)
        Initial point in grid coordinates.
    data : ndarray
        Scalar field data (passed-through to interp_spline).
    target_index : None or int in {0,1,2,3}
        Desired Morse index: number of negative eigenvalues.
          - None: preserve previous behavior using is_maximum flag.
          - 0 => minimum (all eigenvalues > 0)
          - 1 => index-1 saddle (one negative eigenvalue)
          - 2 => index-2 saddle (two negative eigenvalues)
          - 3 => maximum (all eigenvalues < 0)
    is_frac : bool
        Whether or not the provided coordinates are fractional rather than
        in voxel coordinates
    max_iter : int
        Maximum Newton iterations.
    grad_tol : float
        Convergence tolerance on gradient norm (L2).
    max_step : float
        Maximum allowed step length per iteration (in grid units).
    lambda0 : float
        Initial damping parameter (used to clamp eigenvalues).
    lambda_up, lambda_down : float
        Multiplicative factors for increasing / decreasing damping.
    h : float
        Finite-difference spacing used by spline_grad/spline_hess.

    eig_tol : float
        Small threshold used when classifying eigenvalues as negative/positive.

    Returns
    -------
    x : ndarray (3,)
        Refined location (grid coordinates).
    converged : bool
        True if converged (and signature matches when require_signature=True).
    info : dict
        Diagnostics: {'grad_norm': ..., 'evals': ndarray of final Hessian eigenvalues}
    """
        
    nx, ny, nz = data.shape
    
    i, j, k = point

    # convert fractional to voxel coordinates
    if is_frac:
        i = i * nx
        j = j * ny
        k = k * nz

    for it in range(max_iter):
        # gradient and its norm
        gi, gj, gk = spline_grad(i, j, k, data, h)
        g_norm = get_norm(gi,gj,gk)

        # compute Hessian
        H = spline_hess(i, j, k, data, h)

        # check convergence by gradient norm
        if g_norm < grad_tol:
            # check Hessian signature
            evals, _ = np.linalg.eigh(H)
            n_neg = np.sum(evals < -eig_tol)
            if n_neg == target_index:
                if is_frac:
                    # convert back to fractional coords
                    i = i / nx % 1.0
                    j = j / ny % 1.0
                    k = k / nz % 1.0
                return i, j, k, True, g_norm, evals
            # else: we have small gradient but wrong signature -> continue attempting to adjust

        # Diagonalize symmetric Hessian
        evals, vecs = np.linalg.eigh(H)  # evals in ascending order
        # evals ascending: smallest first. We'll enforce first target_index to be negative.

        # Build modified eigenvalues with sign enforcement and clamping magnitude >= lam
        # For i < target_index -> force negative: -max(|evals[i]|, lam)
        # For i >= target_index -> force positive: +max(|evals[i]|, lam)
        # This ensures invertibility and desired signature.
        evals_mod = np.empty_like(evals)
        for idx_ev in range(3):
            mag = max(abs(evals[idx_ev]), lambda0)
            if idx_ev < target_index:
                evals_mod[idx_ev] = -mag
            else:
                evals_mod[idx_ev] = +mag

        # Reconstruct modified Hessian in original basis: H_mod = V diag(evals_mod) V^T
        H_mod = (vecs * evals_mod[np.newaxis, :]) @ vecs.T  # efficient diag-multiply then matmul

        # Solve linear system (H_mod) delta = -g
        try:
            di, dj, dk = np.linalg.solve(H_mod, -np.array((gi,gj,gk),dtype=np.float64))
        except:
            # increase damping and try next iter
            lambda0 *= lambda_up
            if lambda0 > 1e12:
                # avoid runaway and return original point
                if is_frac:
                    # convert back to fractional coords
                    i = i / nx % 1.0
                    j = j / ny % 1.0
                    k = k / nz % 1.0
                return i, j, k, False, g_norm, evals
                # return point[0], point[1], point[2], False, g_norm, evals
            continue

        # clamp step length
        step_norm = get_norm(di,dj,dk)
        if step_norm > max_step:
            adj = max_step / step_norm
            di = di * adj
            dj = dj * adj
            dk = dk * adj

        i_trial = i + di
        j_trial = j + dj
        k_trial = k + dk

        # trial gradient
        gi_trial, gj_trial, gk_trial = spline_grad(i_trial, j_trial, k_trial, data, h)
        g_trial_norm = get_norm(gi_trial, gj_trial, gk_trial)

        # Acceptance rule: accept if gradient norm decreases
        if g_trial_norm < g_norm:
            # accept
            i = i_trial
            j = j_trial
            k = k_trial
            lambda0 = max(lambda0 * lambda_down, 1e-16)
        else:
            # reject and increase damping
            lambda0 *= lambda_up
            # continue to next iteration without updating x
            continue

    # finished loop without meeting convergence criteria
    # return final diagnostic information and original point
    evals_final, _ = np.linalg.eigh(spline_hess(i, j, k, data, h))
    gi_final, gj_final, gk_final = spline_grad(i, j, k, data, h)
    grad_norm_final = get_norm(gi_final, gj_final, gk_final)
    
    n_neg_final = np.sum(evals_final < -eig_tol)
    success = (grad_norm_final < grad_tol) and (n_neg_final == target_index)
    
    if is_frac:
        # convert back to fractional coords
        i = i / nx % 1.0
        j = j / ny % 1.0
        k = k / nz % 1.0

    return i, j, k, success, grad_norm_final, evals_final


# Plan:
    # 1. Get residuals
    # 2. Find minima in residuals
    # 3. Get manifold labels
    # 4. Get smallest residual for each connection

bader = Bader.from_vasp("CHGCAR")
cp = CriticalPoints(bader)
manifold_labels = cp.manifold_labels

final_saddles = cp.saddle2_vox

grid = bader.reference_grid.copy()
data = grid.total

new_coords = []
finished = []
# diagnostics = []
for coords in final_saddles:
    i,j,k, success, _, _ = newton_refine_extremum(
        point=coords.astype(np.float64),
        data=data,
        target_index=2,
        is_frac=False
        )
    print(success)
    new_coords.append((i,j,k))
    finished.append(success)
    
new_coords = np.array(new_coords) / grid.shape

test_structure = bader.structure.copy()
for coord in new_coords:
    test_structure.append("x", coord)
test_structure.to("POSCAR_TEST")
