# -*- coding: utf-8 -*-
import math

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from baderkit.core.utilities.basic import (
    get_norm,
    wrap_point, 
    merge_frac_coords_weighted,
    )

###############################################################################
# Nearest point interpolation
###############################################################################


@njit(inline="always", cache=True, fastmath=True)
def interp_nearest(i, j, k, data, is_frac=True):
    nx, ny, nz = data.shape
    if is_frac:
        # convert to voxel coordinates
        i = i * nx
        j = j * ny
        k = k * nz

    # round and wrap
    ix = int(round(i)) % nx
    iy = int(round(j)) % ny
    iz = int(round(k)) % nz

    return data[ix, iy, iz]


###############################################################################
# Linear interpolation
###############################################################################
@njit(inline="always", cache=True, fastmath=True)
def interp_linear(i, j, k, data, is_frac=True):
    nx, ny, nz = data.shape

    if is_frac:
        # convert to voxel coordinates
        i = i * nx
        j = j * ny
        k = k * nz

    # wrap coord
    i, j, k = wrap_point(i, j, k, nx, ny, nz)

    # get rounded down voxel coords
    ri = int(i // 1.0)
    rj = int(j // 1.0)
    rk = int(k // 1.0)

    # get offset from rounded voxel coord
    di = i - ri
    dj = j - rj
    dk = k - rk

    # get data in 2x2x2 cube surrounding point
    v000 = data[ri, rj, rk]
    v100 = data[(ri + 1) % nx, rj, rk]
    v010 = data[ri, (rj + 1) % ny, rk]
    v001 = data[ri, rj, (rk + 1) % nz]
    v110 = data[(ri + 1) % nx, (rj + 1) % ny, rk]
    v101 = data[(ri + 1) % nx, rj, (rk + 1) % nz]
    v011 = data[ri, (rj + 1) % ny, (rk + 1) % nz]
    v111 = data[(ri + 1) % nx, (rj + 1) % ny, (rk + 1) % nz]

    # interpolate value from linear approximation
    return (
        (1 - di) * (1 - dj) * (1 - dk) * v000
        + di * (1 - dj) * (1 - dk) * v100
        + (1 - di) * dj * (1 - dk) * v010
        + (1 - di) * (1 - dj) * dk * v001
        + di * dj * (1 - dk) * v110
        + di * (1 - dj) * dk * v101
        + (1 - di) * dj * dk * v011
        + di * dj * dk * v111
    )


###############################################################################
# Cubic spline interpolation
###############################################################################


@njit(cache=True, inline="always", fastmath=True)
def cubic_bspline_weights(di, dj, dk):
    weights = np.empty((3, 4), dtype=np.float64)
    for d_idx, d in enumerate((di, dj, dk)):
        for i in range(4):
            x = abs((i - 1) - d)
            if x < 1.0:
                w = (4.0 - 6.0 * x * x + 3.0 * x * x * x) / 6.0
            elif x < 2.0:
                t = 2.0 - x
                w = (t * t * t) / 6.0
            else:
                w = 0.0
            weights[d_idx, i] = w
    return weights


@njit(cache=True, fastmath=True)
def interp_spline(i, j, k, data, is_frac=True):
    nx, ny, nz = data.shape

    # convert fractional to voxel coordinates
    if is_frac:
        i = i * nx
        j = j * ny
        k = k * nz

    # round down to get int value
    ri = int(math.floor(i))  # floor works with negative too
    rj = int(math.floor(j))
    rk = int(math.floor(k))

    # get fractional offsets in [0,1)
    di = i - ri
    dj = j - rj
    dk = k - rk

    # calculate weights
    weights = cubic_bspline_weights(di, dj, dk)

    # separable evaluation:
    # first convolve along x for the 4x4 neighborhood in y,z to produce tmp[4,4]
    tmp = np.zeros((4, 4), dtype=np.float64)  # tmp[j_index, k_index]
    for joff in range(4):
        yj = (rj - 1 + joff) % ny
        for koff in range(4):
            zk = (rk - 1 + koff) % nz
            # convolve along x for this (y,z)
            s = 0.0
            for ioff in range(4):
                xi = (ri - 1 + ioff) % nx
                s += weights[0, ioff] * data[xi, yj, zk]
            tmp[joff, koff] = s

    # now convolve tmp by wy along y and wz along z
    val = 0.0
    for joff in range(4):
        for koff in range(4):
            val += weights[1, joff] * weights[2, koff] * tmp[joff, koff]

    return val


###############################################################################
# Methods to interpolate points depending on requested method
###############################################################################


@njit(cache=True)
def interpolate_point(
    point,
    method,
    data,
    is_frac=True,
):
    i, j, k = point
    if method == "nearest":
        value = interp_nearest(i, j, k, data, is_frac)
    elif method == "linear":
        value = interp_linear(i, j, k, data, is_frac)
    elif method == "cubic":
        value = interp_spline(i, j, k, data, is_frac)
    # elif method == "quintic":
    #     value = interp_spline(i, j, k, data, order=5)

    return value


@njit(parallel=True, cache=True)
def interpolate_points(points, method, data, is_frac=True):
    out = np.empty(len(points))
    if method == "nearest":
        for point_idx in prange(len(points)):
            i, j, k = points[point_idx]
            out[point_idx] = interp_nearest(i, j, k, data, is_frac)
    elif method == "linear":
        for point_idx in prange(len(points)):
            i, j, k = points[point_idx]
            out[point_idx] = interp_linear(i, j, k, data, is_frac)
    elif method == "cubic":
        for point_idx in prange(len(points)):
            i, j, k = points[point_idx]
            out[point_idx] = interp_spline(i, j, k, data, is_frac)
    # elif method == "quintic":
    #     for i in prange(len(points)):
    #         i, j, k = points[i]
    #         out[i] = interp_spline(i, j, k, data, order=5)

    return out


@njit(cache=True)
def linear_slice(
    data,
    p1: NDArray[float],
    p2: NDArray[float],
    n: int = 100,
    is_frac=True,
    method="cubic",
):

    x_pts = np.linspace(p1[0], p2[0], num=n)
    y_pts = np.linspace(p1[1], p2[1], num=n)
    z_pts = np.linspace(p1[2], p2[2], num=n)
    coords = np.column_stack((x_pts, y_pts, z_pts))

    return interpolate_points(coords, method, data, is_frac)


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
                    sign * grid[
                        (ix + dx) % nx,
                        (iy + dy) % ny,
                        (iz + dz) % nz
                    ]
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
    extrema_children,
    data,
    labels,
    lattice,
    use_minima = False,
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
        group = extrema_children[group_idx]
        values = np.empty(len(group), dtype=np.float64)
        for idx, (i,j,k) in enumerate(group):
            values[idx] = data[i,j,k]
        best_value = values.max()
        group_frac = group / shape
        # get average frac weighted by value
        average_frac = merge_frac_coords_weighted(group_frac, values)
        # get equivalent grid point
        ai, aj, ak = np.round(average_frac*shape).astype(np.int64)
        # check if this point is in the right basin and has the highest value
        label = labels[ai, aj, ak]
        value = data[ai,aj,ak]
        if label != group_idx or value != best_value:
            # default to current extrema representing this group
            ai, aj, ak = extrema_coords[group_idx]
            average_frac = extrema_coords[group_idx] / shape
        new_voxel_coords[group_idx] = (ai, aj, ak)

        refined_frac, new_value = refine_frac_extrema_parabolic(data, average_frac, lattice)
        frac_coords[group_idx] = refined_frac
        refined_values[group_idx] = new_value
    # round and wrap coords
    frac_coords = np.round(frac_coords, 6)
    frac_coords %= 1
    return new_voxel_coords, frac_coords, refined_values

###############################################################################
# Newton Refinement
###############################################################################

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
    i = float(i)
    j = float(j)
    k = float(k)

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
                    i = i / nx# % 1.0
                    j = j / ny #% 1.0
                    k = k / nz #% 1.0
                return i, j, k, 0, g_norm, evals
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
                    i = i / nx# % 1.0
                    j = j / ny# % 1.0
                    k = k / nz# % 1.0
                return i, j, k, 2, g_norm, evals
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
    if success:
        success = 0
    else:
        success = 1
    
    if is_frac:
        # convert back to fractional coords
        i = i / nx# % 1.0
        j = j / ny# % 1.0
        k = k / nz# % 1.0

    return i, j, k, success, grad_norm_final, evals_final

@njit(parallel=True, cache=True)
def refine_critical_points(
    points,
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
    
    refined_points = np.empty((len(points), 3), dtype=np.float64)
    refined_status = np.empty(len(points), dtype=np.uint8)
    
    for idx in prange(len(points)):
        point = points[idx]
        
        i, j, k, success, _, _ = newton_refine_extremum(
            point,
            data,
            target_index,
            is_frac,
            max_iter,
            grad_tol,
            max_step,
            lambda0,
            lambda_up,
            lambda_down,
            h,
            eig_tol,
        )
        refined_points[idx, 0] = i
        refined_points[idx, 1] = j
        refined_points[idx, 2] = k
        refined_status[idx] = success
    return refined_points, refined_status
        
@njit(parallel=True, cache=True)
def refine_extrema(
    extrema_coords,
    extrema_children,
    data,
    labels,
    lattice,
    use_minima = False,
):
    shape = np.array(data.shape, dtype=np.int64)
    nx,ny,nz = shape
    # for each group of extrema, we try and merge them into one. If the resulting
    # point is not part of the group or does not have the maximum value of the
    # group, we default to the highest point or lowest index in case of a tie.
    # A newton refinement is then performed
    if use_minima:
        target_index = 0
    else:
        target_index = 3
    
    new_voxel_coords = np.empty_like(extrema_coords, dtype=np.uint16)
    frac_coords = np.empty_like(extrema_coords, dtype=np.float64)
    refined_values = np.empty(len(extrema_coords), dtype=np.float64)
    for group_idx in prange(len(extrema_coords)):
        group = extrema_children[group_idx]
        values = np.empty(len(group), dtype=np.float64)
        for idx, (i,j,k) in enumerate(group):
            values[idx] = data[i,j,k]
        best_value = values.max()
        group_frac = group / shape
        # get average frac weighted by value
        average_frac = merge_frac_coords_weighted(group_frac, values)
        # get equivalent grid point
        ai, aj, ak = np.round(average_frac*shape).astype(np.int64)
        # check if this point is in the right basin and has the highest value
        label = labels[ai, aj, ak]
        value = data[ai,aj,ak]
        if label != group_idx or value != best_value:
            # default to current extrema representing this group
            ai, aj, ak = extrema_coords[group_idx]
            average_frac = extrema_coords[group_idx] / shape
        new_voxel_coords[group_idx] = (ai, aj, ak)
        
        i, j, k, success, _, _ = newton_refine_extremum(
            (ai,aj,ak),
            data,
            target_index,
            is_frac=False,
        )

        frac_coords[group_idx] = (i/nx,j/ny,k/nz)
        # todo: interpolate value? I don't want to overshoot with a spline
        refined_values[group_idx] = best_value
    # round and wrap coords
    frac_coords = np.round(frac_coords, 6)
    # frac_coords %= 1
    return new_voxel_coords, frac_coords, refined_values