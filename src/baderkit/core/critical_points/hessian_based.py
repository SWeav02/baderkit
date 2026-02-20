# -*- coding: utf-8 -*-

from numba import njit, prange
import numpy as np
from baderkit.core.utilities.basic import coords_to_flat, get_norm
from baderkit.core.utilities.interpolation import (
    newton_refine_critical, 
    spline_grad_cart, 
    spline_hess_cart, 
    spline_grad_and_hess,
    
    )

@njit(cache=True)
def gather_cell_gradients(i, j, k, gradients):
    nx, ny, nz, _ = gradients.shape
    G = np.empty((8, 3), dtype=np.float64)

    idx = 0
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                ii = (i + di) % nx
                jj = (j + dj) % ny
                kk = (k + dk) % nz
                G[idx, :] = gradients[ii,jj,kk]
                idx += 1
    return G

@njit(cache=True)
def encloses_zero_component(vals):
    return vals.min() <= 0.0 and vals.max() >= 0.0

@njit(cache=True)
def gradient_enclosure_test(G):
    return (
        encloses_zero_component(G[:, 0]) and
        encloses_zero_component(G[:, 1]) and
        encloses_zero_component(G[:, 2])
    )
@njit(cache=True)
def grad_norm_lower_bound(G):
    return np.sqrt(
        np.min(G[:, 0] * G[:, 0]) +
        np.min(G[:, 1] * G[:, 1]) +
        np.min(G[:, 2] * G[:, 2])
    )


@njit(cache=True)
def classify_critical_point(H):
    eigvals = np.linalg.eigvalsh(H)

    n_pos = np.sum(eigvals > 0.0)
    n_neg = np.sum(eigvals < 0.0)

    if n_pos == 3:
        return 0   # minimum
    if n_neg == 3:
        return 3   # maximum
    if n_pos == 2 and n_neg == 1:
        return 1   # index-1 saddle
    if n_pos == 1 and n_neg == 2:
        return 2   # index-2 saddle

    return -1

###############################################################################
# Fast pass cutoff
###############################################################################
# NOTE: This ongrid method seems to work almost as well as the interpolation
# version even on rough grids
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
    i,j,k = coord
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
        gi * (inv_G[0, 0] * gi + inv_G[0, 1] * gj + inv_G[0, 2] * gk) +
        gj * (inv_G[1, 0] * gi + inv_G[1, 1] * gj + inv_G[1, 2] * gk) +
        gk * (inv_G[2, 0] * gi + inv_G[2, 1] * gj + inv_G[2, 2] * gk)
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
    H_frac_max = max(
        abs(Hxx), abs(Hyy), abs(Hzz),
        abs(Hxy), abs(Hxz), abs(Hyz)
    ) + eps

    H_cart_max = H_frac_max * H_frac_to_cart
    H_cart_max2 = H_cart_max * H_cart_max

    # -----------------------------
    # Stage 1: gradient rejection
    # -----------------------------
    if g_cart2 > H_cart_max2:
        return -1

    # -----------------------------
    # Stage 2: directionally-aware Newton bound
    # -----------------------------
    
    # Hessian-vector product (fractional)
    Hg_i = Hxx * gi + Hxy * gj + Hxz * gk
    Hg_j = Hxy * gi + Hyy * gj + Hyz * gk
    Hg_k = Hxz * gi + Hyz * gj + Hzz * gk
    
    # Rayleigh quotient: curvature along gradient
    gHg = gi * Hg_i + gj * Hg_j + gk * Hg_k
    
    # Allow saddles: only magnitude matters
    lam_eff = abs(gHg) / (gi*gi + gj*gj + gk*gk + eps)
    
    # Convert to Cartesian curvature
    lam_cart = lam_eff * H_frac_to_cart + eps
    
    # ||Δx||² ≈ ||g||² / λ_eff²
    delta_est2 = g_cart2 / (lam_cart * lam_cart)
    
    
    if delta_est2 > r_voxel_cart2:
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
# Newton Refinement
###############################################################################

@njit(fastmath=True)
def cartesian_grad_norm2(g, inv_G):
    return (
        g[0] * (inv_G[0, 0] * g[0] + inv_G[0, 1] * g[1] + inv_G[0, 2] * g[2]) +
        g[1] * (inv_G[1, 0] * g[0] + inv_G[1, 1] * g[1] + inv_G[1, 2] * g[2]) +
        g[2] * (inv_G[2, 0] * g[0] + inv_G[2, 1] * g[1] + inv_G[2, 2] * g[2])
    )

@njit(fastmath=True)
def compute_signature(H, eig_rel_tol):
    evals, _ = np.linalg.eigh(H)
    
    tol = eig_rel_tol * np.max(np.abs(evals))
    
    n_neg = 0
    n_flat = 0
    for i in evals:
        if i < -tol:
            n_neg += 1
        elif abs(i) < tol:
            n_flat += 1

    return n_neg, n_flat

@njit(fastmath=True)
def check_flat_extrema(evals, eig_tol):
    n_flat = 0
    for i in evals:
        if abs(i) < eig_tol:
            n_flat += 1
    return n_flat == 1

@njit(fastmath=True)
def modified_hessian(H, g_norm, Q, g, lambda0, grad_tol):

    evals, vecs = np.linalg.eigh(H)  
    lam = max(lambda0, 0.05 * np.max(np.abs(evals)))
    
    # Only allow clamping once gradient is already small
    if g_norm < 10.0 * grad_tol:
        flat = np.abs(evals) < lam
    else:
        flat = np.zeros_like(evals, dtype=bool)
        
        # Clamp flat directions permanently
    if np.any(flat):
        Q = Q @ vecs[:, ~flat]
        evals = evals[~flat]
        vecs = vecs[:, ~flat]
        
    # Project gradient
    g_proj = Q.T @ g
    
    # If fully flat, return zero Hessian
    if evals.size == 0:
        H_mod = np.zeros_like(H)
        return H_mod, evals, Q, g_proj
    
    # Enforce well-conditioned curvature
    evals_mod = np.empty_like(evals)
    for i in range(len(evals)):
        mag = max(abs(evals[i]), lam)
        evals_mod[i] = mag if evals[i] >= 0 else -mag
        
    # Reconstruct modified Hessian in full space
    H_mod = Q @ (evals_mod[:, None] * Q.T)

    return H_mod, evals, Q, g_proj

@njit(fastmath=True)
def clamp_step(dx, max_step):
    step_norm = get_norm(dx[0], dx[1], dx[2])
    if step_norm < 1e-8:
        return dx, False

    scale = min(1.0, max_step / (step_norm + 1e-12))
    scale = scale ** 0.5
    return dx*scale, True

@njit(fastmath=True)
def outside_voxel(coord, vmin, vmax):
    return (
        np.any(coord < vmin)
        or np.any(coord > vmax)
    )

@njit(cache=True)
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

@njit(fastmath=True)
def newton_refine_in_voxel(
    coord,
    data,
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

    voxel_min = np.array(coord - max_change)
    voxel_max = np.array(coord + max_change)
    
    converged = False

    for _ in range(max_iter):
        # get gradient and hessian at this point
        g, H = spline_grad_and_hess(coord, data, h)
        
        # get step
        dx = flat_aware_newton_step(g, H, eig_rel_tol)
    
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

###############################################################################
# Critical Point Finder
###############################################################################

@njit(parallel=True, cache=True)
def find_saddle_points(
    data,
    matrix,
    saddle_mask,
    max_change=1,
    max_iter=30,
    grad_tol=1e-6,
    h=0.25,
    eig_rel_tol=1e-04,
):
    nx, ny, nz = data.shape
    shape = np.array(data.shape, dtype=np.int64)

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
    
    # get range of values
    # eig_tol = 1e-4*(data.max() - data.min())
    allowed_coords = np.argwhere(saddle_mask == np.iinfo(saddle_mask.dtype).max)
    
    for coord_idx in prange(len(allowed_coords)):
        coord = allowed_coords[coord_idx]

        morse_index =  check_valid_newton_step(
            coord,
            data,
            r_voxel_cart2,
            inv_G,
            H_frac_to_cart,
        )
        
        # skip maxima, minima and invalid points
        if morse_index in (-1, 0, 3):
            continue
                        
        new_coord, success, morse_index = newton_refine_in_voxel(
            coord,
            data=data,
            inv_G=inv_G,
            max_change=max_change,
            max_iter=max_iter,
            grad_tol=grad_tol,
            h=h,
            eig_rel_tol=eig_rel_tol,
            )
        
        if not success:
            continue
        
        new_coord = np.round(new_coord).astype(np.int64) % shape

        saddle_mask[new_coord[0],new_coord[1],new_coord[2]] = morse_index

    return saddle_mask

@njit(parallel=True, cache=True)
def refine_saddle_points(
    saddle_mask,
    data,
    matrix,
        ):
    G = matrix @ matrix.T
    inv_G = np.linalg.inv(G)
    
    # get coordinates of saddles
    saddle1s = np.argwhere(saddle_mask == 1)
    saddle2s = np.argwhere(saddle_mask == 2)
    
    # create arrays to store partial coordinates
    saddle1_coords = np.empty_like(saddle1s, dtype=np.float64)
    saddle2_coords = np.empty_like(saddle2s, dtype=np.float64)

    for coord_idx in prange(len(saddle1s)):
        i,j,k = saddle1s[coord_idx]
        # refine
        ii, jj, kk, success, _, morse_index = newton_refine_in_voxel(
            float(i), float(j), float(k), 
            data=data,
            inv_G=inv_G,
            is_frac = False,
            )
        saddle1_coords[coord_idx] = (ii,jj,kk)

    for coord_idx in prange(len(saddle2s)):
        i,j,k = saddle2s[coord_idx]
        # refine
        ii, jj, kk, success, _, morse_index = newton_refine_in_voxel(
            float(i), float(j), float(k), 
            data=data,
            inv_G=inv_G,
            is_frac = False,
            )
        saddle2_coords[coord_idx] = (ii,jj,kk)
    return saddle1_coords, saddle2_coords


# from baderkit.core import Grid
# import time
# grid = Grid.from_vasp("ELFCAR")
# matrix = grid.matrix
# t0 = time.time()
# test = find_critical_points(
#     grid.total, 
#     matrix,
#     )
# t1 = time.time()
# print(t1-t0)
# test_grid = grid.copy()
# for i in range(5):
#     test_grid.total = test==i
#     test_grid.write_vasp(f"ELFCAR_test_{i}")
    
# Next Steps:
    # 1. Update maxima finding method.
        # a. group by parity like its done now. Record persistence cutoffs
        # b. get convex hull with all points in persistence.
        # c. If all maxima points are within small tolerance (1 voxel?) of hull,
        # they should form a mesh.
        # d. If any non-maxima appear inside the mesh (outside the same tolerance
        # as before) we have a cage/ring
        # e. estimate flatness. flat meshes are rings
        # f. assign all points within tolerance of mesh as maxima/minima
    # 2. Update saddle point construction:
        # a. get maxima/minima with method above. Make separate from bader, but
        # also use it in bader
        # b. get saddles from method in this file. Don't allow points in mesh
        # to be saddles
    # 3. Get connections between saddles and maxima/minima. Either:
        # a. get adjacent basins
        # b. follow gradients
    # 4. Figure out how to get saddle connections. breadth first search from
    # saddle 1s?
    