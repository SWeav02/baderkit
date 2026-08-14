# -*- coding: utf-8 -*-

import numpy as np
from numpy.typing import NDArray
from numba import njit, prange

###############################################################################
# Promolecular Methods
###############################################################################
@njit(fastmath=True)
def get_pbc_displacements(
    diff_cart: np.ndarray,
    inv_lattice: np.ndarray,
    lattice_matrix: np.ndarray,
    rcut_sq: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes fractional wrapping, 27-image Cartesian shift generation, and

    minimum image distance filtering inside Numba.

    Parameters
    ----------
    diff_cart : np.ndarray
        Raw Cartesian displacement vectors (N, 3) from atom to grid points.
    inv_lattice : np.ndarray
        Inverse lattice matrix (3, 3) for Cartesian -> Fractional conversion.
    lattice_matrix : np.ndarray
        Lattice matrix (3, 3) where rows are lattice vectors a1, a2, a3.
    rcut_sq : float
        Square of the cutoff radius (rcut**2) for distance filtering.

    Returns
    -------
    dr_cart_min : np.ndarray
        Minimum Cartesian displacement vectors (M, 3) for points within rcut.
    inside_mask : np.ndarray
        Boolean mask (N,) indicating which input points fell within rcut.
    """
    # 1. Generate all 27 Cartesian shift vectors directly from lattice_matrix
    cart_shifts = np.empty((27, 3), dtype=np.float64)
    shift_idx = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                for d in range(3):
                    cart_shifts[shift_idx, d] = (
                        i * lattice_matrix[0, d]
                        + j * lattice_matrix[1, d]
                        + k * lattice_matrix[2, d]
                    )
                shift_idx += 1

    n_pts = diff_cart.shape[0]
    dr_out = np.empty((n_pts, 3), dtype=np.float64)
    inside_mask = np.zeros(n_pts, dtype=np.bool_)
    n_inside = 0

    for p in range(n_pts):
        # 2. Fractional conversion: f = diff_cart @ inv_lattice
        f0 = (
            diff_cart[p, 0] * inv_lattice[0, 0]
            + diff_cart[p, 1] * inv_lattice[1, 0]
            + diff_cart[p, 2] * inv_lattice[2, 0]
        )
        f1 = (
            diff_cart[p, 0] * inv_lattice[0, 1]
            + diff_cart[p, 1] * inv_lattice[1, 1]
            + diff_cart[p, 2] * inv_lattice[2, 1]
        )
        f2 = (
            diff_cart[p, 0] * inv_lattice[0, 2]
            + diff_cart[p, 1] * inv_lattice[1, 2]
            + diff_cart[p, 2] * inv_lattice[2, 2]
        )

        # 3. Primary fractional centering
        f0 -= round(f0)
        f1 -= round(f1)
        f2 -= round(f2)

        # 4. Base Cartesian position: r_base = f @ lattice_matrix
        x0 = (
            f0 * lattice_matrix[0, 0]
            + f1 * lattice_matrix[1, 0]
            + f2 * lattice_matrix[2, 0]
        )
        y0 = (
            f0 * lattice_matrix[0, 1]
            + f1 * lattice_matrix[1, 1]
            + f2 * lattice_matrix[2, 1]
        )
        z0 = (
            f0 * lattice_matrix[0, 2]
            + f1 * lattice_matrix[1, 2]
            + f2 * lattice_matrix[2, 2]
        )

        # 5. Search all 27 Cartesian shifts for global minimum distance
        best_d2 = 1e300
        best_x = 0.0
        best_y = 0.0
        best_z = 0.0

        for k in range(27):
            dx = x0 + cart_shifts[k, 0]
            dy = y0 + cart_shifts[k, 1]
            dz = z0 + cart_shifts[k, 2]

            d2 = dx * dx + dy * dy + dz * dz
            if d2 < best_d2:
                best_d2 = d2
                best_x = dx
                best_y = dy
                best_z = dz

        # 6. Apply distance cutoff filter
        if best_d2 <= rcut_sq:
            dr_out[n_inside, 0] = best_x
            dr_out[n_inside, 1] = best_y
            dr_out[n_inside, 2] = best_z
            inside_mask[p] = True
            n_inside += 1

    return dr_out[:n_inside], inside_mask

###############################################################################
# Spherical Harmonics
###############################################################################

def evaluate_real_harmonics_multi(
    l: int,
    m: int,
    q_vecs: NDArray,
    compute_gradients: bool = False,
) -> tuple[NDArray, NDArray | None]:
    """Computes standard orthonormal real spherical harmonics Y_lm in Cartesian coordinates.

    Coordinates are normalized on-the-fly with a division-by-zero safeguard at q=0.

    Parameters
    ----------
    l : int
        Angular momentum quantum number (0 <= l <= 4).
    m : int
        Magnetic quantum number (-l <= m <= l).
    q_vecs : NDArray
        Cartesian displacement vectors of shape (N, 3).
    compute_gradients : bool, default=False
        If True, also computes and returns the Cartesian gradient grad(Y_lm) of shape (N, 3).

    Returns
    -------
    y_lm : NDArray
        Values of real spherical harmonic Y_lm of shape (N,).
    grad_y_lm : NDArray or None
        Cartesian gradient of Y_lm of shape (N, 3) if compute_gradients is True, else None.
    """
    norms = np.linalg.norm(q_vecs, axis=1)
    norms_safe = np.where(norms < 1e-12, 1.0, norms)
    u = q_vecs / norms_safe[:, np.newaxis]
    x, y, z = u[:, 0], u[:, 1], u[:, 2]

    d_u = np.zeros_like(q_vecs) if compute_gradients else None

    if l == 0:
        c0 = 0.5 * np.sqrt(1.0 / np.pi)
        y_lm = np.full_like(norms, c0)

    elif l == 1:
        c1 = np.sqrt(3.0 / (4.0 * np.pi))
        if m == -1:    # p_y
            y_lm = c1 * y
            if compute_gradients:
                d_u[:, 1] = c1
        elif m == 0:   # p_z
            y_lm = c1 * z
            if compute_gradients:
                d_u[:, 2] = c1
        elif m == 1:   # p_x
            y_lm = c1 * x
            if compute_gradients:
                d_u[:, 0] = c1

    elif l == 2:
        c2 = 0.5 * np.sqrt(15.0 / np.pi)
        c20 = 0.25 * np.sqrt(5.0 / np.pi)
        c22 = 0.25 * np.sqrt(15.0 / np.pi)
        if m == -2:    # d_xy
            y_lm = c2 * x * y
            if compute_gradients:
                d_u[:, 0], d_u[:, 1] = c2 * y, c2 * x
        elif m == -1:  # d_yz
            y_lm = c2 * y * z
            if compute_gradients:
                d_u[:, 1], d_u[:, 2] = c2 * z, c2 * y
        elif m == 0:   # d_z2
            y_lm = c20 * (3.0 * z**2 - 1.0)
            if compute_gradients:
                d_u[:, 2] = c20 * 6.0 * z
        elif m == 1:   # d_xz
            y_lm = c2 * x * z
            if compute_gradients:
                d_u[:, 0], d_u[:, 2] = c2 * z, c2 * x
        elif m == 2:   # d_x2-y2
            y_lm = c22 * (x**2 - y**2)
            if compute_gradients:
                d_u[:, 0], d_u[:, 1] = c22 * 2.0 * x, -c22 * 2.0 * y

    elif l == 3:
        c3_3 = 0.25 * np.sqrt(35.0 / (2.0 * np.pi))
        c3_2 = 0.5 * np.sqrt(105.0 / np.pi)
        c3_1 = 0.25 * np.sqrt(21.0 / (2.0 * np.pi))
        c3_0 = 0.25 * np.sqrt(7.0 / np.pi)
        c3_22 = 0.25 * np.sqrt(105.0 / (2.0 * np.pi))
        if m == -3:
            y_lm = c3_3 * y * (3.0 * x**2 - y**2)
            if compute_gradients:
                d_u[:, 0] = c3_3 * 6.0 * x * y
                d_u[:, 1] = c3_3 * (3.0 * x**2 - 3.0 * y**2)
        elif m == -2:
            y_lm = c3_2 * x * y * z
            if compute_gradients:
                d_u[:, 0] = c3_2 * y * z
                d_u[:, 1] = c3_2 * x * z
                d_u[:, 2] = c3_2 * x * y
        elif m == -1:
            y_lm = c3_1 * y * (5.0 * z**2 - 1.0)
            if compute_gradients:
                d_u[:, 1] = c3_1 * (5.0 * z**2 - 1.0)
                d_u[:, 2] = c3_1 * 10.0 * y * z
        elif m == 0:
            y_lm = c3_0 * z * (5.0 * z**2 - 3.0)
            if compute_gradients:
                d_u[:, 2] = c3_0 * (15.0 * z**2 - 3.0)
        elif m == 1:
            y_lm = c3_1 * x * (5.0 * z**2 - 1.0)
            if compute_gradients:
                d_u[:, 0] = c3_1 * (5.0 * z**2 - 1.0)
                d_u[:, 2] = c3_1 * 10.0 * x * z
        elif m == 2:
            y_lm = c3_22 * z * (x**2 - y**2)
            if compute_gradients:
                d_u[:, 0] = c3_22 * 2.0 * x * z
                d_u[:, 1] = -c3_22 * 2.0 * y * z
                d_u[:, 2] = c3_22 * (x**2 - y**2)
        elif m == 3:
            y_lm = c3_3 * x * (x**2 - 3.0 * y**2)
            if compute_gradients:
                d_u[:, 0] = c3_3 * (3.0 * x**2 - 3.0 * y**2)
                d_u[:, 1] = -c3_3 * 6.0 * x * y

    elif l == 4:
        c4_4 = 0.75 * np.sqrt(35.0 / np.pi)
        c4_3 = 0.75 * np.sqrt(35.0 / (2.0 * np.pi))
        c4_2 = 0.75 * np.sqrt(5.0 / np.pi)
        c4_1 = 0.75 * np.sqrt(5.0 / (2.0 * np.pi))
        c4_0 = 0.1875 * np.sqrt(1.0 / np.pi)
        c4_22 = 0.375 * np.sqrt(5.0 / np.pi)
        c4_44 = 0.1875 * np.sqrt(35.0 / np.pi)

        if m == -4:
            y_lm = c4_4 * x * y * (x**2 - y**2)
            if compute_gradients:
                d_u[:, 0] = c4_4 * (3.0 * x**2 * y - y**3)
                d_u[:, 1] = c4_4 * (x**3 - 3.0 * x * y**2)
        elif m == -3:
            y_lm = c4_3 * y * z * (3.0 * x**2 - y**2)
            if compute_gradients:
                d_u[:, 0] = c4_3 * 6.0 * x * y * z
                d_u[:, 1] = c4_3 * z * (3.0 * x**2 - 3.0 * y**2)
                d_u[:, 2] = c4_3 * y * (3.0 * x**2 - y**2)
        elif m == -2:
            y_lm = c4_2 * x * y * (7.0 * z**2 - 1.0)
            if compute_gradients:
                d_u[:, 0] = c4_2 * y * (7.0 * z**2 - 1.0)
                d_u[:, 1] = c4_2 * x * (7.0 * z**2 - 1.0)
                d_u[:, 2] = c4_2 * 14.0 * x * y * z
        elif m == -1:
            y_lm = c4_1 * y * z * (7.0 * z**2 - 3.0)
            if compute_gradients:
                d_u[:, 1] = c4_1 * z * (7.0 * z**2 - 3.0)
                d_u[:, 2] = c4_1 * y * (21.0 * z**2 - 3.0)
        elif m == 0:
            y_lm = c4_0 * (35.0 * z**4 - 30.0 * z**2 + 3.0)
            if compute_gradients:
                d_u[:, 2] = c4_0 * (140.0 * z**3 - 60.0 * z)
        elif m == 1:
            y_lm = c4_1 * x * z * (7.0 * z**2 - 3.0)
            if compute_gradients:
                d_u[:, 0] = c4_1 * z * (7.0 * z**2 - 3.0)
                d_u[:, 2] = c4_1 * x * (21.0 * z**2 - 3.0)
        elif m == 2:
            y_lm = c4_22 * (x**2 - y**2) * (7.0 * z**2 - 1.0)
            if compute_gradients:
                d_u[:, 0] = c4_22 * 2.0 * x * (7.0 * z**2 - 1.0)
                d_u[:, 1] = -c4_22 * 2.0 * y * (7.0 * z**2 - 1.0)
                d_u[:, 2] = c4_22 * 14.0 * z * (x**2 - y**2)
        elif m == 3:
            y_lm = c4_3 * x * z * (x**2 - 3.0 * y**2)
            if compute_gradients:
                d_u[:, 0] = c4_3 * z * (3.0 * x**2 - 3.0 * y**2)
                d_u[:, 1] = -c4_3 * 6.0 * x * y * z
                d_u[:, 2] = c4_3 * x * (x**2 - 3.0 * y**2)
        elif m == 4:
            y_lm = c4_44 * (x**4 - 6.0 * x**2 * y**2 + y**4)
            if compute_gradients:
                d_u[:, 0] = c4_44 * (4.0 * x**3 - 12.0 * x * y**2)
                d_u[:, 1] = c4_44 * (-12.0 * x**2 * y + 4.0 * y**3)
    else:
        y_lm = np.zeros_like(norms)

    grad_y_lm = None
    if compute_gradients:
        grad_y_lm = (d_u - l * y_lm[:, np.newaxis] * u) / norms_safe[:, np.newaxis]
        grad_y_lm = np.where(norms[:, np.newaxis] < 1e-12, 0.0, grad_y_lm)

    return y_lm, grad_y_lm

###############################################################################
# Tetrahedron Smearing
###############################################################################
@njit(parallel=True, fastmath=True, cache=True)
def integrate_tetrahedra_spectral_density(
    egrid,
    tetra_indices,
    eigenvalues,
    cached_metrics,
    tetra_weight
):
    """
    Parallelized JIT-compiled linear tetrahedron engine.
    
    Optimized:
    - Bounded energy grid iterations (w_start to w_end)
    - Parallelized over (spin * bands) to utilize all CPU cores
    - Zero-allocation 5-comparator sorting network for 4 vertices
    - Hoisted property metric lookups outside the energy loop
    """
    n_omega = egrid.shape[0]
    n_tetra = tetra_indices.shape[0]
    n_spin = eigenvalues.shape[0]
    n_bands = eigenvalues.shape[2]
    n_metrics = cached_metrics.shape[3]

    delta_e = egrid[1] - egrid[0] if n_omega > 1 else 1.0
    egrid_min = egrid[0]

    # Pre-allocate thread-safe output buffer per band to prevent race conditions
    band_out = np.zeros((n_spin, n_bands, n_metrics, n_omega), dtype=np.float64)

    # Parallelize across all spin channels and bands (saturates all CPU threads)
    for sb in prange(n_spin * n_bands):
        s = sb // n_bands
        b = sb % n_bands

        for t in range(n_tetra):
            k1 = tetra_indices[t, 0]
            k2 = tetra_indices[t, 1]
            k3 = tetra_indices[t, 2]
            k4 = tetra_indices[t, 3]

            v0 = eigenvalues[s, k1, b]
            v1 = eigenvalues[s, k2, b]
            v2 = eigenvalues[s, k3, b]
            v3 = eigenvalues[s, k4, b]

            i0, i1, i2, i3 = k1, k2, k3, k4

            # Zero-allocation 5-comparator sorting network for 4 elements
            if v0 > v1:
                v0, v1 = v1, v0
                i0, i1 = i1, i0
            if v2 > v3:
                v2, v3 = v3, v2
                i2, i3 = i3, i2
            if v0 > v2:
                v0, v2 = v2, v0
                i0, i2 = i2, i0
            if v1 > v3:
                v1, v3 = v3, v1
                i1, i3 = i3, i1
            if v1 > v2:
                v1, v2 = v2, v1
                i1, i2 = i2, i1

            e1, e2, e3, e4 = v0, v1, v2, v3
            idx1, idx2, idx3, idx4 = i0, i1, i2, i3

            # Flat band handling
            if e4 - e1 < 1e-7:
                w_closest = int(np.round((e1 - egrid_min) / delta_e))
                if 0 <= w_closest < n_omega:
                    for m in range(n_metrics):
                        val1 = cached_metrics[s, idx1, b, m]
                        band_out[s, b, m, w_closest] += (val1 * tetra_weight) / delta_e
                continue

            e21 = e2 - e1
            e31 = e3 - e1
            e41 = e4 - e1
            e42 = e4 - e2
            e32 = e3 - e2 if (e3 - e2) > 1e-12 else 1.0
            e43 = e4 - e3 if (e4 - e3) > 1e-12 else 1.0

            # Compute exact energy grid bounds spanned by this tetrahedron
            w_start = max(0, int((e1 - egrid_min) / delta_e))
            w_end = min(n_omega, int((e4 - egrid_min) / delta_e) + 2)

            if w_start >= w_end:
                continue

            # Hoist metric values outside the energy grid loop
            # Uses stack allocation for small n_metrics
            for w in range(w_start, w_end):
                E = egrid[w]
                if E < e1 or E >= e4:
                    continue

                w1, w2, w3, w4 = 0.0, 0.0, 0.0, 0.0

                # CASE 1: e1 <= E < e2
                if E < e2:
                    if e21 > 1e-7 and e31 > 1e-7 and e41 > 1e-7:
                        G = 3.0 * (E - e1)**2 / (e21 * e31 * e41)
                        w1 = G * (1.0 - (E - e1) * (1.0/e21 + 1.0/e31 + 1.0/e41) / 3.0)
                        w2 = G * (E - e1) / (3.0 * e21)
                        w3 = G * (E - e1) / (3.0 * e31)
                        w4 = G * (E - e1) / (3.0 * e41)

                # CASE 2: e2 <= E < e3
                elif E < e3:
                    if e41 > 1e-7 and e31 > 1e-7 and e42 > 1e-7 and e32 > 1e-7:
                        G = 3.0 * (E - e1) * (e3 - E) / (e41 * e31 * e32) + 3.0 * (e4 - E) * (E - e2) / (e41 * e42 * e32)
                        w1 = (1.0 / (e41 * e32)) * ( 
                            ((E - e1) * (e3 - E)**2) / (e31**2) + 
                            ((E - e1) * (e3 - E) * (e4 - E)) / (e41 * e31) + 
                            ((e4 - E)**2 * (E - e2)) / (e41 * e42) 
                        )
                        w4 = (1.0 / (e41 * e32)) * ( 
                            ((e4 - E) * (E - e2)**2) / (e42**2) + 
                            ((e4 - E) * (E - e2) * (E - e1)) / (e41 * e42) + 
                            ((E - e1)**2 * (e3 - E)) / (e41 * e31) 
                        )
                        w2 = (G * (e3 - E) - w1 * e31 + w4 * e43) / e32
                        w3 = G - w1 - w2 - w4

                # CASE 3: e3 <= E < e4
                else:
                    if e41 > 1e-7 and e42 > 1e-7 and e43 > 1e-7:
                        G = 3.0 * (e4 - E)**2 / (e41 * e42 * e43)
                        w1 = G * (e4 - E) / (3.0 * e41)
                        w2 = G * (e4 - E) / (3.0 * e42)
                        w3 = G * (e4 - E) / (3.0 * e43)
                        w4 = G * (1.0 - (e4 - E) * (1.0/e41 + 1.0/e42 + 1.0/e43) / 3.0)

                for m in range(n_metrics):
                    val1 = cached_metrics[s, idx1, b, m]
                    val2 = cached_metrics[s, idx2, b, m]
                    val3 = cached_metrics[s, idx3, b, m]
                    val4 = cached_metrics[s, idx4, b, m]

                    integrated_val = (w1*val1 + w2*val2 + w3*val3 + w4*val4) * tetra_weight
                    band_out[s, b, m, w] += integrated_val

    # Reduce across bands: shape (n_metrics, n_spin, n_omega)
    out = np.zeros((n_metrics, n_spin, n_omega), dtype=np.float64)
    for s in range(n_spin):
        for b in range(n_bands):
            for m in range(n_metrics):
                for w in range(n_omega):
                    out[m, s, w] += band_out[s, b, m, w]

    return out


@njit(parallel=True, fastmath=True, cache=True)
def integrate_tetrahedra_analytic_charge(
    energy_grid: np.ndarray,
    tetra_indices: np.ndarray,
    eigenvalues: np.ndarray,
    tetra_weight: float,
    rspin: float
) -> np.ndarray:
    """
    Analytically integrates the cumulative state weight N(E) across all micro-cell 
    tetrahedra. Loop hierarchy matched perfectly to the spectral density layout 
    for maximum caching efficiency and synchronized boundary mappings.
    """
    num_points = len(energy_grid)
    num_tets = tetra_indices.shape[0]
    nspin = eigenvalues.shape[0]
    nbands = eigenvalues.shape[2]
    
    # Store per-spin arrays to preserve race-condition-free prange scaling
    out_spin = np.zeros((nspin, num_points), dtype=np.float64)
    
    for s in prange(nspin):
        for b in range(nbands):
            local_charge = np.zeros(num_points, dtype=np.float64)
            
            for t in range(num_tets):
                k0 = tetra_indices[t, 0]
                k1 = tetra_indices[t, 1]
                k2 = tetra_indices[t, 2]
                k3 = tetra_indices[t, 3]
                
                e1 = eigenvalues[s, k0, b]
                e2 = eigenvalues[s, k1, b]
                e3 = eigenvalues[s, k2, b]
                e4 = eigenvalues[s, k3, b]
                
                # Zero-allocation inline scalar sorting network
                if e1 > e2: e1, e2 = e2, e1
                if e3 > e4: e3, e4 = e4, e3
                if e1 > e3: e1, e3 = e3, e1
                if e2 > e4: e2, e4 = e4, e2
                if e2 > e3: e2, e3 = e3, e2
                
                # Flat band step-function tracking protects against electron loss
                if e4 - e1 < 1e-7:
                    for w in range(num_points):
                        if energy_grid[w] >= e1:
                            local_charge[w] += tetra_weight
                    continue
                
                e21 = e2 - e1
                e31 = e3 - e1
                e41 = e4 - e1
                e42 = e4 - e2
                e32 = e3 - e2 if (e3 - e2) > 1e-12 else 1.0
                e43 = e4 - e3 if (e4 - e3) > 1e-12 else 1.0
                denom_base = e31 * e41
                
                for w in range(num_points):
                    E = energy_grid[w]
                    
                    # FIXED: Synchronized Left-Closed, Right-Open boundary checks
                    if E < e1:
                        continue
                    elif E >= e4:
                        local_charge[w] += tetra_weight
                    elif e1 <= E < e2:
                        denom = e21 * e31 * e41
                        if denom > 1e-12:
                            local_charge[w] += tetra_weight * ((E - e1) ** 3) / denom
                    elif e2 <= E < e3:
                        if denom_base > 1e-12:
                            h = E - e2
                            v1 = e2 - e1
                            factor = (e31 + e42) / (e32 * e42)
                            term = v1**2 + 3.0*v1*h + 3.0*h**2 - factor * (h**3)
                            local_charge[w] += tetra_weight * term / denom_base
                    elif e3 <= E < e4:
                        denom = e41 * e42 * e43
                        if denom > 1e-12:
                            local_charge[w] += tetra_weight * (1.0 - ((e4 - E) ** 3) / denom)
            
            for w in range(num_points):
                out_spin[s, w] += local_charge[w]
                
    # Contract spin dimensions and scale by system degeneracies
    total_charge = np.zeros(num_points, dtype=np.float64)
    for w in range(num_points):
        for s in range(nspin):
            total_charge[w] += out_spin[s, w]
            
    return total_charge * rspin