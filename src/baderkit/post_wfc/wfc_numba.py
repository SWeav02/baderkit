# -*- coding: utf-8 -*-

import numpy as np
from numpy.typing import NDArray
from numba import njit, prange
from numba.typed import List

###############################################################################
# Promolecular Methods
###############################################################################

@njit(fastmath=True, cache=True)
def find_active_periodic_atoms(lattice_matrix, base_frac_coords, atom_types, r_cut):
    """
    Finds all periodic images of the atoms whose cutoff spheres overlap with the
    primary unit cell volume [0, 1]^3.
    """
    num_atoms = base_frac_coords.shape[0]
    inv_lattice = np.linalg.inv(lattice_matrix)
    
    max_bounds = np.zeros(3, dtype=np.int32)
    for i in range(3):
        max_bounds[i] = int(np.ceil(r_cut * np.linalg.norm(inv_lattice[:, i]))) + 1
        
    est_max_images = num_atoms * (2 * max_bounds[0] + 1) * (2 * max_bounds[1] + 1) * (2 * max_bounds[2] + 1)
    
    out_frac = np.zeros((est_max_images, 3), dtype=np.float64)
    out_cart = np.zeros((est_max_images, 3), dtype=np.float64)
    out_types = np.zeros(est_max_images, dtype=np.int32)
    out_base_indices = np.zeros(est_max_images, dtype=np.int32)
    
    count = 0
    for i in range(num_atoms):
        frac_base = base_frac_coords[i]
        atype = atom_types[i]
        
        for dx in range(-max_bounds[0], max_bounds[0] + 1):
            for dy in range(-max_bounds[1], max_bounds[1] + 1):
                for dz in range(-max_bounds[2], max_bounds[2] + 1):
                    
                    shift = np.array([float(dx), float(dy), float(dz)], dtype=np.float64)
                    img_frac = frac_base + shift
                    img_cart = img_frac @ lattice_matrix
                    
                    closest_frac = np.zeros(3, dtype=np.float64)
                    for d in range(3):
                        if img_frac[d] < 0.0:
                            closest_frac[d] = 0.0
                        elif img_frac[d] > 1.0:
                            closest_frac[d] = 1.0
                        else:
                            closest_frac[d] = img_frac[d]
                            
                    closest_cart = closest_frac @ lattice_matrix
                    distance_to_cell = np.linalg.norm(img_cart - closest_cart)
                    
                    if distance_to_cell < r_cut:
                        out_frac[count] = img_frac
                        out_cart[count] = img_cart
                        out_types[count] = atype
                        out_base_indices[count] = i
                        count += 1
                        
    return out_frac[:count], out_cart[:count], out_types[:count], out_base_indices[:count]

# @njit(cache=True, fastmath=True)
def find_voxels_in_atom_range(atom_frac, lattice_matrix, grid_dims, r_cut):
    nx, ny, nz = grid_dims[0], grid_dims[1], grid_dims[2]
    
    fx, fy, fz = atom_frac[0], atom_frac[1], atom_frac[2]
    
    # get fractional cutoff along each lattice vector
    f_ext_x = r_cut / np.linalg.norm(lattice_matrix[0])
    f_ext_y = r_cut / np.linalg.norm(lattice_matrix[1])
    f_ext_z = r_cut / np.linalg.norm(lattice_matrix[2])
    
    # Get max bounds, allowing points outside the lattice
    imin = int(np.floor((fx - f_ext_x) * nx))
    imax = int(np.ceil((fx + f_ext_x) * nx))
    
    jmin = int(np.floor((fy - f_ext_y) * ny))
    jmax = int(np.ceil((fy + f_ext_y) * ny))
    
    kmin = int(np.floor((fz - f_ext_z) * nz))
    kmax = int(np.ceil((fz + f_ext_z) * nz))
    
    # get the maximum number of voxels in range and create placeholder arrays
    max_possible_voxels = (imax - imin) * (jmax - jmin) * (kmax - kmin)
    if max_possible_voxels <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
        
    out_indices = np.zeros(max_possible_voxels, dtype=np.int64)
    out_distances = np.zeros(max_possible_voxels, dtype=np.float64)
    out_vecs = np.zeros((max_possible_voxels, 3), dtype=np.float64)
    
    # loop over all possible coords and calculate their distance
    count = 0
    for i in range(imin, imax):
        df_x = (float(i) / nx) - fx
        df_x -= np.round(df_x)  # wrap around cell
        i_wrapped = i % nx
        
        for j in range(jmin, jmax):
            df_y = (float(j) / ny) - fy
            df_y -= np.round(df_y)
            j_wrapped = j % ny
            
            for k in range(kmin, kmax):
                df_z = (float(k) / nz) - fz
                df_z -= np.round(df_z)
                k_wrapped = k % nz
                
                # Convert relative fractional distance to cartesian
                dx = df_x * lattice_matrix[0, 0] + df_y * lattice_matrix[1, 0] + df_z * lattice_matrix[2, 0]
                dy = df_x * lattice_matrix[0, 1] + df_y * lattice_matrix[1, 1] + df_z * lattice_matrix[2, 1]
                dz = df_x * lattice_matrix[0, 2] + df_y * lattice_matrix[1, 2] + df_z * lattice_matrix[2, 2]
                
                r = np.sqrt(dx*dx + dy*dy + dz*dz)
                
                if r < r_cut:
                    # Flat index in primary unit cell space [0, nx) x [0, ny) x [0, nz)
                    flat_idx = i_wrapped * (ny * nz) + j_wrapped * nz + k_wrapped
                    out_indices[count] = flat_idx
                    out_distances[count] = r
                    out_vecs[count] = (dx, dy, dz)
                    count += 1
                    
    return out_indices[:count], out_distances[:count], out_vecs[count]

# @njit(parallel=True, fastmath=True, cache=True)
def find_all_voxels_parallel(
        atom_fracs, 
        lattice_matrix, 
        grid_dims, 
        r_cuts,
        ):
    """Executes the voxel range filter in parallel across an array of atomic coordinates."""
    num_atoms = atom_fracs.shape[0]
    all_indices = []
    all_distances = []
    all_vecs = []
    
    # append placeholder arrays for numba typing
    for _ in range(num_atoms):
        all_indices.append(np.empty(0, dtype=np.int64))
        all_distances.append(np.empty(0, dtype=np.float64))
        all_vecs.append(np.empty((0,0), dtype=np.float64))
        
    # loop over atoms in parallel and calculate voxels in range
    for i in prange(num_atoms):
        indices, distances, vecs = find_voxels_in_atom_range(
            atom_fracs[i], lattice_matrix, grid_dims, r_cuts[i]
        )
        all_indices[i] = indices
        all_distances[i] = distances
        all_vecs[i] = vecs
        
    return all_indices, all_distances, all_vecs

@njit(parallel=True, fastmath=True)
def accumulate_augmentation_core(
    psi, grad_psi, lap_psi, idx, overlaps, q_vecs,
    radial_diff, radial_diff_deriv, radial_laplacian,
    y_lm, grad_y_lm, has_lap
):
    """
    Thread-safe, memory-allocation-free multi-threaded JIT contraction kernel 
    for local atomic PAW augmentations.
    """
    n_bands = psi.shape[0]
    n_proj = overlaps.shape[1]
    n_masked = len(idx)
    
    # Parallelize over bands to ensure thread-isolated memory writes
    for i_band in prange(n_bands):
        for proj_idx in range(n_proj):
            olap = overlaps[i_band, proj_idx]
            if olap == 0.0 + 0.0j:
                continue
                
            for i_p in range(n_masked):
                g_idx = idx[i_p]
                
                # Cache scalar properties to stay close to CPU registers
                r_diff = radial_diff[proj_idx, i_p]
                ylm = y_lm[proj_idx, i_p]
                r_diff_deriv = radial_diff_deriv[proj_idx, i_p]
                
                # 1. Phi_aug accumulation
                psi[i_band, g_idx] += olap * r_diff * ylm
                
                # 2. Gradient phi_aug components accumulation
                g_ylm_x = grad_y_lm[proj_idx, i_p, 0]
                g_ylm_y = grad_y_lm[proj_idx, i_p, 1]
                g_ylm_z = grad_y_lm[proj_idx, i_p, 2]
                
                t3c_x = (r_diff_deriv * ylm) * q_vecs[i_p, 0] + r_diff * g_ylm_x
                t3c_y = (r_diff_deriv * ylm) * q_vecs[i_p, 1] + r_diff * g_ylm_y
                t3c_z = (r_diff_deriv * ylm) * q_vecs[i_p, 2] + r_diff * g_ylm_z
                
                grad_psi[i_band, g_idx, 0] += olap * t3c_x
                grad_psi[i_band, g_idx, 1] += olap * t3c_y
                grad_psi[i_band, g_idx, 2] += olap * t3c_z
                
                # 3. Laplacian phi_aug accumulation
                if has_lap:
                    lap_psi[i_band, g_idx] += olap * radial_laplacian[proj_idx, i_p] * ylm

###############################################################################
# Spherical Harmonics
###############################################################################

def evaluate_real_harmonics_multi(l: int, m: int, q_vecs: NDArray) -> NDArray:
    """
    Computes standard orthonormal real spherical harmonics Y_lm in Cartesian coordinates.
    Coordinates are normalized on-the-fly with a division-by-zero safeguard at q=0.
    """
    norms = np.linalg.norm(q_vecs, axis=1)
    norms_safe = np.where(norms < 1e-12, 1.0, norms)
    u = q_vecs / norms_safe[:, np.newaxis]
    x, y, z = u[:, 0], u[:, 1], u[:, 2]
    
    if l == 0:
        return np.full_like(norms, 0.5 * np.sqrt(1.0 / np.pi)), norms
    elif l == 1:
        if m == -1:   return np.sqrt(3.0 / (4.0 * np.pi)) * y, norms      # p_y
        elif m == 0:  return np.sqrt(3.0 / (4.0 * np.pi)) * z, norms      # p_z
        elif m == 1:  return np.sqrt(3.0 / (4.0 * np.pi)) * x, norms      # p_x
    elif l == 2:
        if m == -2:   return 0.5 * np.sqrt(15.0 / np.pi) * x * y, norms   # d_xy
        elif m == -1: return 0.5 * np.sqrt(15.0 / np.pi) * y * z, norms   # d_yz
        elif m == 0:  return 0.25 * np.sqrt(5.0 / np.pi) * (3.0 * z**2 - 1.0), norms # d_z2
        elif m == 1:  return 0.5 * np.sqrt(15.0 / np.pi) * x * z, norms   # d_xz
        elif m == 2:  return 0.25 * np.sqrt(15.0 / np.pi) * (x**2 - y**2), norms # d_x2-y2
    elif l == 3:
        # f-orbitals
        if m == -3:   return 0.25 * np.sqrt(35.0 / (2.0 * np.pi)) * y * (3.0 * x**2 - y**2), norms
        elif m == -2: return 0.5 * np.sqrt(105.0 / np.pi) * x * y * z, norms
        elif m == -1: return 0.25 * np.sqrt(21.0 / (2.0 * np.pi)) * y * (5.0 * z**2 - 1.0), norms
        elif m == 0:  return 0.25 * np.sqrt(7.0 / np.pi) * z * (5.0 * z**2 - 3.0), norms
        elif m == 1:  return 0.25 * np.sqrt(21.0 / (2.0 * np.pi)) * x * (5.0 * z**2 - 1.0), norms
        elif m == 2:  return 0.25 * np.sqrt(105.0 / (2.0 * np.pi)) * z * (x**2 - y**2), norms
        elif m == 3:  return 0.25 * np.sqrt(35.0 / (2.0 * np.pi)) * x * (x**2 - 3.0 * y**2), norms
    elif l == 4:
        # g-orbitals
        if m == -4:   return 0.75 * np.sqrt(35.0 / np.pi) * x * y * (x**2 - y**2), norms
        elif m == -3: return 0.75 * np.sqrt(35.0 / (2.0 * np.pi)) * y * z * (3.0 * x**2 - y**2), norms
        elif m == -2: return 0.75 * np.sqrt(5.0 / np.pi) * x * y * (7.0 * z**2 - 1.0), norms
        elif m == -1: return 0.75 * np.sqrt(5.0 / (2.0 * np.pi)) * y * z * (7.0 * z**2 - 3.0), norms
        elif m == 0:  return 0.1875 * np.sqrt(1.0 / np.pi) * (35.0 * z**4 - 30.0 * z**2 + 3.0), norms
        elif m == 1:  return 0.75 * np.sqrt(5.0 / (2.0 * np.pi)) * x * z * (7.0 * z**2 - 3.0), norms
        elif m == 2:  return 0.375 * np.sqrt(5.0 / np.pi) * (x**2 - y**2) * (7.0 * z**2 - 1.0), norms
        elif m == 3:  return 0.75 * np.sqrt(35.0 / (2.0 * np.pi)) * x * z * (x**2 - 3.0 * y**2), norms
        elif m == 4:  return 0.1875 * np.sqrt(35.0 / np.pi) * (x**4 - 6.0 * x**2 * y**2 + y**4), norms
    return np.zeros_like(norms)

def evaluate_real_harmonics_grad_multi(
    l: int, m: int, q_vecs: NDArray, dists_safe: NDArray
) -> NDArray:
    """
    Computes the spatial gradient (dY/dx, dY/dy, dY/dz) of real spherical harmonics 
    with respect to unnormalized Cartesian coordinates.
    
    Args:
        l: Orbital angular momentum quantum number.
        m: Magnetic quantum number.
        q_vecs: Normalized unit vectors directional array of shape (num_coords, 3).
        dists_safe: Distance array with division-by-zero safeguard of shape (num_coords,).
        
    Returns:
        NDArray of shape (3, num_coords) containing the [x, y, z] gradient components.
    """
    x, y, z = q_vecs[:, 0], q_vecs[:, 1], q_vecs[:, 2]
    
    # Initialize polynomial derivatives with respect to the unit vector components
    da_dx = np.zeros_like(x)
    da_dy = np.zeros_like(x)
    da_dz = np.zeros_like(x)

    if l == 0:
        # Constant function -> gradient is strictly zero
        pass

    elif l == 1:
        c = np.sqrt(3.0 / (4.0 * np.pi))
        if m == -1:    # p_y
            da_dy = np.full_like(x, c)
        elif m == 0:   # p_z
            da_dz = np.full_like(x, c)
        elif m == 1:   # p_x
            da_dx = np.full_like(x, c)

    elif l == 2:
        c2 = 0.5 * np.sqrt(15.0 / np.pi)
        c2_0 = 0.25 * np.sqrt(5.0 / np.pi)
        c2_2 = 0.25 * np.sqrt(15.0 / np.pi)
        
        if m == -2:    # d_xy
            da_dx = c2 * y
            da_dy = c2 * x
        elif m == -1:  # d_yz
            da_dy = c2 * z
            da_dz = c2 * y
        elif m == 0:   # d_z2
            da_dz = c2_0 * 6.0 * z
        elif m == 1:   # d_xz
            da_dx = c2 * z
            da_dz = c2 * x
        elif m == 2:   # d_x2-y2
            da_dx = c2_2 * 2.0 * x
            da_dy = -c2_2 * 2.0 * y

    elif l == 3:
        c3_3 = 0.25 * np.sqrt(35.0 / (2.0 * np.pi))
        c3_2 = 0.5 * np.sqrt(105.0 / np.pi)
        c3_1 = 0.25 * np.sqrt(21.0 / (2.0 * np.pi))
        c3_0 = 0.25 * np.sqrt(7.0 / np.pi)
        c3_2a = 0.25 * np.sqrt(105.0 / (2.0 * np.pi))
        
        if m == -3:    # 3x^2*y - y^3
            da_dx = c3_3 * 6.0 * x * y
            da_dy = c3_3 * (3.0 * x**2 - 3.0 * y**2)
        elif m == -2:  # x*y*z
            da_dx = c3_2 * y * z
            da_dy = c3_2 * x * z
            da_dz = c3_2 * x * y
        elif m == -1:  # y*(5z^2 - 1)
            da_dy = c3_1 * (5.0 * z**2 - 1.0)
            da_dz = c3_1 * 10.0 * y * z
        elif m == 0:   # 5z^3 - 3z
            da_dz = c3_0 * (15.0 * z**2 - 3.0)
        elif m == 1:   # x*(5z^2 - 1)
            da_dx = c3_1 * (5.0 * z**2 - 1.0)
            da_dz = c3_1 * 10.0 * x * z
        elif m == 2:   # z*(x^2 - y^2)
            da_dx = c3_2a * 2.0 * x * z
            da_dy = -c3_2a * 2.0 * y * z
            da_dz = c3_2a * (x**2 - y**2)
        elif m == 3:   # x^3 - 3x*y^2
            da_dx = c3_3 * (3.0 * x**2 - 3.0 * y**2)
            da_dy = -c3_3 * 6.0 * x * y

    elif l == 4:
        c4_4 = 0.75 * np.sqrt(35.0 / np.pi)
        c4_3 = 0.75 * np.sqrt(35.0 / (2.0 * np.pi))
        c4_2 = 0.75 * np.sqrt(5.0 / np.pi)
        c4_1 = 0.75 * np.sqrt(5.0 / (2.0 * np.pi))
        c4_0 = 0.1875 * np.sqrt(1.0 / np.pi)
        c4_2a = 0.375 * np.sqrt(5.0 / np.pi)
        c4_4a = 0.1875 * np.sqrt(35.0 / np.pi)
        
        if m == -4:    # x^3*y - x*y^3
            da_dx = c4_4 * (3.0 * x**2 * y - y**3)
            da_dy = c4_4 * (x**3 - 3.0 * x * y**2)
        elif m == -3:  # 3x^2*y*z - y^3*z
            da_dx = c4_3 * 6.0 * x * y * z
            da_dy = c4_3 * z * (3.0 * x**2 - 3.0 * y**2)
            da_dz = c4_3 * y * (3.0 * x**2 - y**2)
        elif m == -2:  # x*y*(7z^2 - 1)
            da_dx = c4_2 * y * (7.0 * z**2 - 1.0)
            da_dy = c4_2 * x * (7.0 * z**2 - 1.0)
            da_dz = c4_2 * 14.0 * x * y * z
        elif m == -1:  # 7y*z^3 - 3y*z
            da_dy = c4_1 * z * (7.0 * z**2 - 3.0)
            da_dz = c4_1 * y * (21.0 * z**2 - 3.0)
        elif m == 0:   # 35z^4 - 30z^2 + 3
            da_dz = c4_0 * (140.0 * z**3 - 60.0 * z)
        elif m == 1:   # 7x*z^3 - 3x*z
            da_dx = c4_1 * z * (7.0 * z**2 - 3.0)
            da_dz = c4_1 * x * (21.0 * z**2 - 3.0)
        elif m == 2:   # (x^2 - y^2)*(7z^2 - 1)
            da_dx = c4_2a * 2.0 * x * (7.0 * z**2 - 1.0)
            da_dy = -c4_2a * 2.0 * y * (7.0 * z**2 - 1.0)
            da_dz = c4_2a * 14.0 * z * (x**2 - y**2)
        elif m == 3:   # x^3*z - 3x*y^2*z
            da_dx = c4_3 * z * (3.0 * x**2 - 3.0 * y**2)
            da_dy = -c4_3 * 6.0 * x * y * z
            da_dz = c4_3 * x * (x**2 - 3.0 * y**2)
        elif m == 4:   # x^4 - 6x^2*y^2 + y^4
            da_dx = c4_4a * (4.0 * x**3 - 12.0 * x * y**2)
            da_dy = c4_4a * (-12.0 * x**2 * y + 4.0 * y**3)
    else:
        return np.zeros((3, len(x)), dtype=np.float64)

    # Compute directional derivative projection along the unit vector path
    g_dot_u = da_dx * x + da_dy * y + da_dz * z

    # Project onto tangent space and divide by safe real distances
    grad_x = (da_dx - g_dot_u * x) / dists_safe
    grad_y = (da_dy - g_dot_u * y) / dists_safe
    grad_z = (da_dz - g_dot_u * z) / dists_safe

    # Reshape and pack cleanly to match (3, num_coords) expected structure
    return np.stack([grad_x, grad_y, grad_z], axis=0)

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
    Parallelized JIT-compiled linear tetrahedron engine adhering strictly to 
    Blöchl's linear property-weight interpolation scheme.
    
    UPDATED: Synchronized boundary intervals and flat-band delta-spike 
    deposition to guarantee absolute mathematical consistency with the analytic charge engine.
    """
    n_omega = egrid.shape[0]
    n_tetra = tetra_indices.shape[0]
    n_spin = eigenvalues.shape[0]
    n_bands = eigenvalues.shape[2]
    n_metrics = cached_metrics.shape[3]

    out = np.zeros((n_metrics, n_spin, n_omega), dtype=np.float64)
    
    # Calculate energy grid spacing for delta function normalization
    delta_e = egrid[1] - egrid[0] if n_omega > 1 else 1.0

    for s in prange(n_spin):
        for b in range(n_bands):
            local = np.zeros((n_metrics, n_omega), dtype=np.float64)

            for t in range(n_tetra):
                k1 = tetra_indices[t, 0]
                k2 = tetra_indices[t, 1]
                k3 = tetra_indices[t, 2]
                k4 = tetra_indices[t, 3]

                e1 = eigenvalues[s, k1, b]
                e2 = eigenvalues[s, k2, b]
                e3 = eigenvalues[s, k3, b]
                e4 = eigenvalues[s, k4, b]

                idx = np.array([k1, k2, k3, k4])
                
                # Stable bubble sort for vertex alignment
                for i in range(3):
                    for j in range(3 - i):
                        if eigenvalues[s, idx[j], b] > eigenvalues[s, idx[j+1], b]:
                            tmp = idx[j]
                            idx[j] = idx[j+1]
                            idx[j+1] = tmp

                e1 = eigenvalues[s, idx[0], b]
                e2 = eigenvalues[s, idx[1], b]
                e3 = eigenvalues[s, idx[2], b]
                e4 = eigenvalues[s, idx[3], b]

                idx1, idx2, idx3, idx4 = idx[0], idx[1], idx[2], idx[3]

                # FIXED: Handle flat bands as exact Dirac Delta Spikes 
                # instead of discarding them.
                if e4 - e1 < 1e-7:
                    # Find where the flat band energy lands on our discrete grid
                    w_closest = int(np.round((e1 - egrid[0]) / delta_e))
                    if 0 <= w_closest < n_omega:
                        for m in range(n_metrics):
                            val1 = cached_metrics[s, idx1, b, m]
                            # Deposit property weight normalized by dE
                            local[m, w_closest] += (val1 * tetra_weight) / delta_e
                    continue

                e21 = e2 - e1
                e31 = e3 - e1
                e41 = e4 - e1
                e42 = e4 - e2
                e32 = e3 - e2 if (e3 - e2) > 1e-12 else 1.0
                e43 = e4 - e3 if (e4 - e3) > 1e-12 else 1.0

                for w in range(n_omega):
                    E = egrid[w]

                    # Left-Closed, Right-Open boundary tracking matching the charge engine
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
                        local[m, w] += integrated_val

            for m in range(n_metrics):
                for w in range(n_omega):
                    out[m, s, w] += local[m, w]

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