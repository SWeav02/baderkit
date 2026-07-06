# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, prange
from numba.typed import List

@njit(cache=True, fastmath=True)
def precompute_cubic_spline_derivs(x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    """Precomputes second derivatives for a natural cubic spline on a non-uniform grid."""
    n = len(x_grid)
    u = np.zeros(n)
    y_derivs = np.zeros(n)  # Stores the second derivatives (y'')
    
    # Forward pass of Thomas algorithm for tridiagonal system
    for i in range(1, n - 1):
        sig = (x_grid[i] - x_grid[i-1]) / (x_grid[i+1] - x_grid[i-1])
        p = sig * y_derivs[i-1] + 2.0
        y_derivs[i] = (sig - 1.0) / p
        
        # Calculate right-hand side vector
        g = 6.0 * ((y_grid[i+1] - y_grid[i]) / (x_grid[i+1] - x_grid[i]) - 
                   (y_grid[i] - y_grid[i-1]) / (x_grid[i] - x_grid[i-1])) / (x_grid[i+1] - x_grid[i-1])
        u[i] = (g - sig * u[i-1]) / p

    # Backward substitution pass
    for k in range(n - 2, 0, -1):
        y_derivs[k] = y_derivs[k] * y_derivs[k+1] + u[k]
        
    return y_derivs

@njit(cache=True, fastmath=True)
def interp1d_cubic_spline_numba(x_grid: np.ndarray, y_grid: np.ndarray, y_derivs: np.ndarray, x_val: float) -> float:
    """Fast cubic spline evaluation for non-uniform radial grids using precomputed derivatives."""
    # Boundary conditions mimicking your PAW core setup
    if x_val <= x_grid[0]:
        return y_grid[0]
    if x_val >= x_grid[-1]:
        return 0.0  # Drops to exactly zero outside the PAW core boundary
        
    # Binary search bounds lookup (O(log N))
    low = 0
    high = len(x_grid) - 1
    while high - low > 1:
        mid = (low + high) // 2
        if x_grid[mid] > x_val:
            high = mid
        else:
            low = mid
            
    x0, x1 = x_grid[low], x_grid[high]
    y0, y1 = y_grid[low], y_grid[high]
    h = x1 - x0
    
    # Cubic spline interpolation formula
    a = (x1 - x_val) / h
    b = (x_val - x0) / h
    
    y_val = (a * y0 + b * y1 + 
             ((a**3 - a) * y_derivs[low] + (b**3 - b) * y_derivs[high]) * (h**2) / 6.0)
    
    return y_val


@njit(cache=True)
def eval_real_harmonics(l, m, x, y, z):
    """
    Evaluates real spherical harmonics Y_lm directly from a normalized 
    Cartesian direction vector (x, y, z). Matches VASP's internal layout.
    """
    pi = np.pi
    if l == 0:
        return np.sqrt(1.0 / (4.0 * pi))
    elif l == 1:
        const = np.sqrt(3.0 / (4.0 * pi))
        if m == -1: return const * y
        if m == 0:  return const * z
        if m == 1:  return const * x
    elif l == 2:
        if m == -2: return np.sqrt(15.0 / (4.0 * pi)) * x * y
        if m == -1: return np.sqrt(15.0 / (4.0 * pi)) * y * z
        if m == 0:  return np.sqrt(5.0 / (16.0 * pi)) * (3.0 * z * z - 1.0)
        if m == 1:  return np.sqrt(15.0 / (4.0 * pi)) * x * z
        if m == 2:  return np.sqrt(15.0 / (16.0 * pi)) * (x * x - y * y)
    elif l == 3:
        if m == -3: return np.sqrt(35.0 / (32.0 * pi)) * y * (3.0 * x * x - y * y)
        if m == -2: return np.sqrt(105.0 / (4.0 * pi)) * x * y * z
        if m == -1: return np.sqrt(21.0 / (32.0 * pi)) * y * (5.0 * z * z - 1.0)
        if m == 0:  return np.sqrt(7.0 / (16.0 * pi)) * z * (5.0 * z * z - 3.0)
        if m == 1:  return np.sqrt(21.0 / (32.0 * pi)) * x * (5.0 * z * z - 1.0)
        if m == 2:  return np.sqrt(105.0 / (16.0 * pi)) * z * (x * x - y * y)
        if m == 3:  return np.sqrt(35.0 / (32.0 * pi)) * x * (x * x - 3.0 * y * y)
    return 0.0

@njit(parallel=True, cache=True, fastmath=True)
def compute_reciprocal_projectors(
    k_cart, 
    G_basis_cart, 
    atom_cart, 
    max_q, 
    ndata, 
    volume, 
    raw_projectors, 
    projector_l, 
    projector_m
):
    num_G = G_basis_cart.shape[0]
    num_projectors = projector_l.shape[0]
    
    q_mag = np.zeros(num_G, dtype=np.float64)
    nx = np.zeros(num_G, dtype=np.float64)
    ny = np.zeros(num_G, dtype=np.float64)
    nz = np.zeros(num_G, dtype=np.float64)
    
    fakt = 1.0 / np.sqrt(volume)
    
    for g in range(num_G):
        qx = k_cart[0] + G_basis_cart[g, 0]
        qy = k_cart[1] + G_basis_cart[g, 1]
        qz = k_cart[2] + G_basis_cart[g, 2]
        
        mag = np.sqrt(qx*qx + qy*qy + qz*qz)
        q_mag[g] = mag
        if mag > 1e-12:
            nx[g], ny[g], nz[g] = qx / mag, qy / mag, qz / mag

    P_G_matrix = np.zeros((num_projectors, num_G), dtype=np.complex128)
    
    # Replicate VASP's explicit grid scaling multiplier: ARGSC = NPSNL / PSMAXN
    argsc = ndata / max_q
    
    for p_idx in prange(num_projectors):
        l = projector_l[p_idx]
        m = projector_m[p_idx]
        prefactor = (1j) ** l
        y_arr = raw_projectors[p_idx]
        
        for g in range(num_G):
            mag = q_mag[g]
            
            # Compute coordinate matching VASP's exact runtime scaling step
            arg = (mag * argsc) + 1.0
            naddr = int(arg)
            
            if naddr >= (ndata - 2) or naddr < 1:
                radial_val = 0.0
            else:
                rem = arg % 1.0
                
                if naddr == 1:
                    v1 = -y_arr[1] if (l % 2 == 1) else y_arr[1]
                else:
                    v1 = y_arr[naddr - 2]
                    
                v2 = y_arr[naddr - 1]
                v3 = y_arr[naddr]
                v4 = y_arr[naddr + 1]
                
                t0 = v2
                t1 = ((6.0 * v3) - (2.0 * v1) - (3.0 * v2) - v4) / 6.0
                t2 = (v1 + v3 - (2.0 * v2)) / 2.0
                t3 = (v4 - v1 + (3.0 * (v2 - v3))) / 6.0
                
                radial_val = t0 + rem * (t1 + rem * (t2 + rem * t3))
                
            angular_part = eval_real_harmonics(l, m, nx[g], ny[g], nz[g])
            
            g_dot_R = G_basis_cart[g, 0] * atom_cart[0] + G_basis_cart[g, 1] * atom_cart[1] + G_basis_cart[g, 2] * atom_cart[2]
            phase = np.cos(g_dot_R) + 1j * np.sin(g_dot_R)
            
            P_G_matrix[p_idx, g] = radial_val * fakt * angular_part * phase * prefactor
            
    return P_G_matrix

@njit(cache=True)
def enforce_matrix_symmetrization(density_matrix, projector_l, projector_m):
    n = density_matrix.shape[0]
    sym_matrix = np.copy(density_matrix)
    for i in range(n):
        for j in range(n):
            if projector_l[i] != projector_l[j] or projector_m[i] != projector_m[j]:
                sym_matrix[i, j] = 0.0 + 0j
    return sym_matrix


@njit(cache=True, fastmath=True)
def reconstruct_onsite_densities_at_point(
    point_cart: np.ndarray, 
    lattice_matrix: np.ndarray, 
    inv_lattice_matrix: np.ndarray, 
    frac_coords: np.ndarray, 
    site_element_indices: np.ndarray, 
    ae_p_list: List, ps_p_list: List,
    ae_derivs_list: List, 
    ps_derivs_list: List,
    r_list: List, 
    l_list: List, 
    m_list: List, 
    occ_list: List, 
    max_cutoffs: np.ndarray
) -> tuple[float, float]:
    """Calculates localized AE and PS onsite densities at a single coordinate point using cubic splines."""
    
    # Map target point to fractional coordinates
    pf_x = inv_lattice_matrix[0,0]*point_cart[0] + inv_lattice_matrix[0,1]*point_cart[1] + inv_lattice_matrix[0,2]*point_cart[2]
    pf_y = inv_lattice_matrix[1,0]*point_cart[0] + inv_lattice_matrix[1,1]*point_cart[1] + inv_lattice_matrix[1,2]*point_cart[2]
    pf_z = inv_lattice_matrix[2,0]*point_cart[0] + inv_lattice_matrix[2,1]*point_cart[1] + inv_lattice_matrix[2,2]*point_cart[2]

    total_ae = 0.0
    total_ps = 0.0
    num_sites = len(frac_coords)

    for site_idx in range(num_sites):
        elem_idx = site_element_indices[site_idx]
        cutoff = max_cutoffs[elem_idx]
        
        df_x = pf_x - frac_coords[site_idx, 0]
        df_y = pf_y - frac_coords[site_idx, 1]
        df_z = pf_z - frac_coords[site_idx, 2]
        
        df_x -= round(df_x)
        df_y -= round(df_y)
        df_z -= round(df_z)
        
        # Project back to Cartesian displacement relative to nucleus
        dx = lattice_matrix[0,0]*df_x + lattice_matrix[0,1]*df_y + lattice_matrix[0,2]*df_z
        dy = lattice_matrix[1,0]*df_x + lattice_matrix[1,1]*df_y + lattice_matrix[1,2]*df_z
        dz = lattice_matrix[2,0]*df_x + lattice_matrix[2,1]*df_y + lattice_matrix[2,2]*df_z
        
        r = np.sqrt(dx*dx + dy*dy + dz*dz)
        if r >= cutoff:  # REMOVED: or r == 0.0
            continue
            
        # FIX: Handle exact nuclear coordinate grid alignment gracefully
        if r < 1e-12:
            nx_site, ny_site, nz_site = 0.0, 0.0, 1.0
            r = 0.0  # Clean snap to the spline origin index
        else:
            nx_site = dx / r
            ny_site = dy / r
            nz_site = dz / r
            
        ae_pairs = ae_p_list[elem_idx]
        ps_pairs = ps_p_list[elem_idx]
        ae_derivs = ae_derivs_list[elem_idx]
        ps_derivs = ps_derivs_list[elem_idx]
        
        r_grid = r_list[elem_idx]
        l_arr = l_list[elem_idx]
        m_arr = m_list[elem_idx]
        flat_occ = occ_list[site_idx]
        num_channels = len(l_arr)
        
        # Cache angular harmonics evaluations for the local site
        Y_arr = np.zeros(num_channels)
        for i in range(num_channels):
            Y_arr[i] = eval_real_harmonics(l_arr[i], m_arr[i], nx_site, ny_site, nz_site)
            
        # Bilinear cross-product contraction loop
        site_ae = 0.0
        site_ps = 0.0
        for i in range(num_channels):
            Y_i = Y_arr[i]
            for j in range(num_channels):
                rho_ij = flat_occ[i * num_channels + j]
                if rho_ij == 0.0:
                    continue
                
                pair_idx = i * num_channels + j
                
                # Perform cubic spline lookups using cached second derivatives
                p_diff_r = interp1d_cubic_spline_numba(r_grid, ae_pairs[pair_idx], ae_derivs[pair_idx], r)
                p_ps_r = interp1d_cubic_spline_numba(r_grid, ps_pairs[pair_idx], ps_derivs[pair_idx], r)
                
                # Reconstruct the genuine un-subtracted AE contribution layer
                p_ae_r = p_diff_r + p_ps_r
                
                term_base = rho_ij * Y_i * Y_arr[j]
                site_ae += term_base * p_ae_r
                site_ps += term_base * p_ps_r
                
        total_ae += site_ae
        total_ps += site_ps
        
    return total_ae, total_ps


@njit(cache=True, fastmath=True)
def reconstruct_onsite_densities(
    grid_dims: np.ndarray, 
    lattice_matrix: np.ndarray, 
    frac_coords: np.ndarray, 
    site_element_indices: np.ndarray, 
    ae_p_list: List, 
    ps_p_list: List,
    ae_derivs_list: List, 
    ps_derivs_list: List,  # Added derivative lists
    r_list: List, 
    l_list: List, 
    m_list: List, 
    occ_list: List, 
    max_cutoffs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Broadcasts localized onsite all-electron and pseudo fields to a 3D grid layout using cubic splines."""
    Nx, Ny, Nz = grid_dims[0], grid_dims[1], grid_dims[2]
    ae_grid = np.zeros((Nx, Ny, Nz), dtype=np.float64)
    ps_grid = np.zeros((Nx, Ny, Nz), dtype=np.float64)
    
    for site_idx in range(len(frac_coords)):
        elem_idx = site_element_indices[site_idx]
        cutoff = max_cutoffs[elem_idx]
        ae_pairs = ae_p_list[elem_idx]
        ps_pairs = ps_p_list[elem_idx]
        ae_derivs = ae_derivs_list[elem_idx]  # Extract element derivatives
        ps_derivs = ps_derivs_list[elem_idx]
        
        r_grid = r_list[elem_idx]
        l_arr = l_list[elem_idx]
        m_arr = m_list[elem_idx]
        flat_occ = occ_list[site_idx]
        num_channels = len(l_arr)
        
        f_atom = frac_coords[site_idx]
        
        for ix in range(Nx):
            fx = ix / Nx
            for iy in range(Ny):
                fy = iy / Ny
                for iz in range(Nz):
                    fz = iz / Nz
                    
                    df_x = fx - f_atom[0]
                    df_y = fy - f_atom[1]
                    df_z = fz - f_atom[2]
                    
                    df_x -= round(df_x)
                    df_y -= round(df_y)
                    df_z -= round(df_z)
                    
                    dx = lattice_matrix[0,0]*df_x + lattice_matrix[0,1]*df_y + lattice_matrix[0,2]*df_z
                    dy = lattice_matrix[1,0]*df_x + lattice_matrix[1,1]*df_y + lattice_matrix[1,2]*df_z
                    dz = lattice_matrix[2,0]*df_x + lattice_matrix[2,1]*df_y + lattice_matrix[2,2]*df_z
                    
                    r = np.sqrt(dx*dx + dy*dy + dz*dz)
                    if r >= cutoff:  # REMOVED: or r == 0.0
                        continue
                        
                    # FIX: Handle exact nuclear coordinate grid alignment gracefully
                    if r < 1e-12:
                        nx_site, ny_site, nz_site = 0.0, 0.0, 1.0
                        r = 0.0  # Clean snap to the spline origin index
                    else:
                        nx_site = dx / r
                        ny_site = dy / r
                        nz_site = dz / r
                        
                    # Cache angular harmonics evaluations for the local site
                    Y_arr = np.zeros(num_channels)
                    for i in range(num_channels):
                        Y_arr[i] = eval_real_harmonics(l_arr[i], m_arr[i], nx_site, ny_site, nz_site)
                        
                    site_ae = 0.0
                    site_ps = 0.0
                    for i in range(num_channels):
                        Y_i = Y_arr[i]
                        for j in range(num_channels):
                            rho_ij = flat_occ[i * num_channels + j]
                            if rho_ij == 0.0:
                                continue
                            
                            pair_idx = i * num_channels + j
                            
                            # Perform cubic spline lookups using cached second derivatives
                            p_diff_r = interp1d_cubic_spline_numba(r_grid, ae_pairs[pair_idx], ae_derivs[pair_idx], r)
                            p_ps_r = interp1d_cubic_spline_numba(r_grid, ps_pairs[pair_idx], ps_derivs[pair_idx], r)
                            p_ae_r = p_diff_r + p_ps_r
                            
                            term_base = rho_ij * Y_i * Y_arr[j]
                            site_ae += term_base * p_ae_r
                            site_ps += term_base * p_ps_r
                            
                    ae_grid[ix, iy, iz] += site_ae
                    ps_grid[ix, iy, iz] += site_ps
                    
    return ae_grid, ps_grid

###############################################################################
# RECONSTRUCTION PLANE-WAVE COUPLING
###############################################################################

@njit(parallel=True, cache=True, fastmath=True)
def build_g_space_projectors(
    q_linear_grid: np.ndarray,
    reciprocal_projectors: np.ndarray,
    reciprocal_projectors_derivs: np.ndarray,
    angular_momenta: np.ndarray,
    magnetic_nums: np.ndarray,
    k_cart: np.ndarray,
    g_vectors_cart: np.ndarray,
    atom_cart_pos: np.ndarray,
    cell_volume: float
) -> np.ndarray:
    """Parallel kernel executing discrete projection scaling in G+k representation."""
    num_channels = len(angular_momenta)
    num_pw = g_vectors_cart.shape[0]
    
    projectors_g = np.zeros((num_channels, num_pw), dtype=np.complex128)
    norm_factor = 1.0 / np.sqrt(cell_volume)
    
    # Extract the absolute reciprocal grid limit to guard against spline blowups
    q_max = q_linear_grid[-1]
    
    for p in prange(num_pw):
        qx = g_vectors_cart[p, 0] + k_cart[0]
        qy = g_vectors_cart[p, 1] + k_cart[1]
        qz = g_vectors_cart[p, 2] + k_cart[2]
        
        q_mag = np.sqrt(qx*qx + qy*qy + qz*qz)
        
        ux, uy, uz = 0.0, 0.0, 0.0
        if q_mag > 1e-12:
            ux, uy, uz = qx / q_mag, qy / q_mag, qz / q_mag
            
        q_dot_R = qx * atom_cart_pos[0] + qy * atom_cart_pos[1] + qz * atom_cart_pos[2]
        phase = (np.cos(q_dot_R) + 1j * np.sin(q_dot_R)) * norm_factor
        
        for c in range(num_channels):
            l = angular_momenta[c]
            m = magnetic_nums[c]
            
            # --- CRITICAL BOUNDARY GUARD ---
            # If the plane wave momentum exceeds the POTCAR dataset range, 
            # it lives outside the core projector's support domain.
            
            if q_mag > q_max:
                p_q = 0.0
            else:
                p_q = interp1d_cubic_spline_numba(
                    q_linear_grid, 
                    reciprocal_projectors[c], 
                    reciprocal_projectors_derivs[c], 
                    q_mag
                )
            
            y_lm = eval_real_harmonics(l, m, ux, uy, uz)
            
            l_mod = l % 4
            if l_mod == 0:    il_factor = 1.0 + 0.0j
            elif l_mod == 1:  il_factor = 0.0 + 1.0j
            elif l_mod == 2:  il_factor = -1.0 + 0.0j
            else:              il_factor = 0.0 - 1.0j
                
            projectors_g[c, p] = il_factor * phase * p_q * y_lm
            
    return projectors_g
