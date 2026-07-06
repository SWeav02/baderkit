# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, prange
from numba.typed import List
from baderkit.post_wfc.pseudopotentials.augmentation_numba import interp1d_cubic_spline_numba, precompute_cubic_spline_derivs

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
                        out_cart[count] = img_cart
                        out_types[count] = atype
                        out_base_indices[count] = i
                        count += 1
                        
    return out_cart[:count], out_types[:count], out_base_indices[:count]

@njit(cache=True, fastmath=True)
def find_voxels_in_atom_range(atom_cart, lattice_matrix, grid_dims, r_cut):
    """Determines which voxels fall within a cutoff radius of an atom position."""
    nx, ny, nz = grid_dims[0], grid_dims[1], grid_dims[2]
    inv_lattice = np.linalg.inv(lattice_matrix)
    
    f_atom = atom_cart @ inv_lattice
    
    f_ext_x = r_cut * np.linalg.norm(inv_lattice[:, 0])
    f_ext_y = r_cut * np.linalg.norm(inv_lattice[:, 1])
    f_ext_z = r_cut * np.linalg.norm(inv_lattice[:, 2])
    
    imin = max(0, int(np.floor((f_atom[0] - f_ext_x) * nx)))
    imax = min(nx, int(np.ceil((f_atom[0] + f_ext_x) * nx)))
    
    jmin = max(0, int(np.floor((f_atom[1] - f_ext_y) * ny)))
    jmax = min(ny, int(np.ceil((f_atom[1] + f_ext_y) * ny)))
    
    kmin = max(0, int(np.floor((f_atom[2] - f_ext_z) * nz)))
    kmax = min(nz, int(np.ceil((f_atom[2] + f_ext_z) * nz)))
    
    max_possible_voxels = (imax - imin) * (jmax - jmin) * (kmax - kmin)
    if max_possible_voxels <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
        
    out_indices = np.zeros(max_possible_voxels, dtype=np.int64)
    out_distances = np.zeros(max_possible_voxels, dtype=np.float64)
    
    count = 0
    for i in range(imin, imax):
        f_x = float(i) / nx
        for j in range(jmin, jmax):
            f_y = float(j) / ny
            for k in range(kmin, kmax):
                f_z = float(k) / nz
                
                v_cart_x = f_x * lattice_matrix[0, 0] + f_y * lattice_matrix[1, 0] + f_z * lattice_matrix[2, 0]
                v_cart_y = f_x * lattice_matrix[0, 1] + f_y * lattice_matrix[1, 1] + f_z * lattice_matrix[2, 1]
                v_cart_z = f_x * lattice_matrix[0, 2] + f_y * lattice_matrix[1, 2] + f_z * lattice_matrix[2, 2]
                
                dx = v_cart_x - atom_cart[0]
                dy = v_cart_y - atom_cart[1]
                dz = v_cart_z - atom_cart[2]
                
                r = np.sqrt(dx*dx + dy*dy + dz*dz)
                
                if r < r_cut:
                    flat_idx = i * (ny * nz) + j * nz + k
                    out_indices[count] = flat_idx
                    out_distances[count] = r
                    count += 1
                    
    return out_indices[:count], out_distances[:count]

@njit(parallel=True, fastmath=True, cache=True)
def find_all_voxels_parallel(atom_carts, lattice_matrix, grid_dims, r_cut):
    """Executes the voxel range filter in parallel across an array of atomic coordinates."""
    num_atoms = atom_carts.shape[0]
    all_indices = List()
    all_distances = List()
    
    for _ in range(num_atoms):
        all_indices.append(np.empty(0, dtype=np.int64))
        all_distances.append(np.empty(0, dtype=np.float64))
        
    for i in prange(num_atoms):
        indices, distances = find_voxels_in_atom_range(
            atom_carts[i], lattice_matrix, grid_dims, r_cut
        )
        all_indices[i] = indices
        all_distances[i] = distances
        
    return all_indices, all_distances

@njit(fastmath=True, parallel=True, cache=True)
def interpolate_total_rho_multi(rho_vector, r_grid, distances, paw_r1):
    """
    Interpolates a single pre-computed 1D radial density profile for an array of distances.
    Mimics VASP/PAW core singularity flattening for distances closer than paw_r1.
    """
    n_distances = distances.shape[0]
    results = np.zeros(n_distances, dtype=np.float64)
    
    # Precompute spline derivatives
    y_derivs = precompute_cubic_spline_derivs(r_grid, rho_vector)
    
    for i in prange(n_distances):
        # Force the exact origin to evaluate at the flattened PAW grid radius shell
        r_eval = distances[i]
        if r_eval < paw_r1:
            r_eval = paw_r1
            
        results[i] = interp1d_cubic_spline_numba(r_grid, rho_vector, y_derivs, r_eval)
        
    return results

@njit(fastmath=True, cache=True)
def broadcast_atoms_to_grid(
    grid_dims,
    atom_types,
    all_indices,
    all_distances,
    rho_matrices_list,  # Numba List of 1D arrays
    r_grids_list,
    paw_r1,
):
    """Accumulates the net charge density onto a unified 3D volumetric spatial grid."""
    nx, ny, nz = grid_dims[0], grid_dims[1], grid_dims[2]
    total_voxels = nx * ny * nz
    
    grid_flat = np.zeros(total_voxels, dtype=np.float64)
    num_atoms = atom_types.shape[0]
    
    for i in range(num_atoms):
        voxel_indices = all_indices[i]
        distances = all_distances[i]
        
        if voxel_indices.shape[0] == 0:
            continue
            
        atype = atom_types[i]
        rho_vector = rho_matrices_list[atype]
        r_grid = r_grids_list[atype]
        
        # Uses the updated high-fidelity natural cubic spline mapping step
        interpolated_rhos = interpolate_total_rho_multi(
            rho_vector, r_grid, distances, paw_r1
        )
        
        for j in range(voxel_indices.shape[0]):
            flat_idx = voxel_indices[j]
            grid_flat[flat_idx] += interpolated_rhos[j]
            
    return grid_flat.reshape((nx, ny, nz))