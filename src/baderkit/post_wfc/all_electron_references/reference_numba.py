# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, prange
from numba.typed import List

@njit(fastmath=True, cache=True)
def find_active_periodic_atoms(lattice_matrix, base_frac_coords, atom_types, r_cut):
    """
    Finds all periodic images of the atoms whose cutoff spheres overlap with the
    primary unit cell volume [0, 1]^3.
    
    Parameters:
    -----------
    lattice_matrix : np.ndarray (2D, float64)
        3x3 matrix where rows represent the lattice vectors.
    base_frac_coords : np.ndarray (2D, float64)
        Asymmetric unit fractional coordinates (num_atoms x 3).
    atom_types : np.ndarray (1D, int32)
        Integer tracking the element classification/index for each base atom.
    r_cut : float
        The real-space interaction cutoff radius in Angstroms.
    """
    num_atoms = base_frac_coords.shape[0]
    inv_lattice = np.linalg.inv(lattice_matrix)
    
    # Compute conservative image boundaries based on the projection of r_cut onto fractional axes
    max_bounds = np.zeros(3, dtype=np.int32)
    for i in range(3):
        max_bounds[i] = int(np.ceil(r_cut * np.linalg.norm(inv_lattice[:, i]))) + 1
        
    # Pre-allocate large maximum estimation arrays to maintain strict Numba array alignment
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
                    
                    # Project image position to Cartesian space
                    img_cart = img_frac @ lattice_matrix
                    
                    # Find the absolute closest point inside the [0, 1]^3 cell volume to this image
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
                    
                    # If the cutoff sphere touches the cell, record the image
                    if distance_to_cell < r_cut:
                        out_cart[count] = img_cart
                        out_types[count] = atype
                        out_base_indices[count] = i
                        count += 1
                        
    return out_cart[:count], out_types[:count], out_base_indices[:count]

@njit(cache=True, fastmath=True)
def find_voxels_in_atom_range(atom_cart, lattice_matrix, grid_dims, r_cut):
    """
    Determines which voxels in a primary unit cell [0, 1)^3 grid fall within 
    a cutoff radius of an atom position. Follows VASP C-index flattening conventions.

    Parameters:
    -----------
    atom_cart : np.ndarray (1D, float64)
        Cartesian coordinates of the target atom image [x, y, z].
    lattice_matrix : np.ndarray (2D, float64)
        3x3 matrix where rows represent the lattice vectors (pymatgen convention).
    grid_dims : np.ndarray or tuple (1D, int32/int64)
        Dimensions of the 3D grid [nx, ny, nz].
    r_cut : float
        The cutoff search radius in Angstroms.

    Returns:
    --------
    tuple (np.ndarray, np.ndarray)
        - 1D int64 array: Flattened indices of matching voxels.
        - 1D float64 array: Corresponding true Cartesian distances.
    """
    nx, ny, nz = grid_dims[0], grid_dims[1], grid_dims[2]
    inv_lattice = np.linalg.inv(lattice_matrix)
    
    # 1. Map the Cartesian atom position into fractional space
    f_atom = atom_cart @ inv_lattice
    
    # 2. Project the Cartesian sphere radius onto individual fractional axes
    # The gradient magnitude of fractional component d corresponds to column d of inv_lattice
    f_ext_x = r_cut * np.linalg.norm(inv_lattice[:, 0])
    f_ext_y = r_cut * np.linalg.norm(inv_lattice[:, 1])
    f_ext_z = r_cut * np.linalg.norm(inv_lattice[:, 2])
    
    # 3. Determine grid bounding index ranges, bounded strictly within the [0, 1) cell
    imin = max(0, int(np.floor((f_atom[0] - f_ext_x) * nx)))
    imax = min(nx, int(np.ceil((f_atom[0] + f_ext_x) * nx)))
    
    jmin = max(0, int(np.floor((f_atom[1] - f_ext_y) * ny)))
    jmax = min(ny, int(np.ceil((f_atom[1] + f_ext_y) * ny)))
    
    kmin = max(0, int(np.floor((f_atom[2] - f_ext_z) * nz)))
    kmax = min(nz, int(np.ceil((f_atom[2] + f_ext_z) * nz)))
    
    # 4. Pre-allocate flat static arrays based on max possible window size 
    # to avoid resizing allocations inside the loop
    max_possible_voxels = (imax - imin) * (jmax - jmin) * (kmax - kmin)
    if max_possible_voxels <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
        
    out_indices = np.zeros(max_possible_voxels, dtype=np.int64)
    out_distances = np.zeros(max_possible_voxels, dtype=np.float64)
    
    # 5. Evaluate coordinates within the sub-box
    count = 0
    for i in range(imin, imax):
        f_x = float(i) / nx
        for j in range(jmin, jmax):
            f_y = float(j) / ny
            for k in range(kmin, kmax):
                f_z = float(k) / nz
                
                # Manual unrolling of the vector-matrix product (f_voxel @ lattice_matrix)
                # This guarantees zero array allocations inside the inner loop registers
                v_cart_x = f_x * lattice_matrix[0, 0] + f_y * lattice_matrix[1, 0] + f_z * lattice_matrix[2, 0]
                v_cart_y = f_x * lattice_matrix[0, 1] + f_y * lattice_matrix[1, 1] + f_z * lattice_matrix[2, 1]
                v_cart_z = f_x * lattice_matrix[0, 2] + f_y * lattice_matrix[1, 2] + f_z * lattice_matrix[2, 2]
                
                dx = v_cart_x - atom_cart[0]
                dy = v_cart_y - atom_cart[1]
                dz = v_cart_z - atom_cart[2]
                
                r = np.sqrt(dx*dx + dy*dy + dz*dz)
                
                # 6. Accumulate points that cross the strict distance boundary
                if r < r_cut:
                    # Flat indexing using VASP/NumPy row-major C standard
                    flat_idx = i * (ny * nz) + j * nz + k
                    out_indices[count] = flat_idx
                    out_distances[count] = r
                    count += 1
                    
    return out_indices[:count], out_distances[:count]

@njit(parallel=True, fastmath=True, cache=True)
def find_all_voxels_parallel(atom_carts, lattice_matrix, grid_dims, r_cut):
    """
    Executes the voxel range filter in parallel across an array of atomic coordinates.
    Handles irregular/jagged output structures safely without thread contention.

    Parameters:
    -----------
    atom_carts : np.ndarray (2D, float64)
        Cartesian coordinates of all target atom images, shape (num_atoms x 3).
    lattice_matrix : np.ndarray (2D, float64)
        3x3 matrix where rows represent the lattice vectors.
    grid_dims : np.ndarray (1D, int64)
        Dimensions of the 3D grid [nx, ny, nz].
    r_cut : float
        The cutoff search radius in Angstroms.

    Returns:
    --------
    tuple (numba.typed.List, numba.typed.List)
        - A typed List containing 1D int64 arrays of flattened voxel indices for each atom.
        - A typed List containing 1D float64 arrays of true Cartesian distances for each atom.
    """
    num_atoms = atom_carts.shape[0]
    
    # 1. INITIALIZATION & TYPECASTING
    # We instantiate Numba typed lists and pre-populate them sequentially.
    # This explicitly declares the internal type structure to the compiler
    # and sizes the list container to avoid dynamic resizing bottlenecks.
    all_indices = List()
    all_distances = List()
    
    for _ in range(num_atoms):
        all_indices.append(np.empty(0, dtype=np.int64))
        all_distances.append(np.empty(0, dtype=np.float64))
        
    # 2. PARALLEL WORK DISTRIBUTION
    # prange splits the total atom pool across your CPU thread pool.
    # Mutating a list element via explicit index placement (list[i] = val) 
    # is completely thread-safe here because thread domains never overlap.
    for i in prange(num_atoms):
        indices, distances = find_voxels_in_atom_range(
            atom_carts[i], lattice_matrix, grid_dims, r_cut
        )
        all_indices[i] = indices
        all_distances[i] = distances
        
    return all_indices, all_distances


@njit(fastmath=True, cache=True)
def log_linear_interp(x, xp, fp):
    """
    Custom 1D interpolator designed for logarithmically/geometrically spaced grids.
    Interpolates linearly with respect to ln(r) to prevent overshooting in sparse 
    regions and enforces strict physical non-negative bounds.
    """
    # Explicit out-of-bounds guarding (also prevents log(0) for x close to nucleus)
    if x <= xp[0]:
        return max(0.0, fp[0])
    if x >= xp[-1]:
        return max(0.0, fp[-1])
    
    # Custom binary search loop to locate bounding indices inside Numba
    low = 0
    high = xp.shape[0] - 1
    while low <= high:
        mid = (low + high) // 2
        if xp[mid] < x:
            low = mid + 1
        else:
            high = mid - 1
            
    idx = low - 1
    
    x0 = xp[idx]
    x1 = xp[idx + 1]
    y0 = fp[idx]
    y1 = fp[idx + 1]
    
    # Map coordinates to log-space to accurately match the geometric grid scaling
    log_x0 = np.log(x0)
    log_x1 = np.log(x1)
    log_x = np.log(x)
    
    # Perform interpolation in log-space
    weight = (log_x - log_x0) / (log_x1 - log_x0)
    val = y0 + weight * (y1 - y0)
    
    # Enforce strict non-negative physical bounds
    if val < 0.0:
        return 0.0
    return val


@njit(fastmath=True, parallel=True, cache=True)
def interpolate_total_rho_multi(rho_vector, r_grid, distances):
    """
    Interpolates a single pre-computed 1D radial density profile 
    for an array of distances, parallelized across CPU threads.
    """
    n_distances = distances.shape[0]
    results = np.zeros(n_distances, dtype=np.float64)
    
    for i in prange(n_distances):
        results[i] = log_linear_interp(distances[i], r_grid, rho_vector)
        
    return results


@njit(fastmath=True, cache=True)
def broadcast_atoms_to_grid(
    grid_dims,
    atom_types,
    all_indices,
    all_distances,
    rho_matrices_list,  # Numba List of 1D arrays
    r_grids_list
):
    """
    Accumulates the net charge density of the selected range chunk
    onto a unified 3D volumetric spatial grid.
    """
    nx, ny, nz = grid_dims[0], grid_dims[1], grid_dims[2]
    total_voxels = nx * ny * nz
    
    # Initialize a single flat 1D accumulation array for total density
    grid_flat = np.zeros(total_voxels, dtype=np.float64)
    num_atoms = atom_types.shape[0]
    
    # Outer loop runs sequentially to eliminate race conditions during grid writes
    for i in range(num_atoms):
        voxel_indices = all_indices[i]
        distances = all_distances[i]
        
        if voxel_indices.shape[0] == 0:
            continue
            
        atype = atom_types[i]
        
        rho_vector = rho_matrices_list[atype]
        r_grid = r_grids_list[atype]
        
        # Interpolate the 1D net chunk density array using the updated log-linear scheme
        interpolated_rhos = interpolate_total_rho_multi(
            rho_vector, r_grid, distances
        )
        # Accumulate directly into the single total density grid map
        for j in range(voxel_indices.shape[0]):
            flat_idx = voxel_indices[j]
            grid_flat[flat_idx] += interpolated_rhos[j]
            
    return grid_flat.reshape((nx, ny, nz))
