# -*- coding: utf-8 -*-

from numba import njit, prange
import numpy as np
from numpy.typing import NDArray

@njit(cache=True)
def find_site_in_tol(
    chunked_coords,
    site_coords,
    tol,
        ):
    chunked = np.round(site_coords / tol).astype(np.int64)
    # Match wrapped coord to site
    index = -1
    for j, chunked_coord in enumerate(chunked_coords):
        
        d = chunked - chunked_coord
        if np.sum(d) == 0:
            index = j
            break

    return index

@njit(cache=True)
def get_canonical_displacement(bond_displacement, tol):
    # wrap into cell
    bond_displacement -= np.round(bond_displacement)
    
    # quantize to tolerance
    v = np.round(bond_displacement / tol).astype(np.int64)

    # choose lexicographically positive representative
    if (
        v[0] < 0 or
        (v[0] == 0 and v[1] < 0) or
        (v[0] == 0 and v[1] == 0 and v[2] < 0)
    ):
        v[0] = -v[0]
        v[1] = -v[1]
        v[2] = -v[2]

    return v


@njit(cache=True)
def get_canonical_bond(
        site_idx,
        neigh_idx,
        neigh_coords,
        all_frac_coords, 
        equivalent_atoms,
        rotation_matrices,
        translation_vectors,
        included_ops,
        pair_dist,
        tol,
        ):
    
    # get frac coords for the site
    site_coords = all_frac_coords[site_idx]
    
    # create a placeholder for the best canonical bond
    best_rep = np.full(7, np.iinfo(np.int64).max, dtype=np.int64)
    
    if equivalent_atoms[site_idx] <= equivalent_atoms[neigh_idx]:
        best_rep[0] = 0 # bond is not inverted
        best_rep[1] = equivalent_atoms[site_idx]
        best_rep[2] = equivalent_atoms[neigh_idx]
    else:
        best_rep[0] = 1 # bond is inverted
        best_rep[2] = equivalent_atoms[site_idx]
        best_rep[1] = equivalent_atoms[neigh_idx]
    
    # get quantized pair distance to retain information on bond length
    best_rep[6] = round(pair_dist/tol)

    # iterate over valid symmetry operations
    for trans_idx, valid in enumerate(included_ops):
        # skip operations that don't transform to the lowest index equivalent atom
        # (checked ahead of time for speed)
        if not valid:
            continue
        
        # get transformed site and neighbor
        matrix = rotation_matrices[trans_idx]
        vector = translation_vectors[trans_idx]
        
        trans_site_coords = matrix @ site_coords + vector
        trans_neigh_coords = matrix @ neigh_coords + vector
        
        # get the displacement vector
        displacement = trans_neigh_coords - trans_site_coords
        
        # canonize
        displacement = get_canonical_displacement(displacement, tol)
        
        # check if this is a better representation than the current one
        if displacement[0] > best_rep[3]:
            continue
        elif displacement[1] > best_rep[4]:
            continue
        elif displacement[2] > best_rep[5]:
            continue
        
        # if we're still here, this is equal or better than the best rep
        best_rep[3:6] = displacement

    # Choose unique canonical representative
    return best_rep


@njit(cache=True)
def perform_symmop(coords, rotation_matrix: NDArray, translation_vector: NDArray):
    return rotation_matrix @ coords + translation_vector

        
@njit(parallel=True, cache=True)
def get_canonical_bonds(
        site_indices,
        neigh_indices,
        neigh_coords, 
        equivalent_atoms,
        all_frac_coords,
        rotation_matrices,
        translation_vectors,
        pair_dists,
        tol=0.02,
        ):
    
    chunked_coords = np.round(all_frac_coords / tol).astype(np.int64)
    
    # first, we narrow down the symmetry operations by including only those that
    # map to the lowest index equivalent atom
    unique_sites = np.unique(site_indices)
    unique_map = np.empty(unique_sites[-1]+1, dtype=np.uint16)
    unique_map[unique_sites] = np.arange(len(unique_sites))
    
    symm_op_mask = np.zeros((len(unique_sites), len(translation_vectors)), dtype=np.bool_)
    
    for unique_idx in prange(len(unique_sites)):
        site_idx = unique_sites[unique_idx]
        equiv_idx = equivalent_atoms[site_idx]
        site_coords = all_frac_coords[site_idx]
        for op_idx, (matrix, vector) in enumerate(zip(rotation_matrices, translation_vectors)):
            trans_site_coords = matrix @ site_coords + vector
            # wrap into cell
            trans_site_coords %= 1
            # get index after operation
            new_idx = find_site_in_tol(chunked_coords, trans_site_coords, tol)

            # if this is the lowest equivalent atom, we will include this operation
            if new_idx == equiv_idx:
                symm_op_mask[unique_idx, op_idx] = True

    
    # create an array to store canonical bonds
    canonical_bonds = np.empty((len(site_indices), 7), dtype=np.int64)
    
    # each row is in order of:
        # 1. whether or not the canonical rep is the reverse of the original bond
        # 2. The lower site index in the bond
        # 3. The higher site index in the bond
        # 4-6. integer representations of the lowest displacement vector
    
    for bond_idx in prange(len(site_indices)):
        site_idx = site_indices[bond_idx]
        neigh_idx = neigh_indices[bond_idx]
        neigh_coord = neigh_coords[bond_idx]
        pair_dist = pair_dists[bond_idx]
        included_ops = symm_op_mask[unique_map[site_idx]]
        
        canonical_bonds[bond_idx] = get_canonical_bond(
                site_idx=site_idx,
                neigh_idx=neigh_idx,
                neigh_coords=neigh_coord,
                all_frac_coords=all_frac_coords, 
                equivalent_atoms=equivalent_atoms,
                rotation_matrices=rotation_matrices,
                translation_vectors=translation_vectors,
                included_ops=included_ops,
                pair_dist=pair_dist,
                tol=tol,
                )
        
    return canonical_bonds

@njit(parallel=True, cache=True)
def generate_symmetric_bonds(
        site_indices,
        neigh_indices,
        neigh_coords,
        bond_types,
        all_frac_coords,
        fracs,
        rotation_matrices,
        translation_vectors,
        # sym_index_map,
        # sym_image_map,
        # n_transforms,
        shape,
        frac2cart,
        tol,
        ):

    # create an array to store all bond information
    # site, neighbor, plane equation
    n_transforms = len(translation_vectors)
    all_bonds = np.empty((n_transforms*len(site_indices), 13), dtype=np.float64)
    
    chunked_coords = np.round(all_frac_coords/tol).astype(np.int64)
    
    for pair_idx in prange(len(site_indices)):
        site_idx = site_indices[pair_idx]
        neigh_idx = neigh_indices[pair_idx]
        site_coord = all_frac_coords[site_idx]
        neigh_coord = neigh_coords[pair_idx]
        bond_type = bond_types[pair_idx]

        frac = fracs[pair_idx]
        
        # apply each transformation
        bond_idx = pair_idx * n_transforms
        for trans_idx, (matrix, vector) in enumerate(zip(rotation_matrices,translation_vectors)):
            trans_site_coord = matrix @ site_coord + vector
            trans_neigh_coord = matrix @ neigh_coord + vector
            
            # get image of new site
            site_image = np.floor(trans_site_coord + tol).astype(np.int64)
            # move site and neighbor
            trans_site_coord = trans_site_coord - site_image
            trans_neigh_coord = trans_neigh_coord - site_image
            
            # get neigh image
            neigh_image = np.floor(trans_neigh_coord + tol).astype(np.int64)
            # wrap neigh
            trans_neigh_coord = trans_neigh_coord - neigh_image
            
            # get the indices of each site/neigh
            site_idx = find_site_in_tol(
                chunked_coords=chunked_coords,
                site_coords=trans_site_coord,
                tol=tol,
                )
            neigh_idx = find_site_in_tol(
                chunked_coords=chunked_coords,
                site_coords=trans_neigh_coord,
                tol=tol,
                )
            
            # get the exact site/neigh coords
            trans_site_coord = all_frac_coords[site_idx]
            trans_neigh_coord = all_frac_coords[neigh_idx] + neigh_image
            
            # calculate the exact plane vector and point for this neighbor. This
            # can be slightly different if atoms aren't at exact positions
            plane_vector = trans_neigh_coord - trans_site_coord
            
            # get fractional vector
            plane_vector *= frac
            # calculate radius
            cart_vector = plane_vector @ frac2cart
            radius = np.linalg.norm(cart_vector)
            
            # get plane equation
            plane_point = plane_vector + trans_site_coord
            plane_vector = plane_vector / np.linalg.norm(plane_vector)
            # set plane equation
            all_bonds[bond_idx,0] = site_idx
            all_bonds[bond_idx,1] = neigh_idx
            all_bonds[bond_idx,2] = radius
            all_bonds[bond_idx,3] = bond_type
            all_bonds[bond_idx, 4:7] = plane_point
            all_bonds[bond_idx, 7:10] = plane_vector
            all_bonds[bond_idx, 10:] = trans_neigh_coord
            bond_idx += 1

    return np.round(all_bonds, 12)
