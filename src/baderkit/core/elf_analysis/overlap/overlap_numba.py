# -*- coding: utf-8 -*-

from numba import njit, prange

import numpy as np
from numpy.typing import NDArray

from baderkit.core.utilities.transforms import INT_TO_IMAGE, IMAGE_TO_INT

@njit(cache=True, parallel=True)
def get_unique_basins_w_images(
    atom_labels,
    atom_images,
    local_labels,
    local_images,
    num_charge,
    num_local,
    local_frac,
    charge_frac,
        ):
    nx, ny, nz = atom_labels.shape
    labels_w_images = np.zeros((num_charge, 27), dtype=np.bool_)
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                atom_label = atom_labels[i,j,k]
                local_label = local_labels[i,j,k]
                # skip vacuum
                if atom_label == num_charge or local_label == num_local:
                    continue
                
                # get the shift required to move the charge maximum into the
                # same image as the local maximum
                shift = np.round(local_frac[local_label] - charge_frac[atom_label]).astype(np.int64)

                # get charge image relative to local basin
                atom_image = INT_TO_IMAGE[atom_images[i,j,k]]
                local_image = INT_TO_IMAGE[local_images[i,j,k]] + shift
                mi, mj, mk = local_image - atom_image

                image = IMAGE_TO_INT[mi, mj, mk]
                
                labels_w_images[atom_label, image] = True
    # construct label map
    pairs = np.argwhere(labels_w_images)
    label_map = np.empty_like(labels_w_images, dtype=np.int16)
    for idx in prange(len(pairs)):
        i,j = pairs[idx]
        label_map[i,j] = idx
    return pairs, label_map
    
                
@njit(cache=True)
def get_overlap_counts(
    atom_labels: NDArray[np.int64],
    atom_images: NDArray[np.int64],
    local_labels: NDArray[np.int64],
    local_images: NDArray[np.int64],
    charge_data: NDArray[np.float64],
    local_frac: NDArray[np.float64],
    charge_frac: NDArray[np.float64],
    num_charge: int,
    num_local: int,
        ):
    nx, ny, nz = local_labels.shape
    
    # get the total unique labels/images
    label_image_pairs, label_image_map = get_unique_basins_w_images(
        atom_labels=atom_labels,
        atom_images=atom_images,
        local_labels=local_labels,
        local_images=local_images,
        num_charge=num_charge,
        num_local=num_local,
        local_frac=local_frac,
        charge_frac=charge_frac,
        )

    # create array to track total populations
    overlap_counts = np.zeros((len(label_image_pairs), num_local), dtype=np.float64)

    # What we need:
    # Overlap labels (distinct types of overlap between charge/local)
    # Atoms overlapped with each local basin
    # Counts for atoms overlapped with each local basin

    # loop over each voxel and count the number of overlaps
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # get the labels at this point
                atom_label = atom_labels[i, j, k]
                local_label = local_labels[i, j, k]
                
                # skip points in vacuum
                if atom_label == num_charge or local_label == num_local:
                    continue
                
                # get the shift required to move the charge maximum into the
                # same image as the local maximum
                shift = np.round(local_frac[local_label] - charge_frac[atom_label]).astype(np.int64)

                # get charge image relative to local basin
                atom_image = INT_TO_IMAGE[atom_images[i,j,k]]
                local_image = INT_TO_IMAGE[local_images[i,j,k]] + shift
                mi, mj, mk = local_image - atom_image

                image = IMAGE_TO_INT[mi, mj, mk]
                
                # add to our count
                atom_pair = label_image_map[atom_label, image]
                overlap_counts[atom_pair, local_label] += charge_data[i,j,k]
                
    return overlap_counts, label_image_pairs, label_image_map

@njit(parallel=True, cache=True)
def get_overlaps(
    atom_labels: NDArray[np.int64],
    atom_images: NDArray[np.int64],
    local_labels: NDArray[np.int64],
    local_images: NDArray[np.int64],
    charge_data: NDArray[np.float64],
    local_frac: NDArray[np.float64],
    charge_frac: NDArray[np.float64],
    num_charge: int,
    num_local: int,
    tol = 0.001,
):
    nx, ny, nz = local_labels.shape

    # get overlap count matrix. We do this in a separate function so numba doesn't
    # try and parallelize
    overlap_counts, label_image_pairs, label_image_map = get_overlap_counts(
        atom_labels,
        atom_images,
        local_labels,
        local_images,
        charge_data,
        local_frac,
        charge_frac,
        num_charge,
        num_local,
        )

    # create arrays to track the fraction of each type of basin taken up by the
    # other
    overlap_indices = np.argwhere(overlap_counts>0)
    counts = np.empty(len(overlap_indices), dtype=np.int64)
    
    # reduce to important and get total atom overlap charges (remove images)
    atom_counts = np.empty((len(overlap_indices), 4), dtype=np.float64)
    for idx in prange(len(overlap_indices)):
        pair_idx, local_idx = overlap_indices[idx]
        # get the counts
        count = overlap_counts[pair_idx, local_idx]
        counts[idx] = count
        # overwrite overlap counts with this pairs index
        overlap_counts[pair_idx, local_idx] = idx
        # save total count
        atom_idx, atom_image = label_image_pairs[pair_idx]
        atom_counts[idx,0] = float(atom_idx)
        atom_counts[idx,1] = float(atom_image)
        atom_counts[idx,2] = float(local_idx)
        atom_counts[idx,3] = float(count)
    overlap_counts = overlap_counts.astype(np.int64)
        
    # get overlap fractions for each basin/atom
    scratch = np.empty((0,0),dtype=np.float64)       
    local_frac = []
    for i in range(num_local):
        local_frac.append(scratch.copy())
    for idx in prange(num_local):
        # get local neighbors
        indices = np.where(overlap_indices[:,1] == idx)[0]
        
        # get charge basin label and image
        neighs = overlap_indices[indices, 0]
        charge_basin_image = label_image_pairs[neighs].astype(np.float64)
        
        count = counts[indices]        
        # get portion
        fracs = count / count.sum()
        # remove fracs below cutoff
        high_fracs = np.where(fracs > tol)[0]
        fracs = fracs[high_fracs]
        charge_basin_image = charge_basin_image[high_fracs]
        # sort from high to low
        sorted_indices = np.flip(np.argsort(fracs))
        local_frac[idx] = np.column_stack((charge_basin_image[sorted_indices], fracs[sorted_indices]))

    # create an array to store the overlap labels.
    overlap_labels = np.full((nx, ny, nz), len(counts), dtype=np.int64)
    
    # loop over the array again and track labels
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # get the labels at this point
                atom_label = atom_labels[i, j, k]
                local_label = local_labels[i, j, k]
                # skip points in vacuum
                if atom_label == num_charge or local_label == num_local:
                    continue
                
                union_label = overlap_counts[atom_label, local_label]

                # update the label
                overlap_labels[i,j,k] = union_label

    return atom_counts, local_frac, overlap_labels

@njit(parallel=True, cache=True)
def get_atom_access_sets(
        overlap_charge_table, 
        local_basin_charges,
        core_basins,
        num_atoms, 
        num_local,
        tol = 0.001,
        ):
    access_sets = []
    charge_claims = []
    for i in range(num_atoms):
        access_sets.append(np.empty((0),dtype=np.int64))
        charge_claims.append(np.empty((0,0), dtype=np.float64))
    for i in prange(num_atoms):
        set_mask = np.zeros(num_local, dtype=np.bool_)
        # first we find the access set for this atom
        for atom_idx, _, local_idx, charge in overlap_charge_table:
            local_idx = int(local_idx)
            # if this is a different atom, a core basin, or already part of
            # our set, we skip
            if atom_idx != i or core_basins[local_idx]!=-1 or set_mask[local_idx]:
                continue
            set_mask[local_idx] = True
        local_set = np.where(set_mask)[0]
        access_sets[i] = local_set
        
        # now found what portion of the accessible charge is held by each atom
        # in the system.
        atom_claims = np.zeros(num_atoms, dtype=np.float64)
        for atom_idx, _, local_idx, charge in overlap_charge_table:
            if set_mask[int(local_idx)]:
                atom_claims[int(atom_idx)] += charge
        
        # get all atoms with claims
        atoms = np.where(atom_claims>0)[0]
        atom_claims = atom_claims[atoms]
        
        # normalize to get fractional claims
        atom_claims /= atom_claims.sum()
        
        # remove atoms with very small claims
        large_set = np.where(atom_claims > tol)[0]
        atoms = atoms[large_set]
        atom_claims = atom_claims[large_set]
        charge_claims[i] = np.column_stack((atoms.astype(np.float64), atom_claims))
        
    return access_sets, charge_claims

# TODO:
    # add images into this as well
    # calculate connection index
    # calculate nearest neighbor sharing
    # test with total charge densities

# @njit(cache=True)
# def szudzik_pair(a: int, b: int):
#     if a >= b:
#         return a * a + a + b
#     elif a < b:
#         return b * b + a


# @njit(cache=True)
# def szudzik_reverse(z):
#     k = (z ** (1 / 2)) // 1
#     kk = k * k

#     if z - kk < k:
#         a = z - kk
#         b = k
#     else:
#         a = k
#         b = z - kk - k
#     return int(a), int(b)
