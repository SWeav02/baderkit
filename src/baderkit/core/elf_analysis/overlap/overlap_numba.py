# -*- coding: utf-8 -*-

from numba import njit, prange

import numpy as np
from numpy.typing import NDArray

@njit(cache=True)
def get_overlap_counts(
    atom_labels: NDArray[np.int64],
    local_labels: NDArray[np.int64],
    num_charge: int,
    num_local: int,
        ):
    nx, ny, nz = local_labels.shape
    
    # create array to track counts
    overlap_counts = np.zeros((num_charge, num_local), dtype=np.int64)
    
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
                
                # add to our count
                overlap_counts[atom_label, local_label] += 1

    return overlap_counts

@njit(parallel=True, cache=True)
def get_overlaps(
    atom_labels: NDArray[np.int64],
    local_labels: NDArray[np.int64],
    num_charge: int,
    num_local: int,
):
    nx, ny, nz = local_labels.shape

    # get overlap count matrix. We do this in a separate function so numba doesn't
    # try and parallelize
    overlap_counts = get_overlap_counts(
        atom_labels,
        local_labels,
        num_charge,
        num_local,
        )

    # create arrays to track the fraction of each type of basin taken up by the
    # other
    overlap_indices = np.argwhere(overlap_counts)
    counts = np.empty(len(overlap_indices), dtype=np.int64)
    
    # reduce to just the important counts
    for idx in prange(len(overlap_indices)):
        i, j = overlap_indices[idx]
        # get the counts
        counts[idx] = overlap_counts[i,j]
        
        # overwrite overlap counts with this pairs index
        overlap_counts[i,j] = idx
        
    # get overlap fractions for each basin/atom
    scratch = np.empty((0,0),dtype=np.float64)
    atom_frac = []
    for i in range(num_charge):
        atom_frac.append(scratch.copy())
    for idx in prange(num_charge):
        # get local neighbors
        indices = np.where(overlap_indices[:,0] == idx)[0]
        neighs = overlap_indices[indices, 1].astype(np.float64)
        count = counts[indices]
        # get portion
        fracs = count / count.sum()
        atom_frac[idx] = np.column_stack((neighs, fracs))
        
    local_frac = []
    for i in range(num_local):
        local_frac.append(scratch.copy())
    for idx in prange(num_local):
        # get local neighbors
        indices = np.where(overlap_indices[:,1] == idx)[0]
        neighs = overlap_indices[indices, 0].astype(np.float64)
        count = counts[indices]
        # get portion
        fracs = count / count.sum()
        local_frac[idx] = np.column_stack((neighs, fracs))
        
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

    return np.column_stack((overlap_indices, counts)), atom_frac, local_frac, overlap_labels


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
