# -*- coding: utf-8 -*-

from numba import njit, prange

import numpy as np
from numpy.typing import NDArray


@njit
def get_overlap_counts(
    local_labels: NDArray[np.int64],
    atom_labels: NDArray[np.int64],
    vacuum_mask: NDArray[np.bool_],
    num_local: int,
    num_charge: int,
        ):
    
    """
    
    """
    nx, ny, nz = local_labels.shape
    
    # create an array to store the overlap labels
    overlap_array = np.zeros((nx, ny, nz), dtype=np.int64)
    
    # create a 2D array to track total overlap
    overlap_counts = np.zeros((num_local, num_charge), dtype=np.uint32)

    # What we need:
        # Overlap labels (distinct types of overlap between charge/local)
        # Atoms overlapped with each local basin
        # Counts for atoms overlapped with each local basin

    # loop over each voxel
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # if this voxel is part of the vacuum, continue
                if vacuum_mask[i, j, k]:
                    overlap_array[i,j,k] = -1
                    continue
                # get the labels at this point
                local_label = local_labels[i, j, k]
                atom_label = atom_labels[i,j,k]
                
                # add to our count
                overlap_counts[local_label, atom_label] += 1
                
                # get szudzik reduced label
                label = szudzik_pair(local_label, atom_label)
                overlap_array[i,j,k] = label

    return overlap_counts, overlap_array


@njit(cache=True)
def szudzik_pair(a: int, b: int):
    if a >= b:
        return a*a + a + b
    elif a < b:
        return b*b + a
    
@njit(cache=True)
def szudzik_reverse(z):
    k = (z**(1/2)) // 1
    kk = k*k
    
    if z - kk < k:
        a = z - kk
        b = k
    else:
        a = k
        b = z - kk - k
    return int(a), int(b)

    