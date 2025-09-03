# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from baderkit.core.methods.shared_numba import get_best_neighbor, wrap_point


@njit(parallel=True, cache=True)
def get_neighbor_flux(
    data: NDArray[np.float64],
    sorted_coords: NDArray[np.int64],
    sorted_pointers: NDArray[np.int64],
    neighbor_transforms: NDArray[np.int64],
    neighbor_alpha: NDArray[np.float64],
    all_neighbor_transforms,
    all_neighbor_dists,
):
    """
    For a 3D array of data set in real space, calculates the flux accross
    voronoi facets for each voxel to its neighbors, corresponding to the
    fraction of volume flowing to the neighbor.

    Parameters
    ----------
    data : NDArray[np.float64]
        A 3D grid of values for each point.
    sorted_coords : NDArray[np.int64]
        A Nx3 array where each entry represents the voxel coordinates of the
        point. This must be sorted from highest value to lowest.
    sorted_pointers : NDArray[np.int64]
        A 3D array where each entry is the sorted index of the coord
    neighbor_transforms : NDArray[np.int64]
        The transformations from each voxel to its VORONOI neighbors.
    neighbor_alpha : NDArray[np.float64]
        The area of the voronoi facet divided by the distance to the voxel
    all_neighbor_transforms
        The transformations from each voxel to each of its 26 nearest neighbors
    all_neighbor_dists
        The distance from each voxe to each of its 26 neighbors

    Returns
    -------
    flux_array : NDArray[float]
        A 2D array of shape x*y*x by len(neighbor_transforms) where each entry
        f(i, j) is the flux flowing from the voxel at index i to its neighbor
        at transform neighbor_transforms[j]
    neigh_array : NDArray[float]
        A 2D array of shape x*y*x by len(neighbor_transforms) where each entry
        f(i, j) is the index of the neighbor from the voxel at index i to the
        neighbor at transform neighbor_transforms[j]
    maxima_mask : NDArray[bool]
        A 1D array of length N that is True where the sorted voxel indices are
        a maximum

    """
    nx, ny, nz = data.shape
    # create empty 2D arrays to store the volume flux flowing from each voxel
    # to its neighbor and the voxel indices of these neighbors. We ignore the
    # voxels that are below the vacuum value
    # TODO: Alternative for improved memory would be to do this in chunks. The
    # new method doesn't rely on stored information as heavily.
    num_coords = len(sorted_coords)
    flux_array = np.zeros(
        (num_coords, len(neighbor_transforms)), dtype=np.float64
    )
    neigh_array = np.full(flux_array.shape, -1, dtype=np.int64)
    # create a mask for the location of maxima
    maxima_mask = np.zeros(num_coords, dtype=np.bool_)
    # Loop over each voxel in parallel (except the vacuum points)
    for coord_index in prange(num_coords):
        i, j, k = sorted_coords[coord_index]
        # get the initial value
        base_value = data[i, j, k]
        # create a counter for the total flux
        total_flux = 0.0
        # iterate over each neighbor sharing a voronoi facet
        neigh_n = 0
        for (si, sj, sk), alpha in zip(neighbor_transforms, neighbor_alpha):
            # get neighbor and wrap around periodic boundary
            ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
            # get the neighbors value
            neigh_value = data[ii, jj, kk]
            # if this value is below the current points value, continue
            if neigh_value <= base_value:
                continue
            # calculate the flux flowing to this voxel
            flux = (neigh_value - base_value) * alpha
            # assign flux
            flux_array[coord_index, neigh_n] = flux
            total_flux += flux
            # add the pointer to this neighbor
            neigh_array[coord_index, neigh_n] = sorted_pointers[ii, jj, kk]
            # add to our neighbor count
            neigh_n += 1

        # Check that we had at least one assignment. If not, this might be a
        # local maximum
        if total_flux == 0.0:
            # there is no flux flowing to any neighbors. Check if this is a true
            # maximum
            shift, (ni, nj, nk), is_max = get_best_neighbor(
                data=data,
                i=i,
                j=j,
                k=k,
                neighbor_transforms=all_neighbor_transforms,
                neighbor_dists=all_neighbor_dists,
            )
            # if this is a maximum note its a max and continue
            if is_max:
                # We don't need to assign the flux/neighbors
                maxima_mask[coord_index] = True
                continue
            # otherwise, set all of the weight to the highest neighbor and continue
            flux_array[coord_index, 0] = 1
            neigh_array[coord_index, 0] = sorted_pointers[ni, nj, nk]
            continue
        
        # otherwise, normalize the flux
        flux_array[coord_index] /= total_flux

    return flux_array, neigh_array, maxima_mask

@njit(fastmath=True, cache=True)
def get_weight_assignments(
    data,
    sorted_coords,
    sorted_charge,
    original_indices,
    neigh_fluxes,
    neigh_pointers,
    weight_maxima_mask,
        ):
    # create arrays to store charges, volumes, and pointers
    charges = []
    volumes = []
    basin_pointers = np.full(data.shape, -1, dtype=np.int64)
    # create array to store volume
    sorted_volume = np.full(len(sorted_charge),1.0, dtype=np.float64)
    # now iterate over coords from lowest to highest
    for idx in range(len(sorted_charge)):
        # get all info
        i,j,k = sorted_coords[idx]
        charge = sorted_charge[idx]
        volume = sorted_volume[idx]
        is_max = weight_maxima_mask[idx]
        # if this is a maximum, create a new basin
        if is_max:
            charges.append(charge)
            volumes.append(volume)
            basin_pointers[i,j,k] = original_indices[i,j,k]
            continue
        
        # get neighbor/flux info
        pointers = neigh_pointers[idx]
        fluxes = neigh_fluxes[idx]
        # otherwise, add charge and volume to neighbors, and get best neighbor
        highest_flux = 0.0
        best_neighbor = -1
        for pointer, flux in zip(pointers, fluxes):
            if pointer == -1:
                # We have reached the last neighbor and break
                break
            sorted_charge[pointer] += charge * flux
            sorted_volume[pointer] += volume * flux
            # check if flux is greater than or equal than the current max flux
            # within a tolerance
            if flux > highest_flux:
                highest_flux = flux
                best_neighbor = pointer

        # assign this point to the best neighbor
        ni,nj,nk = sorted_coords[best_neighbor]
        basin_pointers[i,j,k] = original_indices[ni,nj,nk]
        
    return (
        basin_pointers, 
        np.array(charges, dtype=np.float64), 
        np.array(volumes, dtype=np.float64),
        )

@njit(fastmath=True, cache=True)
def get_labels(
    pointers,
    sorted_indices,
        ):
    # Assuming sorted_pointers is from high to low, we only need to loop over
    # the values once to assign all of them.
    for idx in sorted_indices:
        # skip vacuum assignments
        if pointers[idx] == -1:
            continue
        # assign to parent
        pointers[idx] = pointers[pointers[idx]]
    return pointers

@njit(fastmath=True, cache=True)
def reduce_charge_volume(
    basin_map,
    charges,
    volumes,
    basin_num,
        ):
    # create a new array for charges and volumes
    new_charges = np.zeros(basin_num, dtype=np.float64)
    new_volumes = np.zeros(basin_num, dtype=np.float64)
    for i in range(len(charges)):
        basin = basin_map[i]
        new_charges[basin] += charges[i]
        new_volumes[basin] += volumes[i]            
    return new_charges, new_volumes

###############################################################################
# Tests for better labeling. The label assignments never converged well so I've
# given this up for now.
###############################################################################

# @njit(fastmath=True)
# def get_labels_fine(
#     label_array,
#     flat_grid_indices,
#     neigh_pointers,
#     neigh_fluxes,
#     neigh_numbers,
#     volumes,
#     charges,
#     sorted_coords,
#     sorted_charge,
#         ):
#     max_idx = len(sorted_coords) - 1
#     # create an array to store approximate volumes
#     # approx_volumes = np.zeros(len(volumes), dtype=np.int64)
#     # Flip the true volumes/charges so that they are in order from highest to
#     # lowest coord
#     volumes = np.flip(volumes)
#     # charges = np.flip(charges)
#     # multiply charges by 2 so we can avoid a lot of divisions later
#     # charges *= 2
#     # Create an array to store the difference from the ideal volume
#     volume_diff = np.ones(len(volumes), dtype=np.float64)
#     # charge_diff = np.ones(len(charges), dtype=np.float64)
#     # diffs = np.ones(len(volumes), dtype=np.float64)
#     # Create an array to store the ratio by which the volume_diff changes when
#     # a new voxel is added to the corresponding basin
#     volume_ratios = 1.0 / volumes
#     # create a list to store neighbor labels
#     all_neighbor_labels = []
#     # split_voxels = np.zeros(len(pointers), dtype=np.bool_)
#     # loop over points from high to low
#     maxima_num = 0
#     for idx in np.arange(max_idx, -1, -1):
#         # get the charge and position
#         # charge = sorted_charge[idx]
#         i,j,k = sorted_coords[idx]
#         # If there are neighs, this is a maximum. We assign a new basin
#         neighbor_num = neigh_numbers[idx]
#         if neighbor_num == 0:
#             # label the voxel
#             label_array[i,j,k] = maxima_num
#             all_neighbor_labels.append([maxima_num])
#             # update the volume/charge diffs
#             volume_diff[maxima_num] -= volume_ratios[maxima_num]
#             # charge_diff[maxima_num] -= charge / charges[maxima_num]
#             # diffs[maxima_num] -= (volume_ratios[maxima_num] + charge / charges[maxima_num]) # divide by 2 is done earlier
#             maxima_num += 1
#             continue
        
#         # otherwise, we are not at a maximum
#         # get the pointers/flux
#         pointers = neigh_pointers[idx]
#         # fluxes = neigh_fluxes[idx]
        
#         # tol = (1/neighbor_num) - 1e-12
#         # reduce to labels/weights
#         labels = []
#         # weights = []
#         # for pointer, flux in zip(pointers, fluxes):
#         for pointer in pointers:
#             # if the pointer is -1 we've reached the end of our list
#             if pointer == -1:
#                 break
#             # if the flux is less than our tolerance, we don't consider this neighbor
#             # if flux < tol:
#             #     continue
#             # otherwise, get the labels at this point
#             neigh_labels = all_neighbor_labels[max_idx-pointer]
#             for label in neigh_labels:
#                 if not label in labels:
#                     labels.append(label)
#             # # otherwise, get the label at this point
#             # ni, nj, nk = sorted_coords[pointer]
#             # label = label_array[ni,nj,nk]
#             # # check if the label exists. If not, add it
#             # found = False
#             # for lidx, rlabel in enumerate(labels):
#             #     if label == rlabel:
#             #         found = True
#             #         # weights[lidx] += flux
#             # if not found:
#             #     # add the new label/weight
#             #     labels.append(label)
#             #     # weights.append(flux)
        
        
#         # If there is 1 label, assign this label
#         if len(labels) == 1:
#             label = labels[0]
#             label_array[i,j,k] = label
#             # update volume/charge diffs
#             volume_diff[label] -= volume_ratios[label]
#             # charge_diff[label] -= charge / charges[label]
#             # diffs[label] -= (volume_ratios[label] + charge / charges[label])
#         # if there is more than 1 label, we have a split voxel. As an approximation,
#         # we check how far from the true volume each possible basin is and add
#         # the voxel to the farthest one.
#         else:
#             best_label = -1
#             best_diff = -1.0
#             for label in labels:
#                 # if diffs[label] > best_diff:
#                 #     best_label = label
#                 #     best_diff = diffs[label]
#                 if volume_diff[label] > best_diff:
#                     best_label = label
#                     best_diff = volume_diff[label]
#                 # if charge_diff[label] > best_diff:
#                 #     best_label = label
#                 #     best_diff = charge_diff[label]
#             # update label
#             label_array[i,j,k] = best_label
#             # update diff
#             volume_diff[best_label] -= volume_ratios[best_label]
#             # charge_diff[best_label] -= charge / charges[best_label]
#             # diffs[best_label] -= (volume_ratios[best_label] + charge / charges[best_label])
            
#         all_neighbor_labels.append(labels)
            
#     return label_array

        
