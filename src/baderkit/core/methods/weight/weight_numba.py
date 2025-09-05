# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from baderkit.core.methods.shared_numba import get_best_neighbor, wrap_point, flat_to_coords, coords_to_flat


@njit(parallel=True, cache=True)
def get_weight_assignments(
    reference_data,
    charge_data,
    sorted_indices,
    neighbor_transforms: NDArray[np.int64],
    neighbor_alpha: NDArray[np.float64],
    all_neighbor_transforms,
    all_neighbor_dists,
        ):
    nx,ny,nz = reference_data.shape
    
    # create an array to sum volumes
    volume_data = np.ones_like(charge_data, dtype=np.float64)
    
    # create array to store labels and maxima
    labels = np.full(reference_data.shape, -1, dtype=np.int64)
    maxima_mask = np.zeros(reference_data.shape, dtype=np.bool_)
    
    # create scratch arrays to store neighbors/fluxes
    num_neighs = len(neighbor_transforms)
    neighs = np.empty((num_neighs, 3), dtype=np.int64)
    fluxes = np.empty(num_neighs, dtype=np.float64)
    
    # create lists to store charges/volumes
    charges = []
    volumes = []
    
    for idx in sorted_indices:
        # get 3D coords
        i, j, k = flat_to_coords(idx, nx, ny, nz)
        # get the reference and charge data
        base_value = reference_data[i, j, k]
        charge = charge_data[i,j,k]
        volume = volume_data[i,j,k]
        # calculate the flux going to each neighbor
        for trans_idx in prange(num_neighs):
            si, sj, sk = neighbor_transforms[trans_idx]
            alpha = neighbor_alpha[trans_idx]
        # for (si, sj, sk), alpha in zip(neighbor_transforms, neighbor_alpha):
            # get neighbor and wrap around periodic boundary
            ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
            # get the neighbors value
            neigh_value = reference_data[ii, jj, kk]
            # if this value is below the current points value, continue
            if neigh_value <= base_value:
                neighs[trans_idx, 0] = -1
                fluxes[trans_idx] = 0.0
                continue
            # neigh_sorted = indices_to_sorted[neigh_index]
            # calculate the flux flowing to this voxel
            flux = (neigh_value - base_value) * alpha
            # assign flux
            fluxes[trans_idx] = flux
            # total_flux += flux
            # add the pointer to this neighbor
            # neigh_index = coords_to_flat(ii,jj,kk,nx,ny,nz)
            neighs[trans_idx] = (ii,jj,kk)
        
        # normalize the flux and assign label
        total_flux = 0.0
        best_index = -1
        best_flux = 0.0
        for neigh_idx, flux in enumerate(fluxes):
            total_flux += flux
            if flux > best_flux:
                best_index = neigh_idx
                best_flux = flux
        
        # check that there is flux
        if total_flux == 0.0:
            # this is a local maximum. Check if its a true max
            shift, (ni, nj, nk), is_max = get_best_neighbor(
                data=reference_data,
                i=i,
                j=j,
                k=k,
                neighbor_transforms=all_neighbor_transforms,
                neighbor_dists=all_neighbor_dists,
            )
            # assign label to highest neighbor (self if true max)
            labels[i,j,k] = coords_to_flat(ni, nj, nk, nx, ny, nz)
            if is_max:
                # note this is a max and sum charges/volumes
                maxima_mask[i,j,k] = True
                charges.append(charge)
                volumes.append(volume)
                continue
            # otherwise, add all of the charge/volume to the best neigh
            charge_data[ni,nj,nk] += charge
            volume_data[ni,nj,nk] += volume
            continue
        
        # otherwise normalize and assign label
        fluxes /= total_flux
        ni,nj,nk = neighs[best_index]
        best_label = coords_to_flat(ni,nj,nk, nx,ny,nz)
        labels[i,j,k] = best_label
        
        # loop over neighbors and assign
        for trans_idx in prange(num_neighs):
            # get neigh and flux
            ni,nj,nk = neighs[trans_idx]
            if ni == -1:
                continue
            # otherwise, add charge/volume
            flux = fluxes[trans_idx]
            charge_data[ni,nj,nk] += charge*flux
            volume_data[ni,nj,nk] += volume*flux
    return (
        labels,
        np.array(charges, dtype=np.float64),
        np.array(volumes, dtype=np.float64),
        maxima_mask,
    )
        
    

# @njit(parallel=True, cache=True)
# def get_neighbor_flux(
#     data: NDArray[np.float64],
#     sorted_indices: NDArray[np.int64],
#     # sorted_pointers: NDArray[np.int64],
#     indices_to_sorted: NDArray[np.int64],
#     neighbor_transforms: NDArray[np.int64],
#     neighbor_alpha: NDArray[np.float64],
#     all_neighbor_transforms,
#     all_neighbor_dists,
# ):
#     """
#     For a 3D array of data set in real space, calculates the flux accross
#     voronoi facets for each voxel to its neighbors, corresponding to the
#     fraction of volume flowing to the neighbor.

#     Parameters
#     ----------
#     data : NDArray[np.float64]
#         A 3D grid of values for each point.
#     sorted_coords : NDArray[np.int64]
#         A Nx3 array where each entry represents the voxel coordinates of the
#         point. This must be sorted from highest value to lowest.
#     sorted_pointers : NDArray[np.int64]
#         A 3D array where each entry is the sorted index of the coord
#     neighbor_transforms : NDArray[np.int64]
#         The transformations from each voxel to its VORONOI neighbors.
#     neighbor_alpha : NDArray[np.float64]
#         The area of the voronoi facet divided by the distance to the voxel
#     all_neighbor_transforms
#         The transformations from each voxel to each of its 26 nearest neighbors
#     all_neighbor_dists
#         The distance from each voxe to each of its 26 neighbors

#     Returns
#     -------
#     flux_array : NDArray[float]
#         A 2D array of shape x*y*x by len(neighbor_transforms) where each entry
#         f(i, j) is the flux flowing from the voxel at index i to its neighbor
#         at transform neighbor_transforms[j]
#     neigh_array : NDArray[float]
#         A 2D array of shape x*y*x by len(neighbor_transforms) where each entry
#         f(i, j) is the index of the neighbor from the voxel at index i to the
#         neighbor at transform neighbor_transforms[j]
#     maxima_mask : NDArray[bool]
#         A 1D array of length N that is True where the sorted voxel indices are
#         a maximum

#     """
#     nx, ny, nz = data.shape
#     # create empty 2D arrays to store the volume flux flowing from each voxel
#     # to its neighbor and the voxel indices of these neighbors. We ignore the
#     # voxels that are below the vacuum value
#     # TODO: Alternative for improved memory would be to do this in chunks. The
#     # new method doesn't rely on stored information as heavily.
#     num_coords = len(sorted_indices)
#     flux_array = np.zeros((num_coords, len(neighbor_transforms)), dtype=np.float64)
#     neigh_array = np.full(flux_array.shape, -1, dtype=np.int64)
#     # create a mask for the location of maxima
#     maxima_mask = np.zeros(data.shape, dtype=np.bool_)
#     # Loop over each voxel in parallel (except the vacuum points)
#     for sorted_index in prange(num_coords):
#         coord_index = sorted_indices[sorted_index]
#         i, j, k = flat_to_coords(coord_index, nx, ny, nz)
#         # get the initial value
#         base_value = data[i, j, k]
#         # create a counter for the total flux
#         total_flux = 0.0
#         # iterate over each neighbor sharing a voronoi facet
#         neigh_n = 0
#         for (si, sj, sk), alpha in zip(neighbor_transforms, neighbor_alpha):
#             # get neighbor and wrap around periodic boundary
#             ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
#             # get the neighbors value
#             neigh_value = data[ii, jj, kk]
#             # if this value is below the current points value, continue
#             if neigh_value <= base_value:
#                 continue
#             # get the neighbors sorted index
#             neigh_index = coords_to_flat(ii,jj,kk,nx,ny,nz)
#             # neigh_sorted = indices_to_sorted[neigh_index]
#             # calculate the flux flowing to this voxel
#             flux = (neigh_value - base_value) * alpha
#             # assign flux
#             flux_array[sorted_index, neigh_n] = flux
#             total_flux += flux
#             # add the pointer to this neighbor
#             neigh_array[sorted_index, neigh_n] = neigh_index
#             # add to our neighbor count
#             neigh_n += 1

#         # Check that we had at least one assignment. If not, this might be a
#         # local maximum
#         if total_flux == 0.0:
#             # there is no flux flowing to any neighbors. Check if this is a true
#             # maximum
#             shift, (ni, nj, nk), is_max = get_best_neighbor(
#                 data=data,
#                 i=i,
#                 j=j,
#                 k=k,
#                 neighbor_transforms=all_neighbor_transforms,
#                 neighbor_dists=all_neighbor_dists,
#             )
#             # if this is a maximum note its a max and continue
#             if is_max:
#                 # We don't need to assign the flux/neighbors
#                 maxima_mask[i,j,k] = True
#                 continue
#             # otherwise, set all of the weight to the highest neighbor and continue
#             flux_array[sorted_index, 0] = 1.0
#             neigh_array[sorted_index, 0] = coords_to_flat(ni,nj,nk,nx,ny,nz)
#             continue

#         # otherwise, normalize the flux
#         flux_array[sorted_index] /= total_flux

#     return flux_array, neigh_array, maxima_mask


# @njit(fastmath=True, cache=True)
# def get_weight_assignments(
#     labels,
#     label_map,
#     charge_array,
#     sorted_indices,
#     indices_to_sorted: NDArray[np.int64],
#     neigh_fluxes,
#     neigh_pointers,
#     maxima_mask,
#     maxima_num,
# ):
#     nx,ny,nz = charge_array.shape
#     # create arrays to store charges, volumes, and pointers
#     charges = np.zeros(maxima_num, dtype=np.float64)
#     volumes = np.zeros(maxima_num, dtype=np.float64)
#     # create array to store volume
#     volume_array = np.full(charge_array.shape, 1.0, dtype=np.float64)
#     # now iterate over coords from lowest to highest
#     for sorted_index in range(len(sorted_indices)):
#         coord_index = sorted_indices[sorted_index]
#         i, j, k = flat_to_coords(coord_index, nx, ny, nz)
#         charge = charge_array[i,j,k]
#         volume = volume_array[i,j,k]
#         is_max = maxima_mask[i,j,k]
#         pointers = neigh_pointers[sorted_index]
#         fluxes = neigh_fluxes[sorted_index]
#         # if this is a maximum, create a new basin
#         if is_max:
#             # get the pre-assigned pointer
#             label = labels[i, j, k]
#             # get the corresponding basin
#             for basin_idx, blabel in enumerate(label_map):
#                 if label == blabel:
#                     charges[basin_idx] += charge
#                     volumes[basin_idx] += volume
#             continue

#         # otherwise, add charge and volume to neighbors, and get best neighbor
#         highest_flux = 0.0
#         best_neighbor = -1
#         for pointer, flux in zip(pointers, fluxes):
#             if pointer == -1:
#                 # We have reached the last neighbor and break
#                 break
#             # get neighbor indices
#             # TODO: Store pointers as actual indices instead of sorted
#             ni,nj,nk = flat_to_coords(pointer, nx, ny, nz)
            
#             charge_array[ni,nj,nk] += charge * flux
#             volume_array[ni,nj,nk] += volume * flux
#             # check if flux is greater than or equal than the current max flux
#             # within a tolerance
#             if flux > highest_flux:
#                 highest_flux = flux
#                 best_neighbor = pointer

#         # assign this point to the best neighbor
#         labels[i, j, k] = best_neighbor

#     return (
#         labels,
#         charges,
#         volumes,
#     )


@njit(fastmath=True, cache=True)
def get_labels(
    pointers,
    sorted_indices,
):
    # Assuming sorted_pointers is from high to low, we only need to loop over
    # the values once to assign all of them.
    # NOTE: We don't need to check for vacuum because we are only looping over
    # the values above the vacuum.
    for idx in sorted_indices:
        # assign to parent
        pointers[idx] = pointers[pointers[idx]]
    return pointers


@njit(cache=True)
def relabel_reduced_maxima(
    labels,
    maxima_num,
    maxima_vox,
    flat_grid_indices,
):
    new_label_map = np.full(maxima_num, -1, dtype=np.int64)

    for i, j, k in maxima_vox:
        label = labels[i, j, k]
        if new_label_map[label] == -1:
            new_label = flat_grid_indices[i, j, k]
            new_label_map[label] = new_label
            labels[i, j, k] = new_label
        else:
            labels[i, j, k] = new_label_map[label]
    return labels, new_label_map


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

###############################################################################
# Parallel attempt. Doesn't scale linearly
###############################################################################

# @njit(parallel=True, cache=True)
# def get_weight_assignments(
#     data,
#     labels,
#     flat_charge,
#     neigh_fluxes,
#     neigh_pointers,
#     weight_maxima_mask,
#     all_neighbor_transforms,
#     all_neighbor_dists,
# ):
#     nx,ny,nz = data.shape
#     # Get the indices corresponding to maxima
#     maxima_indices = np.where(weight_maxima_mask)[0]
#     maxima_num = len(maxima_indices)
#     # We are going to reuse the maxima mask as a mask noting which points don't
#     # need to be checked anymore
#     finished_points = weight_maxima_mask
#     finished_maxima = np.zeros(maxima_num, dtype=np.bool_)
#     # create arrays to store charges, volumes, and pointers
#     charges = flat_charge[maxima_indices]
#     volumes = np.ones(maxima_num, dtype=np.float64)
#     # create array to store the true maximum each local maxima belongs to. This
#     # is used to reduce false weight maxima
#     maxima_map = np.empty(maxima_num, dtype=np.int64)
#     # create array representing total volume
#     flat_volume = np.ones(len(flat_charge), dtype=np.float64)
#     # create secondary arrays to store flow of charge/volume
#     flat_volume1 = np.zeros(len(flat_charge), dtype=np.float64)
#     flat_charge1 = np.zeros(len(flat_charge), dtype=np.float64)
#     # create array to store number of lower neighbors at each point
#     neigh_nums = np.zeros(len(flat_charge), dtype=np.int8)
#     # create counter for if we are on an even/odd loop
#     loop_count = 0
    
#     # Now we begin our while loop
#     while True:
#         # get the indices to loop over
#         current_indices = np.where(~finished_points)[0]
#         current_maxima = np.where(~finished_maxima)[0]
#         num_current = len(current_indices)
#         maxima_current = len(current_maxima)
#         if num_current == 0 and maxima_current == 0:
#             break
#         # get the charge and volume arrays that were accumulated into last cycle
#         # and the ones to accumulate into this cycle
#         if loop_count % 2 == 0:
#             charge_store = flat_charge
#             volume_store = flat_volume
#             charge_new = flat_charge1
#             volume_new = flat_volume1
#         else:
#             charge_store = flat_charge1
#             volume_store = flat_volume1
#             charge_new = flat_charge
#             volume_new = flat_volume
            
#         # loop over maxima and sum their neighbors current accumulated charge
#         for max_idx in prange(maxima_num):
#             if finished_maxima[max_idx]:
#                 continue
#             max_pointer = maxima_indices[max_idx]
#             pointers = neigh_pointers[max_pointer]
#             fluxes = neigh_fluxes[max_pointer]
#             # sum each charge
#             new_charge = 0.0
#             new_volume = 0.0
#             for neigh_idx, (pointer, flux) in enumerate(zip(pointers, fluxes)):
#                 # skip neighbors with no charge
#                 if pointer == -1:
#                     continue
#                 # If charge is 0, remove this neighbor
#                 charge = charge_store[pointer]
#                 if charge == 0.0:
#                     pointers[neigh_idx] = -1
#                 new_charge += charge * flux
#                 new_volume += volume_store[pointer] * flux
#             # If no charge was added, we're done with this maximum
#             if new_charge == 0.0:
#                 finished_maxima[max_idx] = True
#                 # Check if this is a true maximum
#                 i,j,k = flat_to_coords(max_pointer, nx, ny, nz)
#                 mi, mj, mk = climb_to_max(data, i, j, k, all_neighbor_transforms, all_neighbor_dists)
#                 # update maxima map and labels
#                 pointer = coords_to_flat(mi,mj,mk,nx,ny,nz)
#                 labels[i,j,k] = pointer
#                 maxima_map[max_idx] = pointer
            
#             # add charge/volume to total
#             charges[max_idx] += new_charge
#             volumes[max_idx] += new_volume
        
#         # loop over other points, sum their neighbors, reset charge/volume accumulation
#         for point_idx in prange(num_current):
#             point_pointer = current_indices[point_idx]
#             pointers = neigh_pointers[point_pointer]
#             fluxes = neigh_fluxes[point_pointer]
#             # if this is our first cycle, we want to get the number of neighbors
#             # for each point and reorder our pointers/fluxes for faster iteration
#             if loop_count == 0:
#                 n_neighs = 0
#                 for neigh_idx, pointer in enumerate(pointers):
#                     # skip empty neighbors
#                     if pointer == -1:
#                         continue
#                     # move pointer/flux to farthest left point
#                     pointers[n_neighs] = pointer
#                     fluxes[n_neighs] = fluxes[neigh_idx]
#                     n_neighs += 1
#                 neigh_nums[point_pointer] = n_neighs
            
#             # otherwise, sum charge/volume as usual
#             n_neighs = neigh_nums[point_pointer]
#             new_charge = 0.0
#             new_volume = 0.0
#             for neigh_idx in range(n_neighs):
#                 neigh_pointer = pointers[neigh_idx]
#                 if neigh_pointer == -1:
#                     continue
#                 charge = charge_store[neigh_pointer]
#                 # if the charge is 0, we no longer need to accumulate charge
#                 # from this point.
#                 if charge == 0.0:
#                     pointers[neigh_idx] = -1
#                     continue
#                 new_charge += charge_store[neigh_pointer] * fluxes[neigh_idx]
#                 new_volume += volume_store[neigh_pointer] * fluxes[neigh_idx]
#             # set new charge and volume
#             charge_new[point_pointer] = new_charge
#             volume_new[point_pointer] = new_volume
#             # if charge was 0 mark this point as not important
#             if new_charge == 0.0:
#                 finished_points[point_pointer] = True
                    
#         loop_count += 1
    
#     # reduce to true maxima
#     true_maxima = np.unique(maxima_map)
#     reduced_charges = np.zeros(len(true_maxima), dtype=np.float64)
#     reduced_volumes = np.zeros(len(true_maxima), dtype=np.float64)
#     for old_idx, max_label in enumerate(maxima_map):
#         for max_idx, true_max in enumerate(true_maxima):
#             if max_label == true_max:
#                 reduced_charges[max_idx] += charges[old_idx]
#                 reduced_volumes[max_idx] += volumes[old_idx]
    
#     return reduced_charges, reduced_volumes, labels, true_maxima