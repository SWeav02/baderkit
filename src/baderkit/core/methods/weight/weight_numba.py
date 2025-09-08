# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from baderkit.core.methods.shared_numba import (
    coords_to_flat,
    flat_to_coords,
    get_best_neighbor,
    wrap_point,
)


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
    nx, ny, nz = reference_data.shape
    num_coords = len(sorted_indices)
    full_num_coords = nx * ny * nz

    # create arrays to store flux/neighs
    flux_array = np.empty((num_coords, len(neighbor_transforms)), dtype=np.float64)
    neigh_array = np.empty(flux_array.shape, dtype=np.uint32)
    neigh_nums = np.empty(num_coords, dtype=np.uint8)

    # create flat arrays to store volumes/charge
    flat_charge = np.empty(full_num_coords, dtype=np.float64)
    flat_volume = np.ones_like(flat_charge, dtype=np.float64)

    # create array to store labels and maxima
    labels = np.full(full_num_coords, -1, dtype=np.int64)
    # maxima_mask = np.zeros(reference_data.shape, dtype=np.bool_)
    sorted_maxima_mask = np.zeros(num_coords, dtype=np.bool_)

    # create lists to store charges/volumes
    charges = []
    volumes = []

    # calculate flux in parallel
    for sorted_idx in prange(num_coords):
        idx = sorted_indices[sorted_idx]
        # get 3D coords
        i, j, k = flat_to_coords(idx, nx, ny, nz)
        # get the reference and charge data
        base_value = reference_data[i, j, k]
        # set flat charge
        flat_charge[idx] = charge_data[i, j, k]
        # track flux
        total_flux = 0.0
        best_label = -1
        best_flux = 0.0
        equal_neighs = False
        # calculate the flux going to each neighbor
        neigh_num = 0
        for (si, sj, sk), alpha in zip(neighbor_transforms, neighbor_alpha):
            # get neighbor and wrap around periodic boundary
            ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
            # get the neighbors value
            neigh_value = reference_data[ii, jj, kk]
            # if this value is below the current points value, continue
            if neigh_value < base_value:
                continue
            # get this neighbors index
            neigh_idx = coords_to_flat(ii, jj, kk, nx, ny, nz)
            # if this neigbhor has an equal value, check if its label is lower
            # than the current point
            if neigh_value == base_value:
                equal_neighs = True
                continue

            # calculate the flux flowing to this voxel
            flux = (neigh_value - base_value) * alpha
            # assign flux
            flux_array[sorted_idx, neigh_num] = flux
            total_flux += flux
            # add the pointer to this neighbor

            neigh_array[sorted_idx, neigh_num] = neigh_idx
            neigh_num += 1
            # update label
            if flux > best_flux:
                best_label = neigh_idx
                best_flux = flux

        # check that there is flux. If not we have a local maximum
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
            if not is_max:
                # this is not a real maximum. Assign it to the highest neighbor
                neigh_idx = coords_to_flat(ni, nj, nk, nx, ny, nz)
                neigh_nums[sorted_idx] = 1
                neigh_array[sorted_idx, 0] = neigh_idx
                labels[idx] = neigh_idx
                continue

            # otherwise, this is a true maximum. However, it might border another
            # point with the same value as this one.
            # maxima_mask[i,j,k] = True
            neigh_nums[sorted_idx] = 0
            if not equal_neighs:
                # we don't border points with the same value. We note this with
                # a -1 in our first neighbor
                neigh_array[sorted_idx, 0] = full_num_coords
                # assign this point its own label
                labels[idx] = idx
                continue
            # We do border at least one other maximum. Note each of them.
            # NOTE: We do not assign a label in this case and will need to do
            # so later
            neigh_num = 0
            for si, sj, sk in neighbor_transforms:
                # get neighbor and wrap around periodic boundary
                ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
                # get neighbors value. Skip of it doesn't equal the current points
                # value
                neigh_value = reference_data[ii, jj, kk]
                if neigh_value != base_value:
                    continue
                # get this neighbors index and assign it
                neigh_idx = coords_to_flat(ii, jj, kk, nx, ny, nz)
                neigh_array[sorted_idx, neigh_num] = neigh_idx
                neigh_num += 1
            # add a value above the possible neighbor indices to note when we
            # have reached the end of our neighboring maxima.
            if neigh_num != len(neighbor_transforms):
                neigh_array[sorted_idx, neigh_num] = full_num_coords
            continue

        # otherwise we don't have a local maximum.
        # assign the neigh num
        neigh_nums[sorted_idx] = neigh_num
        # normalize and assign label
        flux_array[sorted_idx] /= total_flux
        labels[idx] = best_label

    # Now we loop over and sum charge/volume
    maxima_vox = []
    for sorted_idx, (neighs, fluxes, neigh_num) in enumerate(
        zip(neigh_array, flux_array, neigh_nums)
    ):
        idx = sorted_indices[sorted_idx]
        charge = flat_charge[idx]
        volume = flat_volume[idx]

        # check how many higher neighbors this point has. The most common scenario is
        # to have multiple so we check this first for efficiency
        if neigh_num > 1:
            # loop over neighbors and assign
            for neigh_idx in range(neigh_num):
                neigh = neighs[neigh_idx]
                flux = fluxes[neigh_idx]
                # add charge/volume
                flat_charge[neigh] += charge * flux
                flat_volume[neigh] += volume * flux

        # The next most common is to have one higher neighbor. We can skip some math
        # in this case
        elif neigh_num == 1:
            neigh = neighs[0]
            flat_charge[neigh] += charge
            flat_volume[neigh] += volume

        # Finally, we can have 0 higher neighbors, indicating a maximum
        elif neigh_num == 0:
            # this is a local maximum. Add it to our list
            maxima_vox.append(flat_to_coords(idx, nx, ny, nz))
            # Now we want to check if there are any equivalent neighbors
            best_neigh = idx
            for neigh in neighs:
                # We note the end of equivalent neighs with a value slightly
                # above the possible number
                if neigh == full_num_coords:
                    break
                # check if this neighbor has already been searched. We denote this
                # by setting its charge to 0
                if flat_charge[neigh] == 0.0:
                    continue
                # We have found a maximum that hasn't been searched yet.
                best_neigh = neigh
                break
            # set our label to the best neighbor, whatever it is
            labels[idx] = best_neigh
            # if the best neighbor is the current point, we have a new maximum
            if best_neigh == idx:
                sorted_maxima_mask[sorted_idx] = True
                charges.append(charge)
                volumes.append(volume)
            # otherwise, we add the charge/volume to the new point
            else:
                flat_charge[best_neigh] += charge
                flat_volume[best_neigh] += volume
            # set this points charge to 0 to denote that its been searched
            flat_charge[idx] = 0.0
        else:
            # This shouldn't be possible. Raise an error.
            raise Exception()

    return (
        labels,
        np.array(charges, dtype=np.float64),
        np.array(volumes, dtype=np.float64),
        # maxima_mask,
        np.array(maxima_vox, dtype=np.int64),
        sorted_maxima_mask,
    )


@njit(fastmath=True, cache=True)
def get_labels(
    pointers,
    sorted_indices,
    sorted_maxima_mask,
):
    # Assuming sorted_pointers is from high to low, we only need to loop over
    # the values once to assign all of them.
    # NOTE: We don't need to check for vacuum because we are only looping over
    # the values above the vacuum.
    maxima_num = 0
    for idx, is_max in zip(sorted_indices, sorted_maxima_mask):
        # if this is a maximum, add a new max
        if is_max:
            pointers[idx] = maxima_num
            maxima_num += 1
            continue
        # otherwise, assign to parent
        pointers[idx] = pointers[pointers[idx]]
    return pointers


@njit(parallel=True, cache=True)
def sort_maxima_vox(
    maxima_vox,
    nx,
    ny,
    nz,
):
    flat_indices = np.empty(len(maxima_vox), dtype=np.int64)
    for idx in prange(len(flat_indices)):
        i, j, k = maxima_vox[idx]
        flat_indices[idx] = coords_to_flat(i, j, k, nx, ny, nz)

    # sort flat indices from low to high
    sorted_indices = np.argsort(flat_indices)
    # sort maxima from lowest index to highest
    return maxima_vox[sorted_indices]


# @njit(fastmath=True, cache=True)
# def reduce_charge_volume(
#     basin_map,
#     charges,
#     volumes,
#     basin_num,
#         ):
#     # create a new array for charges and volumes
#     new_charges = np.zeros(basin_num, dtype=np.float64)
#     new_volumes = np.zeros(basin_num, dtype=np.float64)
#     for i in range(len(charges)):
#         basin = basin_map[i]
#         new_charges[basin] += charges[i]
#         new_volumes[basin] += volumes[i]
#     return new_charges, new_volumes

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
