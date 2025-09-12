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
    # create arrays to store neighs. Don't store flux yet
    num_transforms = len(neighbor_transforms)
    neigh_array = np.empty((num_coords, num_transforms), dtype=np.uint32)
    neigh_nums = np.empty(num_coords, dtype=np.uint8)
    # Create a 1D array to store labels
    labels = np.full(full_num_coords, -1, dtype=np.int64)
    # Create 1D arrays to store flattened charge
    flat_charge = np.empty(full_num_coords, dtype=np.float64)
    # Create lists to store basin charges/volumes
    charges = []
    volumes = []

    ###########################################################################
    # Get neighbors
    ###########################################################################

    # loop over points in parallel and calculate neighbors
    for sorted_idx in prange(num_coords):
        idx = sorted_indices[sorted_idx]
        # get 3D coords
        i, j, k = flat_to_coords(idx, nx, ny, nz)
        # get the reference and charge data
        base_value = reference_data[i, j, k]
        # set flat charge
        flat_charge[idx] = charge_data[i, j, k]
        equal_neighs = False
        # get higher neighbors at each point
        neigh_num = 0
        for si, sj, sk in neighbor_transforms:
            # get neighbor and wrap around periodic boundary
            ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
            # get the neighbors value
            neigh_value = reference_data[ii, jj, kk]
            # if this value is below the current points value, continue
            if neigh_value < base_value:
                continue
            # Note if this neighbor has an equal value
            if neigh_value == base_value:
                equal_neighs = True
                continue
            # get this neighbors index and add it to our array
            neigh_idx = coords_to_flat(ii, jj, kk, nx, ny, nz)
            neigh_array[sorted_idx, neigh_num] = neigh_idx
            neigh_num += 1
        neigh_nums[sorted_idx] = neigh_num

        # Check if we had any higher neighbors
        if neigh_num == 0:
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
                neigh_nums[sorted_idx] = 1  # overwrite 0 entry
                neigh_array[sorted_idx, 0] = neigh_idx
                continue

            # Check if we border another maximum
            if not equal_neighs:
                # Note this maximum doesn't border any others by labeling it above the max label
                neigh_array[sorted_idx, 0] = full_num_coords
                continue
            # We do border at least one other maximum and record each of them
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

    ###########################################################################
    # Assign interior
    ###########################################################################
    # create list to store edge indices
    edge_sorted_indices = []
    # create list to store maxima indices
    maxima_vox = []
    # Now we have the neighbors for each point. Loop over them from highest to
    # lowest and assign single basin points
    maxima_num = 0
    for sorted_idx, (idx, neighs, neigh_num) in enumerate(
        zip(sorted_indices, neigh_array, neigh_nums)
    ):
        if neigh_num > 0:
            # This is not a maximum. Check if interior point (single basin)
            best_label = -1
            for neigh_idx, neigh in enumerate(neighs):
                if neigh_idx == neigh_num:
                    break
                label = labels[neigh]
                if label == -1:
                    # This neighbor is not an interior and this one can't be either
                    best_label = -1
                    break
                elif label != best_label and best_label != -1:
                    # We have two different basin assignments and this is not an
                    # interior
                    best_label = -1
                    break
                best_label = label

            # If the best label isn't -1, this is an interior point and we assign
            if best_label != -1:
                labels[idx] = label
                charges[label] += flat_charge[idx]
                volumes[label] += 1
            # Otherwise, this point is an exterior point that is partially assigned
            # to multiple basins. We add it to our list.
            else:
                edge_sorted_indices.append(sorted_idx)
        else:
            # This is a local maximum. Add it to our list of indices
            maxima_vox.append((flat_to_coords(idx, nx, ny, nz)))
            # Check if there are any neighboring maxima that have already been assigned
            best_label = -1
            for neigh in neighs:
                if neigh == full_num_coords:
                    # This is the end of our neighs and we break
                    break
                label = labels[neigh]
                if label != -1:
                    best_label = label
                    break
            # if the best label is still this index, assign a new maximum and
            # increase the counter
            if best_label == -1:
                labels[idx] = maxima_num
                maxima_num += 1
                charges.append(flat_charge[idx])
                volumes.append(1)
            else:
                # assign to best label
                labels[idx] = best_label
                charges[best_label] += flat_charge[idx]
                volumes[best_label] += 1

    # convert charges/volumes to arrays
    charges = np.array(charges, dtype=np.float64)
    volumes = np.array(volumes, dtype=np.float64)

    ###########################################################################
    # Fluxes
    ###########################################################################
    # We only need to calculate the flux for each exterior point. Create an array
    # to store these.
    num_edges = len(edge_sorted_indices)
    flux_array = np.empty((num_edges, num_transforms), dtype=np.float64)
    neigh_array = np.empty_like(flux_array, dtype=np.int64)
    neigh_nums = np.empty(num_edges, dtype=np.uint8)
    # create mask to note if this point has neighbors that are also exterior
    only_interor_neighs = np.zeros(num_edges, dtype=np.bool_)
    # create an array to store pointers from idx to edge idx
    idx_to_edge = np.empty(full_num_coords, dtype=np.uint32)
    # calculate fluxes in parallel. If possible, we will immediately calculate the
    # weight as well
    for edge_idx in prange(len(edge_sorted_indices)):
        sorted_idx = edge_sorted_indices[edge_idx]
        idx = sorted_indices[sorted_idx]
        # set idx to edge value
        idx_to_edge[idx] = edge_idx
        # loop over neighs and get their label. If all of them have labels, we
        # can immediately calculate weights
        # get 3D coords
        i, j, k = flat_to_coords(idx, nx, ny, nz)
        # get the reference and charge data
        base_value = reference_data[i, j, k]
        # set flat charge
        flat_charge[idx] = charge_data[i, j, k]
        # get higher neighbors at each point
        total_flux = 0.0
        neigh_labels = neigh_array[edge_idx]
        neigh_fluxes = flux_array[edge_idx]
        no_exterior_neighs = True
        neigh_num = 0
        for (si, sj, sk), alpha in zip(neighbor_transforms, neighbor_alpha):
            # get neighbor and wrap around periodic boundary
            ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
            # get the neighbors value
            neigh_value = reference_data[ii, jj, kk]
            # if this value is below the current points value, continue
            if neigh_value <= base_value:
                continue
            # get this neighbors index
            neigh_idx = coords_to_flat(ii, jj, kk, nx, ny, nz)
            # calculate the flux flowing to this voxel
            flux = (neigh_value - base_value) * alpha
            total_flux += flux
            # get this neighbors label
            neigh_label = labels[neigh_idx]
            # if the neighbor hasn't been assigned, assign flux to this neighbor
            if neigh_label == -1:
                # at least one neighbor is also an exterior point
                neigh_fluxes[neigh_num] = flux
                neigh_labels[neigh_num] = -neigh_idx - 1
                neigh_num += 1
                no_exterior_neighs = False
                continue
            # otherwise, check if this label already exists in our neighbors
            found = False
            for nidx, nlabel in enumerate(neigh_labels):
                if label == nlabel:
                    neigh_fluxes[nidx] += flux
            if not found:
                neigh_fluxes[neigh_num] = flux
                neigh_labels[neigh_num] = neigh_label
                neigh_num += 1
        neigh_nums[edge_idx] = neigh_num
        if no_exterior_neighs:
            only_interor_neighs[edge_idx] = True
        # normalize fluxes
        neigh_fluxes /= total_flux

    ###########################################################################
    # Edge assignments
    ###########################################################################
    # Now we have the fluxes (and some weights) at each edge point. We loop over
    # them from high to low and assign charges, volumes, and labels
    scratch_weights = np.zeros(len(charges), dtype=np.float64)
    approx_charges = charges.copy()
    all_weights = []
    all_labels = []
    for edge_idx, (fluxes, neighs, neigh_num, only_interior) in enumerate(
        zip(flux_array, neigh_array, neigh_nums, only_interor_neighs)
    ):
        sorted_idx = edge_sorted_indices[edge_idx]
        idx = sorted_indices[sorted_idx]
        charge = flat_charge[idx]

        current_labels = []
        current_weights = []
        # Loop over neighbors and calculate weights for this point
        for neigh_idx, (flux, label) in enumerate(zip(fluxes, neighs)):
            if neigh_idx == neigh_num:
                break
            # NOTE: I was looping over neigh_num here, but sometimes this caused a
            # crash. Maybe numba is trying to us prange even though I didn't ask it to?
            # for neigh_idx in range(neigh_num):
            #     flux = fluxes[neigh_idx]
            #     label = neighs[neigh_idx]
            if label >= 0:
                # This is actually a basin rather than another edge index.
                if scratch_weights[label] == 0.0:
                    current_labels.append(label)
                scratch_weights[label] += flux
                continue
            # otherwise, this is another edge index. Get its weight
            label = -label - 1  # convert back to actual neighbor index
            neigh_edge_idx = idx_to_edge[label]
            neigh_labels = all_labels[neigh_edge_idx]
            neigh_weights = all_weights[neigh_edge_idx]
            # loop over neighbors weights and add the portion assigned to this
            # voxel
            for label, weight in zip(neigh_labels, neigh_weights):
                # if there is no weight at this label yet, its new. Add it to our list
                if scratch_weights[label] == 0.0:
                    current_labels.append(label)
                scratch_weights[label] += flux * weight
                continue

        # Now loop over each label and assign charges, volumes, and labels
        best_label = -1
        best_weight = 0.0
        tied_labels = False
        tol = 1e-6  # for floating point errors
        for label in current_labels:
            weight = scratch_weights[label]
            charges[label] += charge * weight
            volumes[label] += weight
            if weight > best_weight + tol:  # greater than with a tolerance
                best_label = label
                best_weight = weight
                tied_labels = False
            elif weight > best_weight - tol:  # equal to with a tolerance
                tied_labels == True
            # add weight to current weights and reset scratch
            current_weights.append(weight)
            scratch_weights[label] = 0.0
        # add weights/labels for this point to our list
        all_weights.append(current_weights)
        all_labels.append(current_labels)

        # Now we want to assign our label. If there wasn't a tie in our labels,
        # we assign to highest weight
        if not tied_labels:
            labels[idx] = best_label
            approx_charges[best_label] += charge
        else:
            # we have a tie. We assign to the basin where the added charge will
            # most improve the approximate charge
            best_improvement = -1.0
            for label, weight in zip(current_labels, current_weights):
                if weight < best_weight - tol:
                    continue
                # calculate the difference from the current charge before and
                # after adding this point
                diff = approx_charges[label] - charges[label]
                before = abs(diff)
                after = abs(diff + charge)
                improvement = (before - after) / charges[label]
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_label = label
            labels[idx] = best_label
            approx_charges[best_label] += charge

    return (
        labels,
        charges,
        volumes,
        np.array(maxima_vox, dtype=np.int64),
    )


@njit(parallel=True, cache=True)
def sort_maxima_vox(
    maxima_vox,
    nx,
    ny,
    nz,
):
    flat_indices = np.zeros(len(maxima_vox), dtype=np.int64)
    for idx in prange(len(flat_indices)):
        i, j, k = maxima_vox[idx]
        flat_indices[idx] = coords_to_flat(i, j, k, nx, ny, nz)

    # sort flat indices from low to high
    sorted_indices = np.argsort(flat_indices)
    # sort maxima from lowest index to highest
    return maxima_vox[sorted_indices]


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
