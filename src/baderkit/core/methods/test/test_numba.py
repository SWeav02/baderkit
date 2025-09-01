# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from baderkit.core.methods.shared_numba import wrap_point


@njit(fastmath=True, cache=True)
def get_interior_basin_charges_and_volumes(
    data: NDArray[np.float64],
    labels: NDArray[np.int64],
    cell_volume: np.float64,
    maxima_num: np.int64,
    edge_mask: NDArray[np.bool_],
):
    nx, ny, nz = data.shape
    # create variables to store charges/volumes
    charges = np.zeros(maxima_num, dtype=np.float64)
    volumes = np.zeros(maxima_num, dtype=np.float64)
    vacuum_charge = 0.0
    vacuum_volume = 0.0
    # iterate in parallel over each voxel
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if edge_mask[i, j, k]:
                    continue
                charge = data[i, j, k]
                label = labels[i, j, k]
                if label < 0:
                    vacuum_charge += charge
                    vacuum_volume += 1
                else:
                    charges[label] += charge
                    volumes[label] += 1.0
    # calculate charge and volume for vacuum
    # NOTE: Don't normalize volumes/charges yet
    return charges, volumes, vacuum_charge, vacuum_volume

# @njit(cache=True)
# def get_possible_labels(
#         i, j, k,
#         nx, ny, nz,
#         labels,
#         neighbor_transforms,
#         ):
#     # list to store unique labels
#     labels = []
#     weights = []
#     for shift_idx in range(len(neighbor_transforms)):
#         si, sj, sk = neighbor_transforms[shift_idx]
#         # get upper and lower neighbors
#         ui, uj, uk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
#         li, lj, lk = wrap_point(i - si, j - sj, k - sk, nx, ny, nz)
#         # get labels
#         upper_label = labels[ui, uj, uk]
#         lower_label = labels[li, lj, lk]
#         if upper_label not in labels:
#             labels.append(upper_label)
#             weights.append(0.0)
#         if lower_label not in labels:
#             labels.append(lower_label)
#             weights.append(0.0)

#     return labels, weights

@njit(cache=True)
def get_labels_and_weights(
    i,j,k,
    reference_data,
    labels_array,
    first_neighbor_transforms,
    neighbor_transforms,
    first_neighbor_dists,
    neighbor_dists,
        ):
    nx, ny, nz = reference_data.shape
    # get value
    value = reference_data[i,j,k]
    # get tracker for total weight
    total_weight = 0.0
    # create lists to store weights and labels
    weights = []
    labels = []
    # loop over first neighbors
    for (si, sj, sk), dist in zip(first_neighbor_transforms, first_neighbor_dists):
        # get neighbor
        ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
        # get neigh value
        neigh_value = reference_data[ii,jj,kk]
        # check value
        if neigh_value <= value:
            continue
        weight = (neigh_value-value)/dist
        weights.append(weight)
        total_weight += weight
        labels.append(labels_array[ii,jj,kk])
    # reduce labels
    reduced_labels = []
    reduced_weights = []
    for li, label in enumerate(labels):
        found = False
        for ri, rlabel in enumerate(reduced_labels):
            if rlabel == label:
                found = True
                reduced_weights[ri] += weights[li]
                break
        if not found:
            reduced_labels.append(label)
            reduced_weights.append(weights[li])
    # loop over second neighbors
    for (si, sj, sk), dist in zip(neighbor_transforms, neighbor_dists):
        # get neighbor
        ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
        # get neigh value
        neigh_value = reference_data[ii,jj,kk]
        # check value and skip if lower
        if neigh_value <= value:
            continue
        # add to labels if exists
        neigh_label = labels_array[ii,jj,kk]
        for ri, rlabel in enumerate(reduced_labels):
            if neigh_label == rlabel:
                weight = (neigh_value-value)/dist
                reduced_weights[ri] += weight
                total_weight += weight
    # if the current total weight is 0.0, we have a local minimum
    # and return a full weight for the current label
    if total_weight == 0.0:
        reduced_labels.append(labels_array[i,j,k])
        reduced_weights.append(1.0)
        return reduced_labels, reduced_weights
    # normalize weights
    for idx in range(len(reduced_weights)):
        reduced_weights[idx] /= total_weight
    return reduced_labels, reduced_weights

@njit(parallel=True, cache=True)
def get_edge_charges_volumes(
    reference_data,
    charge_data,
    edge_indices,
    labels_array,
    charges,
    volumes,
    first_neighbor_transforms,
    neighbor_transforms,
    first_neighbor_dists,
    neighbor_dists,
):
    nx, ny, nz = reference_data.shape
    # create an array to store labels and weight
    label_array = np.full((len(edge_indices),len(first_neighbor_transforms)), -1, dtype=np.int64)
    weight_array = np.full((len(edge_indices),len(first_neighbor_transforms)), 0.0, dtype=np.float64)
    # loop over edge indices
    for idx in prange(len(edge_indices)):
        # get coordinates of grid point
        i, j, k = edge_indices[idx]
        # get weights/labels
        labels, weights = get_labels_and_weights(
            i,j,k,
            reference_data,
            labels_array,
            first_neighbor_transforms,
            neighbor_transforms,
            first_neighbor_dists,
            neighbor_dists,
            )
        # store weights/labels
        for weight_idx in range(len(labels)):
            label_array[idx, weight_idx] = labels[weight_idx]
            weight_array[idx, weight_idx] = weights[weight_idx]
    
    # Now loop over the weights/charges for our edge indices and sum charge
    for idx, (labels, weights) in enumerate(zip(label_array, weight_array)):
        i, j, k = edge_indices[idx]
        charge = charge_data[i, j, k]
        for label, weight in zip(labels, weights):
            # stop if we hit an empty label
            if label == -1:
                break
            # otherwise add the charge/volume
            charges[label] += weight * charge
            volumes[label] += weight

    return charges, volumes


# sorted using gradients. Converges about as fast as weight method.
# @njit(fastmath=True, cache=True)
# def get_edge_charges_volumes(
#     reference_data,
#     charge_data,
#     edge_indices,
#     sorted_edge_indices,
#     labels,
#     charges,
#     volumes,
#     neighbor_transforms,
#     neighbor_dists,
# ):
#     nx, ny, nz = reference_data.shape
#     # create lists to store weights/lists of previous points
#     weight_lists = []
#     label_lists = []
#     # loop over edge indices
#     for idx in range(len(edge_indices)):
#         # get coordinates of grid point
#         i, j, k = edge_indices[idx]
#         # create a list to store weights/labels in
#         current_labels, current_weights = get_possible_labels(
#             i,j,k,
#             nx,ny,nz,
#             labels,
#             neighbor_transforms
#             )
#         # create a counter for the total weight to normalize
#         # with later
#         total_weight = 0.0
#         # get the value at this data point
#         value = reference_data[i, j, k]

#         # loop over neighbors and assign weight
#         for (si, sj, sk), dist in zip(neighbor_transforms, neighbor_dists):
#             # get upper and lower neighbors
#             ui, uj, uk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
#             li, lj, lk = wrap_point(i - si, j - sj, k - sk, nx, ny, nz)
#             # get gradient to each neighbor
#             upper_value = reference_data[ui, uj, uk]
#             lower_value = reference_data[li, lj, lk]
#             upper_grad = (upper_value - value) / dist
#             lower_grad = (lower_value - value) / dist
#             # If both grads are lower than the current value, skip
#             if upper_grad <= 0.0 and lower_grad <= 0.0:
#                 continue
#             # if one grad is lower, change its neighbor indices to match the
#             # other side. We want to assign the absolute value of its gradient
#             # to the opposite neighbor
#             if upper_grad <= 0.0:
#                 ui = li
#                 uj = lj
#                 uk = lk
#             elif lower_grad <= 0.0:
#                 li = ui
#                 lj = uj
#                 lk = uk
            
#             for (ni,nj,nk), grad in zip(
#                     ((ui,uj,uk),(li,lj,lk)),
#                     (upper_grad,lower_grad)):
#                 # get a pointer to check if this neighbor is also an edge
#                 pointer = sorted_edge_indices[ni, nj, nk]
#                 # check if this neighbor is split
#                 if pointer != -1:
#                     # get the weights/labels for this neighbor
#                     neigh_weights = weight_lists[pointer]
#                     neigh_labels = label_lists[pointer]
#                     # we want to remove any labels that don't border our current
#                     # point.
#                     reduced_labels = []
#                     reduced_weights = []
#                     total_reduced_weight = 0.0
#                     for label, weight in zip(neigh_labels, neigh_weights):
#                         if label in current_labels:
#                             reduced_labels.append(label)
#                             reduced_weights.append(weight)
#                             total_reduced_weight += weight
#                     # Now add the weight to the appropriate label
#                     for label, weight in zip(reduced_labels, reduced_weights):
#                         for weight_idx, clabel in enumerate(current_labels):
#                             if label == clabel:
#                                 added_weight = abs(grad * weight / total_reduced_weight)
#                                 current_weights[weight_idx] += added_weight
#                                 total_weight += added_weight
#                                 break
#                 else:
#                     # we just add the total grad to the appropriate label
#                     label = labels[ni,nj,nk]
#                     for idx, clabel in enumerate(current_labels):
#                         if label == clabel:
#                             current_weights[idx] += abs(grad)
#                             total_weight += abs(grad)

#         # get the charge at this point
#         charge = charge_data[i, j, k]
        
#         # check for the case that there are no weights. This could happen
#         # at a local minimum
#         if total_weight == 0.0:
#             label = labels[i, j, k]
#             charges[label] += charge
#             volumes[label] += 1.0
#             weight_lists.append([1.0])
#             label_lists.append([label])
#             continue
        
#         # normalize weights
#         for weight_idx in range(len(current_weights)):
#             current_weights[weight_idx] /= total_weight
        
#         # assign charge and volume
#         for label, weight in zip(current_labels, current_weights):
#             # update charge and volume
#             charges[label] += weight * charge
#             volumes[label] += weight
#         weight_lists.append(current_weights)
#         label_lists.append(current_labels)
#     return charges, volumes

# ! Unsorted using gradients. Converges faster than neargrid-weight
# @njit(fastmath=True, cache=True)
# def get_edge_charges_volumes(
#     reference_data,
#     charge_data,
#     edge_indices,
#     labels,
#     charges,
#     volumes,
#     neighbor_transforms,
#     neighbor_dists,
# ):
#     nx, ny, nz = reference_data.shape
#     # loop over edge indices
#     for idx in range(len(edge_indices)):
#         # get coordinates of grid point
#         i, j, k = edge_indices[idx]
#         # create a list to store weights/labels in
#         current_weights = []
#         current_labels = []
#         # create a counter for the total weight to normalize
#         # with later
#         total_weight = 0.0
#         # get the value at this data point
#         value = reference_data[i, j, k]
#         # loop over neighbors and assign weight
#         for (si, sj, sk), dist in zip(neighbor_transforms, neighbor_dists):
#             # get upper and lower neighbors
#             ui, uj, uk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
#             li, lj, lk = wrap_point(i - si, j - sj, k - sk, nx, ny, nz)
#             # get gradient to each neighbor
#             upper_value = reference_data[ui, uj, uk]
#             lower_value = reference_data[li, lj, lk]
#             upper_grad = (upper_value - value) / dist
#             lower_grad = (lower_value - value) / dist
#             # If both grads are lower than the current value, skip
#             if upper_grad <= 0.0 and lower_grad <= 0.0:
#                 continue
#             # If only one is lower, use only the higher neighbors label
#             if upper_grad <= 0.0:
#                 upper_label = labels[li, lj, lk]
#                 lower_label = upper_label
#             elif lower_grad <= 0.0:
#                 upper_label = labels[ui, uj, uk]
#                 lower_label = upper_label
#             # otherwise, use the actual labels
#             else:
#                 upper_label = labels[ui, uj, uk]
#                 lower_label = labels[li, lj, lk]
#             # add both sides weights/labels
#             current_weights.append(abs(upper_grad))
#             current_weights.append(abs(lower_grad))
#             current_labels.append(upper_label)
#             current_labels.append(lower_label)
#             # update total weight
#             total_weight += abs(upper_grad) + abs(lower_grad)

#         # check for the case that there are no weights. This could happen
#         # at a local minimum
#         if len(current_labels) == 0:
#             label = labels[i, j, k]
#             charge = charge_data[i, j, k]
#             charges[label] += charge
#             volumes[label] += 1.0
#             continue

#         # normalize weights
#         for weight_idx in range(len(current_weights)):
#             current_weights[weight_idx] /= total_weight

#         charge = charge_data[i, j, k]
#         for label, weight in zip(current_labels, current_weights):
#             # update charge and volume
#             charges[label] += weight * charge
#             volumes[label] += weight
#     return charges, volumes

# ! Sorted using weights. Result is basically the weight method.
# @njit(fastmath=True, cache=True)
# def get_edge_charges_volumes(
#     reference_data,
#     charge_data,
#     edge_indices,
#     sorted_edge_indices,
#     labels,
#     charges,
#     volumes,
#     neighbor_transforms,
#     neighbor_weights,
# ):
#     nx, ny, nz = reference_data.shape
#     # create list to store weights
#     weight_lists = []
#     label_lists = []
#     # loop over edge indices
#     for idx in range(len(edge_indices)):
#         # get coordinates of grid point
#         i, j, k = edge_indices[idx]
#         # create a list to store weights/labels in
#         current_weights = []
#         current_labels = []
#         # create a counter for the total weight to normalize
#         # with later
#         total_weight = 0.0
#         # get the value at this data point
#         value = reference_data[i, j, k]
#         # loop over neighbors and assign weight
#         for (si, sj, sk), frac in zip(neighbor_transforms, neighbor_weights):
#             # get neighbor and wrap
#             ni, nj, nk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
#             # skip if neighbor is lower
#             neigh_value = reference_data[ni, nj, nk]
#             if neigh_value <= value:
#                 continue

#             # check if this voxel has already been split
#             neigh_pointer = sorted_edge_indices[ni,nj,nk]
#             if neigh_pointer != -1:
#                 neigh_weights = weight_lists[neigh_pointer]
#                 neigh_labels = label_lists[neigh_pointer]
#                 for label, weight in zip(neigh_labels, neigh_weights):
#                     flux = weight * (neigh_value - value) * frac
#                     current_weights.append(flux)
#                     current_labels.append(label)
#                     total_weight += flux
#                 continue
#             # otherwise, add the portion of this voxel moving to this
#             # label
#             flux = (neigh_value - value) * frac
#             current_weights.append(flux)
#             current_labels.append(labels[ni, nj, nk])
#             total_weight += flux

#         # normalize the weights
#         for weight_idx in range(len(current_weights)):
#             current_weights[weight_idx] /= total_weight

#         # assign charge and volume and get unique labels
#         unique_weights = []
#         unique_labels = []
#         charge = charge_data[i, j, k]
#         for label, weight in zip(current_labels, current_weights):
#             # update charge and volume
#             charges[label] += weight * charge
#             volumes[label] += weight
#             # update unique lists
#             found = False
#             for i, ulabel in enumerate(unique_labels):
#                 if label == ulabel:
#                     unique_weights[i] += weight
#                     found = True
#             if not found:
#                 unique_weights.append(weight)
#                 unique_labels.append(label)
#         weight_lists.append(unique_weights)
#         label_lists.append(unique_labels)

#     return charges, volumes
