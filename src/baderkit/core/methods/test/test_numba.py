# -*- coding: utf-8 -*-

import numpy as np
from numba import njit
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
    # total_points = nx * ny * nz
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
    # volumes = volumes * cell_volume / total_points
    # charges = charges / total_points
    return charges, volumes, vacuum_charge, vacuum_volume

# @njit(fastmath=True, cache=True)
def get_edge_charges_volumes(
    reference_data,
    charge_data,
    edge_indices,
    sorted_edge_indices,
    labels,
    charges,
    volumes,
    neighbor_transforms,
    neighbor_weights,
):
    nx, ny, nz = reference_data.shape
    # create list to store weights
    weight_lists = []
    label_lists = []
    # loop over edge indices
    for idx in range(len(edge_indices)):
        # get coordinates of grid point
        i, j, k = edge_indices[idx]
        # create a list to store weights/labels in
        current_weights = []
        current_labels = []
        # create a counter for the total weight to normalize
        # with later
        total_weight = 0.0
        # get the value at this data point
        value = reference_data[i, j, k]
        # loop over neighbors and assign weight
        for (si, sj, sk), frac in zip(neighbor_transforms, neighbor_weights):
            # get neighbor and wrap
            ni, nj, nk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
            # skip if neighbor is lower
            neigh_value = reference_data[ni, nj, nk]
            if neigh_value <= value:
                continue
            
            # check if this voxel has already been split
            neigh_pointer = sorted_edge_indices[ni,nj,nk]
            if neigh_pointer != -1:
                neigh_weights = weight_lists[neigh_pointer]
                neigh_labels = label_lists[neigh_pointer]
                for label, weight in zip(neigh_labels, neigh_weights):
                    flux = weight * (neigh_value - value) * frac
                    current_weights.append(flux)
                    current_labels.append(label)
                    total_weight += flux
                continue
            # otherwise, add the portion of this voxel moving to this
            # label
            flux = (neigh_value - value) * frac
            current_weights.append(flux)
            current_labels.append(labels[ni, nj, nk])
            total_weight += flux

        # normalize the weights
        for weight_idx in range(len(current_weights)):
            current_weights[weight_idx] /= total_weight

        # assign charge and volume and get unique labels
        unique_weights = []
        unique_labels = []
        charge = charge_data[i, j, k]
        for label, weight in zip(current_labels, current_weights):
            # update charge and volume
            charges[label] += weight * charge
            volumes[label] += weight
            # update unique lists
            found = False
            for i, ulabel in enumerate(unique_labels):
                if label == ulabel:
                    unique_weights[i] += weight
                    found = True
            if not found:
                unique_weights.append(weight)
                unique_labels.append(label)
        weight_lists.append(unique_weights)
        label_lists.append(unique_labels)

    return charges, volumes
# @njit(fastmath=True, cache=True)
# def get_edge_charges_volumes(
#     reference_data,
#     charge_data,
#     edge_indices,
#     labels,
#     charges,
#     volumes,
#     neighbor_transforms,
#     neighbor_weights,
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
#         for (si, sj, sk), frac in zip(neighbor_transforms, neighbor_weights):
#             # get neighbor and wrap
#             ni, nj, nk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
#             # skip if neighbor is lower
#             neigh_value = reference_data[ni, nj, nk]
#             if neigh_value <= value:
#                 continue
#             # otherwise, add the portion of this voxel moving to this
#             # label
#             flux = (neigh_value - value) * frac
#             current_weights.append(flux)
#             current_labels.append(labels[ni, nj, nk])
#             total_weight += flux

#         # normalize the weighs
#         for weight_idx in range(len(current_weights)):
#             current_weights[weight_idx] /= total_weight

#         # assign charge and volume
#         charge = charge_data[i, j, k]
#         for label, weight in zip(current_labels, current_weights):
#             # update charge and volume
#             charges[label] += weight * charge
#             volumes[label] += weight

#     return charges, volumes
