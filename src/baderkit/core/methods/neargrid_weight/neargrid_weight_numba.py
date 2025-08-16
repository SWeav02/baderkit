# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from baderkit.core.methods.shared_numba import (
    get_best_neighbor,
    get_gradient_overdetermined,
    get_gradient_simple,
    wrap_point,
)


@njit(fastmath=True, cache=True)
def get_interior_basin_charges_and_volumes(
    data: NDArray[np.float64],
    labels: NDArray[np.int64],
    cell_volume: np.float64,
    maxima_num: np.int64,
    edge_mask: NDArray[np.bool_],
):
    nx, ny, nz = data.shape
    total_points = nx * ny * nz
    # create variables to store charges/volumes
    charges = np.zeros(maxima_num, dtype=np.float64)
    volumes = np.zeros(maxima_num, dtype=np.float64)
    vacuum_charge = 0.0
    vacuum_volume = 0.0
    # iterate in parallel over each voxel
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if edge_mask[i,j,k]:
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
    vacuum_volume = vacuum_volume * cell_volume / total_points
    vacuum_charge = vacuum_charge / total_points
    return charges, volumes, vacuum_charge, vacuum_volume

@njit(fastmath=True, cache=True)
def get_edge_charges_volumes(
    reference_data,
    charge_data,
    edge_indices,
    index_map,
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
    for unass_idx, (i,j,k) in enumerate(edge_indices):
        current_weight = []
        current_labels = []
        total_weight = 0.0
        value = reference_data[i,j,k]
        # loop over neighbors and assign weight
        for (si, sj, sk), frac in zip(neighbor_transforms, neighbor_weights):
            # get neighbor and wrap
            ni, nj, nk = wrap_point(i+si, j+sj, k+sk, nx, ny, nz)
            # skip if neighbor is lower
            neigh_value = reference_data[ni, nj, nk]
            if neigh_value <= value:
                continue
            # otherwise we get the labels and fraction of labels for
            # this voxel. First check if it is a single weight label
            neigh_index = index_map[ni,nj,nk]
            if neigh_index == -1:
                # this is a single weight voxel
                current_weight.append(frac)
                current_labels.append(labels[ni,nj,nk])
                total_weight += frac
                continue
            # otherwise, this is another multi weight label.
            neigh_weights = weight_lists[neigh_index]
            neigh_labels = label_lists[neigh_index]
            for label, weight in zip(neigh_labels, neigh_weights):
                frac_part = weight * frac
                current_weight.append(frac_part)
                current_labels.append(label)
                total_weight += frac_part
        # reduce labels and weights to unique, and normalize
        # TODO: The following two loops can probably be combined somehow.
        unique_labels = []
        unique_weights = []
        for li in range(len(current_labels)):
            label = current_labels[li]
            weight = current_weight[li]
            found = False
            for lj in range(len(unique_labels)):
                if unique_labels[lj] == label:
                    unique_weights[lj] += weight / total_weight
                    found = True
                    break
            if not found:
                unique_labels.append(label)
                unique_weights.append(weight / total_weight)
        # assign charge and volume
        charge = charge_data[i,j,k]
        for label, weight in zip(unique_labels, unique_weights):
            # update charge and volume
            charges[label] += weight * charge
            volumes[label] += weight

        # assign this weight row
        weight_lists.append(unique_weights)
        label_lists.append(unique_labels)
    return charges, volumes