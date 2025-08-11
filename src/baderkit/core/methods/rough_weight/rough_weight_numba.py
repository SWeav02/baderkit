# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from baderkit.core.methods.shared_numba import get_best_neighbor, wrap_point

# @njit(parallel=True, cache=True)
# def assign_charge_volume_label(
#         neigh_labels,
#         neigh_weights,
#         basin_weights,
#         labels,
#         charge,
#         basin_charges,
#         basin_volumes,
#         ):
#     basin_weights[:] = 0.0
#     n = len(neigh_labels)

#     # Accumulate weights per label
#     for idx in range(n):
#         label = neigh_labels[idx]
#         basin_weights[label] += neigh_weights[idx]

#     # normalize to get fractions
#     basin_weights /= basin_weights.sum()

#     # assign charge and volume
#     for i in prange(len(basin_weights)):
#         basin_charges[i] += basin_weights[i] * charge
#         basin_volumes[i] += basin_weights[i]

#     # assign label
#     max_val = basin_weights.max()
#     for weight in basin_weights:
#         if weight == max_val:
#             labels

# # Find best label and detect ties in one pass
# best_label = -1
# best_weight = -1.0
# tie_found = False
# eps = 1e-12  # tolerance for float comparison

# for label in range(len(basin_weights)):
#     weight = basin_weights[label]
#     if weight > best_weight + eps:
#         best_weight = weight
#         best_label = label
#         tie_found = False
#     elif abs(weight - best_weight) <= eps and weight > 0:
#         # Tie detected
#         tie_found = True

# if tie_found:
#     return -1
# else:
#     return best_label


@njit(cache=True)
def get_rough_weight_labels(
    data: NDArray[np.float64],
    charge_data: NDArray[np.float64],
    sorted_voxel_coords: NDArray[np.int64],
    neighbor_transforms: NDArray[np.int64],
    neighbor_weights: NDArray[np.float64],
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
    sorted_voxel_coords : NDArray[np.int64]
        A Nx3 array where each entry represents the voxel coordinates of the
        point. This must be sorted from highest value to lowest.
    neighbor_transforms : NDArray[np.int64]
        The transformations from each voxel to its voronoi neighbors.
    neighbor_weights : NDArray[np.float64]
        The weight of each neighbor calculated from the facet area and distance
    all_neighbor_transforms : NDArray[np.int64]
        The transformations from each voxel to each of its 26 neighbors.
    all_neighbor_dists : NDArray[np.int64]
        The distance from each voxel to each of its 26 neighbors.

    Returns
    -------
    labels : NDArray[np.int64]
        A 3D array where each entry represents the basin the voxel belongs to.

    maxima_mask : NDArray[bool]
        A 1D array of length N that is True where the sorted voxel indices are
        a maximum

    """
    nx, ny, nz = data.shape
    # create an empty array to store labels
    labels = np.full(data.shape, -1, dtype=np.int64)
    # create a mask for the location of maxima
    maxima_mask = np.zeros(data.shape, dtype=np.bool_)
    # create counters for maxima
    max_num = 0
    # create a scratch array for storing weights
    basin_weights = np.zeros(1, dtype=np.float64)
    # create lists for storing charges and weights
    basin_charges = []
    basin_volumes = []
    # Loop over each voxel in parallel (except the vacuum points)
    for i, j, k in sorted_voxel_coords:
        # get the initial value for this point
        base_value = data[i, j, k]
        # create lists to store the labels and weights
        neigh_labels = []
        neigh_weights = []
        # iterate over each neighbor sharing a voronoi facet
        for (si, sj, sk), weight in zip(neighbor_transforms, neighbor_weights):
            # get neighbor and wrap
            ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
            # if neighbor doesn't have an assignment, continue on
            neigh_label = labels[ii, jj, kk]
            neigh_data = data[ii, jj, kk]
            if neigh_label == -1 or neigh_data <= base_value:
                continue
            # add this neighbors label and weight to our lists
            neigh_labels.append(neigh_label)
            neigh_weights.append((neigh_data - base_value) * weight)

        # if we have no list entries we might be at a maximum. We use the ongrid
        # method to assign the point
        if len(neigh_labels) == 0:
            shift, (ni, nj, nk), is_max = get_best_neighbor(
                data=data,
                i=i,
                j=j,
                k=k,
                neighbor_transforms=all_neighbor_transforms,
                neighbor_dists=all_neighbor_dists,
            )
            # if this is a maximum, set the row to 0, note its a max, and continue
            if is_max:
                # note this is a max
                maxima_mask[i, j, k] = True
                # set the label
                labels[i, j, k] = max_num
                max_num += 1
                # update our scratch array
                basin_weights = np.zeros(max_num, dtype=np.float64)
                # add charge and volume to assignment array
                basin_charges.append(charge_data[i, j, k])
                basin_volumes.append(1.0)
                continue
            # otherwise, we give the point the same label as its highest neighbor
            labels[i, j, k] = labels[ni, nj, nk]
            continue

        # otherwise we assign charge, volumes, and label by weight.
        # reset label sums

        basin_weights[:] = 0.0
        # Accumulate weights per label
        for label, weight in zip(neigh_labels, neigh_weights):
            basin_weights[label] += weight
        # normalize to get fractions
        basin_weights /= basin_weights.sum()
        # assign charge, volume, and label
        max_val = basin_weights.max()
        label_assigned = False
        for basin, weight in enumerate(basin_weights):
            basin_charges[basin] += weight * charge_data[i, j, k]
            basin_volumes[basin] += weight
            if weight == max_val and not label_assigned:
                labels[i, j, k] = basin
                label_assigned = True

    return (
        labels,
        maxima_mask,
        np.array(basin_charges, dtype=np.float64),
        np.array(basin_volumes, dtype=np.float64),
    )
