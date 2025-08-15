# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from baderkit.core.methods.shared_numba import get_best_neighbor, wrap_point


@njit(parallel=True, cache=True)
def get_neighbor_flux(
    data: NDArray[np.float64],
    sorted_voxel_coords: NDArray[np.int64],
    voxel_indices: NDArray[np.int64],
    neighbor_transforms: NDArray[np.int64],
    neighbor_dists: NDArray[np.float64],
    facet_areas: NDArray[np.float64],
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
    voxel_indices : NDArray[np.int64]
        A 3D array where each entry is the flat voxel index of each
        point.
    neighbor_transforms : NDArray[np.int64]
        The transformations from each voxel to its neighbors.
    neighbor_dists : NDArray[np.float64]
        The distance to each neighboring voxel
    facet_areas : NDArray[np.float64]
        The area of the voronoi facet between the voxel and each neighbor

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
    # TODO: Is it worth moving to lists? This often isn't terrible sparse so it
    # may be ok, but it could require a lot of memory for very large grids.
    flux_array = np.zeros(
        (len(sorted_voxel_coords), len(neighbor_transforms)), dtype=np.float64
    )
    neigh_array = np.full(flux_array.shape, -1, dtype=np.int64)
    # calculate the area/dist for each neighbor to avoid repeat calculation
    neighbor_area_over_dist = facet_areas / neighbor_dists
    # create a mask for the location of maxima
    maxima_mask = np.zeros(len(sorted_voxel_coords), dtype=np.bool_)
    # Loop over each voxel in parallel (except the vacuum points)
    for coord_index in prange(len(sorted_voxel_coords)):
        i, j, k = sorted_voxel_coords[coord_index]
        # get the initial value
        base_value = data[i, j, k]
        # iterate over each neighbor sharing a voronoi facet
        for shift_index, ((si, sj, sk), area_dist) in enumerate(
            zip(neighbor_transforms, neighbor_area_over_dist)
        ):
            # loop
            ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
            # get the neighbors value
            neigh_value = data[ii, jj, kk]
            # if this value is below the current points value, continue
            if neigh_value <= base_value:
                continue
            # calculate the flux flowing to this voxel
            flux = (neigh_value - base_value) * area_dist
            # assign flux
            flux_array[coord_index, shift_index] = flux
            # add this neighbor label
            neigh_array[coord_index, shift_index] = voxel_indices[ii, jj, kk]

        # normalize flux row to 1
        row = flux_array[coord_index]
        row_sum = row.sum()
        if row_sum == 0.0:
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
            neigh_array[coord_index, 0] = voxel_indices[ni, nj, nk]
            continue

        flux_array[coord_index] = row / row_sum

    return flux_array, neigh_array, maxima_mask


@njit(fastmath=True, cache=True)
def get_single_weight_voxels(
    neigh_indices_array: NDArray[np.int64],
    sorted_voxel_coords: NDArray[np.int64],
    data: NDArray[np.float64],
    maxima_num: np.int64,
    sorted_flat_charge_data: NDArray[np.float64],
    voxel_volume: np.float64,
    labels: NDArray[np.int64] = None,
):
    """
    Loops over voxels to find any that have exaclty one weight. We store
    these in a single array the size of the labels to reduce space

    Parameters
    ----------
    neigh_indices_array : NDArray[np.int64]
        A 2D array of shape x*y*x by len(neighbor_transforms) where each entry
        f(i, j) is the index of the neighbor from the voxel at index i to the
        neighbor at transform neighbor_transforms[j]
    sorted_voxel_coords : NDArray[np.int64]
        A Nx3 array where each entry represents the voxel coordinates of the
        point. This must be sorted from highest value to lowest.
    data : NDArray[np.float64]
        A 3D grid of values for each point.
    maxima_num : np.int64
        The number of local maxima in the grid
    sorted_flat_charge_data : NDArray[np.float64]
        The charge density at each value sorted highest to lowest.
    voxel_volume : np.float64
        The volume of a single voxel
    labels : NDArray[np.int64], optional
        A 3D array of preassigned labels.

    Returns
    -------
    labels : NDArray[int]
        A 3D array where each entry represents the basin the voxel belongs to.
        If the basin is split to multiple neighbors it is assigned a value of
        0
    unassigned_mask : NDArray[bool]
        A 1D array of bools representing which voxel indices are not assigned
    charge_array : NDArray[float]
        The charge on each basin that has been assigned so far
    volume_array : NDArray[float]
        The volume on each basin that has been assigned so far

    """
    # get the length of our voxel array and create an empty array for storing
    # data as we collect it
    n_voxels = neigh_indices_array.shape[0]
    # create labels array
    if labels is None:
        labels = np.full(data.shape, -1, dtype=np.int64)
    # create an array to note which of our sorted indices are unassigned
    unassigned_mask = np.zeros(n_voxels, dtype=np.bool_)
    # create arrays for storing volumes and charges
    charge_array = np.zeros(maxima_num, dtype=np.float64)
    volume_array = np.zeros(maxima_num, dtype=np.float64)
    # create counter for maxima
    # maxima = 0
    # loop over voxels
    for vox_idx in range(n_voxels):
        i, j, k = sorted_voxel_coords[vox_idx]
        charge = sorted_flat_charge_data[vox_idx]
        # We assign maxima before this step, so we check if this point has a
        # label already
        label = labels[i, j, k]
        if label != -1:
            # this is a maximum or reduced maximum. assign all of its charge
            charge_array[label] += charge
            volume_array[label] += voxel_volume
            continue
        neighbors = neigh_indices_array[vox_idx]
        # otherwise we check each neighbor and check its label
        current_label = -1
        label_num = 0
        for neigh in neighbors:
            if neigh == -1:
                # This isn't a valid neighbor so we skip it
                continue
            # get this neighbors label
            ni, nj, nk = sorted_voxel_coords[neigh]
            neigh_label = labels[ni, nj, nk]
            # If the label is -1, this neighbor is unassigned due to being split
            # to more than one of it's own neighbors. Therefore, the current voxel
            # also should be split.
            if neigh_label == -1:
                label_num = 2
                break
            # If the label exists and is new, update our label count
            if neigh_label != current_label:
                current_label = neigh_label
                label_num += 1
                # if we have more than one label, immediately break
                if label_num > 1:
                    break
        # if we only have one label, update this point's label
        if label_num == 1:
            # i, j, k = sorted_voxel_coords[vox_idx]
            labels[i, j, k] = current_label
            # assign charge and volume
            charge_array[current_label] += charge
            volume_array[current_label] += voxel_volume
        # otherwise, mark this as a point that has a split assignment.
        else:
            unassigned_mask[vox_idx] = True
    return labels, unassigned_mask, charge_array, volume_array


@njit(fastmath=True, cache=True)
def get_multi_weight_voxels(
    flux_array: NDArray[np.float64],
    neigh_indices_array: NDArray[np.int64],
    labels: NDArray[np.int64],
    unass_to_vox_pointer: NDArray[np.int64],
    vox_to_unass_pointer: NDArray[np.int64],
    sorted_voxel_coords: NDArray[np.int64],
    charge_array: NDArray[np.float64],
    volume_array: NDArray[np.float64],
    sorted_flat_charge_data: NDArray[np.float64],
    voxel_volume: np.float64,
    maxima_num: np.int64,
):
    """
    Assigns charge and volume from each voxel that has multiple weights to each
    of the basins it is split to. The returned labels represent the basin
    that has the largest share of each split voxel.

    Parameters
    ----------
    flux_array : NDArray[np.float64]
        A 2D array of shape x*y*x by len(neighbor_transforms) where each entry
        f(i, j) is the flux flowing from the voxel at index i to its neighbor
        at transform neighbor_transforms[j]
    neigh_indices_array : NDArray[np.int64]
        A 2D array of shape x*y*x by len(neighbor_transforms) where each entry
        f(i, j) is the index of the neighbor from the voxel at index i to the
        neighbor at transform neighbor_transforms[j]
    labels : NDArray[np.int64]
        A 3D array where each entry represents the basin the voxel belongs to.
        If the basin is split to multiple neighbors it is assigned a value of
        0.
    unass_to_vox_pointer : NDArray[np.int64]
        An array pointing each entry in the list of unassigned voxels to their
        original voxel index
    vox_to_unass_pointer : NDArray[np.int64]
        An array pointing each voxel in its original voxel index to its unassigned
        index if it exists.
    sorted_voxel_coords : NDArray[np.int64]
        A Nx3 array where each entry represents the voxel coordinates of the
        point. This must be sorted from highest value to lowest.
    charge_array : NDArray[np.float64]
        The charge on each basin that has been assigned so far
    volume_array : NDArray[np.float64]
        The volume on each basin that has been assigned so far
    sorted_flat_charge_data : NDArray[np.float64]
        The charge density at each value sorted highest to lowest.
    voxel_volume : np.float64
        The volume of a single voxel
    maxima_num : np.int64
        The number of local maxima in the grid

    Returns
    -------
    new_labels : NDArray[np.int64]
        The updated labels.
    charge_array : NDArray[np.float64]
        The final charge on each basin
    volume_array : NDArray[np.float64]
        The final volume of each basin

    """
    # create list to store weights
    weight_lists = []
    label_lists = []
    # create a new labels array to store updated labels
    new_labels = labels.copy()
    for unass_idx, vox_idx in enumerate(unass_to_vox_pointer):
        current_weight = []
        current_labels = []
        # get the important neighbors and their fraction of flow from this vox
        neighbors = neigh_indices_array[vox_idx]
        fracs = flux_array[vox_idx]
        for neighbor, frac in zip(neighbors, fracs):
            # skip if no neighbor
            if neighbor < 0:
                continue
            # otherwise we get the labels and fraction of labels for
            # this voxel. First check if it is a single weight label
            ni, nj, nk = sorted_voxel_coords[neighbor]
            label = labels[ni, nj, nk]
            if label != -1:
                current_weight.append(frac)
                current_labels.append(label)
                continue
            # otherwise, this is another multi weight label.
            neigh_unass_idx = vox_to_unass_pointer[neighbor]
            neigh_weights = weight_lists[neigh_unass_idx]
            neigh_labels = label_lists[neigh_unass_idx]
            for label, weight in zip(neigh_labels, neigh_weights):
                current_weight.append(weight * frac)
                current_labels.append(label)
        # reduce labels and weights to unique
        # TODO: The following two loops can probably be combined somehow.
        unique_labels = []
        unique_weights = []
        for i in range(len(current_labels)):
            label = current_labels[i]
            weight = current_weight[i]
            found = False
            for j in range(len(unique_labels)):
                if unique_labels[j] == label:
                    unique_weights[j] += weight
                    found = True
                    break
            if not found:
                unique_labels.append(label)
                unique_weights.append(weight)
        # assign label, charge, and volume
        best_weight = 0.0
        best_label = -1
        charge = sorted_flat_charge_data[vox_idx]
        for label, weight in zip(unique_labels, unique_weights):
            # update charge and volume
            charge_array[label] += weight * charge
            volume_array[label] += weight * voxel_volume
            if weight >= best_weight:
                best_weight = weight
                best_label = label
        # update label
        i, j, k = sorted_voxel_coords[vox_idx]
        new_labels[i, j, k] = best_label
        # assign this weight row
        weight_lists.append(unique_weights)
        label_lists.append(unique_labels)
    return new_labels, charge_array, volume_array
