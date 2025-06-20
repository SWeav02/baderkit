# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, prange, types
from numpy.typing import NDArray

###############################################################################
# General methods
###############################################################################


@njit(parallel=True, cache=True)
def get_maxima(
    data: NDArray[np.float64],
    neighbor_transforms: NDArray[np.int64],
) -> NDArray[np.bool_]:
    """
    Finds the local maxima in an array.

    Parameters
    ----------
    data : NDArray[np.float64]
        The data to find the maxima in. Must be 3D.
    neighbor_transforms : NDArray[np.int64]
        The transformations to the neighbors to consider while finding maxima.

    Returns
    -------
    NDArray[np.bool_]
        An array of the same shape as the original data that is True where maxima
        are located
    """
    nx, ny, nz = data.shape
    maxima_mask = np.zeros(data.shape, dtype=np.bool_)
    # iterate in parallel over each voxel
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # get the value for this voxel
                base_value = data[i, j, k]
                # iterate over the transformations to neighboring voxels
                is_max = True
                for shift_index, shift in enumerate(neighbor_transforms):
                    ii = (i + shift[0]) % nx  # Loop around box
                    jj = (j + shift[1]) % ny
                    kk = (k + shift[2]) % nz
                    # get the neighbors value
                    neigh_value = data[ii, jj, kk]
                    # If its larger than the base value this isn't a maximum
                    if neigh_value > base_value:
                        is_max = False
                        break
                if is_max:
                    maxima_mask[i, j, k] = True
    return maxima_mask


@njit(parallel=True, cache=True)
def get_edges(
    labeled_array: NDArray[np.int64],
    neighbor_transforms: NDArray[np.int64],
):
    """
    In a 3D array of labeled voxels, finds the voxels that neighbor at
    least one voxel with a different label.

    Parameters
    ----------
    labeled_array : NDArray[np.int64]
        A 3D array where each entry represents the basin label of the point.
    neighbor_transforms : NDArray[np.int64]
        The transformations from each voxel to its neighbors.

    Returns
    -------
    edges : NDArray[np.bool_]
        A mask with the same shape as the input grid that is True at points
        on basin edges.

    """
    nx, ny, nz = labeled_array.shape
    # create 3D array to store edges
    edges = np.zeros_like(labeled_array, dtype=np.bool_)
    # loop over each voxel in parallel
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # get this voxels label
                label = labeled_array[i, j, k]
                # iterate over the neighboring voxels
                for shift_index, shift in enumerate(neighbor_transforms):
                    ii = (i + shift[0]) % nx  # Loop around box
                    jj = (j + shift[1]) % ny
                    kk = (k + shift[2]) % nz
                    # get neighbors label
                    neigh_label = labeled_array[ii, jj, kk]
                    # if any label is different, the current voxel is an edge.
                    # Note this in our edge array and break
                    if neigh_label != label:
                        edges[i, j, k] = True
                        break
    return edges


@njit(parallel=True, cache=True)
def propagate_edges(
    edge_mask: NDArray[np.bool_],
    neighbor_transforms: NDArray[np.int64],
):
    """
    Expand the True values of a grid to their neighbors.

    Parameters
    ----------
    edge_mask : NDArray[np.bool_]
        A 3D array of bools.
    neighbor_transforms : NDArray[np.int64]
        The transformations from each voxel to its neighbors.

    Returns
    -------
    new_edge_mask : NDArray[np.bool_]
        A 3D array of bools.

    """
    new_edge_mask = np.zeros_like(edge_mask, dtype=np.bool_)
    nx, ny, nz = edge_mask.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                if not edge_mask[i, j, k]:
                    # skip voxels that aren't edges
                    continue
                # set as an edge
                new_edge_mask[i, j, k] = True
                for shift in neighbor_transforms:
                    ii = (i + shift[0]) % nx
                    jj = (j + shift[1]) % ny
                    kk = (k + shift[2]) % nz
                    # mark neighbor as an edge
                    new_edge_mask[ii, jj, kk] = True
    return new_edge_mask


@njit(parallel=True, cache=True)
def unmark_isolated_voxels(
    edge_mask: NDArray[np.bool_],
    neighbor_transforms: NDArray[np.int64],
):
    """
    Switch any True entries in a bool mask to False if none of their neighbors
    are also True.

    Parameters
    ----------
    edge_mask : NDArray[np.bool_]
        A 3D array of bools.
    neighbor_transforms : NDArray[np.int64]
        The transformations from each voxel to its neighbors.

    Returns
    -------
    edge_mask : NDArray[np.bool_]
        A 3D array of bools

    """
    nx, ny, nz = edge_mask.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                if not edge_mask[i, j, k]:
                    continue  # Only unmark candidates

                found_edge_neighbor = False
                for shift in neighbor_transforms:
                    ii = (i + shift[0]) % nx
                    jj = (j + shift[1]) % ny
                    kk = (k + shift[2]) % nz
                    if edge_mask[ii, jj, kk]:
                        found_edge_neighbor = True
                        break

                if not found_edge_neighbor:
                    edge_mask[i, j, k] = False
    return edge_mask


@njit(parallel=True, cache=True)
def get_neighbor_diffs(
    data: NDArray[np.float64],
    initial_labels: NDArray[np.int64],
    neighbor_transforms: NDArray[np.int64],
):
    """
    Gets the difference in value between each voxel and its neighbors.
    Does not weight by distance.

    Parameters
    ----------
    data : NDArray[np.float64]
        The data for each voxel.
    initial_labels : NDArray[np.int64]
        A 3D grid representing the flat indices for each voxel.
    neighbor_transforms : NDArray[np.int64]
        The transformations from each voxel to its neighbors.

    Returns
    -------
    diffs : NDArray[float]
        A 2D array of shape x*y*x by len(neighbor_transforms) where each entry
        i, j correspondin to the voxel index and transformation index respectively

    """
    nx, ny, nz = data.shape
    # create empty array for diffs. This is a 2D array with with entries i, j
    # corresponding to the voxel index and transformation index respectively
    diffs = np.zeros((nx * ny * nz, len(neighbor_transforms)), dtype=np.float64)
    # iterate in parallel over each voxel
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # get the value for this voxel as well as its index number
                base_value = data[i, j, k]
                index = initial_labels[i, j, k]
                # iterate over the transformations to neighboring voxels
                for shift_index, shift in enumerate(neighbor_transforms):
                    ii = (i + shift[0]) % nx  # Loop around box
                    jj = (j + shift[1]) % ny
                    kk = (k + shift[2]) % nz
                    # get the neighbors value, the difference, and store in the
                    # diffs array
                    neigh_value = data[ii, jj, kk]
                    diff = neigh_value - base_value
                    diffs[index, shift_index] = diff
    return diffs


###############################################################################
# Functions for on-grid method
###############################################################################
@njit(parallel=True, cache=True)
def get_steepest_pointers(
    data: NDArray[np.float64],
    initial_labels: NDArray[np.int64],
    neighbor_transforms: NDArray[np.int64],
    neighbor_dists: NDArray[np.int64],
):
    """
    For each voxel in a 3D grid of data, finds the index of the neighboring voxel with
    the highest value, weighted by distance.

    Parameters
    ----------
    data : NDArray[np.float64]
        A 3D grid of values for each point.
    initial_labels : NDArray[np.int64]
        A 3D array where each entry represents the basin label of the point.
    neighbor_transforms : NDArray[np.int64]
        The transformations from each voxel to its neighbors.
    neighbor_dists : NDArray[np.int64]
        The distance to each neighboring voxel

    Returns
    -------
    best_label : NDArray[np.int64]
        A 3D array where each entry is the index of the neighbor that had the
        greatest increase in value.

    """
    nx, ny, nz = data.shape
    # create array to store the label of the neighboring voxel with the greatest
    # elf value
    # best_diff  = np.zeros_like(data)
    best_label = initial_labels.copy()
    # loop over each voxel in parallel
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # get the elf value and initial label for this voxel. This defaults
                # to the voxel pointing to itself
                base = data[i, j, k]
                best = 0.0
                label = initial_labels[i, j, k]
                # For each neighbor get the difference in value and if its better
                # than any previous, replace the current best
                for shift, dist in zip(neighbor_transforms, neighbor_dists):
                    ii = (i + shift[0]) % nx  # Loop around box
                    jj = (j + shift[1]) % ny
                    kk = (k + shift[2]) % nz
                    # calculate the difference in value taking into account distance
                    diff = (data[ii, jj, kk] - base) / dist
                    # if better than the current best, note the best and the
                    # current label
                    if diff > best:
                        best = diff
                        label = initial_labels[ii, jj, kk]
                # We've finished our loop. Assing the current best label
                best_label[i, j, k] = label
    return best_label


###############################################################################
# Methods for weight method and hybrid weight method
###############################################################################
@njit(parallel=True, cache=True)
def get_neighbor_flux(
    data: NDArray[np.float64],
    sorted_voxel_coords: NDArray[np.int64],
    voxel_indices: NDArray[np.int64],
    neighbor_transforms: NDArray[np.int64],
    neighbor_dists: NDArray[np.float64],
    facet_areas: NDArray[np.float64],
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
    # to its neighbor and the voxel indices of these neighbors.
    flux_array = np.zeros((nx * ny * nz, len(neighbor_transforms)), dtype=np.float64)
    neigh_array = np.full(flux_array.shape, -1, dtype=np.int64)
    # calculate the area/dist for each neighbor to avoid repeat calculation
    neighbor_area_over_dist = facet_areas / neighbor_dists
    # create a mask for the location of maxima
    maxima_mask = np.zeros(nx * ny * nz, dtype=np.bool_)
    # Loop over each voxel in parallel
    for coord_index in prange(len(sorted_voxel_coords)):
        i, j, k = sorted_voxel_coords[coord_index]
        # get the initial value
        base_value = data[i, j, k]
        # iterate over each neighbor sharing a voronoi facet
        for shift_index, (shift, area_dist) in enumerate(
            zip(neighbor_transforms, neighbor_area_over_dist)
        ):
            ii = (i + shift[0]) % nx  # Loop around box
            jj = (j + shift[1]) % ny
            kk = (k + shift[2]) % nz
            # get the neighbors value
            neigh_value = data[ii, jj, kk]
            # calculate the volume flowing to this voxel
            diff = neigh_value - base_value
            # make sure diff is above a cutoff for rounding errors
            if diff < 1e-12:
                diff = 0.0
            flux = diff * area_dist
            # only assign flux if it is above 0
            if flux > 0.0:
                flux_array[coord_index, shift_index] = flux
                neigh_label = voxel_indices[ii, jj, kk]
                neigh_array[coord_index, shift_index] = neigh_label

        # normalize flux row to 1
        row = flux_array[coord_index]
        row_sum = row.sum()
        if row_sum == 0.0:
            # this is a maximum. Convert from 0 to 1 to avoid division by 0
            maxima_mask[coord_index] = True
            row_sum = 1
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
    maxima = 0
    # loop over voxels
    for vox_idx in range(n_voxels):
        neighbors = neigh_indices_array[vox_idx]
        charge = sorted_flat_charge_data[vox_idx]
        if np.all(neighbors < 0):
            # we have a maximum and assign it to its own label.
            # NOTE: We first check if the point already has a label. We do
            # this because our hybrid weight method assigns maxima beforehand
            i, j, k = sorted_voxel_coords[vox_idx]
            maxima_label = labels[i, j, k]
            if maxima_label == -1:
                labels[i, j, k] = maxima
                # assign charge and volume
                charge_array[maxima] += charge
                volume_array[maxima] += voxel_volume
                # increase our maxima counter
                maxima += 1
            else:
                # just assign charge and volume
                charge_array[maxima_label] += charge
                volume_array[maxima_label] += voxel_volume
            continue
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
        # if we only have one label, update our this point's label
        if label_num == 1:
            i, j, k = sorted_voxel_coords[vox_idx]
            labels[i, j, k] = current_label
            # assign charge and volume
            charge_array[current_label] += charge
            volume_array[current_label] += voxel_volume
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
    charge_array : TYPE
        The final charge on each basin
    volume_array : TYPE
        The final volume of each basin

    """
    # create weight array
    weight_array = np.zeros((len(unass_to_vox_pointer), maxima_num), dtype=np.float64)
    # create a new labels array to store updated labels
    new_labels = labels.copy()
    # create a scratch weight array to store rows in
    scratch_weight_array = np.empty(weight_array.shape[1], dtype=np.float64)
    for unass_idx, vox_idx in enumerate(unass_to_vox_pointer):
        # zero out our weight array
        scratch_weight_array[:] = 0.0
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
                # assign the current frac to this basin
                scratch_weight_array[label] += frac
                continue
            # otherwise, this is another multi weight label.
            neigh_unass_idx = vox_to_unass_pointer[neighbor]
            neigh_weights = weight_array[neigh_unass_idx]
            for label, weight in enumerate(neigh_weights):
                scratch_weight_array[label] += weight * frac
        # assign label, charge, and volume
        best_weight = 0.0
        best_label = -1
        charge = sorted_flat_charge_data[vox_idx]
        for label, weight in enumerate(scratch_weight_array):
            # skip if there is no weight
            if weight == 0.0:
                continue
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
        weight_array[unass_idx] = scratch_weight_array
    return new_labels, charge_array, volume_array


###############################################################################
# Functions for near grid method
###############################################################################


@njit(cache=True)
def ongrid_step(
    data: NDArray[np.float64],
    voxel_coord: NDArray[np.int64],
    neighbor_transforms: NDArray[np.int64],
    neighbor_dists: NDArray[np.float64],
) -> tuple[NDArray[np.int64], np.bool_]:
    """
    Performs a single ongrid step from the provided voxel coordinate.

    Parameters
    ----------
    data : NDArray[np.float64]
        A 3D grid of values for each point.
    voxel_coord : NDArray[np.int64]
        The point to make the step from.
    neighbor_transforms : NDArray[np.int64]
        The transformations from each voxel to its neighbors.
    neighbor_dists : NDArray[np.float64]
        The distance to each neighboring voxel.

    Returns
    -------
    NDArray[np.int64]
        The next voxel to step to.
    bool
        Whether or not the next step is a maximum.

    """
    nx, ny, nz = data.shape
    i, j, k = voxel_coord
    best = 0.0
    init_elf = data[i, j, k]
    best_neighbor = -1
    for shift_index, shift in enumerate(neighbor_transforms):
        # get the new neighbor
        ii = (i + shift[0]) % nx  # Loop around box
        jj = (j + shift[1]) % ny
        kk = (k + shift[2]) % nz
        new_elf = data[ii, jj, kk]
        dist = neighbor_dists[shift_index]
        diff = (new_elf - init_elf) / dist
        if diff > best:
            best = diff
            best_neighbor = shift_index
    if best_neighbor == -1:
        # if this is a maximum, return the original voxel coord and True
        return voxel_coord, True
    else:
        # get the neighbor, wrap, and return
        pointer = neighbor_transforms[best_neighbor]
        # move to next point
        new_coord = voxel_coord + pointer

        # wrap around indices
        ni = (new_coord[0]) % nx  # Loop around box
        nj = (new_coord[1]) % ny
        nk = (new_coord[2]) % nz
        return np.array((ni, nj, nk), dtype=np.int64), False


@njit(cache=True)
def neargrid_step(
    data: NDArray[np.float64],
    labels: NDArray[np.int64],
    max_val: int,
    voxel_coord: NDArray[np.int64],
    total_delta_r: NDArray[np.float64],
    car2lat: NDArray[np.float64],
    neighbor_transforms: NDArray[np.int64],
    neighbor_dists: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.int64], np.bool_]:
    """
    Peforms a neargrid step from the provided voxel coordinate.

    Parameters
    ----------
    data : NDArray[np.float64]
        A 3D grid of values for each point.
    labels : NDArray[np.int64]
        A 3D grid of labels representing current voxel assignments.
    max_val : int
        The current maximum used to track the path.
    voxel_coord : NDArray[np.int64]
        The point to make the step from.
    total_delta_r : NDArray[np.float64]
        A vector pointing from the current ongrid point to the true gradient.
    car2lat : NDArray[np.float64]
        A matrix that converts a coordinate in cartesian space to fractional
        space.
    neighbor_transforms : NDArray[np.int64]
        The transformations from each voxel to its neighbors.
    neighbor_dists : NDArray[np.float64]
        The distance to each neighboring voxel.

    Returns
    -------
    new_coord : NDArray[np.int64]
        The next voxel to step to
    total_delta_r : NDArray[np.int64]
        The vector pointing from the next step to the true gradient after this
        current step
    bool
        Whether or not the next step is a maximum.

    """
    nx, ny, nz = data.shape
    i, j, k = voxel_coord
    # calculate the gradient at this point in voxel coords
    charge000 = data[i, j, k]
    charge001 = data[i, j, (k + 1) % nz]
    charge010 = data[i, (j + 1) % ny, k]
    charge100 = data[(i + 1) % nx, j, k]
    charge00_1 = data[i, j, (k - 1) % nz]
    charge0_10 = data[i, (j - 1) % ny, k]
    charge_100 = data[(i - 1) % nx, j, k]

    charge_grad_vox = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    charge_grad_vox[0] = (charge100 - charge_100) / 2.0
    charge_grad_vox[1] = (charge010 - charge0_10) / 2.0
    charge_grad_vox[2] = (charge001 - charge00_1) / 2.0

    if charge100 < charge000 and charge_100 < charge000:
        charge_grad_vox[0] = 0.0
    if charge010 < charge000 and charge0_10 < charge000:
        charge_grad_vox[1] = 0.0
    if charge001 < charge000 and charge00_1 < charge000:
        charge_grad_vox[2] = 0.0

    # convert to cartesian coordinates
    charge_grad_cart = np.dot(charge_grad_vox, car2lat)
    # express in direct coordinates
    charge_grad_frac = np.dot(car2lat, charge_grad_cart)
    # calculate max gradient in a single direction
    max_grad = np.max(np.abs(charge_grad_frac))
    # check for 0 gradient
    if max_grad < 1e-30:
        # we have no gradient so we reset the total delta r
        total_delta_r[:] = 0
        # Check if this is a maximum and if not step ongrid
        new_coord, is_max = ongrid_step(
            data, voxel_coord, neighbor_transforms, neighbor_dists
        )
        if is_max:
            return new_coord, total_delta_r, True
    else:
        # Normalize
        charge_grad_frac /= max_grad
        # calculate on grid step
        new_coord = voxel_coord + np.rint(charge_grad_frac).astype(np.int64)
        # calculate dr
        total_delta_r += charge_grad_frac - np.round(charge_grad_frac).astype(
            np.float64
        )
        # apply dr
        new_coord += np.rint(total_delta_r).astype(np.int64)
        total_delta_r -= np.rint(total_delta_r).astype(np.int64)

    # wrap
    new_coord[0] %= nx
    new_coord[1] %= ny
    new_coord[2] %= nz

    # check if the new step is already in our path and if so, make an ongrid
    # step instead
    label = labels[new_coord[0], new_coord[1], new_coord[2]]
    # For the first stage we mark our path with the current max value. For the
    # refinement we mark them with the negative of their label to avoid
    # rewriting edge labels
    if label == max_val or label < 0:
        new_coord, is_max = ongrid_step(
            data, voxel_coord, neighbor_transforms, neighbor_dists
        )
        # set dr to 0
        total_delta_r[:] = 0
    # return info
    return new_coord, total_delta_r, False


@njit(fastmath=True, cache=True)
def get_neargrid_labels(
    data: NDArray[np.float64],
    car2lat: NDArray[np.float64],
    neighbor_transforms: NDArray[np.int64],
    neighbor_dists: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """
    Assigns each point to a basin using the neargrid method.

    Parameters
    ----------
    data : NDArray[np.float64]
        A 3D grid of values for each point.
    car2lat : NDArray[np.float64]
        DESCRIPTION.
    car2lat : NDArray[np.float64]
        A matrix that converts a coordinate in cartesian space to fractional
        space.
    neighbor_transforms : NDArray[np.int64]
        The transformations from each voxel to its neighbors.
    neighbor_dists : NDArray[np.float64]
        The distance to each neighboring voxel.

    Returns
    -------
    labels : NDArray[np.int64]
        The assignment for each point on the grid.
    maxima_mask : NDArray[np.bool_]
        A mask that is true at points that are maxima

    """
    nx, ny, nz = data.shape
    # define an array to assign to
    labels = np.zeros(data.shape, dtype=np.int64)
    # define an array for noting maxima
    maxima_mask = np.zeros(data.shape, dtype=np.bool_)
    # create a scratch array for our path
    path = np.empty((nx * ny * nz, 3), dtype=np.int64)
    # create a count of basins
    maxima_num = 1
    # create a scratch value for delta r
    total_delta_r = np.zeros(3, dtype=np.float64)
    # loop over all voxels
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # check if we've already assigned this point
                if labels[i, j, k] != 0:
                    continue
                # reset our delta_r
                total_delta_r[:] = 0.0
                # create a count for the length of the path
                pnum = 0
                # start climbing
                current_coord = np.array([i, j, k]).astype(np.int64)
                while True:
                    ii, jj, kk = current_coord
                    # check if we've hit another label
                    current_label = labels[ii, jj, kk]
                    if current_label != 0:
                        # relabel our path and break the loop
                        for p in range(pnum):
                            x, y, z = path[p]
                            labels[x, y, z] = current_label
                        break
                    # assign the current point to the current max
                    labels[ii, jj, kk] = maxima_num
                    # add it to our path
                    path[pnum] = (ii, jj, kk)
                    pnum = pnum + 1
                    # make a neargrid step
                    new_coord, total_delta_r, is_max = neargrid_step(
                        data=data,
                        labels=labels,
                        max_val=maxima_num,
                        voxel_coord=current_coord,
                        total_delta_r=total_delta_r,
                        car2lat=car2lat,
                        neighbor_transforms=neighbor_transforms,
                        neighbor_dists=neighbor_dists,
                    )
                    # if we reached a maximum, leave our current path assigned
                    # as is and move to the next point
                    if is_max:
                        maxima_mask[ii, jj, kk] = True
                        maxima_num += 1
                        break
                    # otherwise, continue with our loop
                    current_coord = new_coord
    return labels, maxima_mask


@njit(cache=True)
def refine_neargrid(
    data: NDArray[np.float64],
    labels: NDArray[np.int64],
    refinement_indices: NDArray[np.int64],
    refinement_mask: NDArray[np.bool_],
    checked_mask: NDArray[np.bool_],
    maxima_mask: NDArray[np.bool_],
    car2lat: NDArray[np.float64],
    neighbor_transforms: NDArray[np.int64],
    neighbor_dists: NDArray[np.float64],
) -> tuple[NDArray[np.int64], np.int64, NDArray[np.bool_], NDArray[np.bool_]]:
    """
    Refines the provided voxels by running the neargrid method until a maximum
    is found for each.

    Parameters
    ----------
    data : NDArray[np.float64]
        A 3D grid of values for each point.
    labels : NDArray[np.int64]
        A 3D grid of labels representing current voxel assignments.
    refinement_indices : NDArray[np.int64]
        A Nx3 array of voxel indices to perform the refinement on.
    refinement_mask : NDArray[np.bool_]
        A 3D mask that is true at the voxel indices to be refined.
    checked_mask : NDArray[np.bool_]
        A 3D mask that is true at voxels that have already been refined.
    maxima_mask : NDArray[np.bool_]
        A 3D mask that is true at maxima.
    car2lat : NDArray[np.float64]
        A matrix that converts a coordinate in cartesian space to fractional
        space.
    neighbor_transforms : NDArray[np.int64]
        The transformations from each voxel to its neighbors.
    neighbor_dists : NDArray[np.float64]
        The distance to each neighboring voxel.

    Returns
    -------
    new_labels : NDArray[np.int64]
        The updated assignment for each point on the grid.
    reassignments : np.int64
        The number of points that were reassigned.
    refinement_mask : NDArray[np.bool_]
        The updated mask of points that need to be refined
    checked_mask : NDArray[np.bool_]
        The updated mask of points that have been checked.

    """
    # create an array for new labels
    new_labels = labels.copy()
    # get shape
    nx, ny, nz = data.shape
    # create scratch total_delta_r
    total_delta_r = np.zeros(3, dtype=np.float64)
    current_coord = np.empty(3, dtype=np.int64)
    # create scratch path
    path = np.empty((nx * ny * nz, 3), dtype=np.int64)
    # now we reassign any voxel in our refinement mask
    reassignments = 0
    for i, j, k in refinement_indices:
        # get our initial label, just for comparison
        label = labels[i, j, k]
        # Now we do neargrid steps until we reach a point with an label
        total_delta_r[:] = 0.0
        # create a count for the length of the path
        pnum = 0
        # note that we've checked this index
        # start climbing
        current_coord[0] = i
        current_coord[1] = j
        current_coord[2] = k
        while True:
            ii, jj, kk = current_coord
            # check if we've hit a maximum
            if maxima_mask[ii, jj, kk]:
                # add this point to our checked list. We use this to make sure
                # this point doesn't get re-added to our list later in the
                # process.
                checked_mask[i, j, k] = True
                # remove it from the refinement list
                refinement_mask[i, j, k] = False
                current_label = labels[ii, jj, kk]
                # Points along the path are switched to negative. we
                # switch them back here
                for p in range(pnum):
                    x, y, z = path[p]
                    labels[x, y, z] = -labels[x, y, z]
                # Check if this is a reassignment
                if label != current_label:
                    reassignments += 1
                    # add any neighbors to our refinement mask for the next iteration
                    for shift in neighbor_transforms:
                        # get the new neighbor
                        ni = (i + shift[0]) % nx  # Loop around box
                        nj = (j + shift[1]) % ny
                        nk = (k + shift[2]) % nz
                        # If we haven't already checked this point, add it
                        if not checked_mask[ni, nj, nk]:
                            refinement_mask[ni, nj, nk] = True
                # relabel just this voxel then stop the loop
                new_labels[i, j, k] = current_label
                break
            # add this label to our path
            labels[ii, jj, kk] = -labels[ii, jj, kk]
            path[pnum] = (ii, jj, kk)
            pnum = pnum + 1
            # make a neargrid step
            current_coord, total_delta_r, is_max = neargrid_step(
                data=data,
                labels=labels,
                max_val=0,  # should never exist
                voxel_coord=current_coord,
                total_delta_r=total_delta_r,
                car2lat=car2lat,
                neighbor_transforms=neighbor_transforms,
                neighbor_dists=neighbor_dists,
            )

    return new_labels, reassignments, refinement_mask, checked_mask
