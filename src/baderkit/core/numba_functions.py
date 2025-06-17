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
        A 3D array where each entry represents the basin assignment of the point.
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


@njit(parallel=True, cache=True)
def get_basin_charge_volume_from_label(
    basin_labels: NDArray[np.int64],
    charge_data: NDArray[np.float64],
    voxel_volume: np.float64,
    maxima_num: types.int64,
):
    """
    Calculates the charge and volume

    Parameters
    ----------
    basin_labels : NDArray[np.int64]
        A 3D array where each entry represents the basin assignment of the point.
    charge_data : NDArray[np.float64]
        The charge density at each voxel.
    voxel_volume : np.float64
        The volume of each voxel
    maxima_num : types.int64
        The number of maxima in the grid

    Returns
    -------
    charge_array : NDArray[float]
        The total charge for each basin
    volume_array : NDArray[float]
        The total volume for each basin

    """
    charge_array = np.zeros(maxima_num, dtype=types.float64)
    volume_array = np.zeros(maxima_num, dtype=types.float64)
    for basin_index in prange(maxima_num):
        basin_indices = np.argwhere(basin_labels == basin_index)
        for x, y, z in basin_indices:
            charge = charge_data[x, y, z]
            charge_array[basin_index] += charge
            volume_array[basin_index] += voxel_volume
    return charge_array, volume_array


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
        A 3D array where each entry represents the basin assignment of the point.
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
    assignments: NDArray[np.int64] = None,
):
    """
    Loops over voxels to find any that have exaclty one weight. We store
    these in a single array the size of the assignments to reduce space

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
    assignments : NDArray[np.int64], optional
        A 3D array of preassigned labels.

    Returns
    -------
    assignments : NDArray[int]
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
    # create assignments array
    if assignments is None:
        assignments = np.full(data.shape, -1, dtype=np.int64)
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
            # NOTE: We first check if the assignment already has a label. We do
            # this because our hybrid weight method assigns maxima beforehand
            i, j, k = sorted_voxel_coords[vox_idx]
            maxima_assignment = assignments[i, j, k]
            if maxima_assignment == -1:
                assignments[i, j, k] = maxima
                # assign charge and volume
                charge_array[maxima] += charge
                volume_array[maxima] += voxel_volume
                # increase our maxima counter
                maxima += 1
            else:
                # just assign charge and volume
                charge_array[maxima_assignment] += charge
                volume_array[maxima_assignment] += voxel_volume
            continue
        # otherwise we check each neighbor and check its assignment
        current_label = -1
        label_num = 0
        for neigh in neighbors:
            if neigh == -1:
                # This isn't a valid neighbor so we skip it
                continue
            # get this neighbors assignment
            ni, nj, nk = sorted_voxel_coords[neigh]
            neigh_label = assignments[ni, nj, nk]
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
        # if we only have one label, update our assignment
        if label_num == 1:
            i, j, k = sorted_voxel_coords[vox_idx]
            assignments[i, j, k] = current_label
            # assign charge and volume
            charge_array[current_label] += charge
            volume_array[current_label] += voxel_volume
        else:
            unassigned_mask[vox_idx] = True
    return assignments, unassigned_mask, charge_array, volume_array


@njit(fastmath=True, cache=True)
def get_multi_weight_voxels(
    flux_array: NDArray[np.float64],
    neigh_indices_array: NDArray[np.int64],
    assignments: NDArray[np.int64],
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
    of the basins it is split to. The returned assignments represent the basin
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
    assignments : NDArray[np.int64]
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
    new_assignments : NDArray[np.int64]
        The updated assignments.
    charge_array : TYPE
        The final charge on each basin
    volume_array : TYPE
        The final volume of each basin

    """
    # create weight array
    weight_array = np.zeros((len(unass_to_vox_pointer), maxima_num), dtype=np.float64)
    # create a new assignments array to store updated assignments
    new_assignments = assignments.copy()
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
            # otherwise we get the assignments and fraction of assignments for
            # this voxel. First check if it is a single weight assignment
            ni, nj, nk = sorted_voxel_coords[neighbor]
            assignment = assignments[ni, nj, nk]
            if assignment != -1:
                # assign the current frac to this basin
                scratch_weight_array[assignment] += frac
                continue
            # otherwise, this is another multi weight assignment.
            neigh_unass_idx = vox_to_unass_pointer[neighbor]
            neigh_weights = weight_array[neigh_unass_idx]
            for assignment, weight in enumerate(neigh_weights):
                scratch_weight_array[assignment] += weight * frac
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
        # update assignment
        i, j, k = sorted_voxel_coords[vox_idx]
        new_assignments[i, j, k] = best_label
        # assign this weight row
        weight_array[unass_idx] = scratch_weight_array
    return new_assignments, charge_array, volume_array


# @njit(fastmath=True, cache=True)
# def get_hybrid_basin_weights(
#     flux_array: NDArray[np.float64],
#     neigh_indices_array: NDArray[np.int64],
#     weight_array: NDArray[np.float64],
# ):
#     # get the length of our voxel array
#     n_voxels = flux_array.shape[0]
#     # iterate over our voxels. We assume voxels are ordered from highest to lowest
#     # data
#     for i in range(n_voxels):
#         neighbors = neigh_indices_array[i]
#         # Our neighbor indices array is -1 where the neighbors are lower. Maxima
#         # correspond to where this is true for all neighbors
#         if np.all(neighbors < 0):
#             # This is a maximum and should already have been labeled. We continue
#             continue
#         # Otherwise we are at either an interior or edge voxel.
#         # Get a mask where there are neighbors in this row (those that are above -1)
#         mask = neigh_indices_array[i, :] >= 0
#         # Get the relavent neighbors and flux flowing into them
#         fluxes = flux_array[i, mask]
#         # Get the sum of each current_flux*neighbor_flux for each basin and
#         weight_array[i] = fluxes @ weight_array[neighbors[mask]]

#     return weight_array


###############################################################################
# Functions for near grid method
###############################################################################


@njit(fastmath=True, cache=True)
def get_near_grid_assignments(
    data: NDArray[np.float64],
    flat_voxel_coords: NDArray[np.int64],
    pointer_voxel_coords: NDArray[np.int64],
    voxel_indices: NDArray[np.int64],
    zero_grads: NDArray[np.bool_],
    # rgrid: NDArray[np.int64],
    delta_rs: NDArray[np.float64],
    neighbors: NDArray[np.int64],
    neighbor_dists: NDArray[np.float64],
):
    """
    Do an initial assignment of voxels to basins using the neargrid method.

    Parameters
    ----------
    data : NDArray[np.float64]
        A 3D grid of values for each point.
    flat_voxel_coords : NDArray[np.int64]
        A Nx3 array of voxel coords
    pointer_voxel_coords : NDArray[np.int64]
        A 1D array of length N pointing each voxel to the neighboring voxel that
        is nearest to one step along the gradient at the voxel
    voxel_indices : NDArray[np.int64]
        A 3D array where each entry is the flat voxel index of each
        point
    zero_grads : NDArray[np.bool_]
        A 1D array that is True where voxels have zero gradient and are either
        maxima or need to use an on-grid step
    delta_rs : NDArray[np.float64]
        A Nx3 array of differences between the ongrid steps and true gradient
        steps
    neighbors : NDArray[np.int64]
        The transformations from each voxel to its neighbors.
    neighbor_dists : NDArray[np.float64]
        The distance to each neighboring voxel

    Returns
    -------
    assignments : NDArray[int]
        A 3D array where each entry represents the basin the voxel belongs to.

    maxima_mask : NDArray[bool]
        A 1D array of length N that is True where the sorted voxel indices are
        a maximum

    """
    nx, ny, nz = data.shape
    # create array for assigning
    assignments = np.zeros(data.shape, dtype=np.int64)
    # create scratch array for tracking which points have been visited
    visited = np.empty((nx * ny * nz, 3), dtype=np.int64)
    # Create array for storing maxima
    maxima_mask = np.zeros(data.shape, dtype=np.bool_)
    # create counter for number of maxima
    maxima_count = 1
    # iterate over all voxels, their pointers, and delta rs
    for initial_coord in flat_voxel_coords:
        # We manually set the dtype to int64 because older versions of numba seem
        # to convert the dtype to int32 somewhere
        current_coord = np.array(
            (initial_coord[0], initial_coord[1], initial_coord[2]), dtype=np.int64
        )
        # Begin a while loop climbing the gradient and correcting it until we
        # reach either a maximum or an already assigned label
        total_dr = np.zeros(3, dtype=np.float64)
        path_len = 0
        while True:
            i, j, k = current_coord
            # First check if this coord has an assignment already
            current_label = assignments[i, j, k]
            if current_label != 0:
                # If this is our first step we want to immedietly break and
                # continue, as this voxel has been assigned.
                if path_len == 0:
                    break
                # Otherwise we need to reassign all points in the path.
                for visited_idx in range(path_len):
                    xi, yi, zi = visited[visited_idx]
                    assignments[xi, yi, zi] = current_label
                # and we halt this loop
                break

            # Note that we've visited this voxel in this path
            visited[path_len] = (i, j, k)
            path_len += 1
            # otherwise, we have no label. We assign our current maximum value
            assignments[i, j, k] = maxima_count

            # Next we check if the rgrid step is 0 for this point.
            voxel_index = voxel_indices[i, j, k]
            no_grad = zero_grads[voxel_index]
            if no_grad:
                # check that this is a maximum
                best = 0.0
                init_elf = data[i, j, k]
                best_neighbor = -1
                for shift_index, shift in enumerate(neighbors):
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
                    # This is a maximum. We note that we've labeled a new maximum
                    # and break to continue to the next point
                    maxima_count += 1
                    # mark this as a maximum
                    maxima_mask[i, j, k] = True
                    break
                else:
                    # This voxel won't move to a nearby point. We default back
                    # to on-grid assignment
                    pointer = neighbors[best_neighbor]
                    # Reset our total dr since we've arrived at a point with
                    # zero gradient
                    total_dr = np.zeros(3, dtype=np.float64)
                    # move to next point
                    new_coord = current_coord + pointer

            else:
                # move to next point
                new_coord = pointer_voxel_coords[voxel_index]
                # get the delta r between this on-grid gradient and the true gradient
                dr = delta_rs[voxel_index]
                # get new total dr
                total_dr += dr
                # adjust based on total diff
                new_coord += np.round(total_dr).astype(np.int64)
                # adjust total diff
                total_dr -= np.round(total_dr).astype(np.int64)

            # Wrap our new coord and set it as our new coord
            ni = (new_coord[0]) % nx  # Loop around box
            nj = (new_coord[1]) % ny
            nk = (new_coord[2]) % nz

            # Make sure we aren't revisiting coords in our path
            new_label = assignments[ni, nj, nk]
            if new_label == maxima_count:
                # We start climbing with on-grid until we find a voxel that doesn't
                # belong to this path. We also reset dr
                total_dr = np.zeros(3, dtype=np.float64)
                temp_current_coord = current_coord.copy()
                while True:
                    ti, tj, tk = temp_current_coord
                    new_label = assignments[ni, nj, nk]
                    if new_label == maxima_count:
                        # continue on grid steps
                        best = 0.0
                        init_elf = data[ti, tj, tk]
                        best_neighbor = -1
                        for shift_index, shift in enumerate(neighbors):
                            # get the new neighbor
                            ii = (ti + shift[0]) % nx  # Loop around box
                            jj = (tj + shift[1]) % ny
                            kk = (tk + shift[2]) % nz
                            new_elf = data[ii, jj, kk]
                            dist = neighbor_dists[shift_index]
                            diff = (new_elf - init_elf) / dist
                            if diff > best:
                                best = diff
                                best_neighbor = shift_index

                        pointer = neighbors[best_neighbor]
                        # move to next point
                        new_coord = temp_current_coord + pointer
                        # update the pointer for this voxel to avoid repeat calc

                        # wrap around indices
                        ni = (new_coord[0]) % nx  # Loop around box
                        nj = (new_coord[1]) % ny
                        nk = (new_coord[2]) % nz
                        temp_current_coord = np.array((ni, nj, nk), dtype=np.int64)
                    else:
                        # we have reached a voxel outside the current path.
                        break

            current_coord = np.array((ni, nj, nk), dtype=np.int64)
    return assignments, maxima_mask


@njit(fastmath=True, cache=True)
def refine_near_grid_edges(
    data: NDArray[np.float64],
    current_assignments: NDArray[np.int64],
    refined_assignments: NDArray[np.int64],
    edge_voxel_coords: NDArray[np.int64],
    pointer_voxel_coords: NDArray[np.int64],
    voxel_indices: NDArray[np.int64],
    zero_grads: NDArray[np.bool_],
    delta_rs: NDArray[np.float64],
    neighbors: NDArray[np.int64],
    neighbor_dists: NDArray[np.float64],
):
    """
    Refines the assignments at the edges of each basin.

    Parameters
    ----------
    data : NDArray[np.float64]
        A 3D grid of values for each point.
    current_assignments : NDArray[np.int64]
        A 3D array where each entry represents the basin the voxel currently belongs to.
    refined_assignments : NDArray[np.int64]
        A 3D array with assignments anywhere that isn't an edge and 0 at the edge.
    edge_voxel_coords : NDArray[np.int64]
        An Nx3 array of voxel indices
    pointer_voxel_coords : NDArray[np.int64]
        A 1D array of length N pointing each voxel to the neighboring voxel that
        is nearest to one step along the gradient at the voxel
    voxel_indices : NDArray[np.int64]
        A 3D array where each entry is the flat voxel index of each
        point
    zero_grads : NDArray[np.bool_]
        A 1D array that is True where voxels have zero gradient and are either
        maxima or need to use an on-grid step
    delta_rs : NDArray[np.float64]
        A Nx3 array of differences between the ongrid steps and true gradient
        steps
    neighbors : NDArray[np.int64]
        The transformations from each voxel to its neighbors.
    neighbor_dists : NDArray[np.float64]
        The distance to each neighboring voxel

    Returns
    -------
    new_assignments : NDArray[int]
        The updated assignments from the refinement
    changed_labels : int
        The number of labels that changed in the refinement

    """
    nx, ny, nz = data.shape
    # create array for assigning
    new_assignments = refined_assignments.copy()
    # create scratch array for tracking which points have been visited
    visited_indices = np.empty((nx * ny * nz), dtype=np.int64)
    # create counter for edges that have been updated
    changed_labels = 0
    # iterate over all voxels, their pointers, and delta rs
    for initial_coord in edge_voxel_coords:
        current_coord = np.array(
            (initial_coord[0], initial_coord[1], initial_coord[2]), dtype=np.int64
        )
        original_label = current_assignments[
            current_coord[0], current_coord[1], current_coord[2]
        ]
        # Begin a while loop climbing the gradient and correcting it until we
        # reach either a maximum or an already assigned label
        total_dr = np.zeros(3, dtype=np.float64)
        path_len = 0
        while True:
            i, j, k = current_coord[0], current_coord[1], current_coord[2]
            # First check if this coord has an assignment
            current_label = refined_assignments[i, j, k]
            if current_label != 0:
                # We've found the assignment for our original coord and label it
                new_assignments[
                    initial_coord[0], initial_coord[1], initial_coord[2]
                ] = current_label
                # update new label counter
                if current_label != original_label:
                    changed_labels += 1
                # and we halt this loop
                break

            # Now we note that we've visited this voxel in this path
            voxel_index = voxel_indices[i, j, k]
            visited_indices[path_len] = voxel_index
            path_len += 1

            # Next we check if the rgrid step is 0 for this point.
            voxel_index = voxel_indices[i, j, k]
            no_grad = zero_grads[voxel_index]
            if no_grad:
                # We resort to standard hill climbing
                best = 0.0
                init_elf = data[i, j, k]
                best_neighbor = -1
                for shift_index, shift in enumerate(neighbors):
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

                # It should not be possible for this to be a maximum as all maxima
                # were assigned in the previous step
                assert best_neighbor != -1
                # This voxel won't move to a nearby point. We default back
                # to on-grid assignment
                pointer = neighbors[best_neighbor]
                # Reset our total dr since we've arrived at a point with
                # zero gradient
                total_dr = np.zeros(3, dtype=np.float64)
                # move to next point
                new_coord = current_coord + pointer

            else:
                # move to next point
                new_coord = pointer_voxel_coords[voxel_index]
                # get the delta r between this on-grid gradient and the true gradient
                dr = delta_rs[voxel_index]
                # get new total dr
                total_dr += dr
                # adjust based on total diff
                new_coord += np.round(total_dr).astype(np.int64)
                # adjust total diff
                total_dr -= np.round(total_dr).astype(np.int64)

            # Wrap our new coord and set it as our new coord
            ni = (new_coord[0]) % nx  # Loop around box
            nj = (new_coord[1]) % ny
            nk = (new_coord[2]) % nz

            # Make sure we aren't revisiting coords in our path
            new_index = voxel_indices[ni, nj, nk]
            if new_index in visited_indices[:path_len]:
                # We start climbing with on-grid until we find a voxel that doesn't
                # belong to this path. We also reset dr
                total_dr = np.zeros(3, dtype=np.float64)
                temp_current_coord = current_coord.copy()
                while True:
                    ti, tj, tk = temp_current_coord
                    new_index = voxel_indices[ni, nj, nk]
                    if new_index in visited_indices[:path_len]:
                        # continue on grid steps
                        best = 0.0
                        init_elf = data[ti, tj, tk]
                        best_neighbor = -1
                        for shift_index, shift in enumerate(neighbors):
                            # get the new neighbor
                            ii = (ti + shift[0]) % nx  # Loop around box
                            jj = (tj + shift[1]) % ny
                            kk = (tk + shift[2]) % nz
                            new_elf = data[ii, jj, kk]
                            dist = neighbor_dists[shift_index]
                            diff = (new_elf - init_elf) / dist
                            if diff > best:
                                best = diff
                                best_neighbor = shift_index

                        pointer = neighbors[best_neighbor]
                        # move to next point
                        new_coord = temp_current_coord + pointer
                        # update the pointer for this voxel to avoid repeat calc

                        # wrap around indices
                        ni = (new_coord[0]) % nx  # Loop around box
                        nj = (new_coord[1]) % ny
                        nk = (new_coord[2]) % nz
                        temp_current_coord = np.array((ni, nj, nk), dtype=np.int64)
                    else:
                        # we have reached a voxel outside the current path.
                        break

            current_coord = np.array((ni, nj, nk), dtype=np.int64)
    return new_assignments, changed_labels


# @njit(cache=True)
# def refine_near_grid_edges(
#         data: NDArray[np.float64],
#         assignments: NDArray[np.int64],
#         edge_voxel_coords: NDArray[np.int64],
#         pointer_voxel_coords:NDArray[np.int64],
#         voxel_indices: NDArray[np.int64],
#         delta_rs: NDArray[np.float64],
#         neighbors: NDArray[np.int64],
#         neighbor_dists: NDArray[np.float64],
#         ):
#     nx,ny,nz = assignments.shape
#     refined_assignments = assignments.copy()
#     # create scratch array for tracking which points have been visited
#     visited = np.zeros((nx * ny * nz), dtype=np.int64)
#     visited_num = 0
#     # loop over edges
#     for edge_index in range(len(edge_voxel_coords)):
#         initial_coords = edge_voxel_coords[edge_index]
#         current_coord = edge_voxel_coords[edge_index]
#         # start tracking dr
#         total_dr = np.zeros(3, dtype=np.float64)
#         # start hill climbing
#         while True:
#             i,j,k = current_coord
#             label = assignments[i,j,k]

#             # check if this is a labeled voxel
#             if label != 0:
#                 # we want to label our initial coord and break
#                 refined_assignments[initial_coords[0], initial_coords[1], initial_coords[2]] = label
#                 break
#             # otherwise, we want to keep climbing.
#             # Get the flat index of this voxel
#             voxel_index = voxel_indices[i,j,k]
#             # Make sure we haven't visited it yet
#             already_visited = False
#             for already_visited in visited[:visited_num]:
#                 if voxel_index == already_visited:
#                     already_visited = True
#                     break
#             if already_visited:
#                 # We default back to standard hill climbing until we
#                 # reach an already assigned voxel
#                 temp_current_coord = current_coord.copy()
#                 while True:
#                     ti, tj, tk = temp_current_coord
#                     label = assignments[ti, tj, tk]
#                     if label == 0:
#                         # We haven't found an assigned voxel.
#                         # continue on grid steps
#                         best = 0.0
#                         init_elf = data[ti, tj, tk]
#                         best_neighbor = -1
#                         for shift_index, shift in enumerate(neighbors):
#                             # get the new neighbor
#                             ii = (ti + shift[0]) % nx  # Loop around box
#                             jj = (tj + shift[1]) % ny
#                             kk = (tk + shift[2]) % nz
#                             # Check if the difference is better than the current
#                             new_elf = data[ii, jj, kk]
#                             dist = neighbor_dists[shift_index]
#                             diff = (new_elf - init_elf) / dist
#                             if diff > best:
#                                 best = diff
#                                 best_neighbor = shift_index
#                         assert best_neighbor != -1
#                         # get the pointer to the best neighbor
#                         pointer = neighbors[best_neighbor]
#                         # move to next point
#                         new_coord = current_coord + pointer
#                         # TODO: I could update the pointer during the first round
#                         # of calculations so I don't need to do this here
#                         # wrap around indices
#                         ni = (new_coord[0]) % nx  # Loop around box
#                         nj = (new_coord[1]) % ny
#                         nk = (new_coord[2]) % nz
#                         temp_current_coord = np.array((ni, nj, nk), dtype=np.int64)
#                     else:
#                         # we have reached a voxel outside the current path.
#                         refined_assignments[initial_coords[0], initial_coords[1], initial_coords[2]] = label
#                         break
#                 # Stop the current loop
#                 break
#             # Otherwise, note that we've visited this voxel
#             visited[visited_num] = voxel_index
#             visited_num += 1
#             # get the next voxel
#             new_coord = pointer_voxel_coords[voxel_index]
#             dr = delta_rs[voxel_index]
#             total_dr += dr
#             # adjust based on total diff
#             rounded_dr = np.round(total_dr).astype(np.int64)
#             new_coord += rounded_dr
#             # adjust total diff
#             total_dr -= rounded_dr
#             # wrap around the edges
#             ni = (new_coord[0]) % nx # Loop around box
#             nj = (new_coord[1]) % ny
#             nk = (new_coord[2]) % nz
#             # mark as current voxel
#             current_coord = np.array((ni,nj,nk),dtype=np.int64)
#     return refined_assignments
