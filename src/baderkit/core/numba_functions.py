# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, prange  # , types
from numpy.typing import NDArray

###############################################################################
# General methods
###############################################################################


@njit(parallel=True, cache=True)
def get_edges(
    labeled_array: NDArray[np.int64],
    neighbor_transforms: NDArray[np.int64],
    vacuum_mask: NDArray[np.bool_],
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
    vacuum_mask: NDArray[np.bool_]
        A 3D array representing the location of the vacuum

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
                # if this voxel is part of the vacuum, continue
                if vacuum_mask[i, j, k]:
                    continue
                # get this voxels label
                label = labeled_array[i, j, k]
                # iterate over the neighboring voxels
                for shift_index, shift in enumerate(neighbor_transforms):
                    ii = i + shift[0]
                    jj = j + shift[1]
                    kk = k + shift[2]
                    # wrap points
                    ii, jj, kk = wrap_point(ii, jj, kk, nx, ny, nz)
                    # get neighbors label
                    neigh_label = labeled_array[ii, jj, kk]
                    # if any label is different, the current voxel is an edge.
                    # Note this in our edge array and break
                    # NOTE: we also check that the neighbor is not part of the
                    # vacuum
                    if neigh_label != label and not vacuum_mask[ii, jj, kk]:
                        edges[i, j, k] = True
                        break
    return edges


@njit(fastmath=True, cache=True)
def get_basin_charges_and_volumes(
    data: NDArray[np.float64],
    labels: NDArray[np.int64],
    cell_volume: np.float64,
    maxima_num: np.int64,
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
                charge = data[i, j, k]
                label = labels[i, j, k]
                if label < 0:
                    vacuum_charge += charge
                    vacuum_volume += 1
                else:
                    charges[label] += charge
                    volumes[label] += 1.0
    # calculate charge and volume
    volumes = volumes * cell_volume / total_points
    charges = charges / total_points
    vacuum_volume = vacuum_volume * cell_volume / total_points
    vacuum_charge = vacuum_charge / total_points
    return charges, volumes, vacuum_charge, vacuum_volume


@njit(cache=True)
def wrap_point(
    i: np.int64, j: np.int64, k: np.int64, nx: np.int64, ny: np.int64, nz: np.int64
) -> tuple[np.int64, np.int64, np.int64]:
    """
    Wraps a 3D point (i, j, k) into the periodic bounds defined by the grid dimensions (nx, ny, nz).

    If any of the input coordinates are outside the bounds [0, nx), [0, ny), or [0, nz),
    they are wrapped around using periodic boundary conditions.

    Parameters
    ----------
    i : np.int64
        x-index of the point.
    j : np.int64
        y-index of the point.
    k : np.int64
        z-index of the point.
    nx : np.int64
        Number of grid points along x-direction.
    ny : np.int64
        Number of grid points along y-direction.
    nz : np.int64
        Number of grid points along z-direction.

    Returns
    -------
    tuple[np.int64, np.int64, np.int64]
        The wrapped (i, j, k) indices within the bounds.
    """
    if i >= nx:
        i -= nx
    elif i < 0:
        i += nx
    if j >= ny:
        j -= ny
    elif j < 0:
        j += ny
    if k >= nz:
        k -= nz
    elif k < 0:
        k += nz
    return i, j, k


@njit(cache=True)
def get_gradient_simple(
    data: NDArray[np.float64],
    voxel_coord: NDArray[np.int64],
    car2lat: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.int64], np.bool_]:
    """
    Peforms a neargrid step from the provided voxel coordinate.

    Parameters
    ----------
    data : NDArray[np.float64]
        A 3D grid of values for each point.
    voxel_coord : NDArray[np.int64]
        The point to make the step from.
    car2lat : NDArray[np.float64]
        A matrix that converts a coordinate in cartesian space to fractional
        space.

    Returns
    -------
    charge_grad_frac : NDArray[np.float64]
        The gradient in direct space at this voxel coord

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

    if charge100 <= charge000 and charge_100 <= charge000:
        charge_grad_vox[0] = 0.0
    if charge010 <= charge000 and charge0_10 <= charge000:
        charge_grad_vox[1] = 0.0
    if charge001 <= charge000 and charge00_1 <= charge000:
        charge_grad_vox[2] = 0.0

    # convert to cartesian coordinates
    charge_grad_cart = np.dot(charge_grad_vox, car2lat)
    # express in direct coordinates
    charge_grad_frac = np.dot(car2lat, charge_grad_cart)
    # return the gradient
    return charge_grad_frac


# This is an alternative method for calculating the gradient that either doesn't
# work or isn't a valid method.
# @njit(cache=True, parallel=True)
# def combine_neigh_maxima(
#     labels,
#     neighbor_transforms,
#     maxima_vox,
#     maxima_frac,
#     maxima_mask,
# ):
#     nx, ny, nz = labels.shape
#     initial_labels = np.arange(len(maxima_vox), dtype=np.int64)
#     new_labels = np.zeros(len(maxima_vox), dtype=np.int64)

#     # check each neighbor and it its a max with a lower index, update the index
#     for max_idx in range(len(maxima_vox)):
#         i, j, k = maxima_vox[max_idx]
#         best_label = initial_labels[max_idx]
#         for shift in neighbor_transforms:
#             ii = i + shift[0]
#             jj = j + shift[1]
#             kk = k + shift[2]
#             # wrap
#             ii, jj, kk = wrap_point(ii, jj, kk, nx, ny, nz)
#             # check if new point is also a maximum
#             if maxima_mask[ii, jj, kk] and labels[ii, jj, kk] < best_label:
#                 best_label = labels[ii, jj, kk]
#         new_labels[max_idx] = best_label
#     # TODO:
#     # This may not fully reduce maxima. It may be possible for several voxels
#     # in a row to all be maxima. If we had for example four voxels in a line
#     # that are all maxima, the first two may be relabeld to the first and the
#     # other two may be relabeld to the second or some other configuration. I
#     # need to decide if these are physically reasonable situations where an
#     # average should be taken.

#     # Now we want to calculate the new frac coords for each group. We also want
#     # to make sure the labels go from 0, 1, 2, ... so on, while currently they
#     # may skip some (e.g. 0,2,3,5,...)
#     # get unique labels to combine and create an array for storing average frac coords
#     unique_labels = np.unique(new_labels)
#     frac_coords = np.zeros((len(unique_labels), 3), dtype=np.float64)
#     # create array to store the reduce labels
#     reduced_new_labels = np.zeros(len(new_labels), dtype=np.int64)
#     # create a counter to assist in relabeling
#     label_counter = -1
#     for label in unique_labels:
#         # increment our label counter
#         label_counter += 1
#         all_coords = []
#         for max_idx in range(len(new_labels)):
#             new_label = new_labels[max_idx]
#             # skip labels that don't match
#             if new_labels[new_label] != label:
#                 continue
#             # update the label and add this coord
#             reduced_new_labels[max_idx] = label_counter
#             all_coords.append(maxima_frac[max_idx])

#         # if we have only one maxima, add it to the array and continue
#         if len(all_coords) == 1:
#             frac_coords[label_counter] = all_coords[0]
#             continue
#         # use the first coord as our reference and create an array to serve as
#         # our average
#         ref = all_coords[0]
#         total = np.zeros(3, dtype=np.float64)
#         # loop over each coord, shift it to be close to the original point,
#         # add it to our total, then divide to average
#         for i in range(len(all_coords)):
#             coord = all_coords[i]
#             # unwrap each coord relative to ref
#             unwrapped = coord - np.round(coord - ref)
#             total += unwrapped
#         # average and wrap
#         avg = total / len(all_coords)
#         avg %= 1.0
#         frac_coords[label_counter] = avg

#     return reduced_new_labels, frac_coords


@njit(cache=True)
def get_gradient_overdetermined(
    data,
    i,
    j,
    k,
    vox_transforms,
    transform_dists,
    inv_norm_cart_trans,
    inv_lattice_matrix,
):
    nx, ny, nz = data.shape
    # Value at the central point
    point_value = data[i, j, k]
    # Number of neighbor displacements/transforms
    num_transforms = len(vox_transforms)

    # Array to hold finite‐difference estimates along each transform direction
    diffs = np.zeros(num_transforms)
    # Loop over each neighbor transform
    for trans_idx in range(num_transforms):
        # Displacement vector in voxel (grid) coordinates
        x, y, z = vox_transforms[trans_idx]
        # Compute “upper” neighbor index, wrapped by periodic boundaries
        ui, uj, uk = wrap_point(i + x, j + y, k + z, nx, ny, nz)
        # Compute “lower” neighbor index (opposite direction), also wrapped
        li, lj, lk = wrap_point(i - x, j - y, k - z, nx, ny, nz)
        # Values at the neighboring points
        upper_value = data[ui, uj, uk]
        lower_value = data[li, lj, lk]

        # If both neighbors are below or equal to the center, zero out this direction
        # (prevents spurious negative slopes if data dips on both sides)
        if lower_value <= point_value and upper_value <= point_value:
            diffs[trans_idx] = 0.0
        else:
            # Standard central‐difference estimate: (f(i+Δ) – f(i–Δ)) / (2Δ)
            diffs[trans_idx] = (upper_value - lower_value) / (
                2.0 * transform_dists[trans_idx]
            )

    # Solve the overdetermined system to get the Cartesian gradient:
    #   norm_cart_transforms.T @ cart_grad ≈ diffs
    # Use the pseudoinverse to handle more directions than dimensions
    cart_grad = inv_norm_cart_trans @ diffs
    # Convert Cartesian gradient to fractional (lattice) coordinates
    frac_grad = cart_grad @ inv_lattice_matrix
    return frac_grad


###############################################################################
# Functions for ongrid method
###############################################################################
@njit(cache=True)
def get_best_neighbor(
    data: NDArray[np.float64],
    i: np.int64,
    j: np.int64,
    k: np.int64,
    neighbor_transforms: NDArray[np.int64],
    neighbor_dists: NDArray[np.int64],
):
    """
    For a given coordinate (i,j,k) in a grid (data), finds the neighbor with
    the largest gradient.

    Parameters
    ----------
    data : NDArray[np.float64]
        The data for each voxel.
    i : np.int64
        First coordinate
    j : np.int64
        Second coordinate
    k : np.int64
        Third coordinate
    neighbor_transforms : NDArray[np.int64]
        Transformations to apply to get to the voxels neighbors
    neighbor_dists : NDArray[np.int64]
        The distance to each voxels neighbor

    Returns
    -------
    best_transform : NDArray[np.int64]
        The transformation to the best neighbor
    best_neigh : NDArray[np.int64]
        The coordinates of the best neigbhor
    is_max: bool
        Whether or not this voxel is a local maximum

    """
    nx, ny, nz = data.shape
    # get the elf value and initial label for this voxel. This defaults
    # to the voxel pointing to itself
    base = data[i, j, k]
    best = 0.0
    best_transform = np.zeros(3, dtype=np.int64)
    best_neigh = np.array([i, j, k], dtype=np.int64)
    # For each neighbor get the difference in value and if its better
    # than any previous, replace the current best
    for shift, dist in zip(neighbor_transforms, neighbor_dists):
        ii = i + shift[0]
        jj = j + shift[1]
        kk = k + shift[2]
        # loop
        ii, jj, kk = wrap_point(ii, jj, kk, nx, ny, nz)
        # calculate the difference in value taking into account distance
        diff = (data[ii, jj, kk] - base) / dist
        # if better than the current best, note the best and the
        # current label
        if diff > best:
            best = diff
            best_transform = shift
            best_neigh[:] = (ii, jj, kk)
    # We've finished our loop. return the best shift, neighbor, and whether this
    # is a max
    # NOTE: Can't do is_max = best == 0.0 for older numba
    is_max = False
    if best == 0.0:
        is_max = True
    return best_transform, best_neigh, is_max


@njit(parallel=True, cache=True)
def get_steepest_pointers(
    data: NDArray[np.float64],
    initial_labels: NDArray[np.int64],
    neighbor_transforms: NDArray[np.int64],
    neighbor_dists: NDArray[np.int64],
    vacuum_mask: NDArray[np.bool_],
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
    vacuum_mask: NDArray[np.bool_]
        A 3D array representing the location of the vacuum

    Returns
    -------
    best_label : NDArray[np.int64]
        A 3D array where each entry is the index of the neighbor that had the
        greatest increase in value. A value of -1 indicates a vacuum point.

    """
    nx, ny, nz = data.shape
    # create array to store the label of the neighboring voxel with the greatest
    # elf value
    best_label = initial_labels.copy()
    # loop over each voxel in parallel
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # check if this is a vacuum point. If so, we don't even bother
                # with the label.
                if vacuum_mask[i, j, k]:
                    best_label[i, j, k] = -1
                    continue
                # get the best neighbor
                best_transform, best_neigh, _ = get_best_neighbor(
                    data=data,
                    i=i,
                    j=j,
                    k=k,
                    neighbor_transforms=neighbor_transforms,
                    neighbor_dists=neighbor_dists,
                )
                x, y, z = best_neigh
                best_label[i, j, k] = initial_labels[x, y, z]
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
        for shift_index, (shift, area_dist) in enumerate(
            zip(neighbor_transforms, neighbor_area_over_dist)
        ):
            ii = i + shift[0]
            jj = j + shift[1]
            kk = k + shift[2]
            # loop
            ii, jj, kk = wrap_point(ii, jj, kk, nx, ny, nz)
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


@njit(parallel=True, cache=True)
def reduce_weight_maxima(
    maxima_vox_coords: NDArray[np.int64],
    data: NDArray[np.float64],
    neighbor_transforms: NDArray[np.int64],
    neighbor_dists: NDArray[np.float64],
):
    """
    Determines whether each maximum found by the weight method is a true ongrid
    maximum.

    Parameters
    ----------
    maxima_vox_coords : NDArray[np.int64]
        The voxel coordinates of maxima determined from the weight method
    data : NDArray[np.float64]
        A 3D grid of values for each point.
    neighbor_transforms : NDArray[np.int64]
        The transformations from each voxel to its neighbors.
    neighbor_dists : NDArray[np.float64]
        The distance to each neighboring voxel.

    Returns
    -------
    maxima_indices : NDArray[np.int64]
        A mapping of each weight maximum to its corresponding ongrid maximum

    """
    # create tracker for connecting maxima
    maxima_indices = np.zeros(len(maxima_vox_coords), dtype=np.int64)
    for max_idx in prange(len(maxima_vox_coords)):
        # create scratch current coord
        current_coord = maxima_vox_coords[max_idx]
        i, j, k = current_coord
        # check if this is a maximum
        _, _, is_max = get_best_neighbor(
            data, i, j, k, neighbor_transforms, neighbor_dists
        )
        if is_max:
            # this is a real maximum
            maxima_indices[max_idx] = max_idx
            continue

        # otherwise this isn't a true maximum. Hill climb until on is reached
        while True:
            ii, jj, kk = current_coord
            _, new_coord, is_max = get_best_neighbor(
                data=data,
                i=ii,
                j=jj,
                k=kk,
                neighbor_transforms=neighbor_transforms,
                neighbor_dists=neighbor_dists,
            )
            if is_max:
                # this coord is a maximum. We want to set the fake maximum to
                # this true one
                for real_max_idx in range(len(maxima_vox_coords)):
                    rx, ry, rz = maxima_vox_coords[real_max_idx]
                    # check if this maxima matches where we currently are
                    if rx == ii and ry == jj and rz == kk:
                        maxima_indices[max_idx] = real_max_idx
                        break

                break
            # update the coord and continue
            current_coord = new_coord
    return maxima_indices


###############################################################################
# Functions for near grid method
###############################################################################


@njit(cache=True, parallel=True)
def get_ongrid_and_rgrads(
    data: NDArray[np.float64],
    car2lat: NDArray[np.float64],
    neighbor_transforms: NDArray[np.int64],
    neighbor_dists: NDArray[np.float64],
    vacuum_mask: NDArray[np.bool_],
):
    """
    Calculates the ongrid steps and delta r at each point in the grid

    Parameters
    ----------
    data : NDArray[np.float64]
        A 3D grid of values for each point.
    car2lat : NDArray[np.float64]
        A matrix that converts a coordinate in cartesian space to fractional
        space.
    neighbor_transforms : NDArray[np.int64]
        The transformations from each voxel to its neighbors.
    neighbor_dists : NDArray[np.float64]
        The distance to each neighboring voxel.
    vacuum_mask: NDArray[np.bool_]
        A 3D array representing the location of the vacuum.

    Returns
    -------
    highest_neighbors : NDArray[np.int64]
        A 4D array where highest_neighbors[i,j,k] returns the steepest neighbor at
        point (i,j,k)
    all_drs : NDArray[np.float64]
        A 4D array where all_drs[i,j,k] returns the delta r between the true
        gradient and ongrid step at point (i,j,k)
    maxima_mask : NDArray[np.bool_]
        A 3D array that is True at maxima

    """
    nx, ny, nz = data.shape
    # create array for storing maxima
    maxima_mask = np.zeros(data.shape, dtype=np.bool_)
    # Create a new array for storing pointers
    highest_neighbors = np.zeros((nx, ny, nz, 3), dtype=np.int64)
    # Create a new array for storing rgrads
    # Each (i, j, k) index gives the rgrad [x, y, z]
    all_drs = np.zeros((nx, ny, nz, 3), dtype=np.float64)
    # loop over each grid point in parallel
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # check if this point is part of the vacuum. If it is, we can
                # ignore this point.
                if vacuum_mask[i, j, k]:
                    continue
                voxel_coord = np.array([i, j, k], dtype=np.int64)
                # get gradient
                gradient = get_gradient_simple(
                    data=data,
                    voxel_coord=voxel_coord,
                    car2lat=car2lat,
                )
                max_grad = np.max(np.abs(gradient))
                if max_grad < 1e-30:
                    # we have no gradient so we reset the total delta r
                    # Check if this is a maximum and if not step ongrid
                    shift, neigh, is_max = get_best_neighbor(
                        data=data,
                        i=i,
                        j=j,
                        k=k,
                        neighbor_transforms=neighbor_transforms,
                        neighbor_dists=neighbor_dists,
                    )
                    # set pointer
                    highest_neighbors[i, j, k] = neigh
                    # set dr to 0 because we used an ongrid step
                    all_drs[i, j, k] = (0.0, 0.0, 0.0)
                    if is_max:
                        maxima_mask[i, j, k] = True
                    continue
                # Normalize
                gradient /= max_grad
                # get pointer
                pointer = np.round(gradient)
                # get dr
                delta_r = gradient - pointer
                # get neighbor. Don't bother wrapping because we will do this later
                neighbor = voxel_coord + pointer
                # save neighbor and dr
                highest_neighbors[i, j, k] = neighbor
                all_drs[i, j, k] = delta_r
    return highest_neighbors, all_drs, maxima_mask


@njit(fastmath=True, cache=True)
def get_neargrid_labels(
    data: NDArray[np.float64],
    highest_neighbors: NDArray[np.int64],
    all_drs: NDArray[np.float64],
    maxima_mask: NDArray[np.bool_],
    vacuum_mask: NDArray[np.bool_],
    neighbor_transforms: NDArray[np.int64],
    neighbor_dists: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """
    Assigns each point to a basin using the neargrid method.

    Parameters
    ----------
    data : NDArray[np.float64]
        A 3D grid of values for each point.
    highest_neighbors : NDArray[np.int64]
        A 4D array where highest_neighbors[i,j,k] returns the steepest neighbor at
        point (i,j,k)
    all_drs : NDArray[np.float64]
        A 4D array where all_drs[i,j,k] returns the delta r between the true
        gradient and ongrid step at point (i,j,k)
    maxima_mask : NDArray[np.bool_]
        A 3D array that is True at maxima
    vacuum_mask: NDArray[np.bool_]
        A 3D array representing the location of the vacuum.
    neighbor_transforms : NDArray[np.int64]
        The transformations from each voxel to its neighbors.
    neighbor_dists : NDArray[np.float64]
        The distance to each neighboring voxel.

    Returns
    -------
    labels : NDArray[np.int64]
        The assignment for each point on the grid.

    """
    nx, ny, nz = data.shape
    # define an array to assign to
    labels = np.zeros(data.shape, dtype=np.int64)
    # create a scratch array for our path
    path = np.empty((nx * ny * nz, 3), dtype=np.int64)
    # create a count of basins
    maxima_num = 1
    # create a scratch value for delta r
    total_delta_r = np.zeros(3, dtype=np.float64)
    current_coord = np.zeros(3, dtype=np.int64)
    # loop over all voxels
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # check if this point is part of the vacuum. If so, we don't
                # need to relabel it, so we continue.
                if vacuum_mask[i, j, k]:
                    continue
                # check if we've already assigned this point
                if labels[i, j, k] != 0:
                    continue
                # reset our delta_r
                total_delta_r[:] = 0.0
                # reset count for the length of the path
                pnum = 0
                # start climbing
                current_coord[:] = (i, j, k)

                while True:
                    ii, jj, kk = current_coord
                    # It shouldn't be possible to have entered the vacuum because
                    # it will alwasy be lower than valid points
                    assert not vacuum_mask[ii, jj, kk]

                    # check if we've hit another label
                    current_label = labels[ii, jj, kk]
                    if current_label != 0:
                        # relabel everything in our path and move to the next
                        # voxel
                        for p in range(pnum):
                            x, y, z = path[p]
                            # relabel to the this neighbors value
                            labels[x, y, z] = current_label
                        break  # move to next voxel
                    # check if we've hit a maximum
                    if maxima_mask[ii, jj, kk]:
                        # keep the path labeled as is, and update the current
                        # point to the same label, then increment the maxima count
                        labels[ii, jj, kk] = maxima_num
                        maxima_num += 1
                        break

                    # We have an unlabeled, non-max point and need to continue
                    # our climb
                    # Assign this point to the current maximum.
                    # NOTE: We must relabel this as part of the vacuum later
                    labels[ii, jj, kk] = maxima_num
                    # add it to our path
                    path[pnum] = (ii, jj, kk)
                    pnum = pnum + 1
                    # make a neargrid step
                    # 1. get pointer and delta r
                    new_coord = highest_neighbors[ii, jj, kk].copy()
                    delta_r = all_drs[ii, jj, kk]
                    # 2. sum delta r
                    total_delta_r += delta_r
                    # 3. update new coord and total delta r
                    new_coord += np.rint(total_delta_r).astype(np.int64)
                    total_delta_r -= np.round(total_delta_r)
                    # 4. wrap coord
                    new_coord[:] = wrap_point(
                        new_coord[0], new_coord[1], new_coord[2], nx, ny, nz
                    )
                    new_label = labels[new_coord[0], new_coord[1], new_coord[2]]
                    is_vac = vacuum_mask[new_coord[0], new_coord[1], new_coord[2]]
                    # Check if the new coord is already on the path
                    if new_label == maxima_num or is_vac:
                        # we need to make an ongrid step to avoid repeating steps
                        # or wandering into the vacuum
                        _, new_coord, _ = get_best_neighbor(
                            data=data,
                            i=ii,
                            j=jj,
                            k=kk,
                            neighbor_transforms=neighbor_transforms,
                            neighbor_dists=neighbor_dists,
                        )
                    # update the coord
                    current_coord = new_coord
    return labels


@njit(cache=True)
def refine_neargrid(
    data: NDArray[np.float64],
    labels: NDArray[np.int64],
    refinement_indices: NDArray[np.int64],
    refinement_mask: NDArray[np.bool_],
    checked_mask: NDArray[np.bool_],
    maxima_mask: NDArray[np.bool_],
    highest_neighbors: NDArray[np.int64],
    all_drs: NDArray[np.float64],
    neighbor_transforms: NDArray[np.int64],
    neighbor_dists: NDArray[np.float64],
    vacuum_mask: NDArray[np.bool_],
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
    highest_neighbors : NDArray[np.int64]
        A 4D array where highest_neighbors[i,j,k] returns the steepest neighbor at
        point (i,j,k)
    all_drs : NDArray[np.float64]
        A 4D array where all_drs[i,j,k] returns the delta r between the true
        gradient and ongrid step at point (i,j,k)
    neighbor_transforms : NDArray[np.int64]
        The transformations from each voxel to its neighbors.
    neighbor_dists : NDArray[np.float64]
        The distance to each neighboring voxel.
    vacuum_mask: NDArray[np.bool_]
        A 3D array representing the location of the vacuum.

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
        # get our initial label for comparison
        label = labels[i, j, k]
        # reset the delta r tracker
        total_delta_r[:] = 0.0
        # create a count for the length of the path
        pnum = 0
        # set the initial coord
        current_coord[:] = (i, j, k)
        # start climbing
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
                # remove all points from the path
                for p in range(pnum):
                    x, y, z = path[p]
                    labels[x, y, z] = abs(labels[x, y, z])
                # We've hit a maximum.
                current_label = labels[ii, jj, kk]
                # Check if this is a reassignment
                if label != current_label:
                    reassignments += 1
                    # add neighbors to our refinement mask for the next iteration
                    for shift in neighbor_transforms:
                        # get the new neighbor
                        ni = i + shift[0]
                        nj = j + shift[1]
                        nk = k + shift[2]
                        # loop
                        ni, nj, nk = wrap_point(ni, nj, nk, nx, ny, nz)
                        # If we haven't already checked this point, add it.
                        # NOTE: vacuum points are stored in the mask by default
                        if not checked_mask[ni, nj, nk]:
                            refinement_mask[ni, nj, nk] = True
                # relabel just this voxel then stop the loop
                new_labels[i, j, k] = current_label
                break

            # Otherwise, we have not reached a maximum and want to continue
            # climbine
            # add this label to our path by marking it as negative.
            labels[ii, jj, kk] = -labels[ii, jj, kk]
            path[pnum] = (ii, jj, kk)
            pnum = pnum + 1
            # make a neargrid step
            # 1. get pointer and delta r
            new_coord = highest_neighbors[ii, jj, kk].copy()
            delta_r = all_drs[ii, jj, kk]
            # 2. sum delta r
            total_delta_r += delta_r
            # 3. update new coord and total delta r
            new_coord += np.rint(total_delta_r).astype(np.int64)
            total_delta_r -= np.round(total_delta_r)
            # 4. wrap coord
            new_coord[:] = wrap_point(
                new_coord[0], new_coord[1], new_coord[2], nx, ny, nz
            )
            # check if the new coord is already in our path or belongs to the
            # vacuum
            temp_label = labels[new_coord[0], new_coord[1], new_coord[2]]
            is_vac = vacuum_mask[new_coord[0], new_coord[1], new_coord[2]]
            if temp_label < 0 or is_vac:
                # we default back to an ongrid step to avoid repeating steps
                _, new_coord, _ = get_best_neighbor(
                    data=data,
                    i=ii,
                    j=jj,
                    k=kk,
                    neighbor_transforms=neighbor_transforms,
                    neighbor_dists=neighbor_dists,
                )
                # reset delta r to avoid further loops
                total_delta_r[:] = 0.0
                # check again if we're still in the same path. If so, cancel
                # the loop and don't write anything
                if labels[new_coord[0], new_coord[1], new_coord[2]] < 0:
                    for p in range(pnum):
                        x, y, z = path[p]
                        labels[x, y, z] = abs(labels[x, y, z])
                        break
            # update the current coord
            current_coord = new_coord

    return new_labels, reassignments, refinement_mask, checked_mask


#####################################################################################
# Sorted Near-grid method
#####################################################################################


@njit(cache=True)
def get_path_parent(
    flat_labels,
    initial_label,
    neighbor_label,
):
    current_label = neighbor_label
    while True:
        new_label = flat_labels[current_label]
        if new_label == current_label or new_label == initial_label or new_label == -1:
            break
        current_label = new_label
    return new_label


@njit(cache=True, parallel=True)
def get_sorted_neargrid_labels(
    data: NDArray[np.float64],
    sorted_voxel_coords,
    car2lat: NDArray[np.float64],
    neighbor_transforms: NDArray[np.int64],
    neighbor_dists: NDArray[np.float64],
    vacuum_mask: NDArray[np.bool_],
    initial_labels,
):
    nx, ny, nz = data.shape
    # create array for storing maxima
    maxima_mask = np.zeros(data.shape, dtype=np.bool_)
    # Create a new array for storing pointers
    highest_neighbors = np.zeros((nx, ny, nz, 3), dtype=np.int64)
    # Create a new array for storing rgrads
    # Each (i, j, k) index gives the rgrad [x, y, z]
    all_drs = np.zeros((nx, ny, nz, 3), dtype=np.float64)
    # loop over each grid point in parallel
    for vox_idx in prange(len(sorted_voxel_coords)):
        i, j, k = sorted_voxel_coords[vox_idx]
        # check if this point is part of the vacuum. If it is, we can
        # ignore this point.
        if vacuum_mask[i, j, k]:
            continue
        voxel_coord = np.array([i, j, k], dtype=np.int64)
        # get gradient
        gradient = get_gradient_simple(
            data=data,
            voxel_coord=voxel_coord,
            car2lat=car2lat,
        )
        max_grad = np.max(np.abs(gradient))
        if max_grad < 1e-30:
            # we have no gradient so we reset the total delta r
            # Check if this is a maximum and if not step ongrid
            shift, neigh, is_max = get_best_neighbor(
                data=data,
                i=i,
                j=j,
                k=k,
                neighbor_transforms=neighbor_transforms,
                neighbor_dists=neighbor_dists,
            )
            # set pointer
            highest_neighbors[i, j, k] = neigh
            # set dr to 0 because we used an ongrid step
            all_drs[i, j, k] = (0.0, 0.0, 0.0)
            if is_max:
                maxima_mask[i, j, k] = True
            continue
        # Normalize
        gradient /= max_grad
        # get pointer
        pointer = np.round(gradient)
        # get dr
        delta_r = gradient - pointer
        # get neighbor
        ni, nj, nk = voxel_coord + pointer
        ni, nj, nk = wrap_point(ni, nj, nk, nx, ny, nz)
        # save neighbor and dr
        highest_neighbors[i, j, k] = (ni, nj, nk)
        all_drs[i, j, k] += delta_r
        # add drs to total_dr
        all_drs[int(ni), int(nj), int(nk)] += delta_r

    # do another loop to assign pointers
    # create a flat list of labels
    flat_labels = np.full(nx * ny * nz, -1, dtype=np.int64)
    # loop over each grid point in parallel
    for i, j, k in sorted_voxel_coords:
        initial_label = initial_labels[i, j, k]
        # check if this point is part of the vacuum. If it is, we can
        # ignore this point.
        if vacuum_mask[i, j, k]:
            continue
        # if this is a maximum assign to self
        if maxima_mask[i, j, k]:
            flat_labels[initial_label] = initial_label
            continue
        # adjust neighbor
        ni, nj, nk = highest_neighbors[i, j, k]
        ri, rj, rk = all_drs[i, j, k]
        ni += round(ri)
        nj += round(rj)
        nk += round(rk)
        # wrap
        ni, nj, nk = wrap_point(ni, nj, nk, nx, ny, nz)
        # At this point, several things could go wrong.
        # 1. We hit a vacuum point
        # 2. We connect to a path that loops back to the current point
        # Either of these will result in small unrealistic basins. We correct by
        # making an ongrid step
        neighbor_index = initial_labels[ni, nj, nk]
        neighbor_label = flat_labels[neighbor_index]  # current neigh assignment
        is_self = (
            neighbor_index == initial_label
        )  # if the neighbor is the current point
        is_vacuum = vacuum_mask[ni, nj, nk]  # if the neighbor is in the vacuum
        if neighbor_label != -1:  # if the neighbor has an assignment already
            # we need to check that this point doesn't loop to itself
            parent_label = get_path_parent(flat_labels, initial_label, neighbor_index)
            is_self = parent_label == initial_label
        if is_self or is_vacuum:
            shift, neigh, is_max = get_best_neighbor(
                data=data,
                i=i,
                j=j,
                k=k,
                neighbor_transforms=neighbor_transforms,
                neighbor_dists=neighbor_dists,
            )
            # assign to the neighbor's label
            flat_labels[initial_label] = initial_labels[neigh[0], neigh[1], neigh[2]]
            # don't adjust and drs because we used an ongrid step
            continue
        # assign pointer
        flat_labels[initial_label] = neighbor_index
    return flat_labels, maxima_mask
