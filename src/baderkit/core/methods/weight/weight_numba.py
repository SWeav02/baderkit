# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from baderkit.core.methods.shared_numba import get_best_neighbor, wrap_point


@njit(parallel=True, cache=True)
def get_neighbor_flux(
    data: NDArray[np.float64],
    sorted_coords: NDArray[np.int64],
    sorted_pointers: NDArray[np.int64],
    neighbor_transforms: NDArray[np.int64],
    neighbor_alpha: NDArray[np.float64],
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
    sorted_coords : NDArray[np.int64]
        A Nx3 array where each entry represents the voxel coordinates of the
        point. This must be sorted from highest value to lowest.
    sorted_pointers : NDArray[np.int64]
        A 3D array where each entry is the sorted index of the coord
    neighbor_transforms : NDArray[np.int64]
        The transformations from each voxel to its VORONOI neighbors.
    neighbor_alpha : NDArray[np.float64]
        The area of the voronoi facet divided by the distance to the voxel
    all_neighbor_transforms
        The transformations from each voxel to each of its 26 nearest neighbors
    all_neighbor_dists
        The distance from each voxe to each of its 26 neighbors

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
        (len(sorted_coords), len(neighbor_transforms)), dtype=np.float64
    )
    neigh_array = np.full(flux_array.shape, -1, dtype=np.int64)
    # create a mask for the location of maxima
    maxima_mask = np.zeros(len(sorted_coords), dtype=np.bool_)
    # Loop over each voxel in parallel (except the vacuum points)
    for coord_index in prange(len(sorted_coords)):
        i, j, k = sorted_coords[coord_index]
        # get the initial value
        base_value = data[i, j, k]
        # create a counter for the total flux
        total_flux = 0.0
        # iterate over each neighbor sharing a voronoi facet
        neigh_n = 0
        for (si, sj, sk), alpha in zip(neighbor_transforms, neighbor_alpha):
            # get neighbor and wrap around periodic boundary
            ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
            # get the neighbors value
            neigh_value = data[ii, jj, kk]
            # if this value is below the current points value, continue
            if neigh_value <= base_value:
                continue
            # calculate the flux flowing to this voxel
            flux = (neigh_value - base_value) * alpha
            # assign flux
            flux_array[coord_index, neigh_n] = flux
            total_flux += flux
            # add the pointer to this neighbor
            neigh_array[coord_index, neigh_n] = sorted_pointers[ii, jj, kk]
            # add to our neighbor count
            neigh_n += 1

        # Check that we had at least one assignment. If not, this might be a
        # local maximum
        if total_flux == 0.0:
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
            neigh_array[coord_index, 0] = sorted_pointers[ni, nj, nk]
            continue
        
        # otherwise, normalize the flux
        flux_array[coord_index] /= total_flux

    return flux_array, neigh_array, maxima_mask

@njit(fastmath=True, cache=True)
def get_weight_assignments(
    data,
    sorted_coords,
    sorted_charge,
    original_indices,
    neigh_fluxes,
    neigh_pointers,
    weight_maxima_mask,
        ):
    # create arrays to store charges, volumes, and pointers
    charges = []
    volumes = []
    basin_pointers = np.full(data.shape, -1, dtype=np.int64)
    # create array to store volume
    sorted_volume = np.full(len(sorted_charge),1.0, dtype=np.float64)
    # now iterate over coords from lowest to highest
    for idx in range(len(sorted_charge)):
        # get all info
        i,j,k = sorted_coords[idx]
        charge = sorted_charge[idx]
        volume = sorted_volume[idx]
        is_max = weight_maxima_mask[idx]
        # if this is a maximum, create a new basin
        if is_max:
            charges.append(charge)
            volumes.append(volume)
            basin_pointers[i,j,k] = original_indices[i,j,k]
            continue
        
        # get neighbor/flux info
        pointers = neigh_pointers[idx]
        fluxes = neigh_fluxes[idx]
        # otherwise, add charge and volume to neighbors, and get best neighbor
        highest_flux = 0.0
        best_neighbor = -1
        for pointer, flux in zip(pointers, fluxes):
            if pointer == -1:
                # We have reached the last neighbor and break
                break
            sorted_charge[pointer] += charge * flux
            sorted_volume[pointer] += volume * flux
            # check if flux is greater than or equal than the current max flux
            # within a tolerance
            if flux > highest_flux:
                highest_flux = flux
                best_neighbor = pointer

        # assign this point to the best neighbor
        ni,nj,nk = sorted_coords[best_neighbor]
        basin_pointers[i,j,k] = original_indices[ni,nj,nk]
        
    return (
        basin_pointers, 
        np.array(charges, dtype=np.float64), 
        np.array(volumes, dtype=np.float64),
        )

@njit(fastmath=True)
def get_labels(
    pointers,
    sorted_indices,
        ):
    # Assuming sorted_pointers is from high to low, we only need to loop over
    # the values once to assign all of them.
    for idx in sorted_indices:
        # skip vacuum assignments
        if pointers[idx] == -1:
            continue
        # assign to parent
        pointers[idx] = pointers[pointers[idx]]
    return pointers

@njit(fastmath=True, cache=True)
def reduce_charge_volume(
    basin_map,
    charges,
    volumes,
    basin_num,
        ):
    # create a new array for charges and volumes
    new_charges = np.zeros(basin_num, dtype=np.float64)
    new_volumes = np.zeros(basin_num, dtype=np.float64)
    for i in range(len(charges)):
        basin = basin_map[i]
        new_charges[basin] += charges[i]
        new_volumes[basin] += volumes[i]            
    return new_charges, new_volumes

###############################################################################
# For better labeling
###############################################################################

@njit(fastmath=True)
def get_labels_fine(
    label_array,
    maxima_labels,
    flat_grid_indices,
    sorted_indices,
    neigh_pointers,
    # neigh_fluxes,
    true_volumes,
    sorted_coords,
        ):
    # create an array to store approximate volumes
    # approx_volumes = np.zeros(len(volumes), dtype=np.int64)
    # Create an array to store the reorganized volumes (likely to be different)
    volumes = np.empty(len(true_volumes), dtype=np.float64)
    # Create an array to store the difference from the ideal volume
    volume_diff = np.ones(len(true_volumes), dtype=np.float64)
    # Create an array to store the ratio by which the volume_diff changes when
    # a new voxe is added to the corresponding basin
    volume_ratios = np.empty(len(true_volumes), dtype=np.float64)
    # create an array to note if a voxel is split
    # split_voxels = np.zeros(len(pointers), dtype=np.bool_)
    # loop over points from high to low
    maxima_num = 0
    for idx in np.arange(len(sorted_coords)-1, -1, -1):
        # get the pointers/flux
        pointers = neigh_pointers[idx]
        # fluxes = neigh_fluxes[idx]
        i,j,k = sorted_coords[idx]
        # reduce to labels/weights
        labels = []
        # weights = []
        # for pointer, flux in zip(pointers, fluxes):
        for pointer in pointers:
            # if the pointer is -1 we've reached the end of our list
            if pointer == -1:
                break
            # otherwise, get the label at this point
            ni, nj, nk = sorted_coords[pointer]
            label = label_array[ni,nj,nk]
            # check if the label exists. If not, add it
            found = False
            for lidx, rlabel in enumerate(labels):
                if label == rlabel:
                    found = True
                    # weights[lidx] += flux
            if not found:
                # add the new label/weight
                labels.append(label)
                # weights.append(flux)
        
        # If there are 0 labels, this is a maximum. We assign a new basin
        if len(labels) == 0:
            # label the voxel
            label_array[i,j,k] = maxima_num
            # get the true volume for this voxel
            voxel_index = flat_grid_indices[i,j,k]
            for maxima_label, true_volume in zip(maxima_labels, true_volumes):
                if maxima_label == voxel_index:
                    volumes[maxima_num] = true_volume
                    volume_ratios[maxima_num] = 1.0 / true_volume
                    break
            # approx_volumes[maxima_num] += 1
            volume_diff[maxima_num] -= volume_ratios[maxima_num]
            maxima_num += 1
        # If there is 1 label, assign this label
        elif len(labels) == 1:
            label_array[i,j,k] = labels[0]
            # approx_volumes[labels[0]] += 1
            volume_diff[labels[0]] -= volume_ratios[labels[0]]
        # if there is more than 1 label, we have a split voxel. As an approximation,
        # we check how far from the true volume each possible basin is and add
        # the voxel to the farthest one.
        else:
            best_label = -1
            best_diff = -1.0
            for label in labels:
                if volume_diff[label] > best_diff:
                    best_label = label
                    best_diff = volume_diff[label]
            # update label
            label_array[i,j,k] = best_label
            # update diff
            volume_diff[best_label] -= volume_ratios[best_label]
            
            # best_label = -1
            # best_improvement = -1.0
            # for label, weight in zip(labels, weights):
            #     diff = volumes[label] - approx_volumes[label]
            #     improvement = (abs(diff) - abs(diff - 1)) / abs(volumes[label])
            #     if improvement > best_improvement:
            #         best_label = label
            #         best_improvement = improvement       
            # # update label
            # label_array[i,j,k] = best_label
            # approx_volumes[best_label] += 1
    return label_array

# @njit(fastmath=True, cache=True)
# def relabel_edges(
#     to_refine,
#     sorted_coords,
#     sorted_charge,
#     labels,
#     fluxes,
#     neigh_pointers,
#     basin_num,
#     use_charge = False,
#         ):
#     # create tracker for volumes
#     volumes = np.zeros(basin_num, dtype=np.float64)
#     approx_volumes = np.zeros(basin_num, dtype=np.int64)
#     for idx in to_refine:
#         # get fluxes/pointers and charge for this point
#         neigh_fluxes = fluxes[idx]
#         pointers = neigh_pointers[idx]        

#         # Now we loop over the labels to determine the one that would improve
#         # the approximate volume the best. 
#         # create tracker for label that gets the best ratio
#         best_label = -1
#         best_improvement = -1.0
#         charge = sorted_charge[idx]
#         # for flux, label in zip(reduced_fluxes, reduced_labels):
#         for pointer, flux in zip(pointers, neigh_fluxes):
#             if pointer == -1:
#                 continue
#             # get this neighbors label
#             ni,nj,nk = sorted_coords[pointer]
#             label = labels[ni,nj,nk]
#             # assign volume
#             if use_charge:
#                 volumes[label] += flux * charge
#             else:
#                 volumes[label] += flux
#             # calculate how much the approximate volume would improve if we add
#             # the full voxel to the volume. Added volume is always 1
#             diff = volumes[label] - approx_volumes[label]
#             if use_charge:
#                 improvement = (abs(diff) - abs(diff - charge)) / abs(volumes[label])
#             else:
#                 improvement = (abs(diff) - abs(diff - 1)) / abs(volumes[label])

#             # if adding this volume would improve the approximate charge more
#             # than the other labels checked so far, update to use this label
#             if improvement > best_improvement:
#                 best_label = label
#                 best_improvement = improvement
#         # update label for this point
#         i,j,k = sorted_coords[idx]
#         labels[i,j,k] = best_label
#         # update approximate volume
#         if use_charge:
#             approx_volumes[best_label] += charge
#         else:
#             approx_volumes[best_label] += 1
#     return labels
            
        
