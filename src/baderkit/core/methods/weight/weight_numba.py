# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from baderkit.core.methods.shared_numba import wrap_point, coords_to_flat, climb_to_max, flat_to_coords


@njit(parallel=True, cache=True)
def get_neighbor_flux(
    data: NDArray[np.float64],
    neighbor_transforms: NDArray[np.int64],
    neighbor_alpha: NDArray[np.float64],
    sorted_indices: NDArray[np.int64],
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
    num_coords = nx * ny * nz
    # create array to store labels
    labels = np.full(data.shape, -1, dtype=np.int64)
    # create empty 2D arrays to store the volume flux flowing from each voxel
    # to its neighbor and the voxel indices of these neighbors. We ignore the
    # voxels that are below the vacuum value
    flux_array = np.zeros((num_coords, len(neighbor_transforms)), dtype=np.float64)
    neigh_array = np.full(flux_array.shape, -1, dtype=np.int32)
    # create a mask for the location of maxima
    maxima_mask = np.zeros(num_coords, dtype=np.bool_)
    # Loop over each voxel in parallel
    for sorted_pointer in prange(len(sorted_indices)):
        # get regular flat index for this point
        coord_index = sorted_indices[sorted_pointer]
        # get coords
        i, j, k = flat_to_coords(coord_index, nx, ny, nz)
        # get initial value
        base_value = data[i, j, k]
        # get the flat coord index
        coord_index = coords_to_flat(i,j,k,nx,ny,nz)
        # create a counter for the total flux
        total_flux = 0.0
        # create lists for the flux and neighbors
        fluxes = []
        neighs = []
        trans = []
        # iterate over each neighbor sharing a voronoi facet
        best_flux = 0.0
        best_label = -1
        for trans_idx, ((si, sj, sk), alpha) in enumerate(zip(neighbor_transforms, neighbor_alpha)):
            # get neighbor and wrap around periodic boundary
            ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
            # get the neighbors value
            neigh_value = data[ii, jj, kk]
            # if this value is below the current points value, continue
            if neigh_value <= base_value:
                continue
            # get the neighbors flat coord index
            neigh_index = coords_to_flat(ii,jj,kk,nx,ny,nz)
            # calculate the flux flowing to this voxel
            flux = (neigh_value - base_value) * alpha
            total_flux += flux
            if flux > best_flux:
                best_flux = flux
                best_label = neigh_index
            # add flux, neighbor, and transform index to list
            fluxes.append(flux)
            neighs.append(neigh_index)
            trans.append(trans_idx)
        # If total flux is 0.0 this is a maximum (relative to weight methods tranforms)
        if total_flux == 0.0:
            maxima_mask[coord_index] = True
            # set label to self
            labels[i,j,k] = coord_index
            continue
        # set label
        labels[i,j,k] = best_label
        
        # normalize flux
        for flux_idx in range(len(fluxes)):
            fluxes[flux_idx] /= total_flux
        # assign flux/neighs
        for flux, neigh_index, trans_idx in zip(fluxes, neighs, trans):
            # assign flux to the neighbor
            flux_array[neigh_index, trans_idx] = flux
            total_flux += flux
            # point this neighbor back to this voxel
            neigh_array[neigh_index, trans_idx] = sorted_pointer

    return flux_array, neigh_array, labels, maxima_mask

@njit(parallel=True, cache=True)
def get_weight_assignments(
    data,
    labels,
    sorted_charge,
    sorted_indices,
    neigh_fluxes,
    neigh_pointers,
    weight_maxima_mask,
    all_neighbor_transforms,
    all_neighbor_dists,
        ):
    nx, ny, nz = data.shape
    # Get the indices corresponding to maxima
    maxima_pointers = np.where(weight_maxima_mask)[0]
    maxima_num = len(maxima_pointers)
    # create array to store the true maximum each local maxima belongs to. This
    # is used to reduce false weight maxima
    maxima_map = np.empty(maxima_num, dtype=np.int64)
    # Create array for storing labels
    # labels = np.full(data.shape, -1, dtype=np.int64)
    # create a scratch array for storing pointers and fluxes
    # tol = 1e-12
    # tol1 = 1-tol
    
    charges = np.zeros(maxima_num, dtype=np.float64)
    volumes = np.zeros(maxima_num, dtype=np.float64)
    # loop over maxima
    for max_idx in prange(maxima_num):
        max_pointer = maxima_pointers[max_idx]
        # create boolean array to note which points belong partially to this
        # maximum
        included = np.zeros(len(sorted_charge), dtype=np.bool_)
        included[max_pointer] = True
        # create an array to note the fraction of each voxel assigned to this
        # basin
        fracs = np.zeros(len(sorted_charge), dtype=np.float64)
        fracs[max_pointer] = True
        # loop over points in charge order
        for idx in range(len(included)):
            if not included[idx]:
                continue
            
            # get the frac at this point
            frac = fracs[idx]
            # if the frac is below a certain tolerance, continue
            # if frac < tol:
            #     continue
            # get the charge at this point
            charge = sorted_charge[idx]
            # get the pointers/fluxes
            pointers = neigh_pointers[idx]
            fluxes = neigh_fluxes[idx]
            
            # check if the fraction is 1 within some error. If it is, we can
            # simplify math a bit
            # if frac > tol1:
            #     # add charge from this point
            #     charges[max_idx] += charge
            #     volumes[max_idx] += 1.0
            #     # loop over the pointers/fracs
            #     for pointer, flux in zip(pointers, fluxes):
            #         if pointer == -1:
            #             continue
            #         # note this point is part of the basin
            #         included[pointer] = True
            #         # add the flux to this points overall frac
            #         fracs[pointer] += flux
            #     continue
            # otherwise, we do the same but with slightly more laborous math
            charges[max_idx] += charge * frac
            volumes[max_idx] += frac
            for pointer, flux in zip(pointers, fluxes):
                if pointer == -1:
                    continue
                # note this point is part of the basin
                included[pointer] = True
                # add the flux to this points overall frac
                fracs[pointer] += flux * frac
        
        # Now we check if this is a true maximum.
        # hill climb to find the true maximum this point should be assigned to
        i,j,k = flat_to_coords(sorted_indices[max_pointer], nx, ny, nz)
        mi, mj, mk = climb_to_max(data, i, j, k, all_neighbor_transforms, all_neighbor_dists)
        # update maxima map and labels. Note we use the unsorted index here
        pointer = coords_to_flat(mi,mj,mk,nx,ny,nz)
        labels[i,j,k] = pointer
        maxima_map[max_idx] = pointer
        
    # reduce to true maxima
    true_maxima = np.unique(maxima_map)
    reduced_charges = np.zeros(len(true_maxima), dtype=np.float64)
    reduced_volumes = np.zeros(len(true_maxima), dtype=np.float64)
    for old_idx, max_label in enumerate(maxima_map):
        for max_idx, true_max in enumerate(true_maxima):
            if max_label == true_max:
                reduced_charges[max_idx] += charges[old_idx]
                reduced_volumes[max_idx] += volumes[old_idx]
        
    return charges, volumes, labels, true_maxima

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


