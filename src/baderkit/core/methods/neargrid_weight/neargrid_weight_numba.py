# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from baderkit.core.methods.shared_numba import (
    coords_to_flat,
    flat_to_coords,
    get_best_neighbor,
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


@njit(parallel=True, cache=True)
def get_edge_charges_volumes(
    reference_data,
    charge_data,
    edge_indices,
    sorted_indices,
    labels,
    charges,
    volumes,
    neighbor_transforms,
    neighbor_alpha,
    all_neighbor_transforms,
    all_neighbor_dists,
):
    nx, ny, nz = reference_data.shape
    # create an array to store neighbors and fluxes
    num_coords = len(sorted_indices)
    full_num_coords = nx * ny * nz

    # create arrays to store flux/neighs
    flux_array = np.empty((num_coords, len(neighbor_transforms)), dtype=np.float64)
    neigh_array = np.empty(flux_array.shape, dtype=np.uint32)

    # create arrays to store flat charges/volumes of edges
    flat_charge = np.zeros(full_num_coords, dtype=np.float64)
    flat_volume = np.ones(full_num_coords, dtype=np.float64)
    neigh_nums = np.empty(num_coords, dtype=np.uint8)

    # loop over edge indices
    for sorted_idx in prange(num_coords):
        edge_idx = sorted_indices[sorted_idx]
        # get coordinates of grid point
        i, j, k = edge_indices[edge_idx]
        voxel_idx = coords_to_flat(i, j, k, nx, ny, nz)
        # get the reference and charge data
        base_value = reference_data[i, j, k]
        # set flat charge for this point
        flat_charge[voxel_idx] = charge_data[i, j, k]
        # track flux
        total_flux = 0.0
        # calculate the flux going to each neighbor
        neigh_num = 0
        for (si, sj, sk), alpha in zip(neighbor_transforms, neighbor_alpha):
            # get neighbor and wrap around periodic boundary
            ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
            # get the neighbors value
            neigh_value = reference_data[ii, jj, kk]
            # if this value is below the current points value, continue
            if neigh_value <= base_value:
                continue
            # get this neighbors index
            neigh_idx = coords_to_flat(ii, jj, kk, nx, ny, nz)

            # calculate the flux flowing to this voxel
            flux = (neigh_value - base_value) * alpha
            # assign flux
            flux_array[sorted_idx, neigh_num] = flux
            total_flux += flux
            # add the pointer to this neighbor

            neigh_array[sorted_idx, neigh_num] = neigh_idx
            neigh_num += 1

        # check that there is flux. If not, we have a fake local maximum and
        # revert to an ongrid step
        if total_flux == 0.0:
            # this is a local maximum. Check if its a true max
            shift, (ni, nj, nk), is_max = get_best_neighbor(
                data=reference_data,
                i=i,
                j=j,
                k=k,
                neighbor_transforms=all_neighbor_transforms,
                neighbor_dists=all_neighbor_dists,
            )
            neigh_idx = coords_to_flat(ni, nj, nk, nx, ny, nz)
            neigh_nums[sorted_idx] = 1
            neigh_array[sorted_idx, 0] = neigh_idx
            flux_array[sorted_idx, 0] = 1.0
            continue

        # otherwise we assign as normal.
        neigh_nums[sorted_idx] = neigh_num
        # normalize and assign label
        flux_array[sorted_idx] /= total_flux

    # Now loop over points and assign charge/volume
    for sorted_idx, (neighs, fluxes, neigh_num) in enumerate(
        zip(neigh_array, flux_array, neigh_nums)
    ):
        edge_idx = sorted_indices[sorted_idx]
        # get coordinates of grid point
        i, j, k = edge_indices[edge_idx]
        voxel_idx = coords_to_flat(i, j, k, nx, ny, nz)
        # get charge/volume at this point
        charge = flat_charge[voxel_idx]
        volume = flat_volume[voxel_idx]
        # loop over our each neighbor
        for neigh_idx in range(neigh_num):
            neigh = neighs[neigh_idx]
            flux = fluxes[neigh_idx]
            # if the neighbor has no charge in our flat array, it is not an edge
            if flat_charge[neigh] == 0.0:
                ni, nj, nk = flat_to_coords(neigh, nx, ny, nz)
                # get this neighbors label
                neigh_label = labels[ni, nj, nk]
                # assign charge/volume to corresponding basin
                charges[neigh_label] += charge * flux
                volumes[neigh_label] += volume * flux
                continue
            # otherwise, the neighbor is also an edge. Add the charge/volume
            # to our flat arrays
            flat_charge[neigh] += charge * flux
            flat_volume[neigh] += volume * flux

    return charges, volumes
