# -*- coding: utf-8 -*-
import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from baderkit.core.methods.shared_numba import (
    get_best_neighbor,
    get_gradient_simple,
    wrap_point,
)


@njit(cache=True, parallel=True)
def get_pseudo_neargrid_labels(
    data: NDArray[np.float64],
    sorted_voxel_coords: NDArray[np.int64],
    car2lat: NDArray[np.float64],
    neighbor_transforms: NDArray[np.int64],
    neighbor_dists: NDArray[np.float64],
    vacuum_mask: NDArray[np.bool_],
    initial_labels: NDArray[np.int64],
):
    """
    Gets the pointers for the pseudo-neargrid method.

    Parameters
    ----------
    data : NDArray[np.float64]
        A 3D grid of values for each point.
    sorted_voxel_coords : NDArray[np.int64]
        A Nx3 array where each entry represents the voxel coordinates of the
        point. This must be sorted from highest value to lowest.
    car2lat : NDArray[np.float64]
        A matrix that converts a coordinate in cartesian space to fractional
        space.
    neighbor_transforms : NDArray[np.int64]
        The transformations from each voxel to its neighbors.
    neighbor_dists : NDArray[np.float64]
        The distance to each neighboring voxel.
    vacuum_mask : NDArray[np.bool_]
        A 3D array representing the location of the vacuum.
    initial_labels : NDArray[np.int64]
        A 3D array where each entry represents the basin label of the point.

    Returns
    -------
    flat_labels : NDArray[np.int64]
        The pointer for each grid point to its highest neighbors index.
    maxima_mask : NDArray[np.bool_]
        A 3D mask that is true at maxima.

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
        # 2. We point back to this point
        # 3. We point to a point that is lower than this one
        # These will result in unrealistic basins, so we correct with an ongrid
        # step
        neighbor_index = initial_labels[ni, nj, nk]
        # neighbor_label = flat_labels[neighbor_index]  # current neigh assignment
        is_self = (
            neighbor_index == initial_label
        )  # if the neighbor is the current point
        is_vacuum = vacuum_mask[ni, nj, nk]  # if the neighbor is in the vacuum
        is_lower = data[i, j, k] >= data[ni, nj, nk]
        if is_self or is_vacuum or is_lower:
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
