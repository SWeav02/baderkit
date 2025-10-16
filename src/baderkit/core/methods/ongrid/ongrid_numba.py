# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from baderkit.core.methods.shared_numba import coords_to_flat, get_best_neighbor


@njit(parallel=True, cache=True)
def get_steepest_pointers(
    data: NDArray[np.float64],
    labels: NDArray[np.int64],
    neighbor_transforms: NDArray[np.int64],
    neighbor_dists: NDArray[np.int64],
    vacuum_mask: NDArray[np.bool_],
    maxima_mask: NDArray[np.bool_],
):
    """
    For each voxel in a 3D grid of data, finds the index of the neighboring voxel with
    the highest value, weighted by distance.

    Parameters
    ----------
    data : NDArray[np.float64]
        A 3D grid of values for each point.
    labels : NDArray[np.int64]
        A 1D grid of assignments for each point in the grid
    neighbor_transforms : NDArray[np.int64]
        The transformations from each voxel to its neighbors.
    neighbor_dists : NDArray[np.int64]
        The distance to each neighboring voxel
    vacuum_mask : NDArray[np.bool_]
        A 3D array representing the location of the vacuum
    vacuum_mask : NDArray[np.bool_]
        A 3D array representing the location of local maxima in the grid

    Returns
    -------
    pointers : NDArray[np.int64]
        A 1D array where each entry is the index of the neighbor that had the
        greatest increase in value. A value of -1 indicates a vacuum point.
    maxima_mask : NDArray[np.bool_]
        A 3D array that is True at maxima

    """
    nx, ny, nz = data.shape
    ny_nz = ny * nz
    # loop over each voxel in parallel
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # get the flat index of this point
                flat_idx = coords_to_flat(i, j, k, ny_nz, nz)
                # check if this is a vacuum point. If so, we don't even bother
                # with the label.
                if vacuum_mask[i, j, k]:
                    continue
                # check if this is a maximum. If so, we should have assigned
                # this label earlier and we continue
                if maxima_mask[i, j, k]:
                    continue
                # get the best neighbor
                _, (x, y, z) = get_best_neighbor(
                    data=data,
                    i=i,
                    j=j,
                    k=k,
                    neighbor_transforms=neighbor_transforms,
                    neighbor_dists=neighbor_dists,
                )
                labels[flat_idx] = coords_to_flat(x, y, z, ny_nz, nz)

    return labels
