# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from baderkit.core.methods.shared_numba import get_best_neighbor, wrap_point


@njit(parallel=True, cache=True)
def get_gradient_pointers(
    initial_labels: NDArray[np.int64],
    data: NDArray[np.float64],
    neighbor_transforms: NDArray[np.int64],
    neighbor_dists: NDArray[np.float64],
    weighted_cart: NDArray[np.float64],
    # all_neighbor_transforms: NDArray[np.int64],
    # all_neighbor_dists: NDArray[np.float64],
    vacuum_mask: NDArray[np.bool_],
    car2lat: NDArray[np.float64],
):
    """
    Finds each points highest neighbor by summing weighted gradient vectors
    normal to each voronoi surface.

    Parameters
    ----------
    initial_labels : NDArray[np.int64]
        A 3D array represengint the flat indices of each grid point
    data : NDArray[np.float64]
        A 3D grid of values for each point.
    neighbor_transforms : NDArray[np.int64]
        The transformations from each voxel to its voronoi neighbors.
    weighted_cart : NDArray[np.float64]
        Vectors pointing to each voronoi nieghbor weighted by facet area and distance.
    all_neighbor_transforms : NDArray[np.int64]
        The transformations from each voxel to its 26 neighbors.
    all_neighbor_dists : NDArray[np.float64]
        The distance to each of the 26 neighboring voxels.
    vacuum_mask : NDArray[np.bool_]
        A 3D array representing the location of the vacuum.
    car2lat : NDArray[np.float64]
        A matrix for converting from cartesian coordinates to lattice coords

    Returns
    -------
    pointers : NDArray[np.int64]
        A 3D array where each entry is the index of the neighbor that is most
        along the gradient. A value of -1 indicates a vacuum point.
    maxima_mask : NDArray[np.bool_]
        A 3D array that is True at maxima

    """
    nx, ny, nz = data.shape
    # create array to store the label of the neighboring voxel with the greatest
    # elf value
    pointers = initial_labels.copy()
    # create a mask for maxima
    maxima_mask = np.zeros(data.shape, dtype=np.bool_)
    # get number of neighbors so we don't have to do it every loop
    neigh_num = len(neighbor_transforms)
    # loop over each voxel in parallel
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # check if this is a vacuum point. If so, we don't even bother
                # with the label.
                if vacuum_mask[i, j, k]:
                    pointers[i, j, k] = -1
                    continue
                # get the base value of this point
                base_value = data[i, j, k]
                # create a vector to store the total gradient
                ti, tj, tk = 0.0, 0.0, 0.0
                # loop over neighbors
                for x in range(neigh_num):
                    si, sj, sk = neighbor_transforms[x]
                    # wrap
                    ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
                    # get the neighbors value
                    neigh_value = data[ii, jj, kk]
                    # if the value is lower than the current point, ignore it
                    if neigh_value <= base_value:
                        continue

                    # get the weighted cartesian vector for this transform
                    ci, cj, ck = weighted_cart[x] * (neigh_value - base_value)
                    # add to the total
                    ti += ci
                    tj += cj
                    tk += ck
                # convert to frac coords
                # ti, tj, tk = np.array((ti, tj, tk), dtype=np.float64) @ inv_lattice_matrix
                # ti, tj, tk = np.dot(car2lat, np.array((ti, tj, tk), dtype=np.float64))
                ti_new = car2lat[0, 0] * ti + car2lat[0, 1] * tj + car2lat[0, 2] * tk
                tj_new = car2lat[1, 0] * ti + car2lat[1, 1] * tj + car2lat[1, 2] * tk
                tk_new = car2lat[2, 0] * ti + car2lat[2, 1] * tj + car2lat[2, 2] * tk

                ti, tj, tk = ti_new, tj_new, tk_new
                # If the gradient is 0, check if this is a max or default to a
                # ongrid step
                max_grad = 0.0
                for x in (ti, tj, tk):
                    ax = abs(x)
                    if ax > max_grad:
                        max_grad = ax
                if max_grad < 1e-30:
                    # Check if this is a maximum and if not step ongrid
                    shift, (ni, nj, nk), is_max = get_best_neighbor(
                        data=data,
                        i=i,
                        j=j,
                        k=k,
                        # neighbor_transforms=all_neighbor_transforms,
                        # neighbor_dists=all_neighbor_dists,
                        neighbor_transforms=neighbor_transforms,
                        neighbor_dists=neighbor_dists,
                    )
                    # set pointer
                    pointers[i, j, k] = initial_labels[ni, nj, nk]
                    # set dr to 0 because we used an ongrid step
                    if is_max:
                        maxima_mask[i, j, k] = True
                    continue
                # otherwise, normalize and get neighbor
                ni = round(ti / max_grad) + i
                nj = round(tj / max_grad) + j
                nk = round(tk / max_grad) + k
                # wrap
                ni, nj, nk = wrap_point(ni, nj, nk, nx, ny, nz)
                # Check if we've hit a point lower than the current one. If so we
                # default back to an ongrid step
                if data[ni, nj, nk] <= base_value:
                    # get ongrid neighbor
                    _, (ni, nj, nk), _ = get_best_neighbor(
                        data=data,
                        i=i,
                        j=j,
                        k=k,
                        # neighbor_transforms=all_neighbor_transforms,
                        # neighbor_dists=all_neighbor_dists,
                        neighbor_transforms=neighbor_transforms,
                        neighbor_dists=neighbor_dists,
                    )

                # set pointer
                pointers[i, j, k] = initial_labels[ni, nj, nk]
    return pointers, maxima_mask
