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
def get_ongrid_rgrads_pointers(
    data: NDArray[np.float64],
    car2lat: NDArray[np.float64],
    neighbor_transforms: NDArray[np.int64],
    neighbor_dists: NDArray[np.float64],
    vacuum_mask: NDArray[np.bool_],
    initial_labels: NDArray[np.int64],
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
    vacuum_mask : NDArray[np.bool_]
        A 3D array representing the location of the vacuum.
    initial_labels : NDArray[np.int64]
        A 3D array represengint the flat indices of each grid point

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
    # create array for storing pointers
    pointers = initial_labels.copy()
    # loop over each grid point in parallel
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # check if this point is part of the vacuum. If it is, we can
                # ignore this point.
                if vacuum_mask[i, j, k]:
                    pointers[i,j,k] = -1
                    continue
                # voxel_coord = np.array([i, j, k], dtype=np.int64)
                # get gradient
                gi, gj, gk = get_gradient_simple(
                    data=data,
                    voxel_coord=(i, j, k),
                    car2lat=car2lat,
                )
                max_grad = 0.0
                for x in (gi, gj, gk):
                    ax = abs(x)
                    if ax > max_grad:
                        max_grad = ax
                if max_grad < 1e-30:
                    # we have no gradient so we reset the total delta r
                    # Check if this is a maximum and if not step ongrid
                    shift, (ni, nj, nk), is_max = get_best_neighbor(
                        data=data,
                        i=i,
                        j=j,
                        k=k,
                        neighbor_transforms=neighbor_transforms,
                        neighbor_dists=neighbor_dists,
                    )
                    # set neighbor and pointer
                    highest_neighbors[i, j, k] = (ni, nj, nk)
                    pointers[i,j,k] = initial_labels[ni, nj, nk]
                    # set dr to 0 because we used an ongrid step
                    all_drs[i, j, k] = (0.0, 0.0, 0.0)
                    if is_max:
                        maxima_mask[i, j, k] = True
                    continue
                # Normalize
                gi /= max_grad
                gj /= max_grad
                gk /= max_grad
                # get pointer
                pi, pj, pk = round(gi), round(gj), round(gk)
                # get dr
                di = gi - pi
                dj = gj - pj
                dk = gk - pk
                # get neighbor and wrap
                ni, nj, nk = wrap_point(i+pi, j+pj, k+pk, nx, ny, nz)
                # Ensure neighbor is higher than the current point, or backup to
                # ongrid.
                if data[i,j,k] > data[ni, nj, nk]:
                    shift, (ni, nj, nk), is_max = get_best_neighbor(
                        data=data,
                        i=i,
                        j=j,
                        k=k,
                        neighbor_transforms=neighbor_transforms,
                        neighbor_dists=neighbor_dists,
                    )
                    di, dj, dk = (0.0, 0.0, 0.0)
                # save neighbor, dr, and pointer
                highest_neighbors[i, j, k] = (ni, nj, nk)
                all_drs[i, j, k] = (di, dj, dk)
                pointers[i,j,k] = initial_labels[ni,nj,nk]
    return pointers, highest_neighbors, all_drs, maxima_mask

@njit(parallel=True, cache=True)
def refine_fast_neargrid(
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
    # vacuum_mask: NDArray[np.bool_],
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

    Returns
    -------
    labels : NDArray[np.int64]
        The updated assignment for each point on the grid.
    reassignments : np.int64
        The number of points that were reassigned.
    refinement_mask : NDArray[np.bool_]
        The updated mask of points that need to be refined
    checked_mask : NDArray[np.bool_]
        The updated mask of points that have been checked.

    """
    # get shape
    nx, ny, nz = data.shape

    # now we reassign any voxel in our refinement mask
    reassignments = 0
    for vox_idx in prange(len(refinement_indices)):
    # for i, j, k in refinement_indices:
        i,j,k = refinement_indices[vox_idx]
        # get our initial label for comparison
        label = labels[i, j, k]
        # create delta r
        tdi, tdj, tdk = (0.0, 0.0, 0.0)
        # set the initial coord
        # ci, cj, ck = (i, j, k)
        ii, jj, kk = (i, j, k)
        # start climbing
        while True:
            # ii, jj, kk = ci, cj, ck
            # check if we've hit a maximum
            if maxima_mask[ii, jj, kk]:
                # add this point to our checked list. We use this to make sure
                # this point doesn't get re-added to our list later in the
                # process.
                checked_mask[i, j, k] = True
                # remove it from the refinement list
                refinement_mask[i, j, k] = False
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
                labels[i, j, k] = current_label
                break

            # Otherwise, we have not reached a maximum and want to continue
            # climbing
            # make a neargrid step
            # 1. get pointer and delta r
            hi, hj, hk = highest_neighbors[ii, jj, kk]
            di, dj, dk = all_drs[ii, jj, kk]
            # 2. sum delta r
            tdi += di
            tdj += dj
            tdk += dk
            # 3. update new coord and total delta r
            hi += round(tdi)
            hj += round(tdj)
            hk += round(tdk)
            tdi -= round(tdi)
            tdj -= round(tdj)
            tdk -= round(tdk)
            # 4. wrap coord
            hi, hj, hk = wrap_point(hi, hj, hk, nx, ny, nz)
            # make sure the new point has a higher value than the current one
            # or back up to ongrid
            if data[ii, jj, kk] > data[hi ,hj, hk]:
                # we default back to an ongrid step to avoid repeating steps
                _, (hi, hj, hk), _ = get_best_neighbor(
                    data=data,
                    i=ii,
                    j=jj,
                    k=kk,
                    neighbor_transforms=neighbor_transforms,
                    neighbor_dists=neighbor_dists,
                )
                # reset delta r to avoid further loops
                tdi, tdj, tdk = (0.0, 0.0, 0.0)
                # check again if we're still in the same path. If so, cancel
                # the loop and don't write anything
            # update the current coord
            ii, jj, kk = hi, hj, hk

    return labels, reassignments, refinement_mask, checked_mask
