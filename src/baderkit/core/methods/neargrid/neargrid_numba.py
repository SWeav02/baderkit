# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from baderkit.core.methods.shared_numba import (
    get_best_neighbor,
    get_gradient_overdetermined,
    get_gradient_simple,
    wrap_point,
)

# NOTE: I used to calculate and store the ongrid steps and delta rs in this first
# method rather than just the gradient. This was a tiny bit faster but not worth
# the extra memory usage in my opinion. - S. Weaver


@njit(cache=True, parallel=True)
def get_gradient_pointers_simple(
    data: NDArray[np.float64],
    dir2lat: NDArray[np.float64],
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
    dir2lat : NDArray[np.float64]
        A matrix for converting from direct coordinates to lattice coords
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
    pointers : NDArray[np.int64]
        A 3D array where each entry is the index of the neighbor that is most
        along the gradient. A value of -1 indicates a vacuum point.
    gradients : NDArray[np.float32]
        A 4D array where gradients[i,j,k] returns the gradient at point (i,j,k)
    maxima_mask : NDArray[np.bool_]
        A 3D array that is True at maxima

    """
    nx, ny, nz = data.shape
    # create array for storing maxima
    maxima_mask = np.zeros(data.shape, dtype=np.bool_)
    # Create a new array for storing gradients
    # NOTE: I would even do a float16 here but numba doesn't support it. I doubt
    # we need the accuracy.
    gradients = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    # create array for storing pointers
    pointers = initial_labels.copy()
    # loop over each grid point in parallel
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # check if this point is part of the vacuum. If it is, we can
                # ignore this point.
                if vacuum_mask[i, j, k]:
                    pointers[i, j, k] = -1
                    continue
                # get gradient
                gi, gj, gk = get_gradient_simple(
                    data=data,
                    voxel_coord=(i, j, k),
                    dir2lat=dir2lat,
                )
                max_grad = 0.0
                for x in (gi, gj, gk):
                    ax = abs(x)
                    if ax > max_grad:
                        max_grad = ax
                if max_grad < 1e-30:
                    # we have no gradient so we reset the total delta r
                    # Check if this is a maximum and if not step ongrid
                    (si, sj, sk), (ni, nj, nk), is_max = get_best_neighbor(
                        data=data,
                        i=i,
                        j=j,
                        k=k,
                        neighbor_transforms=neighbor_transforms,
                        neighbor_dists=neighbor_dists,
                    )
                    # set gradient and point. Note gradient is exactly ongrid in
                    # this instance
                    gradients[i, j, k] = (si, sj, sk)
                    pointers[i, j, k] = initial_labels[ni, nj, nk]
                    if is_max:
                        maxima_mask[i, j, k] = True
                    continue
                # Normalize
                gi /= max_grad
                gj /= max_grad
                gk /= max_grad
                # get pointer
                pi, pj, pk = round(gi), round(gj), round(gk)
                # get neighbor and wrap
                ni, nj, nk = wrap_point(i + pi, j + pj, k + pk, nx, ny, nz)
                # Ensure neighbor is higher than the current point, or backup to
                # ongrid.
                if data[i, j, k] >= data[ni, nj, nk]:
                    shift, (ni, nj, nk), is_max = get_best_neighbor(
                        data=data,
                        i=i,
                        j=j,
                        k=k,
                        neighbor_transforms=neighbor_transforms,
                        neighbor_dists=neighbor_dists,
                    )
                    if is_max:
                        # NOTE: when would this happen?
                        maxima_mask[i, j, k] = True
                # save neighbor, dr, and pointer
                gradients[i, j, k] = (gi, gj, gk)
                pointers[i, j, k] = initial_labels[ni, nj, nk]
    return pointers, gradients, maxima_mask


# NOTE: This is an alternative method that calculates the gradient using all
# 26 neighbors rather than just the 6 face sharing ones. I didn't find it to
# make an appreciable difference for NaCl or a rotated H2O molecule, but these
# were both on cubic grids, so it's possible it makes a bigger difference in
# more skewed systems.
@njit(cache=True, parallel=True)
def get_gradient_pointers_overdetermined(
    data: NDArray[np.float64],
    car2lat,
    inv_norm_cart_trans,
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
        A matrix for converting from cartesian coordinates to lattice coords
    inv_norm_cart_trans : NDArray[np.float64]
        pseudo inverse of normalized cartesian transforms to neighbors
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
    pointers : NDArray[np.int64]
        A 3D array where each entry is the index of the neighbor that is most
        along the gradient. A value of -1 indicates a vacuum point.
    gradients : NDArray[np.float32]
        A 4D array where gradients[i,j,k] returns the gradient at point (i,j,k)
    maxima_mask : NDArray[np.bool_]
        A 3D array that is True at maxima

    """
    nx, ny, nz = data.shape
    # create array for storing maxima
    maxima_mask = np.zeros(data.shape, dtype=np.bool_)
    # Create a new array for storing gradients
    # NOTE: I would even do a float16 here but numba doesn't support it. I doubt
    # we need the accuracy.
    gradients = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    # create array for storing pointers
    pointers = initial_labels.copy()
    # get half the transforms for overdetermined gradient
    half_trans = neighbor_transforms[:13]
    half_dists = neighbor_dists[:13]
    # loop over each grid point in parallel
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # check if this point is part of the vacuum. If it is, we can
                # ignore this point.
                if vacuum_mask[i, j, k]:
                    pointers[i, j, k] = -1
                    continue
                # get gradient
                gi, gj, gk = get_gradient_overdetermined(
                    data,
                    i,
                    j,
                    k,
                    vox_transforms=half_trans,
                    transform_dists=half_dists,
                    car2lat=car2lat,
                    inv_norm_cart_trans=inv_norm_cart_trans,
                )
                max_grad = 0.0
                for x in (gi, gj, gk):
                    ax = abs(x)
                    if ax > max_grad:
                        max_grad = ax
                if max_grad < 1e-30:
                    # we have no gradient so we reset the total delta r
                    # Check if this is a maximum and if not step ongrid
                    (si, sj, sk), (ni, nj, nk), is_max = get_best_neighbor(
                        data=data,
                        i=i,
                        j=j,
                        k=k,
                        neighbor_transforms=neighbor_transforms,
                        neighbor_dists=neighbor_dists,
                    )
                    # set gradient and point. Note gradient is exactly ongrid in
                    # this instance
                    gradients[i, j, k] = (si, sj, sk)
                    pointers[i, j, k] = initial_labels[ni, nj, nk]
                    if is_max:
                        maxima_mask[i, j, k] = True
                    continue
                # Normalize
                gi /= max_grad
                gj /= max_grad
                gk /= max_grad
                # get pointer
                pi, pj, pk = round(gi), round(gj), round(gk)
                # get neighbor and wrap
                ni, nj, nk = wrap_point(i + pi, j + pj, k + pk, nx, ny, nz)
                # Ensure neighbor is higher than the current point, or backup to
                # ongrid.
                if data[i, j, k] >= data[ni, nj, nk]:
                    shift, (ni, nj, nk), is_max = get_best_neighbor(
                        data=data,
                        i=i,
                        j=j,
                        k=k,
                        neighbor_transforms=neighbor_transforms,
                        neighbor_dists=neighbor_dists,
                    )
                    if is_max:
                        # NOTE: when would this happen?
                        maxima_mask[i, j, k] = True
                # save neighbor, dr, and pointer
                gradients[i, j, k] = (gi, gj, gk)
                pointers[i, j, k] = initial_labels[ni, nj, nk]
    return pointers, gradients, maxima_mask


@njit(parallel=True, cache=True)
def refine_fast_neargrid(
    data: NDArray[np.float64],
    labels: NDArray[np.int64],
    refinement_mask: NDArray[np.bool_],
    maxima_mask: NDArray[np.bool_],
    gradients: NDArray[np.float32],
    neighbor_transforms: NDArray[np.int64],
    neighbor_dists: NDArray[np.float64],
    initial_labels: NDArray[np.int64],
) -> NDArray[np.int64]:
    """
    Refines the provided voxels by running the neargrid method until a maximum
    is found for each.

    Parameters
    ----------
    data : NDArray[np.float64]
        A 3D grid of values for each point.
    labels : NDArray[np.int64]
        A 3D grid of labels representing current voxel assignments.
    refinement_mask : NDArray[np.bool_]
        A 3D mask that is true at the voxel indices to be refined.
    maxima_mask : NDArray[np.bool_]
        A 3D mask that is true at maxima.
    gradients : NDArray[np.float16]
        A 4D array where gradients[i,j,k] returns the gradient at point (i,j,k)
    neighbor_transforms : NDArray[np.int64]
        The transformations from each voxel to its neighbors.
    neighbor_dists : NDArray[np.float64]
        The distance to each neighboring voxel.

    Returns
    -------
    labels : NDArray[np.int64]
        The updated assignment for each point on the grid.

    """
    # get shape
    nx, ny, nz = data.shape
    # refine iteratively until no assignments change
    reassignments = 1
    while reassignments > 0:
        # get refinement indices
        refinement_indices = np.argwhere(refinement_mask)
        if len(refinement_indices) == 0:
            # there's nothing to refine so we break
            break
        print(f"Refining {len(refinement_indices)} points")
        # now we reassign any voxel in our refinement mask
        # NOTE: this reassignment count may not be perfectly accurate if any race
        # conditions occur due to the parallelization
        reassignments = 0
        for vox_idx in prange(len(refinement_indices)):
            i, j, k = refinement_indices[vox_idx]
            # get our initial label for comparison. We need to take absolute value
            # because refined labels are marked as negative
            label = abs(labels[i, j, k])
            # create delta r
            tdi, tdj, tdk = (0.0, 0.0, 0.0)
            # set the initial coord
            ii, jj, kk = (i, j, k)
            # create a list to store the path
            path = []
            # start climbing
            while True:
                # check if we've hit a maximum
                if maxima_mask[ii, jj, kk]:
                    # remove the point from the refinement list
                    refinement_mask[i, j, k] = False
                    # We've hit a maximum.
                    current_label = abs(labels[ii, jj, kk])
                    # Check if this is a reassignment
                    if label != current_label:
                        reassignments += 1
                        # add neighbors to our refinement mask for the next iteration
                        for si, sj, sk in neighbor_transforms:
                            # get new neighbor and wrap
                            ni, nj, nk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
                            # If we haven't already checked this point, add it.
                            # The vacuum and previously checked values are less than
                            # or equal to 0
                            if labels[ni, nj, nk] > 0:
                                refinement_mask[ni, nj, nk] = True
                                # note we don't want to reassign this again in the
                                # future
                                labels[ni, nj, nk] = -abs(labels[ni, nj, nk])
                    # relabel just this voxel then stop the loop
                    labels[i, j, k] = -current_label
                    break

                # Otherwise, we have not reached a maximum and want to continue
                # climbing
                # add this point to our path
                current_index = initial_labels[ii, jj, kk]
                path.append(current_index)
                # make a neargrid step
                # 1. get gradient
                gi, gj, gk = gradients[ii, jj, kk]
                # 2. Round to obtain a pointer to the neighbor most along this gradient
                pi = round(gi)
                pj = round(gj)
                pk = round(gk)
                # get neighbor. Don't wrap yet since we'll do that later anyways
                ni = ii + pi
                nj = jj + pj
                nk = kk + pk
                # 3. Add difference to the total dr
                tdi += gi - pi
                tdj += gj - pj
                tdk += gk - pk
                # 4. update new coord and total delta r
                ni += round(tdi)
                nj += round(tdj)
                nk += round(tdk)
                tdi -= round(tdi)
                tdj -= round(tdj)
                tdk -= round(tdk)
                # 4. wrap coord
                ni, nj, nk = wrap_point(ni, nj, nk, nx, ny, nz)
                # check if we've hit a point in the path or a vacuum point
                if initial_labels[ni, nj, nk] in path or not labels[ni, nj, nk]:
                    _, (ni, nj, nk), _ = get_best_neighbor(
                        data=data,
                        i=ii,
                        j=jj,
                        k=kk,
                        neighbor_transforms=neighbor_transforms,
                        neighbor_dists=neighbor_dists,
                    )
                    # reset delta r because we used an ongrid step
                    tdi, tdj, tdk = (0.0, 0.0, 0.0)
                # update the current coord
                ii, jj, kk = ni, nj, nk
        print(f"{reassignments} values changed")

    return labels
