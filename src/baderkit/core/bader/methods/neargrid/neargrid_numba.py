# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from baderkit.core.utilities.basic import (
    coords_to_flat,
    flat_to_coords,
    wrap_point,
    wrap_point_w_shift,
)
from baderkit.core.utilities.basins import get_best_neighbor_with_shift
from baderkit.core.utilities.transforms import IMAGE_TO_INT


@njit(cache=True)
def get_gradient(
    data: NDArray[np.float64],
    voxel_coord: NDArray[np.int64],
    dir2lat: NDArray[np.float64],
    use_minima: bool = False,
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

    gi = (charge100 - charge_100) / 2.0
    gj = (charge010 - charge0_10) / 2.0
    gk = (charge001 - charge00_1) / 2.0

    if charge100 <= charge000 and charge_100 <= charge000:
        gi = 0.0
    if charge010 <= charge000 and charge0_10 <= charge000:
        gj = 0.0
    if charge001 <= charge000 and charge00_1 <= charge000:
        gk = 0.0

    # convert to direct
    # NOTE: Doing this rather than the original car2lat with two np.dot operations
    # saves about half the time.
    r0 = dir2lat[0, 0] * gi + dir2lat[0, 1] * gj + dir2lat[0, 2] * gk
    r1 = dir2lat[1, 0] * gi + dir2lat[1, 1] * gj + dir2lat[1, 2] * gk
    r2 = dir2lat[2, 0] * gi + dir2lat[2, 1] * gj + dir2lat[2, 2] * gk

    # if finding minima, flip the sign
    if use_minima:
        r0 = -r0
        r1 = -r1
        r2 = -r2

    return r0, r1, r2


# NOTE
# This is an alternative method for calculating the gradient that uses all of
# the neighbors for each grid point to get an overdetermined system with improved
# sampling. I didn't find it made a big difference.
@njit(cache=True, inline="always")
def get_gradient_overdetermined(
    data,
    i,
    j,
    k,
    vox_transforms,
    transform_dists,
    car2lat,
    inv_norm_cart_trans,
    use_minima: bool = False,
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
    # inv_norm_cart_trans = np.linalg.pinv(norm_cart_transforms) where
    # norm_cart_transforms is an N, 3 shaped array pointing to 13 unique neighbors
    ti, tj, tk = inv_norm_cart_trans @ diffs
    # Convert Cartesian gradient to fractional (lattice) coordinates
    ti_new = car2lat[0, 0] * ti + car2lat[0, 1] * tj + car2lat[0, 2] * tk
    tj_new = car2lat[1, 0] * ti + car2lat[1, 1] * tj + car2lat[1, 2] * tk
    tk_new = car2lat[2, 0] * ti + car2lat[2, 1] * tj + car2lat[2, 2] * tk

    ti, tj, tk = ti_new, tj_new, tk_new

    if use_minima:
        ti = -ti
        tj = -tj
        tk = -tk

    return ti, tj, tk


# NOTE: I used to calculate and store the ongrid steps and delta rs in this first
# method rather than just the gradient. This was a tiny bit faster but not worth
# the extra memory usage in my opinion. - S. Weaver
@njit(cache=True, parallel=True)
def get_gradient_pointers_simple(
    data: NDArray[np.float64],
    labels: NDArray[np.int64],
    images: NDArray[np.int64],
    dir2lat: NDArray[np.float64],
    neighbor_transforms: NDArray[np.int64],
    neighbor_dists: NDArray[np.float64],
    vacuum_mask: NDArray[np.bool_],
    extrema_mask: NDArray[np.bool_],
    use_minima: bool = False,
):
    """
    Calculates the ongrid steps and delta r at each point in the grid

    Parameters
    ----------
    data : NDArray[np.float64]
        A 3D grid of values for each point.
    labels : NDArray[np.int64]
        A 1D grid with extrema assignments
    images : NDArray[np.int64]
        A Nx3 array of images tracking cycles around periodic edges
    dir2lat : NDArray[np.float64]
        A matrix for converting from direct coordinates to lattice coords
    neighbor_transforms : NDArray[np.int64]
        The transformations from each voxel to its neighbors.
    neighbor_dists : NDArray[np.float64]
        The distance to each neighboring voxel.
    vacuum_mask : NDArray[np.bool_]
        A 3D array representing the location of the vacuum.
    extrema_mask : NDArray[np.bool_]
        A 3D array representing the location of local extrema in the grid
    use_minima : bool, optional
        Whether or not to use the negative of the gradient to move towards
        minima

    Returns
    -------
    pointers : NDArray[np.int64]
        A 3D array where each entry is the index of the neighbor that is most
        along the gradient. A value of -1 indicates a vacuum point.
    gradients : NDArray[np.float32]
        A 4D array where gradients[i,j,k] returns the gradient at point (i,j,k)
    extrema_mask : NDArray[np.bool_]
        A 3D array that is True at extrema

    """
    nx, ny, nz = data.shape
    ny_nz = ny * nz
    # Create a new array for storing gradients
    # NOTE: I would even do a float16 here but numba doesn't support it. I doubt
    # we need the accuracy.
    gradients = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    if use_minima:
        mult = -1
    else:
        mult = 1
    # loop over each grid point in parallel
    for flat_idx in prange(len(labels)):
        i, j, k = flat_to_coords(flat_idx, ny_nz, nz)
        # check if this point is part of the vacuum. If it is, we can
        # ignore this point.
        if vacuum_mask[i, j, k]:
            continue
        # check if this point is a maximum. If so, we should already have
        # given this point an assignment previously
        if extrema_mask[i, j, k]:
            continue
        # get gradient
        gi, gj, gk = get_gradient(
            data=data,
            voxel_coord=(i, j, k),
            dir2lat=dir2lat,
            use_minima=use_minima,
        )
        # get the largest gradient direction
        max_grad = 0.0
        for x in (gi, gj, gk):
            ax = abs(x)
            if ax > max_grad:
                max_grad = ax
        if max_grad < 1e-30:
            # we have no gradient
            # Check if this is a maximum and if not step ongrid
            (si, sj, sk), (ni, nj, nk), shift = get_best_neighbor_with_shift(
                data=data,
                i=i,
                j=j,
                k=k,
                neighbor_transforms=neighbor_transforms,
                neighbor_dists=neighbor_dists,
                use_minima=use_minima,
            )
            # set gradient and point. Note gradient is exactly ongrid in
            # this instance
            gradients[i, j, k] = (si, sj, sk)
            labels[flat_idx] = coords_to_flat(ni, nj, nk, ny_nz, nz)
            images[flat_idx] = shift

            continue
        # Normalize gradient
        gi /= max_grad
        gj /= max_grad
        gk /= max_grad
        # get pointer
        pi, pj, pk = round(gi), round(gj), round(gk)
        # get neighbor and wrap
        ni, nj, nk, si, sj, sk = wrap_point_w_shift(i + pi, j + pj, k + pk, nx, ny, nz)
        shift = np.array((si, sj, sk), dtype=np.int8)
        # Ensure neighbor is higher than the current point, or backup to
        # ongrid.

        if mult * data[i, j, k] >= mult * data[ni, nj, nk]:
            (gi, gj, gk), (ni, nj, nk), shift = get_best_neighbor_with_shift(
                data=data,
                i=i,
                j=j,
                k=k,
                neighbor_transforms=neighbor_transforms,
                neighbor_dists=neighbor_dists,
                use_minima=use_minima,
            )

        # save neighbor, dr, and pointer
        gradients[i, j, k] = (gi, gj, gk)
        labels[flat_idx] = coords_to_flat(ni, nj, nk, ny_nz, nz)
        images[flat_idx] = shift

    return labels, images, gradients


# NOTE: This is an alternative method that calculates the gradient using all
# 26 neighbors rather than just the 6 face sharing ones. I didn't find it to
# make an appreciable difference for NaCl or a rotated H2O molecule, but these
# were both on cubic grids, so it's possible it makes a bigger difference in
# more skewed systems.
@njit(cache=True, parallel=True)
def get_gradient_pointers_overdetermined(
    data: NDArray[np.float64],
    labels: NDArray[np.int64],
    images: NDArray[np.int64],
    car2lat,
    inv_norm_cart_trans,
    neighbor_transforms: NDArray[np.int64],
    neighbor_dists: NDArray[np.float64],
    vacuum_mask: NDArray[np.bool_],
    extrema_mask: NDArray[np.bool_],
    use_minima: bool = False,
):
    """
    Calculates the ongrid steps and delta r at each point in the grid

    Parameters
    ----------
    data : NDArray[np.float64]
        A 3D grid of values for each point.
    labels : NDArray[np.int64]
        A 1D grid with extrema assignments
    images : NDArray[np.int64]
        A Nx3 array of images tracking cycles around periodic edges
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
    extrema_mask : NDArray[np.bool_]
        A 3D array that is True at extrema
    use_minima : bool, optional
        Whether or not to use the negative of the gradient to move towards
        minima

    Returns
    -------
    pointers : NDArray[np.int64]
        A 3D array where each entry is the index of the neighbor that is most
        along the gradient. A value of -1 indicates a vacuum point.
    gradients : NDArray[np.float32]
        A 4D array where gradients[i,j,k] returns the gradient at point (i,j,k)
    extrema_mask : NDArray[np.bool_]
        A 3D array that is True at extrema

    """
    nx, ny, nz = data.shape
    ny_nz = ny * nz

    # Create a new array for storing gradients
    # NOTE: I would even do a float16 here but numba doesn't support it. I doubt
    # we need the accuracy.
    gradients = np.zeros((nx, ny, nz, 3), dtype=np.float32)

    # get half the transforms for overdetermined gradient
    half_trans = neighbor_transforms[:13]
    half_dists = neighbor_dists[:13]
    # loop over each grid point in parallel
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # get the flat index of this point
                flat_idx = coords_to_flat(i, j, k, ny_nz, nz)
                # check if this point is part of the vacuum. If it is, we can
                # ignore this point.
                if vacuum_mask[i, j, k]:
                    continue
                if extrema_mask[i, j, k]:
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
                    use_minima=use_minima,
                )
                max_grad = 0.0
                for x in (gi, gj, gk):
                    ax = abs(x)
                    if ax > max_grad:
                        max_grad = ax
                if max_grad < 1e-30:
                    # we have no gradient so we reset the total delta r
                    # Check if this is a maximum and if not step ongrid
                    (si, sj, sk), (ni, nj, nk), shift = get_best_neighbor_with_shift(
                        data=data,
                        i=i,
                        j=j,
                        k=k,
                        neighbor_transforms=neighbor_transforms,
                        neighbor_dists=neighbor_dists,
                        use_minima=use_minima,
                    )
                    # set gradient and point. Note gradient is exactly ongrid in
                    # this instance
                    gradients[i, j, k] = (si, sj, sk)
                    labels[flat_idx] = coords_to_flat(ni, nj, nk, ny_nz, nz)
                    images[flat_idx] = shift
                    continue
                # Normalize
                gi /= max_grad
                gj /= max_grad
                gk /= max_grad
                # get pointer
                pi, pj, pk = round(gi), round(gj), round(gk)
                # get neighbor and wrap
                ni, nj, nk, si, sj, sk = wrap_point_w_shift(
                    i + pi, j + pj, k + pk, nx, ny, nz
                )
                shift = np.array((si, sj, sk), dtype=np.int8)
                # Ensure neighbor is higher than the current point, or backup to
                # ongrid.
                if use_minima:
                    mult = -1
                else:
                    mult = 1
                if mult * data[i, j, k] >= mult * data[ni, nj, nk]:
                    _, (ni, nj, nk), shift = get_best_neighbor_with_shift(
                        data=data,
                        i=i,
                        j=j,
                        k=k,
                        neighbor_transforms=neighbor_transforms,
                        neighbor_dists=neighbor_dists,
                        use_minima=use_minima,
                    )
                # save neighbor, dr, and pointer
                gradients[i, j, k] = (gi, gj, gk)
                labels[flat_idx] = coords_to_flat(ni, nj, nk, ny_nz, nz)
                images[flat_idx] = shift

    return labels, images, gradients


@njit(parallel=True, cache=True)
def refine_fast_neargrid(
    data: NDArray[np.float64],
    labels: NDArray[np.int64],
    images: NDArray[np.int64],
    refinement_mask: NDArray[np.bool_],
    extrema_mask: NDArray[np.bool_],
    gradients: NDArray[np.float32],
    neighbor_transforms: NDArray[np.int64],
    neighbor_dists: NDArray[np.float64],
    vacuum_label: int,
    use_minima: bool = False,
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
    images : NDArray[np.int64]
        A Nx3 array of images tracking cycles around periodic edges
    refinement_mask : NDArray[np.bool_]
        A 3D mask that is true at the voxel indices to be refined.
    extrema_mask : NDArray[np.bool_]
        A 3D mask that is true at extrema.
    gradients : NDArray[np.float16]
        A 4D array where gradients[i,j,k] returns the gradient at point (i,j,k)
    neighbor_transforms : NDArray[np.int64]
        The transformations from each voxel to its neighbors.
    neighbor_dists : NDArray[np.float64]
        The distance to each neighboring voxel.
    use_minima : bool, optional
        Whether or not to use the negative of the gradient to move towards
        minima

    Returns
    -------
    labels : NDArray[np.int64]
        The updated assignment for each point on the grid.

    """
    # get shape
    nx, ny, nz = data.shape
    ny_nz = ny * nz
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
            flat_idx = coords_to_flat(i, j, k, ny_nz, nz)
            # get our initial label for comparison. We need to take absolute value
            # because refined labels are marked as negative
            label = abs(labels[i, j, k])
            # get initial image
            mi, mj, mk = images[flat_idx]
            image = IMAGE_TO_INT[mi, mj, mk]
            # initialize shift tracking for wrapping around periodic boundaries
            wi, wj, wk = (0, 0, 0)
            # create delta r
            tdi, tdj, tdk = (0.0, 0.0, 0.0)
            # set the initial coord
            ii, jj, kk = (i, j, k)
            # create a list to store the path
            path = []
            # start climbing
            while True:
                # check if we've hit an extrema or a vacuum in minima mode
                if extrema_mask[ii, jj, kk] or labels[ii, jj, kk] == vacuum_label:
                    # remove the point from the refinement list
                    refinement_mask[i, j, k] = False
                    # We've hit a maximum.
                    current_label = abs(labels[ii, jj, kk])
                    current_image = IMAGE_TO_INT[wi, wj, wk]
                    # Check if this is a reassignment
                    if label != current_label or image != current_image:
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
                    images[flat_idx] = (wi, wj, wk)
                    break

                # Otherwise, we have not reached a maximum and want to continue
                # climbing
                # add this point to our path
                current_index = coords_to_flat(ii, jj, kk, ny_nz, nz)
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
                # 5. wrap coord
                ni, nj, nk, si, sj, sk = wrap_point_w_shift(ni, nj, nk, nx, ny, nz)
                # 6. Get flat index
                new_idx = coords_to_flat(ni, nj, nk, ny_nz, nz)
                # check if we've hit a point in the path or a vacuum point
                if new_idx in path or abs(labels[ni, nj, nk]) == vacuum_label:
                    _, (ni, nj, nk), shift = get_best_neighbor_with_shift(
                        data=data,
                        i=ii,
                        j=jj,
                        k=kk,
                        neighbor_transforms=neighbor_transforms,
                        neighbor_dists=neighbor_dists,
                        use_minima=use_minima,
                    )
                    si, sj, sk = shift
                    # reset delta r because we used an ongrid step
                    tdi, tdj, tdk = (0.0, 0.0, 0.0)

                # update the current coord
                ii, jj, kk = ni, nj, nk
                # add any wrapping to our total
                wi += si
                wj += sj
                wk += sk
        print(f"{reassignments} values changed")

    return labels, images
