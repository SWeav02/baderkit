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
from baderkit.core.utilities.transforms import IMAGE_TO_INT, INT_TO_IMAGE
from baderkit.core.utilities.union_find import find_root

###############################################################################
# Neighbor Finding Methods
###############################################################################


@njit(cache=True, inline="always")
def get_best_neighbor(
    data: NDArray[np.float64],
    i: np.int64,
    j: np.int64,
    k: np.int64,
    neighbor_transforms: NDArray[np.int64],
    neighbor_dists: NDArray[np.int64],
    use_minima: bool = False,
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
    use_min : bool
        Whether or not to search for the lowest neighbor rather than the highest

    Returns
    -------
    best_transform : NDArray[np.int64]
        The transformation to the best neighbor
    best_neigh : NDArray[np.int64]
        The coordinates of the best neigbhor

    """
    nx, ny, nz = data.shape
    # get the value at this point
    base = data[i, j, k]
    # create a tracker for the best increase in value
    best = 0.0
    # create initial best transform. Default to this point
    bti = 0
    btj = 0
    btk = 0
    # create initial best neighbor
    bni = i
    bnj = j
    bnk = k
    # For each neighbor get the difference in value and if its better
    # than any previous, replace the current best
    for (si, sj, sk), dist in zip(neighbor_transforms, neighbor_dists):
        # loop
        ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
        # calculate the difference in value taking into account distance
        if use_minima:
            diff = (base - data[ii, jj, kk]) / dist
        else:
            diff = (data[ii, jj, kk] - base) / dist
        # if better than the current best, note the best and the
        # current label
        if diff > best:
            best = diff
            bti = si
            btj = sj
            btk = sk
            bni = ii
            bnj = jj
            bnk = kk

    # return the best shift and neighbor
    return (
        np.array((bti, btj, btk), dtype=np.int64),
        np.array((bni, bnj, bnk), dtype=np.int64),
    )


@njit(cache=True, inline="always")
def get_best_neighbor_with_shift(
    data: NDArray[np.float64],
    i: np.int64,
    j: np.int64,
    k: np.int64,
    neighbor_transforms: NDArray[np.int64],
    neighbor_dists: NDArray[np.int64],
    use_minima: bool = False,
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

    """
    nx, ny, nz = data.shape
    # get the value at this point
    base = data[i, j, k]
    # create a tracker for the best increase in value
    best = 0.0
    # create initial best transform. Default to this point
    bti = 0
    btj = 0
    btk = 0
    # create initial best neighbor
    bni = i
    bnj = j
    bnk = k
    # create initial best shift
    bsi = 0
    bsj = 0
    bsk = 0
    # For each neighbor get the difference in value and if its better
    # than any previous, replace the current best
    for (si, sj, sk), dist in zip(neighbor_transforms, neighbor_dists):
        # loop
        ii, jj, kk, sii, sjj, skk = wrap_point_w_shift(
            i + si, j + sj, k + sk, nx, ny, nz
        )
        # calculate the difference in value taking into account distance
        if use_minima:
            diff = (base - data[ii, jj, kk]) / dist
        else:
            diff = (data[ii, jj, kk] - base) / dist
        # if better than the current best, note the best and the
        # current label
        if diff > best:
            best = diff
            bti, btj, btk = (si, sj, sk)
            bni, bnj, bnk = (ii, jj, kk)
            bsi, bsj, bsk = (sii, sjj, skk)

    # return the best shift and neighbor
    return (
        np.array((bti, btj, btk), dtype=np.int64),
        np.array((bni, bnj, bnk), dtype=np.int64),
        np.array((bsi, bsj, bsk), dtype=np.int8),
    )


###############################################################################
# Edge Finding Methods
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
    vacuum_mask : NDArray[np.bool_]
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
                for si, sj, sk in neighbor_transforms:
                    # wrap points
                    ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
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


@njit(parallel=True, cache=True)
def get_edges_w_images(
    labeled_array: NDArray[np.int64],
    images: NDArray[np.int64],
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
    images : NDArray[np.int64]
        A 3D array where each entry represents the periodic image of the basin
        the point belongs to
    neighbor_transforms : NDArray[np.int64]
        The transformations from each voxel to its neighbors.
    vacuum_mask : NDArray[np.bool_]
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
                # get this voxels label and image
                label = labeled_array[i, j, k]
                image = images[i, j, k]
                # iterate over the neighboring voxels
                for si, sj, sk in neighbor_transforms:
                    # wrap points
                    ii, jj, kk, ssi, ssj, ssk = wrap_point_w_shift(
                        i + si, j + sj, k + sk, nx, ny, nz
                    )
                    # skip vacuum neighs
                    if vacuum_mask[ii, jj, kk]:
                        continue
                    # get neighbors label and image
                    neigh_label = labeled_array[ii, jj, kk]
                    neigh_image = images[ii, jj, kk]
                    if neigh_label != label:
                        edges[i, j, k] = True
                        break

                    # adjust neigh image
                    si1, sj1, sk1 = INT_TO_IMAGE[neigh_image]
                    si1 += ssi
                    sj1 += ssj
                    sk1 += ssk
                    neigh_image = IMAGE_TO_INT[si1, sj1, sk1]
                    if neigh_image != image:
                        edges[i, j, k] = True
                        break
    return edges


@njit(parallel=True, cache=True)
def get_edges_w_flat_images(
    labeled_array: NDArray[np.int64],
    images: NDArray[np.int64],
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
    images : NDArray[np.int64]
        A 3D array where each entry represents the periodic image of the basin
        the point belongs to
    neighbor_transforms : NDArray[np.int64]
        The transformations from each voxel to its neighbors.
    vacuum_mask : NDArray[np.bool_]
        A 3D array representing the location of the vacuum

    Returns
    -------
    edges : NDArray[np.bool_]
        A mask with the same shape as the input grid that is True at points
        on basin edges.

    """
    nx, ny, nz = labeled_array.shape
    ny_nz = ny * nz
    # create 3D array to store edges
    edges = np.zeros_like(labeled_array, dtype=np.bool_)

    # loop over each voxel in parallel
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # if this voxel is part of the vacuum, continue
                if vacuum_mask[i, j, k]:
                    continue
                # get this voxels label and image
                label = labeled_array[i, j, k]
                flat_idx = coords_to_flat(i, j, k, ny_nz, nz)
                mi, mj, mk = images[flat_idx]
                image = IMAGE_TO_INT[mi, mj, mk]
                # iterate over the neighboring voxels
                for si, sj, sk in neighbor_transforms:
                    # wrap points
                    ii, jj, kk, ssi, ssj, ssk = wrap_point_w_shift(
                        i + si, j + sj, k + sk, nx, ny, nz
                    )
                    # skip vacuum neighs
                    if vacuum_mask[ii, jj, kk]:
                        continue
                    # get neighbors label and image
                    neigh_label = labeled_array[ii, jj, kk]
                    if neigh_label != label:
                        edges[i, j, k] = True
                        break

                    flat_neigh = coords_to_flat(ii, jj, kk, ny_nz, nz)
                    # adjust neigh image
                    mii, mjj, mkk = images[flat_neigh]
                    mii += ssi
                    mjj += ssj
                    mkk += ssk
                    neigh_image = IMAGE_TO_INT[mii, mjj, mkk]
                    if neigh_image != image:
                        edges[i, j, k] = True
                        break
    return edges


@njit(inline="always", cache=True)
def get_differing_neighs(
    i,
    j,
    k,
    nx,
    ny,
    nz,
    labels,
    images,
    neighbor_transforms,
    vacuum_mask,
):
    # get the label at this point
    label0 = labels[i, j, k]
    image0 = images[i, j, k]

    # initialize potential alternative labels
    label1 = -1
    image1 = -1
    unique = 0

    # iterate over transforms
    for trans in range(neighbor_transforms.shape[0]):
        # if we've found more than two neighbors, immediately break
        if unique == 2:
            break

        # get shifts
        si = neighbor_transforms[trans, 0]
        sj = neighbor_transforms[trans, 1]
        sk = neighbor_transforms[trans, 2]

        # wrap around periodic edges and store shift
        ii, jj, kk, ssi, ssj, ssk = wrap_point_w_shift(
            i + si, j + sj, k + sk, nx, ny, nz
        )

        # skip points in the vacuum
        if vacuum_mask[ii, jj, kk]:
            continue

        # get the label and image of this neighbor
        neigh_label = labels[ii, jj, kk]
        neigh_image = images[ii, jj, kk]

        # update image to be relative to the current points transformation
        if ssi == 0 and ssj == 0 and ssk == 0:
            neigh_image = images[ii, jj, kk]
        else:
            neigh_image = images[ii, jj, kk]
            si1 = INT_TO_IMAGE[neigh_image, 0] + ssi
            sj1 = INT_TO_IMAGE[neigh_image, 1] + ssj
            sk1 = INT_TO_IMAGE[neigh_image, 2] + ssk
            neigh_image = IMAGE_TO_INT[si1, sj1, sk1]

        # compare to any previous labels and update our unique number
        if unique == 0:
            if neigh_label != label0 or neigh_image != image0:
                label1 = neigh_label
                image1 = neigh_image
                unique = 1
        elif unique == 1:
            if (neigh_label != label0 or neigh_image != image0) and (
                neigh_label != label1 or neigh_image != image1
            ):
                unique = 2

    return unique


@njit(parallel=True, cache=True)
def get_basin_edges(
    labels: NDArray[np.int64],
    images: NDArray[np.int64],
    neighbor_transforms: NDArray[np.int64],
    vacuum_mask: NDArray[np.bool_],
):
    nx, ny, nz = labels.shape
    # create 3D array to store edges
    edges = np.zeros_like(labels, dtype=np.uint8)

    # loop over each voxel in parallel
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # if this voxel is part of the vacuum, continue
                if vacuum_mask[i, j, k]:
                    continue

                # check if this point has 0, 1, or 2 neighbors with different
                # labels
                num_neighs = get_differing_neighs(
                    i,
                    j,
                    k,
                    nx,
                    ny,
                    nz,
                    labels,
                    images,
                    neighbor_transforms,
                    vacuum_mask,
                )

                if num_neighs == 0:
                    # not an edge
                    continue
                elif num_neighs == 1:
                    # meeting of two basins
                    edges[i, j, k] = 1
                else:
                    # meeting of multiple basins
                    edges[i, j, k] = 2

    return edges


@njit(inline="always", cache=True)
def get_differing_neighs_thin(
    i,
    j,
    k,
    nx,
    ny,
    nz,
    data,
    labels,
    images,
    neighbor_transforms,
    vacuum_mask,
    use_minima,
):

    label0 = labels[i, j, k]
    image0 = images[i, j, k]
    value0 = data[i, j, k]

    label1 = -1
    image1 = -1
    unique = 0

    for si, sj, sk in neighbor_transforms:

        ii, jj, kk, ssi, ssj, ssk = wrap_point_w_shift(
            i + si, j + sj, k + sk, nx, ny, nz
        )

        if vacuum_mask[ii, jj, kk]:
            continue

        val = data[ii, jj, kk]

        if use_minima:
            if val > value0:
                continue
        else:
            if val < value0:
                continue

        neigh_label = labels[ii, jj, kk]

        if ssi == 0 and ssj == 0 and ssk == 0:
            neigh_image = images[ii, jj, kk]
        else:
            neigh_image = images[ii, jj, kk]
            si1 = INT_TO_IMAGE[neigh_image, 0] + ssi
            sj1 = INT_TO_IMAGE[neigh_image, 1] + ssj
            sk1 = INT_TO_IMAGE[neigh_image, 2] + ssk
            neigh_image = IMAGE_TO_INT[si1, sj1, sk1]

        diff0 = (neigh_label != label0) or (neigh_image != image0)

        if unique == 0:
            if diff0:
                label1 = neigh_label
                image1 = neigh_image
                unique = 1

        else:
            if diff0 and ((neigh_label != label1) or (neigh_image != image1)):
                return 2

    return unique


@njit(parallel=True, cache=True)
def get_thin_basin_edges(
    data: NDArray[np.float64],
    labels: NDArray[np.int64],
    images: NDArray[np.int64],
    neighbor_transforms: NDArray[np.int64],
    use_minima: bool,
    vacuum_mask: NDArray[np.bool_],
):
    nx, ny, nz = labels.shape
    # create 3D array to store edges
    edges = np.zeros_like(labels, dtype=np.uint8)

    # loop over each voxel in parallel
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # if this voxel is part of the vacuum, continue
                if vacuum_mask[i, j, k]:
                    continue

                # check if this point has 0, 1, or 2 neighbors with different
                # labels
                num_neighs = get_differing_neighs_thin(
                    i,
                    j,
                    k,
                    nx,
                    ny,
                    nz,
                    data,
                    labels,
                    images,
                    neighbor_transforms,
                    vacuum_mask,
                    use_minima,
                )

                if num_neighs == 0:
                    # not an edge
                    continue
                elif num_neighs == 1:
                    # meeting of two basins
                    edges[i, j, k] = 1
                else:
                    # meeting of multiple basins
                    edges[i, j, k] = 2

    return edges


###############################################################################
# Morse Manifold Methods
###############################################################################


@njit(parallel=True, cache=True)
def get_manifold_labels(
    maxima_labels: NDArray[np.int64],
    minima_labels: NDArray[np.int64],
    maxima_images: NDArray[np.int64],
    minima_images: NDArray[np.int64],
    maxima_groups: list[NDArray],
    minima_groups: list[NDArray],
    neighbor_transforms: NDArray[np.int64],
    vacuum_mask: NDArray[np.bool_],
):
    """
    Takes the 3-manifolds of maxima and minima and determines the rough locations
    of the following manifolds:

        0: minima
        1: 1-saddle
        2: 2-saddle
        3: maxima
        4: meeting of 2 minima basins (saddle-1 unstable manifold)
        5: meeting of 2 maxima basins (saddle-2 stable manifold)
        6: meeting of 2 minima basins and 2 maxima basins (1D connections between critical points)
        7: meeting of at least 3 minima basin borders (saddle-2 unstable manifold)
        8: meeting of at least 3 maxima basin borders (saddle-1 stable manifold)

        255: overlapping maxima/minima basin
    """
    nx, ny, nz = maxima_labels.shape
    # create 3D array to store edges
    edges = np.full_like(maxima_labels, np.iinfo(np.uint8).max, dtype=np.uint8)

    # add maxima/minima
    for group in minima_groups:
        for i, j, k in group:
            edges[i, j, k] = 0

    for group in maxima_groups:
        for i, j, k in group:
            edges[i, j, k] = 3

    # loop over each voxel in parallel
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # if this voxel is part of the vacuum, continue
                if vacuum_mask[i, j, k]:
                    continue

                # if this voxel is part of a minimum or maximum, continue
                if edges[i, j, k] == 0 or edges[i, j, k] == 3:
                    continue

                # check if this point has 0, 1, or 2 neighbors with different
                # labels
                num_neighs = get_differing_neighs(
                    i,
                    j,
                    k,
                    nx,
                    ny,
                    nz,
                    maxima_labels,
                    maxima_images,
                    neighbor_transforms,
                    vacuum_mask,
                )
                opp_num_neighs = get_differing_neighs(
                    i,
                    j,
                    k,
                    nx,
                    ny,
                    nz,
                    minima_labels,
                    minima_images,
                    neighbor_transforms,
                    vacuum_mask,
                )

                if num_neighs == 1 and opp_num_neighs > 1:
                    # saddle 2
                    edges[i, j, k] = 2
                elif num_neighs > 1 and opp_num_neighs == 1:
                    # saddle 1
                    edges[i, j, k] = 1
                elif num_neighs < 1 and opp_num_neighs == 1:
                    # edge of minima manifold
                    edges[i, j, k] = 4
                elif num_neighs == 1 and opp_num_neighs < 1:
                    # edge of maxima manifold
                    edges[i, j, k] = 5
                elif num_neighs == 1 and opp_num_neighs == 1:
                    # edge of both maxima/minima manifold
                    edges[i, j, k] = 6
                elif num_neighs < 1 and opp_num_neighs > 1:
                    # meeting of at least three minima manifolds
                    edges[i, j, k] = 7
                elif num_neighs > 1 and opp_num_neighs < 1:
                    # meeting of at least three maxima manifolds
                    edges[i, j, k] = 8

    return edges


@njit(parallel=True, cache=True)
def get_manifold_labels_thin(
    data: NDArray[np.float64],
    maxima_labels: NDArray[np.int64],
    minima_labels: NDArray[np.int64],
    maxima_images: NDArray[np.int64],
    minima_images: NDArray[np.int64],
    maxima_groups: list[NDArray],
    minima_groups: list[NDArray],
    neighbor_transforms: NDArray[np.int64],
    vacuum_mask: NDArray[np.bool_],
):
    """
    Takes the 3-manifolds of maxima and minima and determines the rough locations
    of the following manifolds:

        0: minima
        1: 1-saddle
        2: 2-saddle
        3: maxima
        4: meeting of 2 minima basins (saddle-1 unstable manifold)
        5: meeting of 2 maxima basins (saddle-2 stable manifold)
        6: meeting of 2 minima basins and 2 maxima basins (1D connections between critical points)
        7: meeting of at least 3 minima basin borders (saddle-2 unstable manifold)
        8: meeting of at least 3 maxima basin borders (saddle-1 stable manifold)

        255: overlapping maxima/minima basin
    """
    nx, ny, nz = maxima_labels.shape
    # create 3D array to store edges
    edges = np.full_like(maxima_labels, np.iinfo(np.uint8).max, dtype=np.uint8)

    # add maxima/minima
    for group in minima_groups:
        for i, j, k in group:
            edges[i, j, k] = 0

    for group in maxima_groups:
        for i, j, k in group:
            edges[i, j, k] = 3

    # loop over each voxel in parallel
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # if this voxel is part of the vacuum, continue
                if vacuum_mask[i, j, k]:
                    continue

                # if this voxel is part of a minimum or maximum, continue
                if edges[i, j, k] == 0 or edges[i, j, k] == 3:
                    continue

                # check if this point has 0, 1, or 2 neighbors with different
                # labels
                num_neighs = get_differing_neighs_thin(
                    i,
                    j,
                    k,
                    nx,
                    ny,
                    nz,
                    data,
                    maxima_labels,
                    maxima_images,
                    neighbor_transforms,
                    vacuum_mask,
                    use_minima=False,
                )
                opp_num_neighs = get_differing_neighs_thin(
                    i,
                    j,
                    k,
                    nx,
                    ny,
                    nz,
                    data,
                    minima_labels,
                    minima_images,
                    neighbor_transforms,
                    vacuum_mask,
                    use_minima=True,
                )

                if num_neighs == 1 and opp_num_neighs > 1:
                    # saddle 2
                    edges[i, j, k] = 2
                elif num_neighs > 1 and opp_num_neighs == 1:
                    # saddle 1
                    edges[i, j, k] = 1
                elif num_neighs < 1 and opp_num_neighs == 1:
                    # edge of minima manifold
                    edges[i, j, k] = 4
                elif num_neighs == 1 and opp_num_neighs < 1:
                    # edge of maxima manifold
                    edges[i, j, k] = 5
                elif num_neighs == 1 and opp_num_neighs == 1:
                    # edge of both maxima/minima manifold
                    edges[i, j, k] = 6
                elif num_neighs < 1 and opp_num_neighs > 1:
                    # meeting of at least three minima manifolds
                    edges[i, j, k] = 7
                elif num_neighs > 1 and opp_num_neighs < 1:
                    # meeting of at least three maxima manifolds
                    edges[i, j, k] = 8

    return edges


###############################################################################
# Property Calculating Methods
###############################################################################


@njit(cache=True)
def get_neighboring_basin_surface_area(
    labeled_array: NDArray[np.int64],
    neighbor_transforms: NDArray[np.int64],
    neighbor_areas: NDArray[np.float64],
    vacuum_mask: NDArray[np.bool_],
    label_num: int,
):
    """
    In a 3D array of labeled voxels, approximately calculates the surface area
    of contact between each basin using the voxel voronoi surface.

    Parameters
    ----------
    labeled_array : NDArray[np.int64]
        A 3D array where each entry represents the basin label of the point.
    neighbor_transforms : NDArray[np.int64]
        The transformations from each voxel to its neighbors.
    neighbor_areas : NDArray[np.int64]
        The surface area of each neighbor at the corresponding transform.
    vacuum_mask : NDArray[np.bool_]
        A 3D array representing the location of the vacuum
    label_num : int,
        The total number of labels

    Returns
    -------
    connection_counts : NDArray[np.bool_]
        A 2D array with indices i, j where i is the label index, j is the neighboring
        label index, and the entry at i, j is the total area in contact between
        these labels. One extra index is added that stores the number of connections
        to the vacuum for each atom.

    """
    nx, ny, nz = labeled_array.shape
    # create a 2D array to store total number of connections
    connection_counts = np.zeros((label_num, label_num + 1), dtype=np.float64)

    # loop over each voxel. We can't do this in parallel as we may write to the
    # same entry and cause a race condition.
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # if this voxel is part of the vacuum, continue
                if vacuum_mask[i, j, k]:
                    continue
                # get this voxels label
                label = labeled_array[i, j, k]
                # iterate over the neighboring voxels
                for (si, sj, sk), area in zip(neighbor_transforms, neighbor_areas):
                    # wrap points
                    ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
                    # get neighbors label
                    neigh_label = labeled_array[ii, jj, kk]
                    # if this is the same label, skip it
                    if label == neigh_label:
                        continue
                    # add to our count for this connection
                    connection_counts[label, neigh_label] += area
    return connection_counts


@njit(fastmath=True, cache=True)
def get_basin_charges_and_volumes(
    data: NDArray[np.float64],
    labels: NDArray[np.int64],
    cell_volume: np.float64,
    extrema_num: np.int64,
):
    nx, ny, nz = data.shape
    total_points = nx * ny * nz
    # create variables to store charges/volumes
    charges = np.zeros(extrema_num, dtype=np.float64)
    volumes = np.zeros(extrema_num, dtype=np.float64)
    vacuum_charge = 0.0
    vacuum_volume = 0.0
    vacuum_val = extrema_num
    # iterate in parallel over each voxel
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                charge = data[i, j, k]
                label = labels[i, j, k]
                if label == vacuum_val:
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
def get_basin_min_and_max(
    data,
    labels,
    num_basins,
    vacuum_mask,
):
    nx, ny, nz = data.shape
    basin_min = np.full(num_basins, np.inf, dtype=np.float64)
    basin_max = np.full(num_basins, -np.inf, dtype=np.float64)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # if this voxel is part of the vacuum, continue
                if vacuum_mask[i, j, k]:
                    continue
                basin = labels[i, j, k]
                value = data[i, j, k]
                if value > basin_max[basin]:
                    basin_max[basin] = value
                if value < basin_min[basin]:
                    basin_min[basin] = value
    return basin_min, basin_max


@njit(cache=True)
def get_min_avg_surface_dists(
    labels,
    frac_coords,
    edge_mask,
    matrix,
    max_value,
):
    nx, ny, nz = labels.shape
    # create array to store best dists, sums, and counts
    dists = np.full(len(frac_coords), max_value, dtype=np.float64)
    dist_sums = np.zeros(len(frac_coords), dtype=np.float64)
    edge_totals = np.zeros(len(frac_coords), dtype=np.float64)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # skip outside edges
                if not edge_mask[i, j, k]:
                    continue
                # get label at edge
                label = labels[i, j, k]
                # add to our count
                edge_totals[label] += 1.0
                # convert from voxel indices to frac
                fi = i / nx
                fj = j / ny
                fk = k / nz
                # calculate the distance to the appropriate frac coord
                ni, nj, nk = frac_coords[label]
                # get differences between each index
                di = ni - fi
                dj = nj - fj
                dk = nk - fk
                # wrap at edges to be as close as possible
                di -= round(di)
                dj -= round(dj)
                dk -= round(dk)
                # convert to cartesian coordinates
                ci = di * matrix[0, 0] + dj * matrix[1, 0] + dk * matrix[2, 0]
                cj = di * matrix[0, 1] + dj * matrix[1, 1] + dk * matrix[2, 1]
                ck = di * matrix[0, 2] + dj * matrix[1, 2] + dk * matrix[2, 2]
                # calculate distance
                dist = (ci**2 + cj**2 + ck**2) ** 0.5
                # add to our total
                dist_sums[label] += dist
                # if this is the lowest distance, update radius
                if dist < dists[label]:
                    dists[label] = dist
    # get average dists
    average_dists = dist_sums / edge_totals
    return dists, average_dists


#############################################################################
# Other Helper Functions
#############################################################################


@njit(parallel=True, cache=True)
def get_extrema(
    data: NDArray[np.float64],
    neighbor_transforms: NDArray[np.int64],
    vacuum_mask: NDArray[np.bool_],
    use_minima: bool = False,
):
    """
    For a 3D array of data, return a mask that is True at local extrema.

    Parameters
    ----------
    data : NDArray[np.float64]
        A 3D array of data.
    neighbor_transforms : NDArray[np.int64]
        The transformations from each voxel to its neighbors.
    vacuum_mask : NDArray[np.bool_]
        A 3D array representing the location of the vacuum
    use_minima : bool, optional
        Whether or not to search for minima instead of extrema.

    Returns
    -------
    extrema : NDArray[np.bool_]
        A mask with the same shape as the input grid that is True at points
        that are local extrema.

    """
    nx, ny, nz = data.shape
    # create 3D array to store extrema
    extrema = np.zeros_like(data, dtype=np.bool_)
    factor = -1 if use_minima else 1

    # loop over each voxel in parallel
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # if this voxel is part of the vacuum, continue
                if vacuum_mask[i, j, k]:
                    continue
                # get this voxels value
                value = data[i, j, k] * factor
                is_max = True
                # iterate over the neighboring voxels
                for si, sj, sk in neighbor_transforms:
                    # wrap points
                    ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
                    if data[ii, jj, kk] * factor > value:
                        is_max = False
                        break

                if is_max:
                    extrema[i, j, k] = True
    return extrema


@njit(cache=True)
def reorder_labels(
    labels,
    data,
    extrema_labels,
    extrema_values,
    use_minima,
):
    nx, ny, nz = data.shape
    ny_nz = ny * nz

    roots = np.empty(len(extrema_labels), dtype=labels.dtype)
    for ext_idx in range(len(extrema_labels)):
        label = extrema_labels[ext_idx]
        root = find_root(labels, label)
        roots[ext_idx] = root

    unique_roots = np.unique(roots)
    final_roots = np.empty(len(unique_roots), dtype=unique_roots.dtype)
    for root_idx, root_label in enumerate(unique_roots):
        i, j, k = flat_to_coords(root_label, ny_nz, nz)
        group = []
        best_point = root_label
        best_idx = -1
        best_value = data[i, j, k]
        for ext_idx, (label, ext_root) in enumerate(zip(extrema_labels, roots)):
            if ext_root != root_label:
                continue
            group.append(label)
            value = extrema_values[ext_idx]
            label = extrema_labels[ext_idx]
            if label == root_label and best_idx == -1:
                best_idx = ext_idx

            if (
                not use_minima
                and value > best_value
                or use_minima
                and value < best_value
                or value == best_value
                and label < root_label
            ):
                best_point = label
                best_value = value
                best_idx = ext_idx
        for label in group:
            labels[label] = best_point
        final_roots[root_idx] = best_idx

    return labels, np.sort(final_roots)


@njit(parallel=True, cache=True)
def update_labels_and_images(
    labels,
    images,
    label_map,
    image_map,
    vacuum_mask,
):
    nx, ny, nz = labels.shape
    vacuum_label = len(np.unique(label_map))

    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # if this voxel is part of the vacuum, relable and continue
                if vacuum_mask[i, j, k]:
                    labels[i, j, k] = vacuum_label
                    continue
                # get current label
                label = labels[i, j, k]
                # get the current shift
                shift = INT_TO_IMAGE[images[i, j, k]]
                # get the shift from this maxima to its root
                maxima_shift = image_map[label]
                # update the image
                si, sj, sk = shift + maxima_shift
                images[i, j, k] = IMAGE_TO_INT[si, sj, sk]
                # update label
                labels[i, j, k] = label_map[label]

    return labels, images


@njit(parallel=True, cache=True)
def update_images(
    labels,
    images,
    image_map,
    important_mask,
    vacuum_mask,
):
    nx, ny, nz = labels.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # if this voxel is part of the vacuum, continue
                if vacuum_mask[i, j, k]:
                    continue
                # get current label
                label = labels[i, j, k]
                if not important_mask[label]:
                    continue
                # get the current shift
                shift = INT_TO_IMAGE[images[i, j, k]]
                # get the shift from this maxima to its root
                maxima_shift = image_map[label]
                # update the image
                si, sj, sk = shift + maxima_shift

                images[i, j, k] = IMAGE_TO_INT[si, sj, sk]

    return images
