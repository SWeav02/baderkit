# -*- coding: utf-8 -*-

import itertools
import math

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from baderkit.core.utilities.basic import (
    coords_to_flat,
    flat_to_coords,
    wrap_point,
    wrap_point_w_shift,
)
from baderkit.core.utilities.interpolation import linear_slice, refine_extrema
from baderkit.core.utilities.persistence import (
    compute_wrap_offset,
    get_conn_vals,
    reduce_by_conn,
)
from baderkit.core.utilities.union_find import (
    find_root,
    find_root_with_shift,
    union,
    union_with_shift,
)

IMAGE_TO_INT = np.empty([3, 3, 3], dtype=np.int64)
INT_TO_IMAGE = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
for shift_idx, (i, j, k) in enumerate(INT_TO_IMAGE):
    IMAGE_TO_INT[i, j, k] = shift_idx
###############################################################################
# General methods
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
    use_minima=False,
):
    # get the label at this point
    label0 = labels[i, j, k]
    image0 = images[i, j, k]
    value0 = data[i, j, k]

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

        # skip points with a higher value
        if (
            use_minima
            and data[ii, jj, kk] > value0
            or not use_minima
            and data[ii, jj, kk] < value0
        ):
            continue

        # get the label and image of this neighbor
        neigh_label = labels[ii, jj, kk]
        neigh_image = images[ii, jj, kk]

        # update image to be relative to the current points transformation
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
                for (si, sj, sk), area in zip(
                    neighbor_transforms, neighbor_areas
                ):
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
    # loop over each voxel in parallel
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # if this voxel is part of the vacuum, continue
                if vacuum_mask[i, j, k]:
                    continue
                # get this voxels value
                value = data[i, j, k]
                is_max = True
                # iterate over the neighboring voxels
                for si, sj, sk in neighbor_transforms:
                    # wrap points
                    ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
                    if not use_minima:
                        if data[ii, jj, kk] > value:
                            is_max = False
                            break
                    else:
                        if data[ii, jj, kk] < value:
                            is_max = False
                            break
                if is_max:
                    extrema[i, j, k] = True
    return extrema

@njit(cache=True)
def reorder_labels(
    labels,
    data,
    extrema_vox,
    extrema_labels,
    extrema_values,
    use_minima,
        ):
    nx, ny, nz = data.shape
    ny_nz = ny*nz

    roots = np.empty(len(extrema_vox), dtype=labels.dtype)
    for ext_idx in range(len(extrema_vox)):
        label = extrema_labels[ext_idx]
        root = find_root(labels, label)
        roots[ext_idx] = root

    unique_roots = np.unique(roots)
    groups = []
    for root_label in unique_roots:
        i, j, k = flat_to_coords(root_label, ny_nz, nz)
        group = []
        best_point = root_label
        best_value = data[i, j, k]
        for ext_idx, (label, ext_root) in enumerate(zip(extrema_labels, roots)):
            if ext_root != root_label:
                continue
            group.append(label)
            value = extrema_values[ext_idx]
            label = extrema_labels[ext_idx]
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
        for label in group:
            labels[label] = best_point
        groups.append(group)
    return labels, groups

@njit(cache=True)
def initialize_labels_from_extrema(
    data,
    labels,
    extrema_mask,
    neighbor_transforms,
    neighbor_dists,
    persistence_tol,
    method,
    matrix,
    max_cart_offset,
    use_minima=False,
):
    shape = np.array(data.shape, dtype=np.int64)
    nx, ny, nz = shape
    ny_nz = ny * nz

    extrema_vox = np.argwhere(extrema_mask)
    extrema_frac = extrema_vox / shape

    # create an array to store values at each maximum
    extrema_values = np.empty(len(extrema_vox), dtype=np.float64)
    extrema_labels = np.empty(len(extrema_vox), dtype=np.int64)

    # create a flat array of shifts for tracking wrapping around edges. These
    # will initially all be (0,0,0)
    images = np.zeros((nx * ny * nz, 3), dtype=np.int8)

    # Now we initialize the extrema
    for ext_idx, (i, j, k) in enumerate(extrema_vox):
        # get value at extremum
        extrema_values[ext_idx] = data[i, j, k]
        # set as initial group root
        flat_ext_idx = coords_to_flat(i, j, k, ny_nz, nz)
        extrema_labels[ext_idx] = flat_ext_idx
        labels[flat_ext_idx] = flat_ext_idx

    ###########################################################################
    # 2. Combine low-persistence extrema
    ###########################################################################
    # With the right shape (e.g. highly anisotropic) a maximum/minimum may lay offgrid
    # and cause two ongrid points to appear to be higher than the region around
    # them. We merge these by taking a linear slice between each point using
    # cubic interpolation and combining those that have no minimum/maximum between
    # them.
    (
        all_conn_neighs,
        all_conn_vals,
        all_conn_coords,
        neigh_cart_coords,
        cursor,
    ) = get_conn_vals(
        data,
        labels,
        extrema_values,
        extrema_labels,
        extrema_frac,
        max_cart_offset,
        use_minima,
        persistence_tol,
        method,
        matrix,
    )

    labels = reduce_by_conn(
        all_conn_neighs,
        all_conn_vals,
        all_conn_coords,
        neigh_cart_coords,
        cursor,
        labels,
        extrema_values,
        extrema_labels,
        use_minima,
        persistence_tol,
    )

    ###########################################################################
    # 3. Remove Flat False Maxima
    ###########################################################################
    # If there is a particularly flat region, a point might have neighbors that
    # are the same value. This point may be mislabeled as a maximum if these
    # neighbors are not themselves extrema. This issue is typically caused by too
    # few sig figs in the data preventing the region from being properly distinguished

    # Now we look for any points that have neighbors with the same value that
    # aren't maxima. We hill climb from that point to find the corresponding
    # maximum
    # flat_false_maxima = np.zeros(len(extrema_vox), dtype=np.bool_)
    for ext_idx, ((i, j, k), value, ext_label) in enumerate(
        zip(extrema_vox, extrema_values, extrema_labels)
    ):

        for si, sj, sk in neighbor_transforms:
            # get neighbor and wrap
            ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
            neigh_value = data[ii, jj, kk]
            # skip points that don't have the same value or that are also
            # extrema
            if neigh_value != value or extrema_mask[ii, jj, kk]:
                continue
            # If we're still here, this point is a false maximum. We follow the
            # path to the maximum and union
            # flat_false_maxima[ext_idx] = True
            while True:
                _, (ni, nj, nk) = get_best_neighbor(
                    data,
                    ii,
                    jj,
                    kk,
                    neighbor_transforms,
                    neighbor_dists,
                    use_minima=use_minima,
                )
                # stop if we hit another maximum
                if extrema_mask[ni, nj, nk]:
                    break
                # otherwise, update to this point
                ii = ni
                jj = nj
                kk = nk

            # make a union
            best_ext = coords_to_flat(ni, nj, nk, ny_nz, nz)
            union(labels, ext_label, best_ext)

    ###########################################################################
    # Root finding
    ###########################################################################
    # update labels to the highest valued point
    labels, extrema_groups = reorder_labels(
        labels,
        data,
        extrema_vox,
        extrema_labels,
        extrema_values,
        use_minima,
            )

    # Find the images each grouped maximum must cross to reach its parent
    for ext_idx, ext_label in enumerate(extrema_labels):
        root = labels[ext_label]
        # get fractional coordinates
        ext_frac = extrema_frac[ext_idx]
        root_idx = np.searchsorted(extrema_labels, root)
        root_frac = extrema_frac[root_idx]
        # get best image to wrap the maxima to its root
        images[extrema_labels[ext_idx]] = compute_wrap_offset(
            ext_frac, root_frac
        )

    root_extrema = np.where(labels[extrema_labels]==extrema_labels)[0]
    extrema_coords = extrema_vox[root_extrema]
    extrema_frac = extrema_frac[root_extrema]

    return (
        labels,
        images,
        extrema_coords,
        extrema_frac,
        extrema_groups,
    )


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
def update_final_images(
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