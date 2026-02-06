# -*- coding: utf-8 -*-

import math
import itertools

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from baderkit.core.utilities.basic import (
    coords_to_flat,
    flat_to_coords,
    wrap_point,
    wrap_point_w_shift,
)
from baderkit.core.utilities.interpolation import linear_slice
from baderkit.core.utilities.union_find import find_root_no_compression, union, find_root

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
    use_min: bool = False,
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
        if use_min:
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
        ii, jj, kk, sii, sjj, skk = wrap_point_w_shift(i + si, j + sj, k + sk, nx, ny, nz)
        # calculate the difference in value taking into account distance
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
        np.array((bsi, bsj, bsk), dtype=np.int8)
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

IMAGE_TO_INT = np.empty([3,3,3], dtype=np.int64)
INT_TO_IMAGE = np.array(list(itertools.product((-1,0,1), repeat=3)))
for shift_idx, (i,j,k) in enumerate(INT_TO_IMAGE):
    IMAGE_TO_INT[i,j,k] = shift_idx

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
                image = images[i,j,k]
                # iterate over the neighboring voxels
                for si, sj, sk in neighbor_transforms:
                    # wrap points
                    ii, jj, kk, ssi, ssj, ssk = wrap_point_w_shift(i + si, j + sj, k + sk, nx, ny, nz)
                    # skip vacuum neighs
                    if vacuum_mask[ii,jj,kk]:
                        continue
                    # get neighbors label and image
                    neigh_label = labeled_array[ii, jj, kk]
                    neigh_image = images[ii, jj, kk]
                    if neigh_label !=label:
                        edges[i,j,k] = True
                        break
                    
                    # adjust neigh image
                    si1, sj1, sk1 = INT_TO_IMAGE[neigh_image]
                    si1 += ssi
                    sj1 += ssj
                    sk1 += ssk
                    neigh_image = IMAGE_TO_INT[si1,sj1,sk1]
                    if neigh_image != image:
                        edges[i,j,k] = True
                        break
    return edges

@njit(cache=True)
def get_neighboring_basin_connections(
    labeled_array: NDArray[np.int64],
    data: NDArray[np.float64],
    neighbor_transforms: NDArray[np.int64],
    edge_mask: NDArray[np.bool_],
    label_num: int,
        ):
    nx, ny, nz = labeled_array.shape
    # create a 2D array to store total number of connections
    connection_values = np.zeros((label_num, label_num + 1), dtype=np.float64)
    
    # remove half of the transforms as we don't need them in this case
    neighbor_transforms = neighbor_transforms[:int(len(neighbor_transforms)/2)]

    # loop over each voxel. We can't do this in parallel as we may write to the
    # same entry and cause a race condition.
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # if this voxel is part of the vacuum, continue
                if not edge_mask[i, j, k]:
                    continue
                # get this voxels label
                label = labeled_array[i, j, k]
                value = data[i,j,k]
                # iterate over the neighboring voxels
                for (si, sj, sk) in neighbor_transforms:
                    # wrap points
                    ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
                    if not edge_mask[ii,jj,kk]:
                        continue
                    # get neighbors label
                    neigh_label = labeled_array[ii, jj, kk]
                    # if this is the same label, skip it
                    if label == neigh_label:
                        continue
                    # the lower value is the value these points connect at. If
                    # this is higher than the current best, we update
                    lower = min(value, data[ii,jj,kk])
                    lower_idx = min(label, neigh_label)
                    upper_idx = max(label, neigh_label)
                    if connection_values[lower_idx, upper_idx] < lower:
                        connection_values[lower_idx, upper_idx] = lower

    connections = np.argwhere(connection_values>0)
    values = np.empty(len(connections), dtype=np.float64)
    for idx, (i,j) in enumerate(connections):
        values[idx] = connection_values[i,j]

    return connections, values

@njit(cache=True)
def get_neighboring_basin_connections_w_images(
    labeled_array: NDArray[np.int64],
    images: NDArray[np.int64],
    data: NDArray[np.float64],
    neighbor_transforms: NDArray[np.int64],
    edge_mask: NDArray[np.bool_],
    label_num: int,
        ):
    nx, ny, nz = labeled_array.shape
    # create a 2D array to store total number of connections
    connection_values = np.zeros((label_num, label_num, 27, 27), dtype=np.float64)
    
    # remove half of the transforms as we don't need them in this case
    neighbor_transforms = neighbor_transforms[:int(len(neighbor_transforms)/2)]

    # loop over each voxel. We can't do this in parallel as we may write to the
    # same entry and cause a race condition.
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # if this voxel is not part of the edge, continue
                if not edge_mask[i, j, k]:
                    continue
                # get this voxels label
                label = labeled_array[i, j, k]
                value = data[i,j,k]
                image = images[i,j,k]
                # iterate over the neighboring voxels
                for (si, sj, sk) in neighbor_transforms:
                    # wrap points
                    ii, jj, kk, ssi, ssj, ssk = wrap_point_w_shift(i + si, j + sj, k + sk, nx, ny, nz)
                    if not edge_mask[ii,jj,kk]:
                        continue
                    # get neighbors label
                    neigh_label = labeled_array[ii, jj, kk]
                    # get neighbors image
                    neigh_image = images[ii,jj,kk]
                    # adjust neigh image
                    si1, sj1, sk1 = INT_TO_IMAGE[neigh_image]
                    si1 += ssi
                    sj1 += ssj
                    sk1 += ssk
                    neigh_image = IMAGE_TO_INT[si1,sj1,sk1]
                    # if this is the same label, skip it
                    if label == neigh_label and image == neigh_image:
                        continue

                    # the lower value is the value these points connect at. If
                    # this is higher than the current best, we update
                    lower = min(value, data[ii,jj,kk])
                    if image != neigh_image:
                        lower_idx = label
                        upper_idx = neigh_label
                        
                    elif label < neigh_label:
                        lower_idx = label
                        upper_idx = neigh_label
                        
                    elif label > neigh_label:
                        lower_idx = neigh_label
                        upper_idx = label

                    if connection_values[lower_idx, upper_idx, image, neigh_image] < lower:
                        connection_values[lower_idx, upper_idx, image, neigh_image] = lower

    connections = np.argwhere(connection_values>0)
    values = np.empty(len(connections), dtype=np.float64)
    for idx, (i,j,k,w) in enumerate(connections):
        values[idx] = connection_values[i,j,k,w]

    return connections, values
    

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
    maxima_num: np.int64,
):
    nx, ny, nz = data.shape
    total_points = nx * ny * nz
    # create variables to store charges/volumes
    charges = np.zeros(maxima_num, dtype=np.float64)
    volumes = np.zeros(maxima_num, dtype=np.float64)
    vacuum_charge = 0.0
    vacuum_volume = 0.0
    vacuum_val = maxima_num
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
def get_maxima(
    data: NDArray[np.float64],
    neighbor_transforms: NDArray[np.int64],
    vacuum_mask: NDArray[np.bool_],
    use_minima: bool = False,
):
    """
    For a 3D array of data, return a mask that is True at local maxima.

    Parameters
    ----------
    data : NDArray[np.float64]
        A 3D array of data.
    neighbor_transforms : NDArray[np.int64]
        The transformations from each voxel to its neighbors.
    vacuum_mask : NDArray[np.bool_]
        A 3D array representing the location of the vacuum
    use_minima : bool, optional
        Whether or not to search for minima instead of maxima.

    Returns
    -------
    maxima : NDArray[np.bool_]
        A mask with the same shape as the input grid that is True at points
        that are local maxima.

    """
    nx, ny, nz = data.shape
    # create 3D array to store maxima
    maxima = np.zeros_like(data, dtype=np.bool_)
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
                    maxima[i, j, k] = True
    return maxima    

@njit(cache=True)
def compute_wrap_offset(point1, point2):
    """
    Computes wrap from point1 to point2

    """
    best_d2 = np.inf
    best_i = 0
    best_j = 0
    best_k = 0

    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                dx = (point2[0] + i) - point1[0]
                dy = (point2[1] + j) - point1[1]
                dz = (point2[2] + k) - point1[2]
                d2 = dx*dx + dy*dy + dz*dz

                if d2 < best_d2:
                    best_d2 = d2
                    best_i = i
                    best_j = j
                    best_k = k

    return best_i, best_j, best_k

@njit(cache=True)
def initialize_labels_from_maxima(
    data,
    labels,
    maxima_mask,
    lattice,
    neighbor_transforms,
    neighbor_dists,
    persistence_tol,
    max_cart_offset=1,
):
    nx, ny, nz = data.shape
    ny_nz = ny * nz
    
    maxima_vox = np.argwhere(maxima_mask)
    
    # create an array to store values at each maximum
    maxima_values = np.empty(len(maxima_vox), dtype=np.float64)
    maxima_labels = np.empty(len(maxima_vox), dtype=np.int64)

    # get the fractional representation of each maximum
    maxima_frac = maxima_vox / np.array(data.shape, dtype=np.int64)

    # create a flat array of shifs for tracking wrapping around edges. These
    # will initially all be (0,0,0)
    images = np.zeros((nx*ny*nz, 3), dtype=np.int8)

    # Now we initialize the maxima
    for max_idx, (i, j, k) in enumerate(maxima_vox):
        # get value at maximum
        maxima_values[max_idx] = data[i, j, k]
        # set as initial group root
        flat_max_idx = coords_to_flat(i, j, k, ny_nz, nz)
        maxima_labels[max_idx] = flat_max_idx
        labels[flat_max_idx] = flat_max_idx
        
    # create an array to store the highest value at which all maxima in a group
    # connect
    # persistence_cutoffs = np.full(len(maxima_vox), np.inf, dtype=np.float64)
    persistence_cutoffs = maxima_values.copy()

    ###########################################################################
    # 1. Remove Flat False Maxima
    ###########################################################################
    # If there is a particularly flat region, a point might have neighbors that
    # are the same value. This point may be mislabeled as a maximum if these
    # neighbors are not themselves maxima. This issue is typically caused by too
    # few sig figs in the data preventing the region from being properly distinguished

    # create an array to store which maxima need to be reduced
    flat_maxima_labels = []
    flat_maxima_mask = np.zeros(len(maxima_vox), dtype=np.bool_)
    best_neigh = []
    num_to_reduce = 0
    # check each maximum to see if it is a true maximum. We do this iteratively
    # in case there is a flat area larger than a couple of voxels across
    while True:
        for max_idx, ((i, j, k), value, max_label) in enumerate(
            zip(maxima_vox, maxima_values, maxima_labels)
        ):
            # skip points that are not maxima
            if not maxima_mask[i, j, k]:
                continue

            for si, sj, sk in neighbor_transforms:
                # get neighbor and wrap
                ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
                neigh_value = data[ii, jj, kk]
                # skip lower points or points that are also true maxima
                if neigh_value < value or maxima_mask[ii, jj, kk]:
                    continue
                # note this is a false maximum
                flat_maxima_labels.append(max_label)
                flat_maxima_mask[max_idx] = True
                # temporarily set maxima_mask to false
                maxima_mask[i, j, k] = False
                # check if this neighbor is also in our flat set
                neigh_label = coords_to_flat(ii, jj, kk, ny_nz, nz)
                found = False
                for max_label, max_neigh in zip(flat_maxima_labels, best_neigh):
                    if neigh_label == max_label:
                        # give this max the same neighbor as this point
                        best_neigh.append(max_neigh)
                        found = True
                        break
                if not found:
                    best_neigh.append(neigh_label)
                # we only need one neighbor to match so we break
                break
        # check if anything has changed. If not we're done
        new_num_to_reduce = len(flat_maxima_labels)
        if new_num_to_reduce == num_to_reduce:
            break
        num_to_reduce = new_num_to_reduce
    # find the ongrid maximum each false maximum corresponds to
    unique_neighs = np.unique(np.array(best_neigh, dtype=np.int64))
    for unique_neigh_label in unique_neighs:
        i, j, k = flat_to_coords(unique_neigh_label, ny_nz, nz)
        # hill climb to best max
        while True:
            _, (ni, nj, nk) = get_best_neighbor(
                data,
                i,
                j,
                k,
                neighbor_transforms,
                neighbor_dists,
            )
            if maxima_mask[ni, nj, nk]:
                break
            if i == ni and j == nj and k == nk:
                # we've hit another group of flat maxima. get their best neighbor
                # and continue
                flat_neigh = coords_to_flat(ni, nj, nk, ny_nz, nz)
                for max_label, neigh_label in zip(flat_maxima_labels, best_neigh):
                    if max_label == flat_neigh:
                        ni, nj, nk = flat_to_coords(neigh_label, ny_nz, nz)
                        break
            i = ni
            j = nj
            k = nk
        best_max = coords_to_flat(ni, nj, nk, ny_nz, nz)
        # union each corresponding point
        for max_label, neigh_label in zip(flat_maxima_labels, best_neigh):
            if neigh_label != unique_neigh_label:
                continue
            union(labels, max_label, best_max)

    # add maxima back to mask (required for things like the weight method)
    for max_label in flat_maxima_labels:
        i, j, k = flat_to_coords(max_label, ny_nz, nz)
        maxima_mask[i, j, k] = True

    ###########################################################################
    # 2. Combine low-persistence maxima
    ###########################################################################
    # With the right shape (e.g. highly anisotropic) a maximum may lay offgrid
    # and cause two ongrid points to appear to be higher than the region around
    # them. To merge these, we use linear spline interpolation and a persistence metric
    # to combine those that have insignificant maxima between them

    # sort maxima from high to low with preference for lower indices
    sorted_indices = np.flip(np.argsort(maxima_values))
    sorted_values = maxima_values[sorted_indices]
    # get points indices where values differ
    breaks = np.where(sorted_values[1:] != sorted_values[:-1])[0] + 1
    breaks = np.append(0, breaks)
    breaks = np.append(breaks, len(sorted_values))
    for break_idx in range(len(breaks)-1):
        min_idx = breaks[break_idx]
        max_idx = breaks[break_idx+1]
        equal_value_set = sorted_indices[min_idx:max_idx]
        # sort indices
        sorted_indices[min_idx:max_idx] = np.sort(equal_value_set)

    # Iterate over each maximum (except the first) and check for nearby maxima
    # above them
    for sorted_max_idx, max_idx in enumerate(sorted_indices[1:]):
        # skip fake flat maxima (we've already found their higher neighbors)
        if flat_maxima_mask[max_idx]:
            continue
        sorted_max_idx += 1
        max_frac = maxima_frac[max_idx]
        value = maxima_values[max_idx]
        # flat_idx = maxima_labels[max_idx]
        # root = find_root(labels, flat_idx)
        # label = labels[maxima_labels[max_idx]]
        # iterate over maxima above this point
        for neigh_max_idx in sorted_indices:
            # skip if this is the same point
            if neigh_max_idx == max_idx:
                continue

            # break if we reach a point lower than the current one
            neigh_value = maxima_values[neigh_max_idx]
            if neigh_value < value:
                break
            
            # skip if these points are already unioned
            # neigh_label = labels[maxima_labels[neigh_max_idx]]
            # if label == neigh_label:
            #     continue

            neigh_frac = maxima_frac[neigh_max_idx]
            # unwrap relative to central
            fi, fj, fk = neigh_frac - np.round(neigh_frac - max_frac)
            # get offset in frac coords
            oi = fi - max_frac[0]
            oj = fj - max_frac[1]
            ok = fk - max_frac[2]

            # calculate the distance in cart coords
            ci = lattice[0, 0] * oi + lattice[1, 0] * oj + lattice[2, 0] * ok
            cj = lattice[0, 1] * oi + lattice[1, 1] * oj + lattice[2, 1] * ok
            ck = lattice[0, 2] * oi + lattice[1, 2] * oj + lattice[2, 2] * ok
            dist = (ci**2 + cj**2 + ck**2) ** (1 / 2)

            # if above our cutoff, continue
            if dist > max_cart_offset:
                continue

            # check if there is a minimum between this point and its neighbor
            # set number of interpolation points to ~20/A with a minimum of 5
            n_points = max(math.ceil(dist * 20), 5)
            values = linear_slice(
                data, max_frac, (fi, fj, fk), n=n_points, is_frac=True, method="nearest"
            )
            
            # get the maxima belonging to each point. These should be the first
            # points on the left and right
            maxima0_val = values[0]
            maxima1_val = values[-1]
            # get the minimum value reached between these points
            lowest = values.min()
            
            # get the persistence of this pair. We score it relative to the
            # higher value and scale it by cartesian distance. 
            persistence_score = dist*(min(maxima0_val, maxima1_val)-lowest)/max(maxima0_val, maxima1_val)
            
            if persistence_score < persistence_tol:
                # we consider these to be the same maximum and combine them
                max_label = maxima_labels[max_idx]
                neigh_label = maxima_labels[neigh_max_idx]
                # Get the higher maximum. If theres a tie, get the lower index
                i, j, k = maxima_vox[max_idx]
                ni, nj, nk = maxima_vox[neigh_max_idx]

                if data[i,j,k] > data[ni,nj,nk]:
                    lower = max_label
                    upper = neigh_label
                elif data[i,j,k] < data[ni,nj,nk]:
                    lower = neigh_label
                    upper = max_label
                else:
                    lower = min(max_label, neigh_label)
                    upper = max(max_label, neigh_label)
                union(labels, upper, lower)
                # update lowest cutoff for these maxima
                if lowest < persistence_cutoffs[max_idx]:
                    persistence_cutoffs[max_idx] = lowest
                if lowest < persistence_cutoffs[neigh_max_idx]:
                    persistence_cutoffs[neigh_max_idx] = lowest

    # get the remaining maxima after reduction
    maxima_roots = []
    for max_idx in maxima_labels:
        root = find_root_no_compression(labels, max_idx)
        labels[max_idx] = root
        maxima_roots.append(root)
    maxima_roots = np.array(maxima_roots, dtype=np.int64)
    
    root_maxima = []
    # now we get the images across unit cell borders for each maximum.
    for max_idx, max_root in enumerate(maxima_roots):
        if maxima_labels[max_idx] == max_root:
            root_maxima.append(max_idx)
            continue
        max_frac = maxima_frac[max_idx]
        root_idx = np.searchsorted(maxima_labels, max_root)
        root_frac = maxima_frac[root_idx]
        images[maxima_labels[max_idx]] = compute_wrap_offset(max_frac, root_frac)
    root_maxima = np.array(root_maxima, dtype=np.uint16)

    # Finally, we group our maxima so we have a history of which false maxima
    # are joined to the final list of "true" maxima 
    child_maxima = []
    reduced_persistence_cutoffs = []
    for root_idx in root_maxima:
        children = []
        unique_maximum = maxima_labels[root_idx]
        lowest_persistence = persistence_cutoffs[root_idx]
        for max_idx, max_root in enumerate(maxima_roots):
            if max_root == unique_maximum:
                children.append(max_idx)
                # if this child has a lower persistence cutoff, update the parents
                # cutoff. Only do this if the combination was from persistence
                if flat_maxima_mask[max_idx]:
                    continue
                if persistence_cutoffs[max_idx] < lowest_persistence:
                    lowest_persistence = persistence_cutoffs[max_idx]
        reduced_persistence_cutoffs.append(lowest_persistence)
        children = np.array(children, dtype=np.uint16)
        child_maxima.append(children)
    
    maxima_coords = maxima_vox[root_maxima]
    maxima_children = [maxima_vox[i] for i in child_maxima]
    reduced_persistence_cutoffs = np.array(reduced_persistence_cutoffs, dtype=np.float64)

    return labels, images, maxima_coords, maxima_children, reduced_persistence_cutoffs

@njit(cache=True)
def get_persistence_scores(
        data,
        critical_frac,
        critical_values,
        connections,
        connection_values,
        lattice,
        ):
    nx, ny, nz = data.shape
        
    # create array to store persistence scores
    persistence_scores = np.empty(len(connections), dtype=np.float64)
    
    for pair_idx in prange(len(connection_values)):
        crit1, crit2, image1, image2 = connections[pair_idx]
        
        saddle_val = connection_values[pair_idx]
    
        # We define a persistence score as follows:
            # dist * (lower_crit - saddle) / higher_crit
            # the distance is used to penalize very sharp peaks. The higher_crit
            # penalizes shallow connections with high absolute values
        
        # get critical values
        val1 = critical_values[crit1]
        val2 = critical_values[crit2]
        
        # get distance
        crit_frac = critical_frac[crit1] + INT_TO_IMAGE[image1]
        neigh_frac = critical_frac[crit2] + INT_TO_IMAGE[image2]
        
        oi, oj, ok = neigh_frac - crit_frac
    
        # calculate the distance in cart coords
        ci = lattice[0, 0] * oi + lattice[1, 0] * oj + lattice[2, 0] * ok
        cj = lattice[0, 1] * oi + lattice[1, 1] * oj + lattice[2, 1] * ok
        ck = lattice[0, 2] * oi + lattice[1, 2] * oj + lattice[2, 2] * ok
        dist = (ci**2 + cj**2 + ck**2) ** (1 / 2)
        
        # calculate persistence score
        persistence_score = dist * (min(val1, val2) - saddle_val) / max(val1, val2)
        persistence_scores[pair_idx] = persistence_score

    return persistence_scores

@njit(cache=True)
def group_by_persistence(
        data,
        critical_vox,
        connections,
        connection_values,
        lattice,
        persistence_tol,
        persistence_cutoffs,
        ):
    num_critical = len(critical_vox)
    unions = np.arange(num_critical)
    # create an array to track crit and their periodic images
    crit_image_map = np.full((num_critical, 27), -1, dtype=np.int64)
    for i in range(num_critical):
        crit_image_map[i,13] = i
    for crit1, crit2, image1, image2 in connections:
        if crit_image_map[crit1,image1] == -1:
            crit_image_map[crit1, image1] = num_critical
            num_critical += 1
        if crit_image_map[crit2,image2] == -1:
            crit_image_map[crit2, image2] = num_critical
            num_critical += 1
    image_crit_map = np.argwhere(crit_image_map != -1)
    crit_vals = np.empty(len(image_crit_map), dtype=np.int64)
    for idx, (i,j) in enumerate(image_crit_map):
        crit_vals[idx] = crit_image_map[i,j]
    crit_sort = np.argsort(crit_vals)
    image_crit_map = image_crit_map[crit_sort]
    
    # create array to track unions
    unions_w_image = np.arange(num_critical)
    
    # get the fractional representation of each critical point
    critical_frac = critical_vox / np.array(data.shape, dtype=np.int64)
    
    # get values for each critical point
    critical_values = np.empty(len(critical_vox), dtype=np.float64)
    for crit_idx, (i, j, k) in enumerate(critical_vox):
        # get value at maximum
        critical_values[crit_idx] = data[i, j, k]
    
    current_connections = connections.copy()
    current_connection_values = connection_values.copy()
    
    while True:
        
        connection_mask = np.zeros(len(current_connections), dtype=np.bool_)
        
        # get current persistence
        persistence_scores = get_persistence_scores(
            data,
            critical_frac,
            critical_values,
            current_connections,
            current_connection_values,
            lattice,
            )
        # loop over persistence and combine crit below the tolerance
        for pair_idx, ((crit1, crit2, image1, image2), score, conn_val) in enumerate(zip(current_connections, persistence_scores, current_connection_values)):
            # skip anything with a high score
            if score > persistence_tol:
                continue
            if crit1 == crit2 and image1==image2:
                continue
            # we want to union the lower maximum to the higher one. For ties,
            # we use the lower index
            value1 = critical_values[crit1]
            value2 = critical_values[crit2]
            if value1 >= value2:
                higher = crit1
                lower = crit2
                higher_image = crit_image_map[crit1, image1]
                lower_image = crit_image_map[crit2, image2]
            else:
                higher = crit2
                lower = crit1
                higher_image = crit_image_map[crit2, image2]
                lower_image = crit_image_map[crit1, image1]
            # make the union
            union(unions, lower, higher)
            union(unions_w_image, lower_image, higher_image)
            
            connection_mask[pair_idx] = True
            
            # update persistence cutoffs if the connection value is similar
            higher_val = max(value1, value2)
            # if abs((higher_val-conn_val)/higher_val) < 0.05:
            if conn_val < persistence_cutoffs[crit1]:
                persistence_cutoffs[crit1] = conn_val
            if conn_val < persistence_cutoffs[crit2]:
                persistence_cutoffs[crit2] = conn_val

        # get unchanged connections
        connection_indices = np.where(~connection_mask)[0]
        if len(connection_indices) == len(current_connections):
            break

        new_connections = current_connections[connection_indices]
        new_connection_values = current_connection_values[connection_indices]
        for pair_idx, (crit1, crit2, image1, image2) in enumerate(new_connections):
            # update to roots
            root1 = find_root(unions_w_image, crit1)
            root2 = find_root(unions_w_image, crit2)
            basin1, image1 = image_crit_map[root1]
            basin2, image2 = image_crit_map[root2]
            new_connections[pair_idx, 0] = basin1
            new_connections[pair_idx, 1] = basin2
            new_connections[pair_idx, 2] = image1
            new_connections[pair_idx, 3] = image2

        # update our connections for the next round
        current_connections = new_connections
        current_connection_values = new_connection_values
    # get the roots of all maxima and update persistence
    roots = np.empty(len(unions), dtype=np.int64)
    for idx in range(len(roots)):
        root = find_root(unions, idx)
        roots[idx] = root
        # if this maxima has a similar value to its root, we also update the
        # persistence
        value1 = critical_values[idx]
        value2 = critical_values[root]
        higher_val = max(value1, value2)
        conn_val = persistence_cutoffs[idx]
        # if abs((higher_val-conn_val)/higher_val) < 0.05:
        if conn_val < persistence_cutoffs[root]:
            persistence_cutoffs[root] = conn_val

    return roots, persistence_cutoffs

@njit(cache=True)
def get_persistence_groups(
    labels,
    data,
    persistence_cutoffs,
    maxima_vox,
        ):
    nx, ny, nz = data.shape
    max_val = len(maxima_vox)
    
    persistence_groups = []
    for vox in maxima_vox:
        persistence_groups.append([vox])
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # get label
                label = labels[i, j, k]
                # skip unlabeled points
                if label >= max_val:
                    continue
                # get value and cutoff
                value = data[i,j,k]
                cutoff = persistence_cutoffs[label]
                # if value is above the cutoff, add to the group
                if value >= cutoff:
                    point = np.array((i,j,k), dtype=np.uint16)
                    persistence_groups[label].append(point)
    # convert to arrays
    array_groups = []
    for i in persistence_groups:
        new_group = np.empty((len(i), 3), dtype=np.uint16)
        for idx, val in enumerate(i):
            new_group[idx] = val
        array_groups.append(new_group)
    return array_groups
    
    
                        
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
