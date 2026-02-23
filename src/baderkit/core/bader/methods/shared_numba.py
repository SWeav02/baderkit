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
from baderkit.core.utilities.union_find import (
    find_root_with_shift,
    union_with_shift,
    union, 
    find_root)

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
        ii, jj, kk, sii, sjj, skk = wrap_point_w_shift(i + si, j + sj, k + sk, nx, ny, nz)
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

@njit(inline='always', cache=True)
def get_differing_neighs(
    i, j, k,
    nx, ny, nz,
    labels,
    images,
    neighbor_transforms,
    vacuum_mask,
):
    # get the label at this point
    label0 = labels[i,j,k]
    image0 = images[i,j,k]

    # initialize potential alternative labels
    label1 = -1; image1 = -1
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
            i+si, j+sj, k+sk, nx, ny, nz
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
            if ((neigh_label != label0 or neigh_image != image0) and
                (neigh_label != label1 or neigh_image != image1)):
                unique = 2

    return unique

@njit(parallel=True,  cache=True)
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
                    i, j, k, 
                    nx, ny, nz, 
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
                    edges[i,j,k] = 1
                else:
                    # meeting of multiple basins
                    edges[i,j,k] = 2

    return edges


@njit(inline='always', cache=True)
def get_extrema_saddle_connections(
    i, j, k,
    nx, ny, nz,
    ny_nz,
    labels,
    images,
    data,
    neighbor_transforms,
    edge_mask,
    max_val,
    use_minima = False,
):

    # iterate over transforms
    label0 = labels[i,j,k]
    image0 = images[i,j,k]
    value0 = data[i,j,k]
    
    label1 = -1; image1 = -1; value1 = -1.0
    n = 0

    for trans in range(neighbor_transforms.shape[0]):
        # get shifts
        si = neighbor_transforms[trans, 0]
        sj = neighbor_transforms[trans, 1]
        sk = neighbor_transforms[trans, 2]

        # wrap around periodic edges and store shift
        ii, jj, kk, ssi, ssj, ssk = wrap_point_w_shift(
            i+si, j+sj, k+sk, nx, ny, nz
        )
        # skip neighbors that aren't part of the edge
        if edge_mask[ii, jj, kk] == 0:
            continue

        # get the label and image of this neighbor
        neigh_label = labels[ii, jj, kk]
        neigh_image = images[ii, jj, kk]
        neigh_value = data[ii,jj,kk]

        # update image to be relative to the current points transformation
        si1 = INT_TO_IMAGE[neigh_image, 0] + ssi
        sj1 = INT_TO_IMAGE[neigh_image, 1] + ssj
        sk1 = INT_TO_IMAGE[neigh_image, 2] + ssk
        neigh_image = IMAGE_TO_INT[si1, sj1, sk1]

        # skip neighbors in the same basin
        if label0 == neigh_label and image0 == neigh_image:
            continue
            
        # if we haven't already, note the correct neighboring basin
        if n == 0:
            label1 = neigh_label
            image1 = neigh_image
            value1 = neigh_value
            n = 1
            continue
        
        # check if this point improves the connection value
        if use_minima and neigh_value < value1:
            value1 = neigh_value
            if value1 <= value0:
                break
        elif not use_minima and neigh_value > value1:
            value1 = neigh_value
            if value1 >= value0:
                break

    # if no neighbor was found, we just return a fake value
    if label1 == -1:
        return max_val, max_val, max_val, False, 0.0
    
    # otherwise we get the unit cell transform across which these extrema connect
    i, j, k = INT_TO_IMAGE[image0]
    ii, jj, kk = INT_TO_IMAGE[image1]
    image = IMAGE_TO_INT[ii-i, jj-j, kk-k]
    inv_image = IMAGE_TO_INT[i-ii, j-jj, k-kk]
    
    # get best value
    if use_minima:
        best_value = max(value0, value1)
    else:
        best_value = min(value0, value1)
    
    # determine if the canonical image is reversed. We flip it if the neighboring
    # label is lower and if the neighboring image is lower.
    is_reversed = (label0 > label1) != (image > inv_image)

    return (
        min(label0, label1), 
        max(label0, label1), 
        min(image, inv_image), 
        is_reversed,
        best_value)


@njit(parallel=True, cache=True)
def get_canonical_saddle_connections(
    labels: NDArray[np.int64],
    images: NDArray[np.int64],
    data: NDArray[np.float64],
    neighbor_transforms: NDArray[np.int64],
    edge_mask: NDArray[np.uint8],
    use_minima: bool = False,
):
    nx, ny, nz = labels.shape
    ny_nz = ny*nz
    
    # get the points that may be saddles
    saddle_coords = np.argwhere(edge_mask==1)
    
    # create an array to track connections between these points.
    # For each entry we will have:
        # 1: the lower label index
        # 2: the higher label index
        # 3: the connection image between basins
        # 4: whether or not the connection image is lower -> higher (0) or higher -> lower (1)
    saddle_connections = np.empty((len(saddle_coords),4),dtype=np.uint16)
    connection_vals = np.empty(len(saddle_coords), dtype=np.float64)
    # create a mask to track important connections
    important = np.ones(len(saddle_coords), dtype=np.bool)
    max_val = np.iinfo(np.uint16).max
    for idx in prange(len(saddle_coords)):
        i,j,k = saddle_coords[idx]
        lower, higher, shift, is_reversed, connection_value = get_extrema_saddle_connections(
            i, j, k,
            nx, ny, nz,
            ny_nz,
            labels,
            images,
            data,
            neighbor_transforms,
            edge_mask,
            max_val,
            use_minima,
        )
        if lower == max_val:
            # note this wasn't a true saddle
            important[idx] = False
            continue
        saddle_connections[idx, 0] = lower
        saddle_connections[idx, 1] = higher
        saddle_connections[idx, 2] = shift
        saddle_connections[idx, 3] = is_reversed
        connection_vals[idx] = connection_value
        
    # get only the connections that are important
    important = np.where(important)[0]
    saddle_coords = saddle_coords[important]
    saddle_connections = saddle_connections[important]
    connection_vals = connection_vals[important]
                
    return saddle_coords, saddle_connections, connection_vals

@njit(cache=True)
def get_single_point_saddles(
    data,
    connection_values,
    saddle_coords,
    connection_indices,
    num_connections,
    use_minima = False,
):
    # create an array to store best points
    saddles = np.empty(num_connections, dtype=np.uint16)
    if use_minima:
        best_vals = np.full(num_connections, np.inf, dtype=np.float64)
    else:
        best_vals = np.full(num_connections, -np.inf, dtype=np.float64)
    
    for saddle_idx, (idx, connection_value) in enumerate(zip(connection_indices, connection_values)):
        if not use_minima and  connection_value > best_vals[idx]:
            best_vals[idx] = connection_value
            saddles[idx] = saddle_idx
        elif use_minima and connection_value < best_vals[idx]:
            best_vals[idx] = connection_value
            saddles[idx] = saddle_idx            
    
    return saddles, best_vals

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
def initialize_labels_from_extrema(
    data,
    labels,
    extrema_frac,
    extrema_mask,
    neighbor_transforms,
    neighbor_dists,
    persistence_tol,
    max_vox_offset=5,
    use_minima=False,
):
    nx, ny, nz = data.shape
    ny_nz = ny * nz
    
    extrema_vox = np.argwhere(extrema_mask)
    
    # create an array to store values at each maximum
    extrema_values = np.empty(len(extrema_vox), dtype=np.float64)
    extrema_labels = np.empty(len(extrema_vox), dtype=np.int64)

    # create a flat array of shifs for tracking wrapping around edges. These
    # will initially all be (0,0,0)
    images = np.zeros((nx*ny*nz, 3), dtype=np.int8)

    # Now we initialize the extrema
    for ext_idx, (i, j, k) in enumerate(extrema_vox):
        # get value at extremum
        extrema_values[ext_idx] = data[i, j, k]
        # set as initial group root
        flat_ext_idx = coords_to_flat(i, j, k, ny_nz, nz)
        extrema_labels[ext_idx] = flat_ext_idx
        labels[flat_ext_idx] = flat_ext_idx
        
    ###########################################################################
    # 1. Remove Flat False Maxima
    ###########################################################################
    # If there is a particularly flat region, a point might have neighbors that
    # are the same value. This point may be mislabeled as a maximum if these
    # neighbors are not themselves extrema. This issue is typically caused by too
    # few sig figs in the data preventing the region from being properly distinguished

    # first we union any maxima that are within one voxel of each other
    for (i, j, k), ext_label in zip(extrema_vox, extrema_labels):
        for si, sj, sk in neighbor_transforms:
            # get neighbor and wrap
            ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
            if extrema_mask[ii,jj,kk]:
                # union
                neigh_label = coords_to_flat(ii,jj,kk,ny_nz,nz)
                union(labels, ext_label, neigh_label)

    # Now we look for any points that have neighbors with the same value that
    # aren't maxima. We hill climb from that point to find the corresponding
    # maximum
    flat_false_maxima = np.zeros(len(extrema_vox), dtype=np.bool_)
    for ext_idx, ((i, j, k), value, ext_label) in enumerate(
        zip(extrema_vox, extrema_values, extrema_labels)
    ):

        for si, sj, sk in neighbor_transforms:
            # get neighbor and wrap
            ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
            neigh_value = data[ii, jj, kk]
            # skip points that don't have the same value or that are also
            # extrema
            if neigh_value != value or extrema_mask[ii,jj,kk]:
                continue
            # If we're still here, this point is a false maximum. We follow the
            # path to the maximum and union
            flat_false_maxima[ext_idx] = True
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
    # 2. Combine low-persistence extrema
    ###########################################################################
    # With the right shape (e.g. highly anisotropic) a maximum/minimum may lay offgrid
    # and cause two ongrid points to appear to be higher than the region around
    # them. We merge these by taking a linear slice between each point using
    # cubic interpolation and combining those that have no minimum/maximum between
    # them.

    # sort extrema from from most to least extreme
    if not use_minima:
        sorted_indices = np.flip(np.argsort(extrema_values))
    else:
        sorted_indices = np.argsort(extrema_values)

    # record the points and values that connect
    # connections = []
    
    # Iterate over each maximum (except the first) and check for nearby extrema
    # above/below (extrema/minima)
    for sorted_ext_idx, ext_idx in enumerate(sorted_indices[1:]):
        
        # skip flat maxima
        if flat_false_maxima[ext_idx]:
            continue

        sorted_ext_idx += 1
        max_frac = extrema_frac[ext_idx]
        value = extrema_values[ext_idx]

        # iterate over extrema above this point
        for neigh_ext_idx in sorted_indices:
            # skip if this is the same point
            if neigh_ext_idx == ext_idx:
                continue

            # break if we reach a point lower than the current one
            neigh_value = extrema_values[neigh_ext_idx]
            if not use_minima and neigh_value < value:
                break
            elif use_minima and neigh_value > value:
                break

            neigh_frac = extrema_frac[neigh_ext_idx]
            # unwrap relative to central
            fi, fj, fk = neigh_frac - np.round(neigh_frac - max_frac)
            # get offset in frac coords
            oi = fi - max_frac[0]
            oj = fj - max_frac[1]
            ok = fk - max_frac[2]

            # covert to voxel coords and get distance
            ci = oi * nx
            cj = oj * ny
            ck = ok * nz            
            dist = (ci**2 + cj**2 + ck**2) ** (1/2)

            # # if above our cutoff, continue
            # if dist > max_vox_offset:
            #     continue

            # check if there is a minimum between this point and its neighbor
            n_points = max(math.ceil(dist * 3), 5)
            values = linear_slice(
                data, max_frac, (fi, fj, fk), n=n_points, is_frac=True, method="cubic"
            )

            # get the number of extrema
            s = np.sign(np.diff(values))

            if use_minima:
                # add end points
                s = np.append(-1, s)
                s = np.append(s, 1)
                # get min indices
                min_indices = np.where((s[:-1] <= 0) & (s[1:] >= 0))[0]
                if len(min_indices) < 2:
                    persistence_score = 0
                    conn_val = values.max()
                else:
                    min0 = min_indices[0]
                    min1 = min_indices[-1]
                    val = max(values[min0], values[min1])
                    conn_val = values[min0:min1].max()
                    persistence_score = (conn_val - val) / (conn_val + 1e-12)
            else:
                # add end points
                s = np.append(1, s)
                s = np.append(s, -1)
                # get max indices
                max_indices = np.where((s[:-1] >= 0) & (s[1:] <= 0))[0]
                if len(max_indices) < 2:
                    persistence_score = 0
                    conn_val = values.min()
                else:
                    max0 = max_indices[0]
                    max1 = max_indices[-1]
                    val = min(values[max0], values[max1])
                    conn_val = values[max0:max1].min()
                    persistence_score = (val - conn_val) / (conn_val + 1e-12)
            # if we don't have at least two of the corresponding extrema, these
            # points are not separated and we union.
            if persistence_score < persistence_tol:
                union(labels, extrema_labels[ext_idx], extrema_labels[neigh_ext_idx])
                # connections.append((ext_idx, conn_val))
    
    ###########################################################################
    # Root finding
    ###########################################################################

    # get each extremas root
    extrema_roots = np.empty(len(extrema_labels), dtype=np.int64)
    for ext_idx, ext_label in enumerate(extrema_labels):
        root = find_root(labels, ext_label)
        labels[ext_label] = root
        extrema_roots[ext_idx] = root
    unique_roots = np.unique(extrema_roots)
    
    # group the maxima and set them to the highest member in their group
    extrema_groups = []
    root_extrema = []

    for root_label in unique_roots:
        i,j,k = flat_to_coords(root_label, ny_nz, nz)
        group = []
        group_labels = []
        best_point = root_label
        best_value = data[i,j,k]
        for ext_idx, ext_root in enumerate(extrema_roots):
            if ext_root == root_label:
                # add to group
                label = extrema_labels[ext_idx]
                group_labels.append(label)
                # check if the value of this point is higher/lower
                ii, jj, kk = flat_to_coords(label, ny_nz, nz)
                group.append(np.array((ii,jj,kk), dtype=np.uint16))
                value = data[ii,jj,kk]
                if (
                    not use_minima and value > best_value
                    or use_minima and value < best_value
                    or value == best_value and label < root_label
                    ):
                    best_point = label
                    best_value = value
        # relabel each point in the group
        for label in group_labels:
            labels[label] = best_point
        # add group
        group_array = np.empty((len(group), 3), dtype=np.uint16)
        for idx, array in enumerate(group):
            group_array[idx] = array
        extrema_groups.append(group_array)
        # find the ext_idx of the best point
        root_idx = np.searchsorted(extrema_labels, best_point)
        root_extrema.append(root_idx)
    root_extrema = np.array(root_extrema, np.uint32)

    # Update roots again, and note the images each grouped maximum must cross
    for ext_idx, ext_label in enumerate(extrema_labels):
        root = labels[ext_label]
        # get fractional coordinates
        ext_frac = extrema_frac[ext_idx]
        root_idx = np.searchsorted(extrema_labels, root)
        root_frac = extrema_frac[root_idx]
        # get best image to wrap the maxima to its root
        images[extrema_labels[ext_idx]] = compute_wrap_offset(ext_frac, root_frac)
    
    # root_labels = extrema_labels[root_extrema]
    # root_values = extrema_values[root_extrema]
    extrema_coords = extrema_vox[root_extrema]
    extrema_frac = extrema_frac[root_extrema]

    # persistence_cutoffs = extrema_values[root_extrema]
    # # get the lowest/highest connection values for each root maxima/minima
    # for ext_idx, conn_val in connections:
    #     root = extrema_roots[ext_idx]
    #     root_idx = np.searchsorted(root_labels, root)
        
    #     # check that these extrema are reasonably close in value
    #     val = extrema_values[ext_idx]
    #     root_val = root_values[root_idx]
        
    #     if (root_val-val)/val > persistence_tol:
    #         continue

    #     if use_minima:
    #         persistence_cutoffs[root_idx] = max(persistence_cutoffs[root_idx], conn_val)
    #     else:
    #         persistence_cutoffs[root_idx] = min(persistence_cutoffs[root_idx], conn_val)


    return labels, images, extrema_coords, extrema_frac, extrema_groups#, persistence_cutoffs

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
                basin = labels[i,j,k]
                value = data[i,j,k]
                if value > basin_max[basin]:
                    basin_max[basin] = value
                if value < basin_min[basin]:
                    basin_min[basin] = value
    return basin_min, basin_max

@njit(cache=True)
def get_persistence_scores(
        critical_values,
        connections,
        connection_values,
        use_minima=False,
        ):
        
    # create array to store persistence scores
    persistence_scores = np.empty(len(connections), dtype=np.float64)
    
    # persistence criteria:
        # 1. persistence = abs(extremum_val - saddle_val)
        # 2. divide by saddle value
    
    for pair_idx in prange(len(connection_values)):
        crit1, crit2, _, _ = connections[pair_idx]
        
        # get the value at the saddle connecting them
        saddle_val = connection_values[pair_idx]
        
        # get each critical points value
        val1 = critical_values[crit1]
        val2 = critical_values[crit2]
        
        if use_minima:
            val = max(val1, val2)
        else:
            val = min(val1, val2)
        
        persistence_score = abs(saddle_val - val) / (saddle_val + 1e-12)
        persistence_scores[pair_idx] = persistence_score

    return persistence_scores

@njit(cache=True)
def group_by_persistence(
        data,
        critical_vox,
        basin_connections,
        saddle_values,
        persistence_tol,
        # persistence_cutoffs,
        use_minima = False,
        ):
    num_critical = len(critical_vox)
    
    critical_frac = critical_vox / np.array(data.shape)
    
    # create array to track unions between basins
    unions = np.arange(num_critical)
    
    # create array to track the important saddles
    important_saddles = np.ones(len(saddle_values), dtype=np.bool_)
    active_connections = np.zeros(len(saddle_values), dtype=np.bool_)
    saddle_indices = np.arange(len(important_saddles))
    
    # get values at critical points
    critical_values = np.empty(len(critical_vox), dtype=np.float64)
    for idx, (i,j,k) in enumerate(critical_vox):
        critical_values[idx] = data[i,j,k]

    # initialize temporary copies of connections
    current_connections = basin_connections.copy()
    current_connection_values = saddle_values.copy()
    current_indices = saddle_indices.copy()
    
    while True:
        
        connection_mask = np.zeros(len(current_connections), dtype=np.bool_)
        
        # get current persistence
        persistence_scores = get_persistence_scores(
            critical_values,
            current_connections,
            current_connection_values,
            use_minima=use_minima,
            )

        # loop over persistence and combine crit below the tolerance
        for pair_idx, ((crit1, crit2, _, _), score, conn_val, saddle_idx) in enumerate(zip(
                current_connections, 
                persistence_scores, 
                current_connection_values,
                current_indices,
                )):
            # skip pairs that are already part of the same basin, but note that
            # they have been unioned
            if crit1 == crit2:
                connection_mask[pair_idx] = True
                important_saddles[saddle_idx] = False
                continue
            # skip anything with a high score
            if score > persistence_tol:
                continue
            # we want to union the lower maximum to the higher one. For ties,
            # we use the lower index
            if not use_minima:
                value1 = critical_values[crit1]
                value2 = critical_values[crit2]
            else:
                value1 = -critical_values[crit1]
                value2 = -critical_values[crit2]
                
            if value1 >= value2:
                higher = crit1
                lower = crit2
            else:
                higher = crit2
                lower = crit1
            # make the union
            union(unions, lower, higher)
            
            connection_mask[pair_idx] = True
            important_saddles[saddle_idx] = False
            active_connections[saddle_idx] = True

        # get unchanged connections
        connection_indices = np.where(~connection_mask)[0]
        if len(connection_indices) == len(current_connections):
            break

        new_connections = current_connections[connection_indices]
        new_connection_values = current_connection_values[connection_indices]
        new_indices = current_indices[connection_indices]
        for pair_idx, (crit1, crit2, _, _) in enumerate(new_connections):
            root1 = find_root(unions, crit1)
            root2 = find_root(unions, crit2)
            
            new_connections[pair_idx, 0] = root1
            new_connections[pair_idx, 1] = root2

        # update our connections for the next round
        current_connections = new_connections
        current_connection_values = new_connection_values
        current_indices = new_indices
        
    # get the roots of all extrema
    roots = np.empty(num_critical, dtype=np.int64)
    for idx in range(num_critical):
        root = find_root(unions, idx)
        roots[idx] = root
        
    root_transforms = np.empty((len(roots), 3), dtype=np.int8)
    # Get the transformations from each merged point to its parents
    for ext_idx, root_idx in enumerate(roots):
        crit_frac = critical_frac[ext_idx]
        root_frac = critical_frac[root_idx]
        root_transforms[ext_idx] = compute_wrap_offset(crit_frac, root_frac)

    # # update persistence cutoffs
    # for active, (crit1, crit2, _, _), value in zip(active_connections, basin_connections, saddle_values):
    #     # skip connections that didn't activate
    #     if not active:
    #         continue

    #     # get the root
    #     root = roots[crit1]
        
    #     # check that these extrema are reasonably close in value
    #     val = critical_values[ext_idx]
    #     root_val = critical_values[root_idx]
        
    #     if (root_val-val)/val > persistence_tol:
    #         continue
        
    #     if use_minima:
    #         persistence_cutoffs[root] = max(persistence_cutoffs[root], value)
    #     else:
    #         persistence_cutoffs[root] = min(persistence_cutoffs[root], value)

    return roots, root_transforms#, persistence_cutoffs

@njit(cache=True, parallel=True)
def get_persistence_cutoffs(
    data,
    groups,
    max_dist=5
        ):
    persistence_cutoffs = np.full(len(groups), np.inf, dtype=np.float64)
    shape = np.array(data.shape)
    for group_idx in prange(len(groups)):
        # get group
        group = groups[group_idx]
        # wrap all to the same region
        group_frac = group / shape
        ref = group_frac[0]
        group_frac = group_frac - np.round(group_frac - ref)
        # convert back to vox
        group = group_frac * shape
        # get distances between them
        neighs = np.empty((len(group)), dtype=np.int32)
        dists = np.empty((len(group)), dtype=np.float64)
        for i in range(len(group)):
            best_dist = np.inf
            best_neigh = -1
            ci,cj,ck = group[i]
            for j in range(len(group)):
                if i == j:
                    continue
                ci1,cj1,ck1 = group[j]
                dist = ci*ci1 + cj*cj1 + ck*ck1
                if dist < best_dist:
                    best_dist = dist
                    best_neigh = j
            neighs[i] = best_neigh
            dists[i] = best_dist
        # for each point, get the closest point
        lowest_val = np.inf
        for j, (i, dist) in enumerate(zip(neighs, dists)):
            # skip dists above our cutoff
            if dist > max_dist:
                continue
            p1 = group[i]
            p2 = group[j]
            n = math.ceil(dist*3)
            # otherwise get values between
            values=linear_slice(data=data, p1=p1, p2=p2, n=n, is_frac=False,method="nearest")
            lowest = values.min()
            lowest_val = min(lowest, lowest_val)
        persistence_cutoffs[group_idx] = lowest_val
    return persistence_cutoffs
        
        

@njit(parallel=True, cache=True)
def update_labels_and_images(
    labels,
    images,
    label_map,
    image_map,
    vacuum_mask,
        ):
    nx,ny,nz = labels.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # if this voxel is part of the vacuum, continue
                if vacuum_mask[i, j, k]:
                    continue
                # get current label
                label = labels[i,j,k]
                # get the current shift
                shift = INT_TO_IMAGE[images[i,j,k]]
                # get the shift from this maxima to its root
                maxima_shift = image_map[label]
                # update the image
                si,sj,sk = shift+maxima_shift
                images[i,j,k] = IMAGE_TO_INT[si,sj,sk]
                # update label
                labels[i,j,k] = label_map[label]
                
    return labels, images

@njit(parallel=True, cache=True)
def update_final_images(
    labels,
    images,
    image_map,
    important_mask,
    vacuum_mask,
        ):
    nx,ny,nz = labels.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # if this voxel is part of the vacuum, continue
                if vacuum_mask[i, j, k]:
                    continue
                # get current label
                label = labels[i,j,k]
                if not important_mask[label]:
                    continue
                # get the current shift
                shift = INT_TO_IMAGE[images[i,j,k]]
                # get the shift from this maxima to its root
                maxima_shift = image_map[label]
                # update the image
                si,sj,sk = shift+maxima_shift
                images[i,j,k] = IMAGE_TO_INT[si,sj,sk]
                
    return images

# @njit(cache=True)
# def group_by_persistence(
#         data,
#         critical_vox,
#         connections,
#         connection_values,
#         lattice,
#         persistence_tol,
#         persistence_cutoffs,
#         use_minima = False,
#         ):
#     num_critical = len(critical_vox)
#     unions = np.arange(num_critical)
#     # create an array to track crit and their periodic images
#     crit_image_map = np.full((num_critical, 27), -1, dtype=np.int64)
#     for i in range(num_critical):
#         crit_image_map[i,13] = i
#     for crit1, crit2, image1, image2 in connections:
#         if crit_image_map[crit1,image1] == -1:
#             crit_image_map[crit1, image1] = num_critical
#             num_critical += 1
#         if crit_image_map[crit2,image2] == -1:
#             crit_image_map[crit2, image2] = num_critical
#             num_critical += 1
#     image_crit_map = np.argwhere(crit_image_map != -1)
#     crit_vals = np.empty(len(image_crit_map), dtype=np.int64)
#     for idx, (i,j) in enumerate(image_crit_map):
#         crit_vals[idx] = crit_image_map[i,j]
#     crit_sort = np.argsort(crit_vals)
#     image_crit_map = image_crit_map[crit_sort]
    
#     # create array to track unions
#     unions_w_image = np.arange(num_critical)
    
#     # get the fractional representation of each critical point
#     critical_frac = critical_vox / np.array(data.shape, dtype=np.int64)
    
#     # get values for each critical point
#     critical_values = np.empty(len(critical_vox), dtype=np.float64)
#     for crit_idx, (i, j, k) in enumerate(critical_vox):
#         # get value at maximum
#         critical_values[crit_idx] = data[i, j, k]
    
#     current_connections = connections.copy()
#     current_connection_values = connection_values.copy()
    
#     while True:
        
#         connection_mask = np.zeros(len(current_connections), dtype=np.bool_)
        
#         # get current persistence
#         persistence_scores = get_persistence_scores(
#             data,
#             critical_frac,
#             critical_values,
#             current_connections,
#             current_connection_values,
#             lattice,
#             )
#         # loop over persistence and combine crit below the tolerance
#         for pair_idx, ((crit1, crit2, image1, image2), score, conn_val) in enumerate(zip(current_connections, persistence_scores, current_connection_values)):
#             # skip anything with a high score
#             if score > persistence_tol:
#                 continue
#             if crit1 == crit2 and image1==image2:
#                 continue
#             # we want to union the lower maximum to the higher one. For ties,
#             # we use the lower index
#             value1 = critical_values[crit1]
#             value2 = critical_values[crit2]
#             if value1 >= value2:
#                 higher = crit1
#                 lower = crit2
#                 higher_image = crit_image_map[crit1, image1]
#                 lower_image = crit_image_map[crit2, image2]
#             else:
#                 higher = crit2
#                 lower = crit1
#                 higher_image = crit_image_map[crit2, image2]
#                 lower_image = crit_image_map[crit1, image1]
#             # make the union
#             union(unions, lower, higher)
#             union(unions_w_image, lower_image, higher_image)
            
#             connection_mask[pair_idx] = True
            
#             # update persistence cutoffs if the connection value is similar
#             higher_val = max(value1, value2)
#             # if abs((higher_val-conn_val)/higher_val) < 0.05:
#             if conn_val < persistence_cutoffs[crit1]:
#                 persistence_cutoffs[crit1] = conn_val
#             if conn_val < persistence_cutoffs[crit2]:
#                 persistence_cutoffs[crit2] = conn_val

#         # get unchanged connections
#         connection_indices = np.where(~connection_mask)[0]
#         if len(connection_indices) == len(current_connections):
#             break

#         new_connections = current_connections[connection_indices]
#         new_connection_values = current_connection_values[connection_indices]
#         for pair_idx, (crit1, crit2, image1, image2) in enumerate(new_connections):
#             # update to roots
#             root1 = find_root(unions_w_image, crit1)
#             root2 = find_root(unions_w_image, crit2)
#             basin1, image1 = image_crit_map[root1]
#             basin2, image2 = image_crit_map[root2]
#             new_connections[pair_idx, 0] = basin1
#             new_connections[pair_idx, 1] = basin2
#             new_connections[pair_idx, 2] = image1
#             new_connections[pair_idx, 3] = image2

#         # update our connections for the next round
#         current_connections = new_connections
#         current_connection_values = new_connection_values
#     # get the roots of all extrema and update persistence
#     roots = np.empty(len(unions), dtype=np.int64)
#     for idx in range(len(roots)):
#         root = find_root(unions, idx)
#         roots[idx] = root
#         # if this extrema has a similar value to its root, we also update the
#         # persistence
#         value1 = critical_values[idx]
#         value2 = critical_values[root]
#         higher_val = max(value1, value2)
#         conn_val = persistence_cutoffs[idx]
#         # if abs((higher_val-conn_val)/higher_val) < 0.05:
#         if conn_val < persistence_cutoffs[root]:
#             persistence_cutoffs[root] = conn_val

#     return roots, persistence_cutoffs

@njit(cache=True)
def get_persistence_groups(
    labels,
    data,
    persistence_cutoffs,
    extrema_vox,
    use_minima,
        ):
    nx, ny, nz = data.shape
    max_val = len(extrema_vox)
    
    persistence_groups = []
    for vox in extrema_vox:
        temp = [vox]
        temp = temp[1:]
        persistence_groups.append(temp)
    
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
                if (
                    not use_minima and value >= cutoff
                    or use_minima and value <= cutoff
                    ):
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
