# -*- coding: utf-8 -*-
from baderkit.core.utilities.basic import (
    wrap_point_w_shift, 
    flat_to_coords
    )

import itertools
import numpy as np
from numpy.typing import NDArray
from numba import njit, prange



IMAGE_TO_INT = np.empty([3,3,3], dtype=np.int64)
INT_TO_IMAGE = np.array(list(itertools.product((-1,0,1), repeat=3)))
for shift_idx, (i,j,k) in enumerate(INT_TO_IMAGE):
    IMAGE_TO_INT[i,j,k] = shift_idx

FACE_TRANSFORMS = np.array([
    [1,0,0],
    [-1,0,0],
    [0,1,0],
    [0,-1,0],
    [0,0,1],
    [0,0,-1],
    ], dtype=np.int64)

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
        for i,j,k in group:
            edges[i,j,k] = 0
            
    for group in maxima_groups:
        for i,j,k in group:
            edges[i,j,k] = 3
    
    # loop over each voxel in parallel
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # if this voxel is part of the vacuum, continue
                if vacuum_mask[i, j, k]:
                    continue
                
                # if this voxel is part of a minimum or maximum, continue
                if edges[i,j,k] == 0 or edges[i,j,k] == 3:
                    continue
                
                # check if this point has 0, 1, or 2 neighbors with different
                # labels
                num_neighs = get_differing_neighs(
                    i, j, k, 
                    nx, ny, nz, 
                    maxima_labels, 
                    maxima_images, 
                    neighbor_transforms, 
                    vacuum_mask,
                    )
                opp_num_neighs = get_differing_neighs(
                    i, j, k, 
                    nx, ny, nz, 
                    minima_labels, 
                    minima_images, 
                    neighbor_transforms, 
                    vacuum_mask,
                    )
                
                if num_neighs == 1 and opp_num_neighs > 1:
                    # saddle 1
                    edges[i,j,k] = 1
                elif num_neighs > 1 and opp_num_neighs == 1:
                    # saddle 2
                    edges[i,j,k] = 2
                elif num_neighs < 1 and opp_num_neighs == 1:
                    # edge of minima manifold
                    edges[i,j,k] = 4
                elif num_neighs == 1 and opp_num_neighs <1:
                    # edge of maxima manifold
                    edges[i,j,k] = 5
                elif num_neighs == 1 and opp_num_neighs == 1:
                    # edge of both maxima/minima manifold
                    edges[i,j,k] = 6
                elif num_neighs < 1 and opp_num_neighs > 1:
                    # meeting of at least three minima manifolds
                    edges[i,j,k] = 7
                elif num_neighs > 1 and opp_num_neighs < 1:
                    # meeting of at least three maxima manifolds
                    edges[i,j,k] = 8

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
    if use_minima:
        best_value = np.inf
        edge_indices = (1,2,4,6)
    else:
        best_value = -np.inf
        edge_indices = (1,2,5,6)

    best_neigh_label = -1
    best_neigh_image = -1
    inv_image_idx = -1
    # iterate over transforms
    label = labels[i,j,k]
    image = images[i,j,k]
    value = data[i,j,k]

    mi, mj, mk = INT_TO_IMAGE[image]

    for trans in range(neighbor_transforms.shape[0]):
        # get shifts
        si = neighbor_transforms[trans, 0]
        sj = neighbor_transforms[trans, 1]
        sk = neighbor_transforms[trans, 2]

        # wrap around periodic edges and store shift
        ii, jj, kk, ssi, ssj, ssk = wrap_point_w_shift(
            i+si, j+sj, k+sk, nx, ny, nz
        )
        # skip neighbors that are not also part of the edge
        if not edge_mask[ii, jj, kk] in edge_indices:
            continue

        
        # get the label and image of this neighbor
        neigh_label = labels[ii, jj, kk]
        neigh_image = images[ii, jj, kk]

        # update image to be relative to the current points transformation
        si1 = INT_TO_IMAGE[neigh_image, 0] + ssi
        sj1 = INT_TO_IMAGE[neigh_image, 1] + ssj
        sk1 = INT_TO_IMAGE[neigh_image, 2] + ssk
        neigh_image = IMAGE_TO_INT[si1, sj1, sk1]

        # skip neighbors in the same basin
        if label == neigh_label and image == neigh_image:
            continue
        
        # get the value of the neighbor
        neigh_value = data[ii,jj,kk]
        
        if use_minima:
            best_val = max(value, neigh_value)
            improved = best_val <= best_value
        else:
            best_val = min(value, neigh_value)
            improved = best_val >= best_value
        
        if improved:
            best_value = best_val
            best_neigh_label = neigh_label
            # adjust image to point
            si1 -= mi
            sj1 -= mj
            sk1 -= mk
            best_neigh_image = IMAGE_TO_INT[si1, sj1, sk1]
            inv_image_idx = IMAGE_TO_INT[-si1, -sj1, -sk1]
            
            # we can't improve beyond this points value so we can break
            if best_value == value:
                break

    # if no neighbor was found, we just return a fake value
    if best_neigh_label == -1:
        return max_val, max_val, max_val, False, 0.0
    
    is_reversed = best_neigh_image > inv_image_idx
    
    return (
        min(label, best_neigh_label), 
        max(label, best_neigh_label), 
        min(best_neigh_image, inv_image_idx), 
        is_reversed,
        best_value)

@njit(cache=True)
def get_ongrid_gradient_cart(i, j, k, data, dir2car):
    nx, ny, nz = data.shape
    
    c000 = data[i,j,k]
    c100 = data[(i + 1) % nx,j,k]
    c_100 = data[(i - 1) % nx,j,k]
    c010 = data[i,(j + 1) % ny,k]
    c0_10 = data[i,(j - 1) % ny,k]
    c001 = data[i,j,(k + 1) % nz]
    c00_1 = data[i,j,(k - 1) % nz]

    # central differences in voxel coordinates
    gi = (c100 - c_100) / (2.0)
    gj = (c010 - c0_10) / (2.0)
    gk = (c001 - c00_1) / (2.0)

    # optional extrema clamping
    if c100 <= c000 and c_100 <= c000:
        gi = 0.0
    if c010 <= c000 and c0_10 <= c000:
        gj = 0.0
    if c001 <= c000 and c00_1 <= c000:
        gk = 0.0

    # convert to fractional-coordinate gradient
    gi *= nx
    gj *= ny
    gk *= nz

    # convert to Cartesian gradient
    gx = dir2car[0, 0] * gi + dir2car[0, 1] * gj + dir2car[0, 2] * gk
    gy = dir2car[1, 0] * gi + dir2car[1, 1] * gj + dir2car[1, 2] * gk
    gz = dir2car[2, 0] * gi + dir2car[2, 1] * gj + dir2car[2, 2] * gk

    return gx, gy, gz

@njit(parallel=True, cache=True)
def get_canonical_saddle_connections(
    labels: NDArray[np.int64],
    images: NDArray[np.int64],
    data: NDArray[np.float64],
    neighbor_transforms: NDArray[np.int64],
    edge_mask: NDArray[np.uint8],
    dir2car: NDArray[np.float64],
    use_minima: bool = False,
):
    nx, ny, nz = labels.shape
    ny_nz = ny*nz
    
    # get the points that may be saddles
    if use_minima:
        saddle_coords = np.argwhere(np.isin(edge_mask,(2)))
    else:
        saddle_coords = np.argwhere(np.isin(edge_mask,(1)))
    
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
    nx,ny,nz = data.shape
    ny_nz = ny*nz
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
        else:
            i,j,k = flat_to_coords(idx, ny_nz, nz)
            value = data[i,j,k]
            if value == connection_value:
                best_vals[idx] = connection_value
                saddles[idx] = saddle_idx
    
    return saddles, best_vals

@njit(cache=True)
def get_saddle_connections(
    saddle1_coords,
    saddle2_coords,
    neighbor_transforms: NDArray[np.int64],
    edge_mask: NDArray[np.uint8],
):
    nx, ny, nz = edge_mask.shape
    
    # get the number of possible edges
    num_edges = len(np.where(np.isin(edge_mask, (1,2,6)))[0])
    num_edges += len(saddle1_coords) + len(saddle2_coords)
    
    # create an empty queue for storing which points are next
    queue = np.empty((num_edges, 3), dtype=np.uint16)
    
    # create arrays to store flood filled labels
    max_val = np.iinfo(np.uint16).max
    flood_labels = np.full_like(edge_mask, max_val, dtype=np.uint16)
    flood_images = np.full_like(edge_mask, 13, dtype=np.uint8)
    
    # seed saddles
    saddle_idx = 0
    for i,j,k in saddle2_coords:
        flood_labels[i,j,k] = saddle_idx
        queue[saddle_idx] = (i,j,k)
        saddle_idx += 1
    for i,j,k in saddle1_coords:
        flood_labels[i,j,k] = saddle_idx
        queue[saddle_idx] = (i,j,k)
        saddle_idx += 1
    
    # create lists to store connections
    connections = []
    connection_coords = []
    queue_start = 0
    queue_end = saddle_idx
    
    while queue_start != queue_end:
        next_end = queue_end
        for edge_idx in range(queue_start, queue_end):
            i,j,k = queue[edge_idx]
            # get label and image
            label = flood_labels[i,j,k]
            image = flood_images[i,j,k]
            mi, mj, mk = INT_TO_IMAGE[image]
            
            # iterate over each neighbor. if unlabeled, assign it the same label
            # if labeled, note a new connection
            for trans, (si, sj, sk) in  enumerate(neighbor_transforms):
                # get the neighbor
                ii, jj, kk, ssi, ssj, ssk = wrap_point_w_shift(
                    i+si, j+sj, k+sk, nx, ny, nz
                )
                # skip points that can't be part of our connections
                if not edge_mask[ii,jj,kk] in (1,2,6):
                    continue
                
                neigh_label = flood_labels[ii,jj,kk]
                shift = IMAGE_TO_INT[ssi+mi, ssj+mj, ssk+mk]
                
                # skip points that haven't been labeled
                if neigh_label == max_val:
                    flood_labels[ii,jj,kk] = label
                    flood_images[ii,jj,kk] = shift
                    queue[next_end] = (ii,jj,kk)
                    next_end += 1

                else:
                    # get where this image has wrapped around periodic edges
                    neigh_image = flood_images[ii,jj,kk]
    
                    # if this belongs to a different saddle, we note a connection
                    if neigh_label != label or neigh_image != shift:
                        # get shift difference
                        ni, nj, nk = INT_TO_IMAGE[neigh_image]
                        si, sj, sk = INT_TO_IMAGE[shift]
                        
                        bi = ni - si
                        bj = nj - sj
                        bk = nk - sk
                        best_image = IMAGE_TO_INT[bi,bj,bk]
                        inv_image = IMAGE_TO_INT[-bi, -bj, -bk]
                        
                        connections.append((
                            min(label, neigh_label),
                            max(label, neigh_label),
                            min(best_image, inv_image),
                            best_image < inv_image # whether the connection is reversed or not
                            ))
                        # add coord
                        connection_coords.append(queue[edge_idx])
        queue_start = queue_end
        queue_end = next_end
    
    
    return connections, connection_coords

