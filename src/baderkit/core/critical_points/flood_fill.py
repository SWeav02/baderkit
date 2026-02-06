# -*- coding: utf-8 -*-
import time
from baderkit.core.utilities.basic import (
    wrap_point_w_shift, 
    wrap_point, 
    coords_to_flat, 
    flat_to_coords,
    )
from baderkit.core.utilities.union_find import union, find_root_no_compression, find_root
from baderkit.core.bader.methods.shared_numba import get_best_neighbor
from baderkit.core import Bader
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

@njit(inline='always')
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

@njit(parallel=True)
def get_possible_saddles(
    labels: NDArray[np.int64],
    opposite_labels: NDArray[np.int64],
    images: NDArray[np.int64],
    opposite_images: NDArray[np.int64],
    neighbor_transforms: NDArray[np.int64],
    vacuum_mask: NDArray[np.bool_],
):
    """
    In a 3D array of labeled voxels, finds the voxels that neighbor at
    least one voxel with a different label.

    Parameters
    ----------
    labels : NDArray[np.int64]
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
                opp_num_neighs = get_differing_neighs(
                    i, j, k, 
                    nx, ny, nz, 
                    opposite_labels, 
                    opposite_images, 
                    neighbor_transforms, 
                    vacuum_mask,
                    )
                
                if num_neighs == 1 and opp_num_neighs > 1:
                    edges[i,j,k] = 1
                elif num_neighs > 1 and opp_num_neighs == 1:
                    edges[i,j,k] = 2
                elif num_neighs == 1 and opp_num_neighs <1:
                    edges[i,j,k] = 3
                elif num_neighs < 1 and opp_num_neighs == 1:
                    edges[i,j,k] = 4
                elif num_neighs == 1 and opp_num_neighs == 1:
                    edges[i,j,k] = 5
    return edges


@njit(inline='always')
def get_saddle_connection(
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
        edge_indices = (2,4,5)
    else:
        best_value = -np.inf
        edge_indices = (1,3,5)
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
        if edge_mask[ii, jj, kk] not in edge_indices:
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
            # we only allow one neighbor type, so if our point is the lower
            # one, we can break
            if best_val == value:
                break

    # if no neighbor was found, we just return a fake value
    if best_neigh_label == -1:
        return max_val, max_val, max_val, 0.0
    # assert best_neigh_label != -1
    
    return (
        min(label, best_neigh_label), 
        max(label, best_neigh_label), 
        min(best_neigh_image, inv_image_idx), 
        best_value)


@njit(parallel=True)
def get_possible_saddle_vals(
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
    if use_minima:
        possible_saddles = np.argwhere(edge_mask==2)
    else:
        possible_saddles = np.argwhere(edge_mask==1)
    
    # create an array to track connections between these points.
    # For each entry we will have:
        # 1: the lower label index
        # 2: the higher label index
        # 3: the connection image from lower basin to higher basin
    saddle_connections = np.empty((len(possible_saddles),3),dtype=np.uint16)
    connection_vals = np.empty(len(possible_saddles), dtype=np.float64)
    # create a mask to track important connections
    important = np.ones(len(possible_saddles), dtype=np.bool)
    max_val = np.iinfo(np.uint16).max
    for idx in prange(len(possible_saddles)):
        i,j,k = possible_saddles[idx]
        lower, higher, shift, connection_value = get_saddle_connection(
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
        connection_vals[idx] = connection_value
        
    # get only the connections that are important
    important = np.where(important)[0]
    possible_saddles = possible_saddles[important]
    saddle_connections = saddle_connections[important]
    connection_vals = connection_vals[important]
                
    return edge_mask, possible_saddles, saddle_connections, connection_vals

# @njit
def get_saddles(
    connection_values,
    saddle_coords,
    connection_indices,
    num_connections,
    edges,
    use_minima = False,
):
    # create an array to store best points
    saddles = np.empty(num_connections, dtype=np.uint16)
    if use_minima:
        best_vals = np.full(num_connections, np.inf, dtype=np.float64)
    else:
        best_vals = np.full(num_connections, -np.inf, dtype=np.float64)
    
    for saddle_idx, (idx, connection_value) in enumerate(zip(connection_indices, connection_values)):
        if not use_minima and best_vals[idx] < connection_value:
            best_vals[idx] = connection_value
            saddles[idx] = saddle_idx
        elif use_minima and best_vals[idx] > connection_value:
            best_vals[idx] = connection_value
            saddles[idx] = saddle_idx
            
    # update edge labels
    for idx in saddles:
        i,j,k = saddle_coords[idx]
        edges[i,j,k] = 3
    
    return saddles, best_vals

# @njit(parallel=True)
# def get_possible_1saddles(
#     labels: NDArray[np.int64],
#     images: NDArray[np.int64],
#     data: NDArray[np.float64],
#     neighbor_transforms: NDArray[np.int64],
#     edge_mask: NDArray[np.uint8],
# ):
#     nx, ny, nz = labels.shape
#     ny_nz = ny*nz
    
#     # get the points that may be saddles
#     possible_saddles = np.argwhere(edge_mask==1)
    
#     # create an array to track connections between these points.
#     # For each entry we will have:
#         # 1: the lower label index
#         # 2: the higher label index
#         # 3: the connection image from lower basin to higher basin
#     saddle_connections = np.empty((len(possible_saddles),3),dtype=np.uint16)
#     connection_vals = np.empty(len(possible_saddles), dtype=np.float64)
#     # create a mask to track important connections
#     important = np.ones(len(possible_saddles), dtype=np.bool)
#     max_val = np.iinfo(np.uint16).max
#     for idx in prange(len(possible_saddles)):
#         i,j,k = possible_saddles[idx]
#         lower, higher, shift, connection_value = get_saddle_connection(
#             i, j, k,
#             nx, ny, nz,
#             ny_nz,
#             labels,
#             images,
#             data,
#             neighbor_transforms,
#             edge_mask,
#             max_val,
#         )
#         if lower == max_val:
#             # note this wasn't a true saddle
#             important[idx] = False
#             continue
#         saddle_connections[idx, 0] = lower
#         saddle_connections[idx, 1] = higher
#         saddle_connections[idx, 2] = shift
#         connection_vals[idx] = connection_value
        
#     # get only the connections that are important
#     important = np.where(important)[0]
#     possible_saddles = possible_saddles[important]
#     saddle_connections = saddle_connections[important]
#     connection_vals = connection_vals[important]
                
#     return edge_mask, possible_saddles, saddle_connections, connection_vals

# # @njit
# def get_1saddles(
#     connection_values,
#     saddle_coords,
#     connection_indices,
#     num_connections,
#     edges,
# ):
#     # create an array to store best points
#     saddles = np.empty(num_connections, dtype=np.uint16)
#     best_vals = np.full(num_connections, -np.inf, dtype=np.float64)
    
#     for saddle_idx, (idx, connection_value) in enumerate(zip(connection_indices, connection_values)):
#         if best_vals[idx] < connection_value:
#             best_vals[idx] = connection_value
#             saddles[idx] = saddle_idx
            
#     # update edge labels
#     for idx in saddles:
#         i,j,k = saddle_coords[idx]
#         edges[i,j,k] = 3
    
#     return saddles, best_vals
# @njit(parallel=True)
# def get_possible_1saddles(
#     labels: NDArray[np.int64],
#     images: NDArray[np.int64],
#     data: NDArray[np.float64],
#     neighbor_transforms: NDArray[np.int64],
#     edge_mask: NDArray[np.uint8],
#     # minima_vox: NDArray[np.uint16],
# ):
#     nx, ny, nz = labels.shape
    
#     # get the points that may be saddles
#     possible_saddles = np.argwhere(edge_mask==2)
#     # get values
#     edge_values = np.empty(len(possible_saddles), dtype=np.float64)
#     for idx, (i,j,k) in enumerate(possible_saddles):
#         edge_values[idx] = data[i,j,k]
        
#     # sort from lowest to highest
#     sorted_indices = np.argsort(edge_values)
    
#     # flood fill, labeling new minima as we go
#     max_val = np.iinfo(np.uint16).max
#     flood_labels = np.full_like(data, max_val, dtype=np.uint16)
#     flood_images = np.full_like(data, 13, dtype=np.uint8)
    
#     minima_val = 0
#     connected_minima = []
#     connection_images = []
#     connection_coords = []
#     minima = []
#     for edge_idx in sorted_indices:
#         i,j,k = possible_saddles[edge_idx]
#         # make new minima if unlabeled
#         label = flood_labels[i,j,k]
#         image = flood_images[i,j,k]
#         mi, mj, mk = INT_TO_IMAGE[image]
#         if label == max_val:
#             flood_labels[i,j,k] = minima_val
#             label = minima_val
#             minima.append(possible_saddles[edge_idx])
#             minima_val += 1
        
#         # iterate over each neighbor. if unlabeled, assign it the same label
#         # if labeled, note a new connection
#         for trans, (si, sj, sk) in  enumerate(neighbor_transforms):
#             # get the neighbor
#             ii, jj, kk, ssi, ssj, ssk = wrap_point_w_shift(
#                 i+si, j+sj, k+sk, nx, ny, nz
#             )
#             # skip non-edges
#             if edge_mask[ii,jj,kk] != 2:
#                 continue
            
#             neigh_label = flood_labels[ii,jj,kk]
#             shift = IMAGE_TO_INT[ssi+mi, ssj+mj, ssk+mk]
            
#             # skip points that haven't been labeled
#             if neigh_label == max_val:
#                 flood_labels[ii,jj,kk] = label
#                 flood_images[ii,jj,kk] = shift
#             else:
#                 # get where this image has wrapped around periodic edges
#                 neigh_image = flood_images[ii,jj,kk]

#                 # if this belongs to a different minima, we note a connection
#                 if neigh_label != label or neigh_image != shift:
#                     connected_minima.append(np.array((min(label, neigh_label), max(label,neigh_label)), dtype=np.uint16))
#                     # get shift difference
#                     ni, nj, nk = INT_TO_IMAGE[neigh_image]
#                     si, sj, sk = INT_TO_IMAGE[shift]
                    
#                     bi = ni - si
#                     bj = nj - sj
#                     bk = nk - sk
#                     best_image = IMAGE_TO_INT[bi,bj,bk]
#                     inv_image = IMAGE_TO_INT[-bi, -bj, -bk]
                    
#                     connection_images.append(min(best_image, inv_image))
#                     # add coord
#                     connection_coords.append(possible_saddles[edge_idx])
    
#     connected_minima = np.array(connected_minima, dtype=np.uint16)
#     connection_images = np.array(connection_images, dtype=np.uint16)
#     connections = np.column_stack((connected_minima, connection_images))
    
#     return np.array(minima, dtype=np.uint16), connections, np.array(connection_coords, dtype=np.uint16)


bader = Bader.from_vasp("CHGCAR", method="neargrid", persistence_tol=0.01)

# get basins
maxima_labels = bader.maxima_basin_labels
maxima_images = bader.maxima_basin_images

minima_labels = bader.minima_basin_labels
minima_images = bader.minima_basin_images

neighbor_transforms, neighbor_dists,_,_ = bader.reference_grid.point_neighbor_voronoi_transforms
# neighbor_transforms1, _ = bader.reference_grid.point_neighbor_transforms

edges = get_possible_saddles(
    maxima_labels,
    minima_labels,
    maxima_images,
    minima_images,
    neighbor_transforms,
    bader.vacuum_mask
    )

edges, possible_saddles, saddle_connections, connection_vals = get_possible_saddle_vals(
    maxima_labels,
    maxima_images,
    bader.reference_grid.total,
    neighbor_transforms,
    edges,
    use_minima=False,
    )

unique_connections, inverse = np.unique(saddle_connections,axis=0, return_inverse=True)

saddle2_indices, saddle_vals = get_saddles(
    connection_values=connection_vals,
    saddle_coords=possible_saddles,
    connection_indices=inverse,
    num_connections=len(unique_connections),
    edges=edges,
)

saddle2_coords = possible_saddles[saddle2_indices]
saddle2_connections = saddle_connections[saddle2_indices]
saddle2_vals = connection_vals[saddle2_indices]

edges, possible_saddles, saddle_connections, connection_vals = get_possible_saddle_vals(
    minima_labels,
    minima_images,
    bader.reference_grid.total,
    neighbor_transforms,
    edges,
    use_minima=True,
    )

unique_connections, inverse = np.unique(saddle_connections,axis=0, return_inverse=True)

saddle1_indices, saddle_vals = get_saddles(
    connection_values=connection_vals,
    saddle_coords=possible_saddles,
    connection_indices=inverse,
    num_connections=len(unique_connections),
    edges=edges,
)

saddle1_coords = possible_saddles[saddle1_indices]
saddle1_connections = saddle_connections[saddle1_indices]
saddle1_vals = connection_vals[saddle1_indices]


test_mask = np.zeros_like(edges, dtype=np.uint8)
test_mask[saddle1_coords[:,0],saddle1_coords[:,1],saddle1_coords[:,2]] = 1
test_mask[saddle2_coords[:,0],saddle2_coords[:,1],saddle2_coords[:,2]] = 2


test_grid = bader.reference_grid.copy()
test_structure = bader.structure.copy()
for coord in bader.minima_frac:
    # coord = coord / bader.reference_grid.shape
    test_structure.append("x", coord)
test_grid.structure = test_structure
for i in range(2):
    test_grid.total = test_mask==i+1
    # test_grid.total = edges==i+1
    test_grid.write_vasp(f"ELFCAR_test_{i+1}")