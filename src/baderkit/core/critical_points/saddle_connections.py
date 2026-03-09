# -*- coding: utf-8 -*-

import itertools

import numpy as np
from numpy.typing import NDArray
from numba import njit, prange

from baderkit.core.utilities.basic import wrap_point_w_shift, wrap_point
from baderkit.core.utilities.interpolation import spline_hess_cart
from baderkit.core.utilities.interpolation import newton_refine_critical

NEIGHBOR_TRANSFORMS = np.array(
    [(dx, dy, dz)
     for dx in (-1, 0, 1)
     for dy in (-1, 0, 1)
     for dz in (-1, 0, 1)
     if not (dx == dy == dz == 0)],
    dtype=np.int8
)

IMAGE_TO_INT = np.empty([3,3,3], dtype=np.int64)
INT_TO_IMAGE = np.array(list(itertools.product((-1,0,1), repeat=3)))
for shift_idx, (i,j,k) in enumerate(INT_TO_IMAGE):
    IMAGE_TO_INT[i,j,k] = shift_idx
    
@njit(cache=True)
def get_saddle_most_connections(
    i,j,k,
    labels,
    images,
    neighbor_transforms,
    neighbor_dists,
    vacuum_mask,
        ):
    nx,ny,nz = labels.shape
    neigh_labels = []
    neigh_label_counts = []
    max_dist = neighbor_dists.max()
    for trans, ((si, sj, sk), dist) in  enumerate(zip(
            neighbor_transforms,
            neighbor_dists,
             )):
        # get the neighbor
        ii, jj, kk, ssi, ssj, ssk = wrap_point_w_shift(
            i+si, j+sj, k+sk, nx, ny, nz
        )
        # skip points that can't be part of our connections
        if vacuum_mask[ii,jj,kk]:
            continue

        # get the label of the neighbor
        neigh_label = labels[ii,jj,kk]
        ni, nj, nk = INT_TO_IMAGE[images[ii,jj,kk]]
        neigh_image = IMAGE_TO_INT[ssi+ni, ssj+nj, ssk+nk]
        
        comb = tuple((neigh_label, neigh_image))
        if not comb in neigh_labels:
            neigh_labels.append(comb)
            neigh_label_counts.append(1/(1+dist/max_dist))
        else:
            for label_idx, label in enumerate(neigh_labels):
                if comb == label:
                    neigh_label_counts[label_idx] += 1/(1+dist/max_dist)
                    break
    # get two most found neighbors
    neigh_label_counts = np.array(neigh_label_counts, dtype=np.uint8)
    b0, b1 = np.argsort(neigh_label_counts)[-2:]
    return neigh_labels[b0][0], neigh_labels[b0][1], neigh_labels[b1][0], neigh_labels[b1][1]
    
    
@njit(cache=True)
def get_saddle_extrema_connections(
    labels,
    images,
    saddle_coords,
    saddle_indices,
    neighbor_transforms: NDArray[np.int64],
    neighbor_dists,
    vacuum_mask,
):
    nx, ny, nz = labels.shape

    connections = []
    problem_saddles = []
    for saddle_idx in saddle_indices:
        i,j,k = saddle_coords[saddle_idx]
        
        label0 = -1; image0 = -1
        label1 = -1; image1 = -1
        n = 0
        
        
        for trans, (si, sj, sk) in  enumerate(neighbor_transforms):
            # get the neighbor
            ii, jj, kk, ssi, ssj, ssk = wrap_point_w_shift(
                i+si, j+sj, k+sk, nx, ny, nz
            )
            # skip points that can't be part of our connections
            if vacuum_mask[ii,jj,kk]:
                continue

            # get the label of the neighbor
            neigh_label = labels[ii,jj,kk]
            ni, nj, nk = INT_TO_IMAGE[images[ii,jj,kk]]
            neigh_image = IMAGE_TO_INT[ssi+ni, ssj+nj, ssk+nk]
            
            if n == 0:
                label0 = neigh_label
                image0 = neigh_image
                n=1
            elif n == 1:
                if label0 != neigh_label or image0 != neigh_image:
                    label1 = neigh_label
                    image1 = neigh_image
                    n=2
            elif n == 2:
                if (
                    neigh_label == label0 and neigh_image == image0
                    or neigh_label == label1 and neigh_image == image1
                    ):
                    continue
                # backup to best count of labels
                label0, image0, label1, image1 = get_saddle_most_connections(
                    i, j, k, 
                    labels, 
                    images, 
                    neighbor_transforms, 
                    neighbor_dists,
                    vacuum_mask)
                break

        if n == 2:
            connections.append((saddle_idx, label0, image0))
            connections.append((saddle_idx, label1, image1))
        else:
            problem_saddles.append(saddle_idx)

    return connections, problem_saddles


@njit(cache=True)
def get_saddle_saddle_connections(
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
    queue = np.empty((num_edges, 3), dtype=np.uint32)
    
    # create arrays to store flood filled labels
    max_val = np.iinfo(np.int32).max
    flood_labels = np.full_like(edge_mask, max_val, dtype=np.int32)
    flood_images = np.full_like(edge_mask, 13, dtype=np.uint8)
    
    # seed saddles
    saddle_idx = 0
    saddle1_idx = 1
    saddle2_idx = -1
    for i,j,k in saddle1_coords:
        flood_labels[i,j,k] = saddle1_idx
        queue[saddle_idx] = (i,j,k)
        saddle1_idx += 1
        saddle_idx += 1
    for i,j,k in saddle2_coords:
        flood_labels[i,j,k] = saddle2_idx
        queue[saddle_idx] = (i,j,k)
        saddle2_idx -= 1
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
            mi, mj, mk = INT_TO_IMAGE[flood_images[i,j,k]]
            
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

                # get the label of the neighbor
                neigh_label = flood_labels[ii,jj,kk]
                
                # get total image of the current path
                mi1 = mi+ssi
                mj1 = mj+ssj
                mk1 = mk+ssk

                shift = IMAGE_TO_INT[mi1, mj1, mk1]
                
                # if unlabeled, label immediately
                if neigh_label == max_val:
                    flood_labels[ii,jj,kk] = label
                    flood_images[ii,jj,kk] = shift
                    queue[next_end] = (ii,jj,kk)
                    next_end += 1
                # skip points that are both the same type of saddle
                elif (
                    (neigh_label < 0 and label < 0) or
                    (neigh_label > 0 and label > 0) or
                    (neigh_label == label)
                    ):
                    continue

                else:
                    # this point belongs to a different saddle
                    # get this points image
                    ni, nj, nk = INT_TO_IMAGE[flood_images[ii,jj,kk]]
                    bi = mi1 - ni
                    bj = mj1 - nj
                    bk = mk1 - nk
                    
                    # order from saddle1 to saddle2
                    if label > 0 and neigh_label < 0:
                        best_image = IMAGE_TO_INT[bi,bj,bk]
                        lower = abs(label)-1
                        upper = abs(neigh_label)-1
                    
                    elif label < 0 and neigh_label > 0:
                        best_image = IMAGE_TO_INT[-bi, -bj, -bk]
                        lower = abs(neigh_label)-1
                        upper = abs(label)-1
                    else:
                        # we should never have the same label or two labels
                        # with the same sign
                        continue
                    
                    connections.append((
                        lower,
                        upper,
                        best_image,
                        ))
                    # add coord
                    connection_coords.append(queue[edge_idx])
        queue_start = queue_end
        queue_end = next_end
    
    
    return connections, connection_coords
    
# @njit(cache=True)
# def get_saddle_extrema_connections(
#     saddle_coords,
#     extrema_groups,
#     allowed_indices,
#     neighbor_transforms: NDArray[np.int64],
#     edge_mask: NDArray[np.uint8],
# ):
#     nx, ny, nz = edge_mask.shape
    
#     # get the number of possible edges
#     num_edges = len(np.where(np.isin(edge_mask, allowed_indices))[0])
    
#     # create arrays to store flood filled labels
#     max_val = np.iinfo(np.int32).max
#     flood_labels = np.full_like(edge_mask, max_val, dtype=np.int32)
#     flood_images = np.full_like(edge_mask, 13, dtype=np.uint8)
    
#     # create an empty queue for storing which points are next
#     queue = np.empty((num_edges, 3), dtype=np.uint32)
    
#     # seed saddles
#     extra_idx = 0
#     saddle_idx = 1
#     extrema_idx = -1
#     for i,j,k in saddle_coords:
#         flood_labels[i,j,k] = saddle_idx
#         queue[extra_idx] = (i,j,k)
#         saddle_idx += 1
#     for group in extrema_groups:
#         for i,j,k in group:
#             flood_labels[i,j,k] = extrema_idx
#             queue[extra_idx] = (i,j,k)
#             extra_idx += 1
#         extrema_idx -= 1
    
#     # extend queue
#     queue = np.row_stack((queue, np.empty((extra_idx,3), dtype=np.uint32)))
    
#     # create lists to store connections
#     connections = []
#     connection_coords = []
#     queue_start = 0
#     queue_end = extra_idx
    
#     while queue_start != queue_end:
#         next_end = queue_end
#         for edge_idx in range(queue_start, queue_end):
#             i,j,k = queue[edge_idx]
#             # get label and image
#             label = flood_labels[i,j,k]
#             flat_image = flood_images[i,j,k]
#             mi, mj, mk = INT_TO_IMAGE[flat_image]
            
#             # iterate over each neighbor. if unlabeled, assign it the same label
#             # if labeled, note a new connection
#             for trans, (si, sj, sk) in  enumerate(neighbor_transforms):
#                 # get the neighbor
#                 ii, jj, kk, ssi, ssj, ssk = wrap_point_w_shift(
#                     i+si, j+sj, k+sk, nx, ny, nz
#                 )
#                 # skip points that can't be part of our connections
#                 if not edge_mask[ii,jj,kk] in allowed_indices:
#                     continue

#                 # get the label of the neighbor
#                 neigh_label = flood_labels[ii,jj,kk]
#                 shift = IMAGE_TO_INT[ssi+mi, ssj+mj, ssk+mk]
                
#                 # if unlabeled, label immediately
#                 if neigh_label == max_val:
#                     flood_labels[ii,jj,kk] = label
#                     flood_images[ii,jj,kk] = shift
#                     queue[next_end] = (ii,jj,kk)
#                     next_end += 1
#                 # skip points that are both the same type of critical point
#                 elif (
#                     (neigh_label < 0 and label < 0) or
#                     (neigh_label > 0 and label > 0)
#                     ):
#                     continue

#                 else:
#                     neigh_image = flood_images[ii,jj,kk]
#                     # get shift difference
#                     ni, nj, nk = INT_TO_IMAGE[neigh_image]
#                     si, sj, sk = INT_TO_IMAGE[shift]
#                     bi = si - ni
#                     bj = sj - nj
#                     bk = sk - nk
                   
#                     # if this belongs to a different saddle, we note a connection
#                     if neigh_label != label:
#                         # order from saddle1 to saddle2
#                         if label > 0 and neigh_label < 0:
#                             lower = abs(label)-1
#                             upper = abs(neigh_label)-1
#                             image = IMAGE_TO_INT[bi, bj, bk]
                        
#                         elif label < 0 and neigh_label > 0:
#                             lower = abs(neigh_label)-1
#                             upper = abs(label)-1
#                             image = IMAGE_TO_INT[-bi, -bj, -bk]
#                         else:
#                             # we should never have the same labels or two labels
#                             # with the same sign
#                             continue
                        
#                         connections.append((
#                             lower,
#                             upper,
#                             image
#                             ))
#                         # add coord
#                         connection_coords.append(queue[edge_idx])
#         queue_start = queue_end
#         queue_end = next_end
    
    
#     return connections, connection_coords

# # @njit(parallel=True, cache=True)
# def get_saddle_extrema_connections(
#     data,
#     labels,
#     matrix,
#     images,
#     saddle_coords,
#         ):
#     nx, ny, nz = labels.shape
    
#     dir2car = np.linalg.inv(matrix).T
    
#     connections0 = np.empty((len(saddle_coords), 2), dtype=np.uint32)
#     connections1 = np.empty((len(saddle_coords), 2), dtype=np.uint32)
    
#     for saddle_idx in prange(len(saddle_coords)):
#         coord = saddle_coords[saddle_idx]
#         i,j,k = coord
        
#         # get hessian
#         H = spline_hess_cart(
#             i,j,k,
#             data, 
#             dir2car,
#             )
        
#         # get eigval and eigvec
#         eigvals, eigvecs = np.linalg.eigh(H)
        
#         # get type
#         saddle_type = np.count_nonzero(eigvals<0)
        
#         # get vector
#         if saddle_type == 1:
#             v = eigvecs.T[np.where(eigvals < 0)[0][0]]
#         if saddle_type == 2:
#             v = eigvecs.T[np.where(eigvals > 0)[0][0]]
            
#         # convert back to voxel coord
#         v_dir = v @ matrix
        
#         # normalize
#         v_dir = v_dir / np.linalg.norm(v_dir)
        
#         # create trackers for unique labels
#         lower_label = labels[i,j,k]
#         lower_image = images[i,j,k]
#         upper_label = -1; upper_image = -1

#         coord0 = coord.astype(np.float64)
#         coord1 = coord.astype(np.float64)
#         image0 = np.zeros(3, dtype=np.int8)
#         image1 = np.zeros(3, dtype=np.int8)
                
#         for _ in range(10):
#             # update coords for this step
#             coord0 = coord0 + v_dir
#             coord1 = coord1 - v_dir

#             # round down
#             c0 = np.floor(coord0).astype(np.int32)
#             c1 = np.floor(coord1).astype(np.int32)

#             # wrap
#             i0, j0, k0, si0, sj0, sk0  = wrap_point_w_shift(c0[0],c0[1],c0[2], nx, ny, nz)
#             i1, j1, k1, si1, sj1, sk1  = wrap_point_w_shift(c1[0],c1[1],c1[2], nx, ny, nz)

#             # update images
#             image0[0] += si0
#             image0[1] += sj0
#             image0[2] += sk0

#             image1[0] += si1
#             image1[1] += sj1
#             image1[2] += sk1

#             # get labels and assignment image at this point
#             label0 = labels[i0, j0, k0]
#             pimage0 = INT_TO_IMAGE[images[i0, j0, k0]].copy()

#             label1 = labels[i1, j1, k1]
#             pimage1 = INT_TO_IMAGE[images[i1, j1, k1]].copy()

#             # update image relative to the current point
#             pimage0 += image0
#             pimage1 += image1

#             pimage0 = IMAGE_TO_INT[pimage0[0],pimage0[1],pimage0[2]]
#             pimage1 = IMAGE_TO_INT[pimage1[0],pimage1[1],pimage1[2]]

                
#             if lower_label != label0 or lower_image != pimage0:
#                 upper_label = label0
#                 upper_image = pimage0
#                 break
#             elif lower_label != label1 or lower_image != pimage1:
#                 upper_label = label1
#                 upper_image = pimage1
#                 break
                    
#         connections0[saddle_idx, 0] = lower_label
#         connections0[saddle_idx, 1] = lower_image
#         connections1[saddle_idx, 0] = upper_label
#         connections1[saddle_idx, 1] = upper_image

#     return connections0, connections1

