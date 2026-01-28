# -*- coding: utf-8 -*-

from numba import njit, prange
import numpy as np

from baderkit.core.utilities.basic import wrap_point

def f_edge(vertices):
    # give value of edge based on highest and second highest facets
    if vertices[0] > vertices[1]:
        return vertices
    else:
        return np.flip(vertices)
    
def f_poly(vertices):
    # get the edge with the highest value
    edges = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0]
        ])
    best_edge_idx = 0
    best_edge = f_edge(vertices[edges[0]])

    for edge_idx, indices in enumerate(edges[1:]):
        new_edge = f_edge(vertices[indices])
        if new_edge[0] > best_edge[0]:
            best_edge_idx = edge_idx + 1
            best_edge = new_edge
        elif new_edge[0] == best_edge[0] and new_edge[1] > best_edge[1]:
            best_edge_idx = edge_idx + 1
            best_edge = new_edge
            
    # Now get the edge disjoint from the one we found
    best_edge_idx = (best_edge_idx + 2) % 4
    next_edge = f_edge(vertices[edges[edge_idx]])
    
    # replace vertices and return
    vertices[:2] = best_edge
    vertices[2:] = next_edge

    return vertices
            
def f_voxel(vertices):
    # get the polynomial with the highest value
    polys = np.array([
        [0, 1, 2, 3],
        [0, 1, 5, 4],
        [0, 3, 7, 4],
        [6, 3, 4, 7],
        [6, 2, 3, 7],
        [6, 2, 1, 3],
        ])
    
    best_poly_idx = 0
    best_poly = f_poly(vertices[polys[0]])
    
    for poly_idx, indices in enumerate(polys[1:]):
        new_poly = f_poly(vertices[indices])
        for i in range(4):
            if new_poly[i] < best_poly[i]:
                break
            elif new_poly[i] > best_poly[i]:
                best_poly = new_poly
                best_poly_idx = poly_idx
                break
            
    # Now get the polynomial on the opposite side of the voxel
    best_poly_idx = (best_poly_idx + 3) % 6
    next_poly = f_edge(vertices[polys[best_poly_idx]])
    
    # replace vertices and return
    vertices[:4] = best_poly
    vertices[4:] = next_poly

    return vertices

@njit
def get_uint8_flag(crit: int, face: int, neigh: int):
    """
    Converts connection info to an 8bit integer.
    Assumes following order:
        0: empty
        1: 1 if crit point
        2-4: transform index of pointer to highest valued face.
        5-7: transform index of pointer to unioned cofacet
    
    """
    value = np.uint8(0)
    
    # if critical, add 64 (1 index of bit)
    value += 64*crit
    
    # update face pointer. 32, 16, and 8 (2, 3, 4 indices of bit)
    value += 8*face
    
    # update union pointer. 4, 2, and 1
    value += neigh

    return value

def get_connection_from_flag(value: np.uint8):
    
    crit = value // 64
    value %= 64
    
    face = value // 8
    value %= 8
    
    #neigh = value
    return crit, face, value

def pair_vertex(i, j, k, data, f):
    nx, ny, nz = data.shape
    # convert to data units
    di = i/2
    dj = j/2
    dk = k/2
    value = data[di, dj, dk]
    # check neighbors
    neighbor_transforms = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1],
        [-1,0,0],
        [0,-1,0],
        [0,0,-1],
        ])
    best_idx = -1
    best_value = None
    for transform_idx, (si, sj, sk) in enumerate(neighbor_transforms):
        ni, nj, nk = wrap_point(di+si, dj+sj, dk+sk, nx, ny, nz)
        # skip edges that are above the vertices point
        neigh_value = data[ni, nj, nk]
        if neigh_value > value:
            continue
        if best_value is None or neigh_value > best_value:
            best_idx = transform_idx
            best_value = neigh_value
    # get 8-bit int representation
    if best_idx == -1:
        crit = 1
        best_idx = 0
        best_value = 0
    else:
        crit = 0
    flag = get_uint8_flag(crit, 0, best_idx)
    f[i,j,k] = flag
    
def pair_edge(i, j, k, data, f):
    nx, ny, nz = data
    # get the direction of the edge
    for vec, x in enumerate((i,j,k)):
        if x % 2 == 1:
            break
    
    # get transformations to neighboring polynomials
    neighbor_transforms = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1],
        [-1,0,0],
        [0,-1,0],
        [0,0,-1],
        ])
    
    # get transforms to vertices
    vertices = np.empty((10, 3), dtype=np.int8)
    vertices[0,vec] = 1
    vertices[1,vec] = -1
    for trans_idx, trans in enumerate(neighbor_transforms):
        if trans[vec] != 0:
            continue
        # get next available vertex row
        vec_idx = (trans_idx+1) * 2
        # add vertices with negative vec first to ensure a proper loop
        vertices[vec_idx] = vertices[1] + trans * 2
        vertices[vec_idx+1] = vertices[0] + trans * 2
    
    values = np.empty(10, dtype=np.float64)
    
    # get vertex values
    for vert_idx, (si, sj, sk) in enumerate(vertices):
        ni, nj, nk = wrap_point((i+si)/2, (j+sj)/2, (k+sk)/2, nx, ny, nz)
        values[vert_idx] = data[ni, nj, nk]
        
    # get main edge value
    value = f_edge(values[0,1])
    
    best_idx = -1
    best_value = None
    # get best poly
    for trans_idx, trans in enumerate(neighbor_transforms):
        # skip irrelavent transformations
        if trans[vec] != 0:
            continue
        vec_idx = (trans_idx+1) * 2
        neigh_value = f_poly(values[0, 1, vec_idx, vec_idx+1])
        # if this edge isn't the highest facet of this poly, we skip
        for i in range(2):
            if neigh_value[i] != value[i]:
                continue
            
        # if this is our first value, set it as out best
        if best_value is None:
            best_idx = trans_idx
            best_value = neigh_value
        
        # if any extra values improve the value, replace it
        for i in range(2, 4):
            if neigh_value[i] < best_value[i]:
                best_idx = trans_idx
                best_value = neigh_value
                break
            elif neigh_value[i] > best_value[i]:
                break
    # get 8-bit int representation
    if best_idx == -1:
        crit = 1
        best_idx = 0
        best_value = 0
    else:
        crit = 0
    flag = get_uint8_flag(crit, 0, best_idx)
    f[i,j,k] = flag
    

        
        
    
    # get transformations to relavent vertices
    # get all vertex values
    # skip any transformations in direction parallel to edge
    # get 2 digit value of this edge
    # get best neighbor:
        # get proper polynomial representation matching required order
        # get 4 digit value of polynomial
        # reject if first 2 digits don't match this edge
        # reject if second 2 digits are higher than any previous
    # convert to flag and store
    
def pair_poly(i, j, k, data, f):
    pass
    # get transformations to neighboring voxels
    # get transformations to relavent vertices
    # get all vertex values
    # skip any transformations parallel to plane
    # get 4 digit value of this poly
    # get best neighbor:
        # get proper voxel representation matching required order. May be tricky
        # get 8 digit value of polynomial
        # reject if first 4 digits don't match this edge
        # reject if second 4 digits are higher than any previous
    # convert to flag and store

# figure out gradient traversal and connection mapping
# pull out critical points and smoothed path arcs? Any other info? Decending/ascending manifolds?
# Any way to rigorously use results to make bifurcation plot?
# implement into bifurcations?