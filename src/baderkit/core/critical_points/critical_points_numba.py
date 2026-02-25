# -*- coding: utf-8 -*-
from baderkit.core.utilities.basic import (
    wrap_point,
    wrap_point_w_shift, 
    flat_to_coords
    )

import itertools
import numpy as np
from numpy.typing import NDArray
from numba import njit, prange

def get_trans_maps() -> tuple:
    """

    Returns
    -------
    tuple
        A tuple with two parts:
            trans_to_int
                An array which maps the 6 unit vector transforms to a single
                integer representation
            int_to_trans
                An array which maps the integer representation of a 6 unit vector
                to the 3 entry transformation

    """
    trans_to_int = np.zeros((3, 3, 3), dtype=np.int8)
    int_to_trans = np.zeros((6, 3), dtype=np.int8)
    for i in range(6):
        j = i // 2
        value = (-1) ** (i & 1)  # -1 for even, 1 for odd
        int_to_trans[i, j] = value
        x, y, z = [0 if k != j else value for k in (0, 1, 2)]
        trans_to_int[x, y, z] = i
    return trans_to_int, int_to_trans


TRANS_TO_INT, INT_TO_TRANS = get_trans_maps()

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

# PARITIES = np.array(
#     [
#         [0, 0, 0],  # 3D
#         [0, 0, 1],  # 2D
#         [0, 1, 0],  # 2D
#         [0, 1, 1],  # 1D
#         [1, 0, 0],  # 2D
#         [1, 0, 1],  # 1D
#         [1, 1, 0],  # 1D
#         [1, 1, 1],  # 0D
#     ],
#     dtype=np.uint8,
# )
# PARITY_DIMS = np.array([3, 2, 2, 1, 2, 1, 1, 0], dtype=np.uint8)
PARITIES = np.array(
    [
        [0, 0, 0],  # 0D
        [0, 0, 1],  # 1D
        [0, 1, 0],  # 1D
        [0, 1, 1],  # 2D
        [1, 0, 0],  # 1D
        [1, 0, 1],  # 2D
        [1, 1, 0],  # 2D
        [1, 1, 1],  # 3D
    ],
    dtype=np.uint8,
)
PARITY_DIMS = np.array([0, 1, 1, 2, 1, 2, 2, 3], dtype=np.uint8)

# Conversion from parity to its index
PARITY_TO_INT = np.zeros((2, 2, 2), dtype=np.int8)
for idx, (i, j, k) in enumerate(PARITIES):
    PARITY_TO_INT[i, j, k] = idx
    
# Generate transformations to vertices for each parity type.
CUBE_VERTICES = [
    [-1, -1, -1],
    [1, -1, -1],
    [1, 1, -1],
    [-1, 1, -1],
    # [-1, -1, 1],
    # [1, -1, 1],
    # [1, 1, 1],
    # [-1, 1, 1],
]
POLY_VERTICES = [
    [-1, -1],
    [1, -1],
    # [1, 1],
    # [-1, 1],
]
EDGE_VERTICES = [
    [-1],
    # [1],
]
VERTEX_VERTICES = [[]]
VERTICES = [CUBE_VERTICES, POLY_VERTICES, EDGE_VERTICES, VERTEX_VERTICES]


def parity_to_vertices(parity: NDArray[int]) -> NDArray[int]:
    """

    Parameters
    ----------
    parity : NDArray[int]
        The parity of a cell.

    Returns
    -------
    NDArray[int]
        A Nx3 array where each row is the relative transform from a cell with
        the input parity to each of its vertices.

    """
    zero = np.where(parity == 0)[0]
    vertex_type = len(zero)
    base_transforms = VERTICES[vertex_type]
    transforms = []
    for trans in base_transforms:
        new_trans = trans.copy()
        for idx in zero:
            new_trans.insert(idx, 0)
        transforms.append(new_trans)
    return np.array(transforms)


def get_parity_vertices() -> list[NDArray]:
    """

    Returns
    -------
    transforms : list[NDArray]
        The relative transforms for each parity that generate a corresponding
        cells vertices.

    """
    transforms = []
    for parity in PARITIES:
        transforms.append(parity_to_vertices(parity))
    return tuple(transforms)


PARITY_VERTICES = get_parity_vertices()

###############################################################################
# Facets
###############################################################################
# Generate transformations to facets for each parity type.

CUBE_FACETS = [[0, 0, -1], [0, 0, 1], [0, -1, 0], [0, 1, 0], [-1, 0, 0], [1, 0, 0]]
POLY_FACETS = [
    [0, -1],
    [1, 0],
    [0, 1],
    [-1, 0],
]
EDGE_FACETS = [[-1], [1]]
VERTEX_FACETS = []
FACETS = [CUBE_FACETS, POLY_FACETS, EDGE_FACETS, VERTEX_FACETS]


def parity_to_facets(parity: NDArray[int]) -> NDArray[int]:
    """

    Parameters
    ----------
    parity : NDArray[int]
        The parity of a cell.

    Returns
    -------
    NDArray[int]
        A Nx3 array where each row is the relative transform from a cell with
        the input parity to each of its facets.

    """
    zero = np.where(parity == 0)[0]
    vertex_type = len(zero)
    base_transforms = FACETS[vertex_type]
    transforms = []
    for trans in base_transforms:
        new_trans = trans.copy()
        for idx in zero:
            new_trans.insert(idx, 0)
        transforms.append(new_trans)
    return np.array(transforms)


def get_parity_facets() -> list[NDArray]:
    """

    Returns
    -------
    transforms : list[NDArray]
        The relative transforms for each parity that points to a corresponding
        cell's facets.

    """
    transforms = []
    for parity in PARITIES:
        transforms.append(parity_to_facets(parity))
    return tuple(transforms)


PARITY_FACETS = get_parity_facets()

PARITY_FACETS_INT = tuple(
    [
        np.array([TRANS_TO_INT[x, y, z] for x, y, z in parity], dtype=np.int8)
        for parity in PARITY_FACETS
    ]
)



###############################################################################
# marching cubes
###############################################################################

# corner offsets
CORNERS = np.array([
    (0,0,0), (1,0,0), (0,1,0), (1,1,0),
    (0,0,1), (1,0,1), (0,1,1), (1,1,1)
], dtype=np.int8)

TETS = np.array([
    (0, 1, 3, 7),
    (0, 3, 2, 7),
    (0, 2, 6, 7),
    (0, 6, 4, 7),
    (0, 4, 5, 7),
    (0, 5, 1, 7),
], dtype=np.int8)

# #@njit(cache=True)
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
    # if c100 <= c000 and c_100 <= c000:
    #     gi = 0.0
    # if c010 <= c000 and c0_10 <= c000:
    #     gj = 0.0
    # if c001 <= c000 and c00_1 <= c000:
    #     gk = 0.0

    # convert to fractional-coordinate gradient
    gi *= nx
    gj *= ny
    gk *= nz

    # convert to Cartesian gradient
    gx = dir2car[0, 0] * gi + dir2car[0, 1] * gj + dir2car[0, 2] * gk
    gy = dir2car[1, 0] * gi + dir2car[1, 1] * gj + dir2car[1, 2] * gk
    gz = dir2car[2, 0] * gi + dir2car[2, 1] * gj + dir2car[2, 2] * gk

    return gx, gy, gz

def get_cart_gradients(
    data,
    dir2car
        ):
    nx,ny,nz = data.shape
    gradients = np.empty((nx,ny,nz,3), dtype=np.float32)
    for i in prange(nx):
        for j in prange(ny):
            for k in prange(nz):
                gradients[i,j,k] = get_ongrid_gradient_cart(i, j, k, data, dir2car)
    return gradients


# Plan:
    # 1. Get vertices
    # 2. check where edges meet, depending on parity
    # 3. Count unique differing neighbor pairs at meeting point
        # 1: meeting of 2 basins
        # 2: meeting of 3 basins
        # 3. meeting of 4 basins

def get_differing_neighs_doublegrid(
    i, j, k,
    nx, ny, nz,
    labels,
    images,
    vacuum_mask,
):
    
    # initialize stored labels
    label0 = -1; image0 = -1
    label1 = -1; image1 = -1
    unique = -1
    
    # get parity
    pi = i&1
    pj = j&1
    pk = k&1
    parity = PARITY_TO_INT[pi,pj,pk]
    # get transforms
    neighbor_transforms = PARITY_VERTICES[parity]

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
            int((i+si)/2), int((j+sj)/2), int((k+sk)/2), nx, ny, nz
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
        if unique == -1:
            label0 = neigh_label
            image0 = neigh_image
            unique = 0
        if unique == 0:
            if neigh_label != label0 or neigh_image != image0:
                label1 = neigh_label
                image1 = neigh_image
                unique = 1
        elif unique == 1:
            if ((neigh_label != label0 or neigh_image != image0) and
                (neigh_label != label1 or neigh_image != image1)):
                unique = 2
                break

    return unique

# #@njit(parallel=True,  cache=True)
def get_manifold_labels_doublegrid(
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
    nx2 = nx*2
    ny2 = ny*2
    nz2 = nz*2
    # create 3D array to store edges
    edges = np.full((nx*2,ny*2,nz*2), np.iinfo(np.uint8).max, dtype=np.uint8)
    
    # add maxima/minima
    for group in minima_groups:
        for i,j,k in group:
            # shift to double grid
            i*=2
            j*=2
            k*=2
            edges[i,j,k] = 0
            
    for group in maxima_groups:
        for i,j,k in group:
            # shift to double grid
            i*=2
            j*=2
            k*=2
            edges[i,j,k] = 3

    
    # loop over each voxel in parallel
    for i in prange(nx2):
        for j in range(ny2):
            for k in range(nz2):
                
                # if this voxel is part of a minimum or maximum, continue
                if edges[i,j,k] == 0 or edges[i,j,k] == 3:
                    continue
                
                pi = i&1
                pj = j&1
                pk = k&1
                parity = PARITY_TO_INT[pi,pj,pk]
                dim = PARITY_DIMS[parity]
                if dim != 1:
                    continue
                
                # check if this point has 0, 1, or 2 neighbors with different
                # labels
                num_neighs = get_differing_neighs_doublegrid(
                    i, j, k, 
                    nx, ny, nz, 
                    maxima_labels, 
                    maxima_images, 
                    vacuum_mask,
                    )
                opp_num_neighs = get_differing_neighs_doublegrid(
                    i, j, k, 
                    nx, ny, nz, 
                    minima_labels, 
                    minima_images, 
                    vacuum_mask,
                    )
                label = 10
                if num_neighs == 1:
                    label = 1
                elif num_neighs > 1:
                    label = 2
                if num_neighs == 1 and opp_num_neighs > 1:
                    # saddle 1
                    label = 1
                elif num_neighs > 1 and opp_num_neighs == 1:
                    # saddle 2
                    label = 2
                elif num_neighs < 1 and opp_num_neighs == 1:
                    # edge of minima manifold
                    label = 4
                elif num_neighs == 1 and opp_num_neighs <1:
                    # edge of maxima manifold
                    label = 5
                elif num_neighs == 1 and opp_num_neighs == 1:
                    # edge of both maxima/minima manifold
                    label = 6
                elif num_neighs < 1 and opp_num_neighs > 1:
                    # meeting of at least three minima manifolds
                    label = 7
                elif num_neighs > 1 and opp_num_neighs < 1:
                    # meeting of at least three maxima manifolds
                    label = 8
                edges[i,j,k] = label

    return edges

###############################################################################
# double grid test
###############################################################################

# #@njit(inline='always', cache=True)
def get_differing_neighs_doublegrid(
    i, j, k,
    nx, ny, nz,
    labels,
    images,
    vacuum_mask,
):
    
    # initialize stored labels
    label0 = -1; image0 = -1
    label1 = -1; image1 = -1
    unique = -1
    
    # get parity
    pi = i&1
    pj = j&1
    pk = k&1
    parity = PARITY_TO_INT[pi,pj,pk]
    # get transforms
    neighbor_transforms = PARITY_VERTICES[parity]

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
            int((i+si)/2), int((j+sj)/2), int((k+sk)/2), nx, ny, nz
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
        if unique == -1:
            label0 = neigh_label
            image0 = neigh_image
            unique = 0
        if unique == 0:
            if neigh_label != label0 or neigh_image != image0:
                label1 = neigh_label
                image1 = neigh_image
                unique = 1
        elif unique == 1:
            if ((neigh_label != label0 or neigh_image != image0) and
                (neigh_label != label1 or neigh_image != image1)):
                unique = 2
                break

    return unique

# #@njit(parallel=True,  cache=True)
def get_manifold_labels_doublegrid(
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
    nx2 = nx*2
    ny2 = ny*2
    nz2 = nz*2
    # create 3D array to store edges
    edges = np.full((nx*2,ny*2,nz*2), np.iinfo(np.uint8).max, dtype=np.uint8)
    
    # add maxima/minima
    for group in minima_groups:
        for i,j,k in group:
            # shift to double grid
            i*=2
            j*=2
            k*=2
            edges[i,j,k] = 0
            
    for group in maxima_groups:
        for i,j,k in group:
            # shift to double grid
            i*=2
            j*=2
            k*=2
            edges[i,j,k] = 3

    
    # loop over each voxel in parallel
    for i in prange(nx2):
        for j in range(ny2):
            for k in range(nz2):
                
                # if this voxel is part of a minimum or maximum, continue
                if edges[i,j,k] == 0 or edges[i,j,k] == 3:
                    continue
                
                pi = i&1
                pj = j&1
                pk = k&1
                parity = PARITY_TO_INT[pi,pj,pk]
                dim = PARITY_DIMS[parity]
                if dim != 1:
                    continue
                
                # check if this point has 0, 1, or 2 neighbors with different
                # labels
                num_neighs = get_differing_neighs_doublegrid(
                    i, j, k, 
                    nx, ny, nz, 
                    maxima_labels, 
                    maxima_images, 
                    vacuum_mask,
                    )
                opp_num_neighs = get_differing_neighs_doublegrid(
                    i, j, k, 
                    nx, ny, nz, 
                    minima_labels, 
                    minima_images, 
                    vacuum_mask,
                    )
                label = 10
                if num_neighs == 1:
                    label = 1
                elif num_neighs > 1:
                    label = 2
                if num_neighs == 1 and opp_num_neighs > 1:
                    # saddle 1
                    label = 1
                elif num_neighs > 1 and opp_num_neighs == 1:
                    # saddle 2
                    label = 2
                elif num_neighs < 1 and opp_num_neighs == 1:
                    # edge of minima manifold
                    label = 4
                elif num_neighs == 1 and opp_num_neighs <1:
                    # edge of maxima manifold
                    label = 5
                elif num_neighs == 1 and opp_num_neighs == 1:
                    # edge of both maxima/minima manifold
                    label = 6
                elif num_neighs < 1 and opp_num_neighs > 1:
                    # meeting of at least three minima manifolds
                    label = 7
                elif num_neighs > 1 and opp_num_neighs < 1:
                    # meeting of at least three maxima manifolds
                    label = 8
                edges[i,j,k] = label

    return edges


# #@njit(inline='always', cache=True)
def get_extrema_saddle_connections_doublegrid(
    i, j, k,
    nx, ny, nz,
    ny_nz,
    labels,
    images,
    data,
    vacuum_mask,
    max_val,
    use_minima = False,
):
    if use_minima:
        best_value0 = np.inf
        best_value1 = np.inf

    else:
        best_value0 = -np.inf
        best_value1 = -np.inf

        
    # initialize stored labels
    label0 = -1; image0 = -1
    label1 = -1; image1 = -1
    unique = -1
    
    # get parity
    pi = i&1
    pj = j&1
    pk = k&1
    parity = PARITY_TO_INT[pi,pj,pk]
    # get transforms
    neighbor_transforms = PARITY_VERTICES[parity]

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
            int((i+si)/2), int((j+sj)/2), int((k+sk)/2), nx, ny, nz
        )
        
        # skip points in the vacuum
        if vacuum_mask[ii, jj, kk]:
            continue
        
        # get the label, image, and value of this neighbor
        neigh_label = labels[ii, jj, kk]
        neigh_image = images[ii, jj, kk]
        neigh_value = data[ii,jj,kk]

        # update image to be relative to the current points transformation
        si1 = INT_TO_IMAGE[neigh_image, 0] + ssi
        sj1 = INT_TO_IMAGE[neigh_image, 1] + ssj
        sk1 = INT_TO_IMAGE[neigh_image, 2] + ssk
        neigh_image = IMAGE_TO_INT[si1, sj1, sk1]
        
        # compare to any previous labels and update our unique number
        if unique == -1:
            label0 = neigh_label
            image0 = neigh_image
            best_value0 = neigh_value
            unique = 0
        if unique == 0:
            if neigh_label != label0 or neigh_image != image0:
                label1 = neigh_label
                image1 = neigh_image
                best_value1 = neigh_value
                unique = 1
        elif unique == 1:
            if neigh_label == label0 and neigh_image == image0:
                if neigh_value > best_value0:
                    best_value0 = neigh_value
                
            elif neigh_label == label1 and neigh_image == image1:
                if neigh_value > best_value1:
                    best_value1 = neigh_value
        

    # if no neighbor was found, we just return a fake value
    if label0 == -1 or label1 == -1:
        return max_val, max_val, max_val, False, max_val, max_val, max_val, max_val

    best_value = min(best_value0, best_value1)
    
    # get image from label0 to label1
    i0, j0, k0 = INT_TO_IMAGE[image0]
    i1, j1, k1 = INT_TO_IMAGE[image1]
    crit_vec0 = IMAGE_TO_INT[i1-i0, j1-j0, k1-k0]
    crit_vec1 = IMAGE_TO_INT[i0-i1, j0-j1, k0-k1]

    return (
        min(label0, label1), 
        max(label0, label1), 
        min(crit_vec0, crit_vec1), 
        best_value,
        # also return the actual label and image of the atoms this critical point
        # connects
        label0,
        label1,
        image0,
        image1,
        )

# #@njit(parallel=True, cache=True)
def get_canonical_saddle_connections_doublegrid(
    labels: NDArray[np.int64],
    images: NDArray[np.int64],
    data: NDArray[np.float64],
    vacuum_mask: NDArray[np.uint8],
    edge_mask: NDArray[np.uint8],
    dir2car: NDArray[np.float64],
    use_minima: bool = False,
):
    nx, ny, nz = labels.shape
    ny_nz = ny*nz
    
    # get the points that may be saddles
    if use_minima:
        saddle_coords = np.argwhere(np.isin(edge_mask,(2, 5, 6)))
    else:
        # saddle_coords = np.argwhere(np.isin(edge_mask,(1, 4, 6)))
        saddle_coords = np.argwhere(np.isin(edge_mask,(1)))
    
    # create an array to track connections between these points.
    # For each entry we will have:
        # 1: the lower label index
        # 2: the higher label index
        # 3: the connection image between basins
        # 4: whether or not the connection image is lower -> higher (0) or higher -> lower (1)
    canonical_saddle_connections = np.empty((len(saddle_coords),3),dtype=np.uint32)
    extrema_connections = np.empty((len(saddle_coords),4),dtype=np.uint32)
    connection_vals = np.empty(len(saddle_coords), dtype=np.float64)

    # create a mask to track important connections
    important = np.ones(len(saddle_coords), dtype=np.bool)
    max_val = np.iinfo(np.uint32).max
    for idx in prange(len(saddle_coords)):
        i,j,k = saddle_coords[idx]
        
        lower, higher, shift, connection_value, label0, label1, image0, image1 = get_extrema_saddle_connections_doublegrid(
            i, j, k,
            nx, ny, nz,
            ny_nz,
            labels,
            images,
            data,
            vacuum_mask,
            max_val,
            use_minima,
        )
        if lower == max_val:
            # note this wasn't a true saddle
            important[idx] = False
            continue

        # add canoncial connection
        canonical_saddle_connections[idx, 0] = lower
        canonical_saddle_connections[idx, 1] = higher
        canonical_saddle_connections[idx, 2] = shift
        # add neighbor info
        extrema_connections[idx, 0] = label0
        extrema_connections[idx, 1] = image0
        extrema_connections[idx, 2] = label1
        extrema_connections[idx, 3] = image1

        connection_vals[idx] = connection_value

    # get only the connections that are important
    important = np.where(important)[0]
    saddle_coords = saddle_coords[important]
    canonical_saddle_connections = canonical_saddle_connections[important]
    connection_vals = connection_vals[important]
    extrema_connections = extrema_connections[important]
                
    return saddle_coords, canonical_saddle_connections, extrema_connections, connection_vals


# #@njit(cache=True)
def get_single_point_saddles_doublegrid(
    connection_values,
    saddle_coords,
    connection_indices,
    num_connections,
    use_minima = False,
):
    # create an array to store best points
    saddles = np.empty(num_connections, dtype=np.uint32)
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

# #@njit(cache=True)
def get_saddle_connections_doublegrid(
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
    max_val = np.iinfo(np.uint32).max
    flood_labels = np.full_like(edge_mask, max_val, dtype=np.uint32)
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
                            (label<neigh_label) == (best_image < inv_image) # whether the connection is reversed or not
                            ))
                        # add coord
                        connection_coords.append(queue[edge_idx])
        queue_start = queue_end
        queue_end = next_end
    
    
    return connections, connection_coords

###############################################################################
# Single Grid
###############################################################################






# #@njit(inline='always', cache=True)
# def get_extrema_saddle_connections(
#     i, j, k,
#     nx, ny, nz,
#     ny_nz,
#     labels,
#     images,
#     data,
#     neighbor_transforms,
#     edge_mask,
#     max_val,
#     use_minima = False,
# ):
#     if use_minima:
#         best_value = np.inf
#         edge_indices = (1,2,4,6)
#     else:
#         best_value = -np.inf
#         edge_indices = (1,2,5,6)

#     best_neigh_label = -1
#     best_neigh_image = -1
#     inv_image_idx = -1
#     # iterate over transforms
#     label = labels[i,j,k]
#     image = images[i,j,k]
#     value = data[i,j,k]

#     mi, mj, mk = INT_TO_IMAGE[image]

#     for trans in range(neighbor_transforms.shape[0]):
#         # get shifts
#         si = neighbor_transforms[trans, 0]
#         sj = neighbor_transforms[trans, 1]
#         sk = neighbor_transforms[trans, 2]

#         # wrap around periodic edges and store shift
#         ii, jj, kk, ssi, ssj, ssk = wrap_point_w_shift(
#             i+si, j+sj, k+sk, nx, ny, nz
#         )
#         # skip neighbors that are not also part of the edge
#         if not edge_mask[ii, jj, kk] in edge_indices:
#             continue

        
#         # get the label and image of this neighbor
#         neigh_label = labels[ii, jj, kk]
#         neigh_image = images[ii, jj, kk]

#         # update image to be relative to the current points transformation
#         si1 = INT_TO_IMAGE[neigh_image, 0] + ssi
#         sj1 = INT_TO_IMAGE[neigh_image, 1] + ssj
#         sk1 = INT_TO_IMAGE[neigh_image, 2] + ssk
#         neigh_image = IMAGE_TO_INT[si1, sj1, sk1]

#         # skip neighbors in the same basin
#         if label == neigh_label and image == neigh_image:
#             continue
        
#         # get the value of the neighbor
#         neigh_value = data[ii,jj,kk]
        
#         if use_minima:
#             best_val = max(value, neigh_value)
#             improved = best_val <= best_value
#         else:
#             best_val = min(value, neigh_value)
#             improved = best_val >= best_value
        
#         if improved:
#             best_value = best_val
#             best_neigh_label = neigh_label
#             # adjust image to point
#             si1 -= mi
#             sj1 -= mj
#             sk1 -= mk
#             best_neigh_image = IMAGE_TO_INT[si1, sj1, sk1]
#             inv_image_idx = IMAGE_TO_INT[-si1, -sj1, -sk1]
            
#             # we can't improve beyond this points value so we can break
#             if best_value == value:
#                 break

#     # if no neighbor was found, we just return a fake value
#     if best_neigh_label == -1:
#         return max_val, max_val, max_val, False, 0.0
    
#     is_reversed = (best_neigh_image > inv_image_idx) != (best_neigh_image > inv_image_idx)
    
#     return (
#         min(label, best_neigh_label), 
#         max(label, best_neigh_label), 
#         min(best_neigh_image, inv_image_idx), 
#         is_reversed,
#         best_value)

# # #@njit(cache=True)
# def get_ongrid_gradient_cart(i, j, k, data, dir2car):
#     nx, ny, nz = data.shape
    
#     c000 = data[i,j,k]
#     c100 = data[(i + 1) % nx,j,k]
#     c_100 = data[(i - 1) % nx,j,k]
#     c010 = data[i,(j + 1) % ny,k]
#     c0_10 = data[i,(j - 1) % ny,k]
#     c001 = data[i,j,(k + 1) % nz]
#     c00_1 = data[i,j,(k - 1) % nz]

#     # central differences in voxel coordinates
#     gi = (c100 - c_100) / (2.0)
#     gj = (c010 - c0_10) / (2.0)
#     gk = (c001 - c00_1) / (2.0)

#     # optional extrema clamping
#     if c100 <= c000 and c_100 <= c000:
#         gi = 0.0
#     if c010 <= c000 and c0_10 <= c000:
#         gj = 0.0
#     if c001 <= c000 and c00_1 <= c000:
#         gk = 0.0

#     # convert to fractional-coordinate gradient
#     gi *= nx
#     gj *= ny
#     gk *= nz

#     # convert to Cartesian gradient
#     gx = dir2car[0, 0] * gi + dir2car[0, 1] * gj + dir2car[0, 2] * gk
#     gy = dir2car[1, 0] * gi + dir2car[1, 1] * gj + dir2car[1, 2] * gk
#     gz = dir2car[2, 0] * gi + dir2car[2, 1] * gj + dir2car[2, 2] * gk

#     return gx, gy, gz

# # #@njit(parallel=True, cache=True)
# def get_canonical_saddle_connections(
#     labels: NDArray[np.int64],
#     images: NDArray[np.int64],
#     data: NDArray[np.float64],
#     neighbor_transforms: NDArray[np.int64],
#     edge_mask: NDArray[np.uint8],
#     use_minima: bool = False,
# ):
#     nx, ny, nz = labels.shape
#     ny_nz = ny*nz
    
#     # get the points that may be saddles
#     if use_minima:
#         saddle_coords = np.argwhere(np.isin(edge_mask,(2)))
#     else:
#         saddle_coords = np.argwhere(np.isin(edge_mask,(1)))
    
#     # create an array to track connections between these points.
#     # For each entry we will have:
#         # 1: the lower label index
#         # 2: the higher label index
#         # 3: the connection image between basins
#         # 4: whether or not the connection image is lower -> higher (0) or higher -> lower (1)
#     saddle_connections = np.empty((len(saddle_coords),4),dtype=np.uint32)
#     connection_vals = np.empty(len(saddle_coords), dtype=np.float64)

#     # create a mask to track important connections
#     important = np.ones(len(saddle_coords), dtype=np.bool)
#     max_val = np.iinfo(np.uint32).max
#     for idx in prange(len(saddle_coords)):
#         i,j,k = saddle_coords[idx]
        
#         lower, higher, shift, is_reversed, connection_value = get_extrema_saddle_connections(
#             i, j, k,
#             nx, ny, nz,
#             ny_nz,
#             labels,
#             images,
#             data,
#             neighbor_transforms,
#             edge_mask,
#             max_val,
#             use_minima,
#         )
#         if lower == max_val:
#             # note this wasn't a true saddle
#             important[idx] = False
#             continue
#         saddle_connections[idx, 0] = lower
#         saddle_connections[idx, 1] = higher
#         saddle_connections[idx, 2] = shift
#         saddle_connections[idx, 3] = is_reversed
#         connection_vals[idx] = connection_value
        
#     # get only the connections that are important
#     important = np.where(important)[0]
#     saddle_coords = saddle_coords[important]
#     saddle_connections = saddle_connections[important]
#     connection_vals = connection_vals[important]
                
#     return saddle_coords, saddle_connections, connection_vals


# # #@njit(cache=True)
# def get_single_point_saddles(
#     data,
#     connection_values,
#     saddle_coords,
#     connection_indices,
#     num_connections,
#     use_minima = False,
# ):
#     nx,ny,nz = data.shape
#     ny_nz = ny*nz
#     # create an array to store best points
#     saddles = np.empty(num_connections, dtype=np.uint32)
#     if use_minima:
#         best_vals = np.full(num_connections, np.inf, dtype=np.float64)
#     else:
#         best_vals = np.full(num_connections, -np.inf, dtype=np.float64)

#     for saddle_idx, (idx, connection_value) in enumerate(zip(connection_indices, connection_values)):
#         if not use_minima and  connection_value > best_vals[idx]:
#             best_vals[idx] = connection_value
#             saddles[idx] = saddle_idx
#         elif use_minima and connection_value < best_vals[idx]:
#             best_vals[idx] = connection_value
#             saddles[idx] = saddle_idx
#         else:
#             i,j,k = flat_to_coords(idx, ny_nz, nz)
#             value = data[i,j,k]
#             if value == connection_value:
#                 best_vals[idx] = connection_value
#                 saddles[idx] = saddle_idx
    
#     return saddles, best_vals

# # #@njit(cache=True)
# def get_saddle_connections(
#     saddle1_coords,
#     saddle2_coords,
#     neighbor_transforms: NDArray[np.int64],
#     edge_mask: NDArray[np.uint8],
# ):
#     nx, ny, nz = edge_mask.shape
    
#     # get the number of possible edges
#     num_edges = len(np.where(np.isin(edge_mask, (1,2,6)))[0])
#     num_edges += len(saddle1_coords) + len(saddle2_coords)
    
#     # create an empty queue for storing which points are next
#     queue = np.empty((num_edges, 3), dtype=np.uint32)
    
#     # create arrays to store flood filled labels
#     max_val = np.iinfo(np.uint32).max
#     flood_labels = np.full_like(edge_mask, max_val, dtype=np.uint32)
#     flood_images = np.full_like(edge_mask, 13, dtype=np.uint8)
    
#     # seed saddles
#     saddle_idx = 0
#     for i,j,k in saddle2_coords:
#         flood_labels[i,j,k] = saddle_idx
#         queue[saddle_idx] = (i,j,k)
#         saddle_idx += 1
#     for i,j,k in saddle1_coords:
#         flood_labels[i,j,k] = saddle_idx
#         queue[saddle_idx] = (i,j,k)
#         saddle_idx += 1
    
#     # create lists to store connections
#     connections = []
#     connection_coords = []
#     queue_start = 0
#     queue_end = saddle_idx
    
#     while queue_start != queue_end:
#         next_end = queue_end
#         for edge_idx in range(queue_start, queue_end):
#             i,j,k = queue[edge_idx]
#             # get label and image
#             label = flood_labels[i,j,k]
#             image = flood_images[i,j,k]
#             mi, mj, mk = INT_TO_IMAGE[image]
            
#             # iterate over each neighbor. if unlabeled, assign it the same label
#             # if labeled, note a new connection
#             for trans, (si, sj, sk) in  enumerate(neighbor_transforms):
#                 # get the neighbor
#                 ii, jj, kk, ssi, ssj, ssk = wrap_point_w_shift(
#                     i+si, j+sj, k+sk, nx, ny, nz
#                 )
#                 # skip points that can't be part of our connections
#                 if not edge_mask[ii,jj,kk] in (1,2,6):
#                     continue
                
#                 neigh_label = flood_labels[ii,jj,kk]
#                 shift = IMAGE_TO_INT[ssi+mi, ssj+mj, ssk+mk]
                
#                 # skip points that haven't been labeled
#                 if neigh_label == max_val:
#                     flood_labels[ii,jj,kk] = label
#                     flood_images[ii,jj,kk] = shift
#                     queue[next_end] = (ii,jj,kk)
#                     next_end += 1

#                 else:
#                     # get where this image has wrapped around periodic edges
#                     neigh_image = flood_images[ii,jj,kk]
    
#                     # if this belongs to a different saddle, we note a connection
#                     if neigh_label != label or neigh_image != shift:
#                         # get shift difference
#                         ni, nj, nk = INT_TO_IMAGE[neigh_image]
#                         si, sj, sk = INT_TO_IMAGE[shift]
                        
#                         bi = ni - si
#                         bj = nj - sj
#                         bk = nk - sk
#                         best_image = IMAGE_TO_INT[bi,bj,bk]
#                         inv_image = IMAGE_TO_INT[-bi, -bj, -bk]
                        
#                         connections.append((
#                             min(label, neigh_label),
#                             max(label, neigh_label),
#                             min(best_image, inv_image),
#                             (label<neigh_label) == (best_image < inv_image) # whether the connection is reversed or not
#                             ))
#                         # add coord
#                         connection_coords.append(queue[edge_idx])
#         queue_start = queue_end
#         queue_end = next_end
    
    
#     return connections, connection_coords

