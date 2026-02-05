# -*- coding: utf-8 -*-
import time
from baderkit.core import Bader
from baderkit.core.utilities.basic import wrap_point, wrap_point_w_shift, coords_to_flat, flat_to_coords
from baderkit.core.utilities.union_find import union, find_root
import numpy as np
from numpy.typing import NDArray
from numba import njit, prange
import itertools

@njit
def szudzik_pair(a, b):
    """
    Szudzik pairing for uint16 inputs.
    Returns uint32.
    """
    a = np.uint16(a)
    b = np.uint16(b)
    if a >= b:
        return a * a + a + b
    else:
        return b * b + a

@njit
def szudzik_unpair(z):
    """
    Inverse Szudzik pairing.
    z : uint32
    Returns (a, b) as uint16.
    """
    k = np.int32(np.sqrt(z))
    t = z - k * k

    if t < k:
        a = t
        b = k
    else:
        a = k
        b = t - k

    return a, b

@njit(parallel=True)
def szudzik_unpair_array(z):
    n = z.size
    a = np.empty(n, dtype=np.uint16)
    b = np.empty(n, dtype=np.uint16)

    for i in prange(n):
        zi = z[i]
        k = np.uint32(np.sqrt(zi))
        t = zi - k * k
        if t < k:
            a[i] = np.uint16(t)
            b[i] = np.uint16(k)
        else:
            a[i] = np.uint16(k)
            b[i] = np.uint16(t - k)

    return a, b

###############################################################################
# Parity mappings
###############################################################################
PARITIES = np.array(
    [
        [0, 0, 0],  # 3D
        [0, 0, 1],  # 2D
        [0, 1, 0],  # 2D
        [0, 1, 1],  # 1D
        [1, 0, 0],  # 2D
        [1, 0, 1],  # 1D
        [1, 1, 0],  # 1D
        [1, 1, 1],  # 0D
    ],
    dtype=np.uint8,
)

PARITY_DIMS = np.array([3,2,2,1,2,1,1,0], dtype=np.uint8)

# Conversion from parity to its index
PARITY_TO_INT = np.zeros((2, 2, 2), dtype=np.int8)
for idx, (i, j, k) in enumerate(PARITIES):
    PARITY_TO_INT[i, j, k] = idx

###############################################################################
# Voxel Transforms
###############################################################################
# Generate transformations to vertices for each parity type.
VERTEX_VOXELS = [
    [-1, -1, -1],
    [1, -1, -1],
    [1, 1, -1],
    [-1, 1, -1],
    [-1, -1, 1],
    [1, -1, 1],
    [1, 1, 1],
    [-1, 1, 1],
]
EDGE_VOXELS = [
    [-1, -1],
    [1, -1],
    [1, 1],
    [-1, 1],
]
FACE_VOXELS = [
    [-1],
    [1],
]
VOXEL_VOXELS = [[]]
VOXELS = [VERTEX_VOXELS, EDGE_VOXELS, FACE_VOXELS, VOXEL_VOXELS]

def parity_to_voxels(parity: NDArray[int]) -> NDArray[int]:
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
    base_transforms = VOXELS[vertex_type]
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
        transforms.append(parity_to_voxels(parity))
    return tuple(transforms)


PARITY_VOXELS = get_parity_vertices()


IMAGE_TO_INT = np.empty([3,3,3], dtype=np.int64)
INT_TO_IMAGE = np.array(list(itertools.product((-1,0,1), repeat=3)))
for shift_idx, (i,j,k) in enumerate(INT_TO_IMAGE):
    IMAGE_TO_INT[i,j,k] = shift_idx

# @njit(cache=True)
# def get_cell_type(
#     i,j,k,
#     nx,ny,nz,
#     labels,
#     images,
#     vacuum_mask,
#         ):
#     # get this cells parity
#     pi = i&1
#     pj = j&1
#     pk = k&1
#     parity = PARITY_TO_INT[pi,pj,pk]
    
#     # get transforms to voxels
#     transforms = PARITY_VOXELS[parity]

#     # iterate over the neighboring voxels
#     current_labels = [-1]
#     current_images = [-1]
#     unique = 0
#     for si, sj, sk in transforms:
#         if unique == 3:
#             break
#         # wrap points
#         ii, jj, kk, ssi, ssj, ssk = wrap_point_w_shift(int((i + si)/2), int((j + sj)/2), int((k + sk)/2), nx, ny, nz)
#         # skip vacuum points
#         if vacuum_mask[ii,jj,kk]:
#             continue

#         # get neighbors label and image
#         neigh_label = labels[ii, jj, kk]
#         neigh_image = images[ii, jj, kk]
        
#         # adjust neigh image
#         si1, sj1, sk1 = INT_TO_IMAGE[neigh_image]
#         si1 += ssi
#         sj1 += ssj
#         sk1 += ssk
#         neigh_image = IMAGE_TO_INT[si1,sj1,sk1]
        
#         # check if this is a new label/image
#         new = True
#         for label, image in zip(current_labels, current_images):
#             if neigh_label == label and neigh_image == image:
#                 new = False
#                 break
#         if new:
#             current_labels.append(neigh_label)
#             current_images.append(neigh_image)
#             unique +=1

#     return unique
@njit(cache=True, inline='always')
def get_cell_type(
    i, j, k,
    nx, ny, nz,
    labels,
    images,
    vacuum_mask,
):
    # get the parity of this point
    pi = i & 1
    pj = j & 1
    pk = k & 1
    parity = PARITY_TO_INT[pi, pj, pk]
    
    # get the corresponding transforms
    transforms = PARITY_VOXELS[parity]

    # initialize potential labels
    label0 = -1; image0 = -1
    label1 = -1; image1 = -1
    unique = 0

    # iterate over transforms
    for trans in range(transforms.shape[0]):
        # if we've found more than two neighbors, immediately break
        if unique == 3:
            break
        
        # get shifts
        si = transforms[trans, 0]
        sj = transforms[trans, 1]
        sk = transforms[trans, 2]

        # move to original grid (same as (i+si)/2)
        ii = (i + si) >> 1
        jj = (j + sj) >> 1
        kk = (k + sk) >> 1

        # wrap around periodic edges and store shift
        ii, jj, kk, ssi, ssj, ssk = wrap_point_w_shift(
            ii, jj, kk, nx, ny, nz
        )

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
            label0 = neigh_label
            image0 = neigh_image
            unique = 1
        elif unique == 1:
            if neigh_label != label0 or neigh_image != image0:
                label1 = neigh_label
                image1 = neigh_image
                unique = 2
        elif unique == 2:
            if ((neigh_label != label0 or neigh_image != image0) and
                (neigh_label != label1 or neigh_image != image1)):
                unique = 3

    return unique


@njit(parallel=True, cache=True)
def get_manifold_labels(
    maxima_labels: NDArray[np.int64],
    minima_labels: NDArray[np.int64],
    maxima_images: NDArray[np.int64],
    minima_images: NDArray[np.int64],
    maxima_groups: NDArray[np.int64],
    minima_groups: NDArray[np.int64],
    vacuum_mask: NDArray[np.bool_],
):
    """
    0: minima
    1: 1-saddle
    2: 2-saddle
    3: maxima
    4: 1-saddle ascending 2-manifold
    5: 1-saddle descending 1-manifold
    6: 2-saddle ascending 1-manifold
    7: 2-saddle descending 2-manifold
    8: saddle connecting 1-manifold
    9: 1 manifold in both min/max. is this meaningful?
    10: blank
    """
    
    nx, ny, nz = maxima_labels.shape
    nx2 = nx*2
    ny2 = ny*2
    nz2 = nz*2

    # create 3D array, twice the size of the original, to store critical points
    # and manifolds
    manifold_labels = np.full((nx2,ny2,nz2), np.iinfo(np.uint8).max, dtype=np.uint8)
    
    # initialize maxima and minima and get the double grid groups
    new_maxima_groups = []
    for group in maxima_groups:
        # adjust to double grid coords
        group2 = group*2
        new_group = []
        for i,j,k in group2:
            new_group.append(np.array((i,j,k), dtype=np.uint16))
            for si, sj, sk in INT_TO_IMAGE:
                ii, jj, kk = wrap_point(i+si, j+sj, k+sk, nx2, ny2, nz2)
                manifold_labels[ii, jj, kk] = 3
                new_group.append(np.array((ii,jj,kk), dtype=np.uint16))
        new_maxima_groups.append(new_group)
        
    new_minima_groups = []
    for group in minima_groups:
        # adjust to double grid coords
        group2 = group*2
        new_group = []
        for i,j,k in group2:
            new_group.append(np.array((i,j,k), dtype=np.uint16))
            for si, sj, sk in INT_TO_IMAGE:
                ii, jj, kk = wrap_point(i+si, j+sj, k+sk, nx2, ny2, nz2)
                manifold_labels[ii, jj, kk] = 0
                new_group.append(np.array((ii,jj,kk), dtype=np.uint16))
        new_minima_groups.append(new_group)

    # loop over each voxel in parallel
    for i in prange(nx2):
        for j in range(ny2):
            for k in range(nz2):
                
                # skip maxima/minima
                if manifold_labels[i,j,k] == 0 or manifold_labels[i,j,k] == 3:
                    continue
                
                # get the number of neighboring voxels (or vertices if you prefer)
                # in different basins
                unique_max = get_cell_type(
                    i, j, k, 
                    nx, ny, nz, 
                    maxima_labels, 
                    maxima_images, 
                    vacuum_mask)
                unique_min = get_cell_type(
                    i, j, k, 
                    nx, ny, nz, 
                    minima_labels, 
                    minima_images, 
                    vacuum_mask)

                if unique_min < 2 and unique_max < 2:
                    # this is not part of a 1 or 2 manifold
                    label = 10
                
                elif unique_min < 2 and unique_max == 2:
                    # this is a separating plane between two maxima 3-manifolds.
                    # It represents the descending 2-manifold from a 2-saddle
                    label = 7
                    
                elif unique_min == 2 and unique_max < 2:
                    # this is a separating plane between two minima 3-manifolds.
                    # It represents the ascending 2-manifold from a 1-saddle
                    label = 4
                
                elif unique_min < 2 and unique_max > 2:
                    # this is a bounding edge of the descending 2-manifold of
                    # a 2-saddle. it is a 1-manifold between a minimum and 1-saddle
                    label = 5
                
                elif unique_min > 2 and unique_max < 2:
                    # this is a bounding edge of the ascending 2-manifold of
                    # a 1-saddle. it is a 1-manifold between a maximum and 2-saddle
                    label = 6
                
                elif unique_min == 2 and unique_max == 2:
                    # this is a 1-manifold connecting two saddles
                    label = 8
                
                elif unique_min == 2 and unique_max > 2:
                    # this is a 1-saddle
                    label = 1
                
                elif unique_min > 2 and unique_max == 2:
                    # this is a 2-saddle
                    label = 2
                    
                elif unique_min > 2 and unique_max > 2:
                    # unsure what this means or if it would just result from
                    # voxelation
                    label = 9

                manifold_labels[i,j,k] = label
    return manifold_labels, new_maxima_groups, new_minima_groups

FACE_TRANSFORMS = np.array([
    [1,0,0],
    [-1,0,0],
    [0,1,0],
    [0,-1,0],
    [0,0,1],
    [0,0,-1],
    ], dtype=np.int64)

@njit(cache=True)
def group_saddles(
    manifold_labels,
        ):
    nx, ny, nz = manifold_labels.shape
    ny_nz = ny*nz
    
    saddle_indices_3d = np.argwhere(np.isin(manifold_labels, (1,2))).astype(np.uint16)
    num_cells = len(saddle_indices_3d)
    
    # create arrays to store ordered indices of saddles
    saddle_unions = np.arange(num_cells)
    saddle_indices_1d = np.empty(num_cells, dtype=np.uint32)
    saddle_types = np.empty(num_cells, dtype=np.uint8)
    for idx in prange(num_cells):
        i,j,k = saddle_indices_3d[idx]
        saddle_indices_1d[idx] = coords_to_flat(i,j,k,ny_nz,nz)
    
    # iterate over saddle cells and group them with any other adjacent cells
    # of the same type
    for union_idx in range(num_cells):
        i, j, k = saddle_indices_3d[union_idx]
        saddle_idx = saddle_indices_1d[union_idx]
        saddle_type = manifold_labels[i,j,k]
        saddle_types[union_idx] = saddle_type
        for si, sj, sk in FACE_TRANSFORMS:
            # get neighbor
            ni, nj, nk = wrap_point(i+si, j+sj, k+sk, nx, ny, nz)
            # skip points that aren't this type of saddle
            neigh_type = manifold_labels[ni,nj,nk]
            if neigh_type != saddle_type:
                continue
            # otherwise get this points flat index
            neigh_idx = coords_to_flat(ni, nj, nk, ny_nz, nz)
            # skip neighbors with higher indices
            if neigh_idx > saddle_idx:
                continue
            # find this points union index and create a union
            neigh_union_idx = np.searchsorted(saddle_indices_1d, neigh_idx)
            # pair them
            union(saddle_unions, union_idx, neigh_union_idx)

    # get the roots
    roots = saddle_unions.copy()
    for union_idx in range(len(saddle_unions)):
        root = find_root(saddle_unions, union_idx)
        roots[union_idx] = root
    unique_roots = np.unique(roots)
    # create groups
    saddle1_groups = []
    saddle2_groups = []
    for root_idx in unique_roots:
        children = [saddle_indices_3d[root_idx]]
        saddle_type = saddle_types[root_idx]
        for saddle_idx, saddle_root in enumerate(roots):
            if saddle_root == root_idx:
                children.append(saddle_indices_3d[saddle_idx])
        if saddle_type == 1:
            saddle1_groups.append(children)
        else:
            saddle2_groups.append(children)
    return saddle1_groups, saddle2_groups

bader = Bader.from_vasp("CHGCAR")

# get basins
maxima_labels = bader.maxima_basin_labels
maxima_images = bader.maxima_basin_images

minima_labels = bader.minima_basin_labels
minima_images = bader.minima_basin_images

image_to_int = np.empty([3,3,3], dtype=np.int64)
int_to_image = np.array(list(itertools.product((-1,0,1), repeat=3)))
for shift_idx, (i,j,k) in enumerate(int_to_image):
    image_to_int[i,j,k] = shift_idx
    
neighbor_transforms, _ = bader.reference_grid.point_neighbor_transforms

maxima_groups, minima_groups = bader.get_persistence_groups()

t0 = time.time()
manifold_labels, maxima_groups, minima_groups = get_manifold_labels(
    maxima_labels, 
    minima_labels, 
    maxima_images, 
    minima_images, 
    maxima_groups,
    minima_groups,
    vacuum_mask=bader.vacuum_mask
    )
t1 = time.time()
print(f"Manifold labeling: {t1-t0}")
saddle1_groups, saddle2_groups = group_saddles(manifold_labels)
t2=time.time()
print(f"Saddle grouping: {t2-t1}")

# test_grid = bader.reference_grid.copy()
# for i in range(11):
#     test_grid.total = manifold_labels==i
#     test_grid.write_vasp(f"ELFCAR_test_{i}")




