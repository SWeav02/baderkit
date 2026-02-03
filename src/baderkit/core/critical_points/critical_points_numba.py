from baderkit.core.utilities.basic import wrap_point_w_shift, wrap_point, coords_to_flat
import itertools
import numpy as np
from numpy.typing import NDArray
from numba import njit, prange

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

@njit(cache=True)
def get_cell_type(
    i,j,k,
    nx,ny,nz,
    labels,
    images,
    vacuum_mask,
        ):
    # get this cells parity
    pi = i&1
    pj = j&1
    pk = k&1
    parity = PARITY_TO_INT[pi,pj,pk]
    
    # get transforms to voxels
    transforms = PARITY_VOXELS[parity]

    # iterate over the neighboring voxels
    current_labels = [-1]
    current_images = [-1]
    unique = 0
    for si, sj, sk in transforms:
        if unique == 3:
            break
        # wrap points
        ii, jj, kk, ssi, ssj, ssk = wrap_point_w_shift(int((i + si)/2), int((j + sj)/2), int((k + sk)/2), nx, ny, nz)
        # skip vacuum points
        if vacuum_mask[ii,jj,kk]:
            continue

        # get neighbors label and image
        neigh_label = labels[ii, jj, kk]
        neigh_image = images[ii, jj, kk]
        
        # adjust neigh image
        si1, sj1, sk1 = INT_TO_IMAGE[neigh_image]
        si1 += ssi
        sj1 += ssj
        sk1 += ssk
        neigh_image = IMAGE_TO_INT[si1,sj1,sk1]
        
        # check if this is a new label/image
        new = True
        for label, image in zip(current_labels, current_images):
            if neigh_label == label and neigh_image == image:
                new = False
                break
        if new:
            current_labels.append(neigh_label)
            current_images.append(neigh_image)
            unique +=1

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
    0: maxima
    1: minima
    2: 1-saddle
    3: 2-saddle
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
    manifold_labels = np.empty((nx2,ny2,nz2), dtype=np.uint8)
    
    # initialize maxima and minima
    for group in maxima_groups:
        # adjust to double grid coords
        group2 = group*2
        for i,j,k in group2:
            for si, sj, sk in INT_TO_IMAGE:
                ii, jj, kk = wrap_point(i+si, j+sj, k+sk, nx2, ny2, nz2)
                manifold_labels[ii, jj, kk] = 0
    for group in minima_groups:
        # adjust to double grid coords
        group2 = group*2
        for i,j,k in group2:
            for si, sj, sk in INT_TO_IMAGE:
                ii, jj, kk = wrap_point(i+si, j+sj, k+sk, nx2, ny2, nz2)
                manifold_labels[ii, jj, kk] = 1
    
    # loop over each voxel in parallel
    for i in prange(nx2):
        for j in range(ny2):
            for k in range(nz2):
                
                # skip maxima/minima
                if manifold_labels[i,j,k] == 0 or manifold_labels[i,j,k] == 1:
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
                    label = 2
                
                elif unique_min > 2 and unique_max == 2:
                    # this is a 2-saddle
                    label = 3
                    
                elif unique_min > 2 and unique_max > 2:
                    # unsure what this means or if it would just result from
                    # voxelation
                    label = 9

                manifold_labels[i,j,k] = label
    return manifold_labels

def combine_adjacent_saddles(
    manifold_labels,
        ):
    nx, ny, nz = manifold_labels.shape
    ny_nz = ny*nz
    
    saddles = np.argwhere(np.isin(manifold_labels, (2,3)))
    
    # get integer representations
    saddle_indices = np.empty(len(saddles), dtype=np.int64)
    for idx, (i,j,k) in enumerate(saddles):
        saddle_indices[idx] = coords_to_flat(i, j, k, ny_nz, nz)
    pass
    # get saddle point locations
    # create some method of tracking unions ideally without labeling full grid
    # get groups
    # get one rep per group. most central probably
    # relabel non-roots to be part of 1-manifolds

# Things to do:
    # x get grid with labels for each type of critical point and manifold
    # x refine positions of CPs and manifolds? e.g. saddle points as single points
    # - combine low-persistence saddles
    # - label arcs by seeding from minima, then 1-saddles, then 2-saddles
    # - create connection map between critical points
    # - determine what else is useful to make easily available. 
        # - integral lines?
        # - bader bonded atoms
        # - types of maxima/minima: point, ring, cage
            # - points should be consolidated by doing a single pass over and marking
            # any points in a basin with values above the persistence minimum as
            # maxima.
            # - once all points that are part of the shape are included, there
            # should be a way to get the shape. rings have saddle in center, cages
            # have minima in center?
        # - etc.

# @njit(cache=True)
# def get_cell_type(
#     i,j,k,
#     nx,ny,nz,
#     transforms,
#     labels,
#     images,
#     vacuum_mask,
#         ):
#     # get this voxels label and image
#     label = labels[i, j, k]
#     image = images[i, j, k]

#     # iterate over the neighboring voxels
#     current_label = -1
#     current_image = -1
#     unique = 0
#     for si, sj, sk in transforms:
#         if unique == 2:
#             break
#         # wrap points
#         ii, jj, kk, ssi, ssj, ssk = wrap_point_w_shift(i + si, j + sj, k + sk, nx, ny, nz)
#         # skip vacuum
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
        
#         # if the neighbor has a different label, this is an edge
#         if (
#             neigh_label != label and current_label != neigh_label
#             or neigh_image != image and current_image != neigh_image
#             ):
#             current_label = neigh_label
#             current_image = neigh_image
#             unique +=1

#     return unique

# @njit(parallel=True, cache=True)
# def get_manifold_labels(
#     maxima_labels: NDArray[np.int64],
#     minima_labels: NDArray[np.int64],
#     maxima_images: NDArray[np.int64],
#     minima_images: NDArray[np.int64],
#     neighbor_transforms: NDArray[np.int64],
#     vacuum_mask: NDArray[np.bool_],
# ):
#     """
#     0: maxima
#     1: minima
#     2: 1-saddle
#     3: 2-saddle
#     4: 1-saddle ascending 2-manifold
#     5: 1-saddle descending 1-manifold
#     6: 2-saddle ascending 1-manifold
#     7: 2-saddle descending 2-manifold
#     8: saddle connecting 1-manifold
#     9: blank
#     """
    
#     nx, ny, nz = maxima_labels.shape

#     # create 3D array to store labels
#     manifold_labels = np.empty(maxima_labels, dtype=np.uint8)
#     # loop over each voxel in parallel
#     for i in prange(nx):
#         for j in range(ny):
#             for k in range(nz):
#                 # if this voxel is part of the vacuum, continue
#                 if vacuum_mask[i, j, k]:
#                     manifold_labels[i,j,k] = 10
#                     continue
                
#                 # get the number of neighbors with different labels
#                 unique_max = get_cell_type(
#                     i, j, k, 
#                     nx, ny, nz, 
#                     neighbor_transforms, 
#                     maxima_labels, 
#                     maxima_images, 
#                     vacuum_mask)
#                 unique_min = get_cell_type(
#                     i, j, k, 
#                     nx, ny, nz, 
#                     neighbor_transforms, 
#                     minima_labels, 
#                     minima_images, 
#                     vacuum_mask)

#                 if unique_min == 0 and unique_max == 0:
#                     # this is not part of a 1 or 2 manifold
#                     label = 9
                
#                 elif unique_min == 0 and unique_max == 1:
#                     # this is a separating plane between two maxima 3-manifolds.
#                     # It represents the descending 2-manifold from a 2-saddle
#                     label = 7
                
#                 elif unique_min == 0 and unique_max > 1:
#                     # this is a bounding edge of the descending 2-manifold of
#                     # a 2-saddle. it is a 1-manifold between a minimum and 1-saddle
#                     label = 5
                
#                 elif unique_min == 1 and unique_max == 0:
#                     # this is a separating plane between two minima 3-manifolds.
#                     # It represents the ascending 2-manifold from a 1-saddle
#                     label = 4
                
#                 elif unique_min > 1 and unique_max == 0:
#                     # this is a bounding edge of the ascending 2-manifold of
#                     # a 1-saddle. it is a 1-manifold between a maximum and 2-saddle
#                     label = 6
                
#                 elif unique_min == 1 and unique_max == 1:
#                     # this is a 1-manifold connecting two saddles
#                     label = 8
                
#                 elif unique_min == 1 and unique_max > 1:
#                     # this is a 1-saddle
#                     label = 2
                
#                 elif unique_min > 1 and unique_max == 1:
#                     # this is a 2-saddle
#                     label = 3
                
#                 # is both > 2 possible?
#                 manifold_labels[i,j,k] = label
#     return manifold_labels