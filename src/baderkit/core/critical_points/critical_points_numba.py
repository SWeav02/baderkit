from baderkit.core.utilities.basic import (
    wrap_point_w_shift, 
    wrap_point, 
    coords_to_flat, 
    flat_to_coords,
    )
from baderkit.core.utilities.union_find import union, find_root_no_compression, find_root
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

FACE_TRANSFORMS = np.array([
    [1,0,0],
    [-1,0,0],
    [0,1,0],
    [0,-1,0],
    [0,0,1],
    [0,0,-1],
    ], dtype=np.int64)

@njit(cache=True, inline='always')
def get_cell_type(
    i, j, k,
    nx, ny, nz,
    parity,
    labels,
    images,
    vacuum_mask,
):
    
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
    4: 1-saddle unstable 2-manifold
    5: 1-saddle stable 1-manifold
    6: 2-saddle unstable 1-manifold
    7: 2-saddle stable 2-manifold
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
    
    # initialize maxima and minima
    # for group in maxima_groups:
    #     # adjust to double grid coords
    #     group2 = group*2
    #     for i,j,k in group2:
    #         for si, sj, sk in INT_TO_IMAGE:
    #             ii, jj, kk = wrap_point(i+si, j+sj, k+sk, nx2, ny2, nz2)
    #             manifold_labels[ii, jj, kk] = 3
        
    # for group in minima_groups:
    #     # adjust to double grid coords
    #     group2 = group*2
    #     for i,j,k in group2:
    #         for si, sj, sk in INT_TO_IMAGE:
    #             ii, jj, kk = wrap_point(i+si, j+sj, k+sk, nx2, ny2, nz2)
    #             manifold_labels[ii, jj, kk] = 0
    
    # loop over each voxel in parallel
    for i in prange(nx2):
        for j in range(ny2):
            for k in range(nz2):
                
                # skip maxima/minima
                if manifold_labels[i,j,k] == 0 or manifold_labels[i,j,k] == 3:
                    continue
                
                # get the parity of this point
                pi = i & 1
                pj = j & 1
                pk = k & 1
                parity = PARITY_TO_INT[pi, pj, pk]
                dim = PARITY_DIMS[parity]
                
                # skip potential critical points for now
                if dim == 0 or dim == 3:
                    continue
                
                # get the number of neighboring voxels that have different
                # values
                unique_max = get_cell_type(
                    i, j, k, 
                    nx, ny, nz, 
                    parity,
                    maxima_labels, 
                    maxima_images, 
                    vacuum_mask)
                unique_min = get_cell_type(
                    i, j, k, 
                    nx, ny, nz, 
                    parity,
                    minima_labels, 
                    minima_images, 
                    vacuum_mask)
                
                # default to a blank entry of 10
                label = 10
                
                # label faces. These can only be part of 2 manifolds
                if dim == 2:
                    if unique_min < 2 and unique_max == 2:
                        # this is a separating plane between two maxima 3-manifolds.
                        # It represents the descending 2-manifold from a 2-saddle
                        label = 7
                        
                    elif unique_min == 2 and unique_max < 2:
                        # this is a separating plane between two minima 3-manifolds.
                        # It represents the ascending 2-manifold from a 1-saddle
                        label = 4
                
                    elif unique_min == 2 and unique_max == 2:
                        # this is the intersection of the 2-manifolds of 1-saddles
                        # and 2-saddles.
                        # This is a 1-manifold connecting two saddles
                        label = 8
                
                # label edges. These can only be part of 1 manifolds
                elif dim == 1:
                
                    if unique_min < 2 and unique_max > 2:
                        # this is a bounding edge of the descending 2-manifold of
                        # a 2-saddle. it is a 1-manifold between a minimum and 1-saddle
                        label = 5
                    
                    elif unique_min > 2 and unique_max < 2:
                        # this is a bounding edge of the ascending 2-manifold of
                        # a 1-saddle. it is a 1-manifold between a maximum and 2-saddle
                        label = 6
                    
                    # elif unique_min == 2 and unique_max > 2:
                    #     # this is a 1-saddle
                    #     label = 1
                    
                    # elif unique_min > 2 and unique_max == 2:
                    #     # this is a 2-saddle
                    #     label = 2
                        
                    # elif unique_min > 2 and unique_max > 2:
                    #     # unsure what this means or if it would just result from
                    #     # voxelation
                    #     label = 9

                manifold_labels[i,j,k] = label
    return manifold_labels

# @njit(parallel=True, cache=True)
# def get_manifold_labels(
#     maxima_labels: NDArray[np.int64],
#     minima_labels: NDArray[np.int64],
#     maxima_images: NDArray[np.int64],
#     minima_images: NDArray[np.int64],
#     maxima_groups: NDArray[np.int64],
#     minima_groups: NDArray[np.int64],
#     vacuum_mask: NDArray[np.bool_],
# ):
#     """
#     0: minima
#     1: 1-saddle
#     2: 2-saddle
#     3: maxima
#     4: 1-saddle unstable 2-manifold
#     5: 1-saddle stable 1-manifold
#     6: 2-saddle unstable 1-manifold
#     7: 2-saddle stable 2-manifold
#     8: saddle connecting 1-manifold
#     9: 1 manifold in both min/max. is this meaningful?
#     10: blank
#     """
    
#     nx, ny, nz = maxima_labels.shape
#     nx2 = nx*2
#     ny2 = ny*2
#     nz2 = nz*2

#     # create 3D array, twice the size of the original, to store critical points
#     # and manifolds
#     manifold_labels = np.full((nx2,ny2,nz2), np.iinfo(np.uint8).max, dtype=np.uint8)
    
#     # initialize maxima and minima
#     for group in maxima_groups:
#         # adjust to double grid coords
#         group2 = group*2
#         for i,j,k in group2:
#             for si, sj, sk in INT_TO_IMAGE:
#                 ii, jj, kk = wrap_point(i+si, j+sj, k+sk, nx2, ny2, nz2)
#                 manifold_labels[ii, jj, kk] = 3
        
#     for group in minima_groups:
#         # adjust to double grid coords
#         group2 = group*2
#         for i,j,k in group2:
#             for si, sj, sk in INT_TO_IMAGE:
#                 ii, jj, kk = wrap_point(i+si, j+sj, k+sk, nx2, ny2, nz2)
#                 manifold_labels[ii, jj, kk] = 0
    
#     # loop over each voxel in parallel
#     for i in prange(nx2):
#         for j in range(ny2):
#             for k in range(nz2):
                
#                 # skip maxima/minima
#                 if manifold_labels[i,j,k] == 0 or manifold_labels[i,j,k] == 3:
#                     continue
                
#                 # get the number of neighboring voxels (or vertices if you prefer)
#                 # in different basins
#                 unique_max = get_cell_type(
#                     i, j, k, 
#                     nx, ny, nz, 
#                     maxima_labels, 
#                     maxima_images, 
#                     vacuum_mask)
#                 unique_min = get_cell_type(
#                     i, j, k, 
#                     nx, ny, nz, 
#                     minima_labels, 
#                     minima_images, 
#                     vacuum_mask)

#                 if unique_min < 2 and unique_max < 2:
#                     # this is not part of a 1 or 2 manifold
#                     label = 10
                
#                 elif unique_min < 2 and unique_max == 2:
#                     # this is a separating plane between two maxima 3-manifolds.
#                     # It represents the descending 2-manifold from a 2-saddle
#                     label = 7
                    
#                 elif unique_min == 2 and unique_max < 2:
#                     # this is a separating plane between two minima 3-manifolds.
#                     # It represents the ascending 2-manifold from a 1-saddle
#                     label = 4
                
#                 elif unique_min == 2 and unique_max == 2:
#                     # this is the intersection of the 2-manifolds of 1-saddles
#                     # and 2-saddles.
#                     # This is a 1-manifold connecting two saddles
#                     label = 8
                
#                 elif unique_min < 2 and unique_max > 2:
#                     # this is a bounding edge of the descending 2-manifold of
#                     # a 2-saddle. it is a 1-manifold between a minimum and 1-saddle
#                     label = 5
                
#                 elif unique_min > 2 and unique_max < 2:
#                     # this is a bounding edge of the ascending 2-manifold of
#                     # a 1-saddle. it is a 1-manifold between a maximum and 2-saddle
#                     label = 6
                
#                 elif unique_min == 2 and unique_max > 2:
#                     # this is a 1-saddle
#                     label = 1
                
#                 elif unique_min > 2 and unique_max == 2:
#                     # this is a 2-saddle
#                     label = 2
                    
#                 elif unique_min > 2 and unique_max > 2:
#                     # unsure what this means or if it would just result from
#                     # voxelation
#                     label = 9

#                 manifold_labels[i,j,k] = label
#     return manifold_labels
    
    
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
            saddle_unions = union(saddle_unions, union_idx, neigh_union_idx)
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

# @njit(cache=True)
def expand_critical_point(
    manifold_labels,
    critical_groups,
    critical_type,
    allowed_manifold_types,
    nx,ny,nz
        ):
    # expand our group to fill nearby 1D manifolds until they don't touch
    new_groups = []
    for group in critical_groups:
        new_group = [i for i in group]
        new_points = [i for i in group]
        while len(new_points) !=0:
            adjacent_points = []
            for i,j,k in new_points:
                # add bordering points to the critical point
                for si, sj, sk in FACE_TRANSFORMS:
                    # get this point
                    ni, nj, nk = wrap_point(i+si, j+sj, k+sk, nx, ny, nz)
                    label = manifold_labels[ni,nj,nk]
                    if label == critical_type or not label in allowed_manifold_types:
                        continue
                    adjacent_points.append(np.array((ni,nj,nk), dtype=np.uint16))
                    manifold_labels[ni,nj,nk] += 20
            
            next_points = []
            reset_points = []
            # go through our next possible points. if any of them touch each other
            # we add them to our adjacent points
            for point in adjacent_points:
                i,j,k = point
                # add bordering points to the critical point
                touches=False
                for si, sj, sk in FACE_TRANSFORMS:
                    # get this point
                    ni, nj, nk = wrap_point(i+si, j+sj, k+sk, nx, ny, nz)
                    label = manifold_labels[ni,nj,nk]
                    if label > 19:
                        touches=True
                        break
                if touches:
                    next_points.append(point)
                    new_group.append(point)
                else:
                    reset_points.append(point)
            # now we go back through and update labels
            for i, j, k in next_points:
                manifold_labels[i,j,k] = critical_type
            for i, j, k in reset_points:
                manifold_labels[i,j,k] % 20
            new_points = next_points
        new_groups.append(new_group)
    return new_groups

# @njit(cache=True)
def seed_manifolds(
    manifold_labels,
    critical_groups,
    critical_type,
    nx,ny,nz,
    critical_indices,
    allowed_manifold_types,
        ):
    paths = []
    path_types = []
    path_starts = []
    path_ends = []
    max_val = np.iinfo(np.uint16).max
    for group, group_idx in zip(critical_groups, critical_indices):
        # iterate over the cells making up this group
        for i,j,k in group:
            # iterate over each neighboring point. Each one is a potential new
            # path
            for si, sj, sk in FACE_TRANSFORMS:
                # get this point
                ni, nj, nk = wrap_point(i+si, j+sj, k+sk, nx, ny, nz)
                # skip cells that are not part of a 1-d arc
                label = manifold_labels[ni,nj,nk]
                if not label in allowed_manifold_types:
                    continue
                
                # label this point as having been visited. We do this by increasing
                # its value to something that is not typically allowed
                manifold_labels[ni,nj,nk] += 20
                
                # initialize path
                path = []
                
                # we want to find adjacent points in contact with our critical
                # group that should also be part of this path. We do this to
                # avoid double paths.
                new_points = [np.array((ni,nj,nk), dtype=np.uint16)]

                # flood fill any neighbors with the same type that are touching
                # the critical point group
                while len(new_points) != 0:
                    next_points = []
                    for point in new_points:
                        ni1, nj1, nk1 = point
                        # check if this point borders the critical group
                        neighbors_crit = False
                        # check each neighbor
                        for si1, sj1, sk1 in FACE_TRANSFORMS:
                            ni2, nj2, nk2 = wrap_point(ni1+si1, nj1+sj1, nk1+sk1, nx, ny, nz)
                            # get this neighbor points type
                            label1 = manifold_labels[ni2,nj2,nk2]
                            # check if its part of our critical point
                            if label1 == critical_type:
                                # this is part of the critical group. We note
                                # that the current path point is valid and continue
                                neighbors_crit = True
                                break
                        # if this point doesn't border the critical group, we
                        # don't add any of its neighbors and we unlabel it
                        if not neighbors_crit:
                            manifold_labels[ni1,nj1,nk1] -= 20
                            continue
                        path.append(point)
                        # otherwise we add it to the path and add any neighbors 
                        # that are adjacent to it to our next round
                        for si1, sj1, sk1 in FACE_TRANSFORMS:
                            ni2, nj2, nk2 = wrap_point(ni1+si1, nj1+sj1, nk1+sk1, nx, ny, nz)
                            # get this neighbor points type
                            label1 = manifold_labels[ni2,nj2,nk2]
                            # skip if the point is not part of the same manifold
                            # or if its already been checked
                            if label1 != label or label1 > 19:
                                continue
                            # If we are still here, we add this point to our next
                            # set and note that its been visited
                            manifold_labels[ni2,nj2,nk2] += 20
                            next_points.append(np.array((ni2,nj2,nk2), dtype=np.uint16))
                            
                    new_points = next_points

                # add this new path
                paths.append(path)
                path_types.append(label)
                path_starts.append(group_idx)
                path_ends.append(max_val)
        # reset all neighbors of this group
        # for i,j,k in group:
        #     for si, sj, sk in INT_TO_IMAGE:
        #         ni, nj, nk = wrap_point(i+si, j+sj, k+sk, nx, ny, nz)
        #         manifold_labels[ni,nj,nk] %= 20
    return paths, path_types, path_starts, path_ends
        
@njit(parallel=True)
def get_manifold_groups(
    manifold_labels,
    minima_groups,
    saddle1_groups,
    saddle2_groups,
    maxima_groups,
        ):
    nx,ny,nz = manifold_labels.shape
    ny_nz = ny*nz
    
    # first we expand our minima and maxima points to fill poorly behaved 1D
    # manifolds
    minima_groups = expand_critical_point(
        manifold_labels,
        minima_groups,
        0,
        (5,),
        nx,ny,nz
            )
    maxima_groups = expand_critical_point(
        manifold_labels,
        maxima_groups,
        3,
        (6,),
        nx,ny,nz
            )
    
    # create a list to store the voxels of all critical points and manifolds
    all_manifolds = []
    all_manifold_types = []
    all_manifold_starts = []
    all_manifold_ends = []
    
    # get initial labels for each critical point group
    group_labels = []
    group_idx = 0
    crit_indices = []
    crit_index_labels = []
    for crit_type, groups in enumerate((minima_groups, saddle1_groups, saddle2_groups, maxima_groups)):
        labels = []
        for group in groups:
            # add to manifold lists
            all_manifolds.append(group)
            all_manifold_types.append(np.uint8(crit_type))
            all_manifold_starts.append(group_idx)
            all_manifold_ends.append(np.array([group_idx], dtype=np.uint16))
            labels.append(group_idx)
            # add to index maps
            for i,j,k in group:
                idx = coords_to_flat(i, j, k, ny_nz, nz)
                crit_indices.append(idx)
                crit_index_labels.append(group_idx)
            group_idx += 1
        group_labels.append(labels)
    
    max_value = np.iinfo(np.uint16).max
    # We already have the stable/unstable manifolds for maxima and minima as these
    # correspond to their basins and the points themselves. Now we need to find
    # the stable/unstable manifolds for the saddle points. We will not label the
    # entire 2-manifolds, only the lines connecting saddles
    for crit_label, critical_groups, allowed_types in zip(
            (1,2), 
            (saddle1_groups, saddle2_groups),
            ((5,8), (6))
            ):
    # for crit_label, critical_groups in enumerate((minima_groups, saddle1_groups, saddle2_groups)):
        # first we create seeds at each point bordering our current group
        paths, path_types, path_starts, path_ends = seed_manifolds(
            manifold_labels,
            critical_groups,
            crit_label,
            nx, ny, nz,
            group_labels[crit_label],
            allowed_types,
            )
        # add path starts/types
        all_manifold_types.extend(path_types)
        all_manifold_starts.extend(path_starts)
        # Each path represents a manifold. We perform breadth-first searches for
        # each
        for path_idx in prange(len(paths)):
            path_points = paths[path_idx]
            path_type = path_types[path_idx]
            
            # create a tracker for end points of this manifold
            end_point = max_value
            
            # create a queue
            new_points = [i for i in path_points]
            
            # start iterating
            maxima_found = False
            cycles = 0
            while len(new_points) != 0 and not maxima_found:
                # create new list for next iteration
                next_points = []
                # check if any of our current points border a critical point
                for i,j,k in new_points:
                    for trans_idx, (si, sj, sk) in enumerate(INT_TO_IMAGE):
                        ni, nj, nk = wrap_point(i+si, j+sj, k+sk, nx, ny, nz)
                        label = manifold_labels[ni,nj,nk]
                        # if this is a critical point, add it to our list of end
                        # points
                        if label < 4 and cycles != 0:
                            flat_idx = coords_to_flat(ni,nj,nk,ny_nz,nz)
                            end_point = flat_idx
                            maxima_found = True
                            break
                    if maxima_found:
                        break
                # if no maxima was found, get the new points surrounding the
                # current ones
                for i,j,k in new_points:
                    # iterate over neighbors
                    for trans_idx, (si, sj, sk) in enumerate(FACE_TRANSFORMS):
                        ni, nj, nk = wrap_point(i+si, j+sj, k+sk, nx, ny, nz)
                        # get the label at this point
                        label = manifold_labels[ni,nj,nk]
                        # skip already visited points
                        if label > 19:
                            continue
                        # skip points that are not of the same manifold type
                        if label != path_type:
                            continue
                        # add the point to our queue and full path
                        point = np.array((ni,nj,nk), dtype=np.uint16)
                        path_points.append(point)
                        next_points.append(point)
                        manifold_labels[ni,nj,nk] += 20
                new_points = next_points
                cycles += 1
            # get the unique end points and update our end points list
            paths[path_idx] = path_points
            path_ends[path_idx] = end_point
        
        # right now our path ends are mislabeled with their voxel index rather
        # than their critical index. We want to go through and relabel them
        new_end_points = []
        for end_vox_idx in path_ends:
            if end_vox_idx == max_value:
                new_end_points.append(max_value)
                continue
            for crit_vox_idx, crit_idx in zip(crit_indices, crit_index_labels):
                if crit_vox_idx == end_vox_idx:
                    new_end_points.append(crit_idx)
                    break
                                
        
        # append paths and endpoints
        all_manifolds.extend(paths)
        all_manifold_ends.extend(new_end_points)
    return all_manifolds, all_manifold_types, all_manifold_starts, all_manifold_ends
        