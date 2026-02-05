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
                new_group.append(np.array(ii,jj,kk), dtype=np.uint16)
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
                new_group.append(np.array(ii,jj,kk), dtype=np.uint16)
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

# @njit(parallel=True)
# def label_manifolds(
#     manifold_labels,
#         ):
#     nx, ny, nz = manifold_labels.shape
#     ny_nz = ny*nz
    
#     saddle_indices_3d = np.argwhere(manifold_labels!=10).astype(np.uint16)
#     num_cells = len(saddle_indices_3d)
    
#     # create arrays to store ordered indices of manifolds
#     saddle_unions = np.arange(num_cells)
#     saddle_indices_1d = np.empty(num_cells, dtype=np.uint32)
#     for idx in prange(num_cells):
#         i,j,k = saddle_indices_3d[idx]
#         saddle_indices_1d[idx] = coords_to_flat(i,j,k,ny_nz,nz)
        
#     needs_refinement = np.zeros(num_cells, dtype=np.bool_)
#     multiple_lower = np.zeros(num_cells, dtype=np.bool_)
    
#     # iterate over manifold cells and group them with any other adjacent cells
#     # of the same type
#     for union_idx in prange(num_cells):
#         i, j, k = saddle_indices_3d[union_idx]
#         saddle_idx = saddle_indices_1d[union_idx]
#         saddle_type = manifold_labels[i,j,k]
#         lowest = saddle_idx
#         num_lower = 0
#         touches_different_critical = False
#         for si, sj, sk in FACE_TRANSFORMS:
#             # get neighbor
#             ni, nj, nk = wrap_point(i+si, j+sj, k+sk, nx, ny, nz)
#             # skip points that aren't this type of saddle. If this type is a
#             # critical point, we fully break and label connections in a later
#             # step to avoid grouping manifolds that contact the same point
#             neigh_type = manifold_labels[ni,nj,nk]
#             if neigh_type != saddle_type:
#                 if neigh_type < 4:
#                     touches_different_critical = True
#                     break
#                 continue
#             # otherwise get this points flat index
#             neigh_idx = coords_to_flat(ni, nj, nk, ny_nz, nz)
#             # check if this point is lower
#             if neigh_idx < saddle_idx:
#                 num_lower += 1
#             if neigh_idx < lowest:
#                 lowest = neigh_idx
#         # if this point contacts a different critical point, we don't group for now
#         # and note that we need to check it later
#         if touches_different_critical:
#             needs_refinement[union_idx] = True
#             continue
#         if num_lower > 1:
#             multiple_lower[union_idx] = True 
#         # get the union index of this neighbor
#         neigh_union_idx = np.searchsorted(saddle_indices_1d, lowest)
#         # pair them
#         saddle_unions[union_idx] = neigh_union_idx
        
#     # refine cells that are in contact with a critical point. They can only be
#     # unioned to cells that contact them that are not also touching the critical
#     # point
#     unrefined = np.where(needs_refinement)[0]
#     for union_idx in unrefined:
#         i, j, k = saddle_indices_3d[union_idx]
#         saddle_idx = saddle_indices_1d[union_idx]
#         saddle_type = manifold_labels[i,j,k]
#         lowest = saddle_idx
#         lowest_union = union_idx
#         for si, sj, sk in INT_TO_IMAGE:
#             # get neighbor
#             ni, nj, nk = wrap_point(i+si, j+sj, k+sk, nx, ny, nz)
#             # skip points that aren't this type of saddle
#             if manifold_labels[ni,nj,nk] != saddle_type:
#                 continue
#             # otherwise get this points flat index
#             neigh_idx = coords_to_flat(ni, nj, nk, ny_nz, nz)
#             # skip points that have higher indices than our current best
#             if neigh_idx > lowest:
#                 continue
#             # get the union index of this neighbor
#             neigh_union_idx = np.searchsorted(saddle_indices_1d[:union_idx], lowest)
#             # skip this point if it is also in contact with a critical point
#             if needs_refinement[neigh_union_idx]:
#                 continue
#             # update lowest
#             lowest = neigh_idx
#             lowest_union = neigh_union_idx
            
#         # make union
#         union(saddle_unions, union_idx, lowest_union)
    
#     # # get indices that may still need pairing
#     # multiple_pairs = np.where(multiple_below)[0]
#     # for union_idx in multiple_pairs:
#     #     i, j, k = saddle_indices_3d[union_idx]
#     #     saddle_idx = saddle_indices_1d[union_idx]
#     #     saddle_type = manifold_labels[i,j,k]
#     #     num_below = 0
#     #     lowest = saddle_idx
#     #     for si, sj, sk in INT_TO_IMAGE:
#     #         # get neighbor
#     #         ni, nj, nk = wrap_point(i+si, j+sj, k+sk, nx, ny, nz)
#     #         # skip points that aren't this type of saddle
#     #         if manifold_labels[ni,nj,nk] != saddle_type:
#     #             continue
#     #         # otherwise get this points flat index
#     #         neigh_idx = coords_to_flat(ni, nj, nk, ny_nz, nz)
#     #         # skip points that have higher indices
#     #         if neigh_idx > saddle_idx:
#     #             continue
#     #         # get the union index of this neighbor
#     #         neigh_union_idx = np.searchsorted(saddle_indices_1d[:union_idx], lowest)
#     #         # make union
#     #         union(saddle_unions, union_idx, neigh_union_idx)
    
#     # get the roots
#     roots = saddle_unions.copy()
#     for union_idx in prange(len(saddle_unions)):
#         root = find_root_no_compression(saddle_unions, union_idx)
#         roots[union_idx] = root

    # update labels to point to main index
    
    
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

        
@njit(cache=True)
def seed_manifolds(
    manifold_labels,
    groups,
    group_type,
    nx,ny,nz,
    critical_indices,
        ):
    paths = []
    path_types = []
    path_starts = []
    path_ends = []
    max_val = np.iinfo(np.uint16).max
    for group, group_idx in zip(groups, critical_indices):
        # iterate over the cells making up this group
        for i,j,k in group:
            # iterate over each neighboring point. Each one is a potential new
            # path
            for si, sj, sk in FACE_TRANSFORMS:
                # get this point
                ni, nj, nk = wrap_point(i+si, j+sj, k+sk, nx, ny, nz)
                # skip if this neighbor is part of the critical group or has been visited
                # already
                label = manifold_labels[ni,nj,nk]
                if label < 4 or label == 11 or label == 10:
                    continue
                
                # label this point as having been visited
                manifold_labels[ni,nj,nk] = 11
                
                # initialize path
                path = [np.array((ni,nj,nk), dtype=np.uint16)]
                new_points = []
                # add valid neighbors of this point
                for si1, sj1, sk1 in FACE_TRANSFORMS:
                    ni1, nj1, nk1 = wrap_point(ni+si1, nj+sj1, nk+sk1, nx, ny, nz)
                    label1 = manifold_labels[ni1,nj1,nk1]
                    # if this label isn't part of the same type of manifold or
                    # is part of a critical point we skip it
                    if label != label1 or label1 < 4:
                        continue
                    # add this point to our path and note that its been visited
                    manifold_labels[ni1,nj1,nk1] = 12
                    new_points.append(np.array((ni1,nj1,nk1), dtype=np.uint16))
                # flood fill any neighbors with the same type that are touching
                # the critical point group
                while len(new_points) != 0:
                    next_points = []
                    for point in new_points:
                        ni1, nj1, nk1 = point
                        # track if this point neighbors our critical index
                        neighbors_crit = False
                        # check each neighbor
                        for si1, sj1, sk1 in FACE_TRANSFORMS:
                            ni2, nj2, nk2 = wrap_point(ni1+si1, nj1+sj1, nk1+sk1, nx, ny, nz)
                            # skip points in a different manifold
                            label1 = manifold_labels[ni2,nj2,nk2]
                            if label1 == group_type:
                                # the current point borders our critical group
                                neighbors_crit = True
                                continue
                            if label != label1 or label1 < 4 or label1 == 12 or label1 == 11:
                                continue
                            # If we are still here, we add this point to our next
                            # set and note that its been visited
                            manifold_labels[ni2,nj2,nk2] = 12
                            next_points.append(np.array((ni2,nj2,nk2), dtype=np.uint16))
                        if neighbors_crit:
                            path.append(point)
                            manifold_labels[ni1,nj1,nk1] = 11
                    new_points = next_points
                # reset all neighbors next to our path seeds
                for ni, nj, nk in path:
                    for si1, sj1, sk1 in FACE_TRANSFORMS:
                        ni1, nj1, nk1 = wrap_point(ni+si1, nj+sj1, nk+sk1, nx, ny, nz)
                        if manifold_labels[ni1,nj1,nk1] == 12:
                            manifold_labels[ni1,nj1,nk1] = label
                # add this new path
                paths.append(path)
                path_types.append(label)
                path_starts.append(group_idx)
                path_ends.append(np.array([max_val], dtype=np.uint16))
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
    
    
    for crit_label, critical_groups in enumerate((minima_groups, saddle1_groups, saddle2_groups)):
        
        # first we create seeds at each point bordering our current group
        paths, path_types, path_starts, path_ends = seed_manifolds(
            manifold_labels,
            critical_groups,
            crit_label,
            nx, ny, nz,
            group_labels[crit_label],
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
            end_points = []
            
            # create a queue
            new_points = [path_points[0]]
            
            # start iterating
            while len(new_points) != 0:
                # create new list for next iteration
                next_points = []
                for i,j,k in new_points:
                    # iterate over neighbors
                    for si, sj, sk in FACE_TRANSFORMS:
                        ni, nj, nk = wrap_point(i+si, j+sj, k+sk, nx, ny, nz)
                        # get the label at this point
                        label = manifold_labels[ni,nj,nk]
                        # skip already visited points
                        if label == 11:
                            continue
                        # if this is a critical point, add it to our list of end
                        # points
                        if label < 4:
                            flat_idx = coords_to_flat(ni,nj,nk,ny_nz,nz)
                            end_points.append(flat_idx)
                            continue
                        # skip points that are not of the same manifold type
                        if label != path_type:
                            continue
                        # add the point to our queue and full path
                        point = np.array((ni,nj,nk), dtype=np.uint16)
                        path_points.append(point)
                        next_points.append(point)
                        # mark as visited
                        manifold_labels[ni,nj,nk] = 11
                new_points = next_points
            # get the unique end points and update our end points list
            end_points = np.array(end_points, dtype=np.uint32)
            end_points = np.unique(end_points)
            paths[path_idx] = path_points
            path_ends[path_idx] = end_points
        
        # right now our path ends are mislabeled with their voxel index rather
        # than their critical index. We want to go through and relabel them
        new_end_points = []
        for end_points in path_ends:
            new_points = np.empty(len(end_points), dtype=np.uint16)
            for end_idx, end_crit_idx in enumerate(end_points):
                for crit_idx, crit_label in zip(crit_indices, crit_index_labels):
                    if crit_idx == end_crit_idx:
                        new_points[end_idx] = crit_label
                        break
            new_points = np.unique(new_points)
            new_end_points.append(new_points)
                                
        
        # append paths and endpoints
        all_manifolds.extend(paths)
        all_manifold_ends.extend(new_end_points)
    return all_manifolds, all_manifold_types, all_manifold_starts, all_manifold_ends
        
        

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