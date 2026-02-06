import time
from baderkit.core.utilities.basic import (
    wrap_point_w_shift, 
    wrap_point, 
    coords_to_flat, 
    flat_to_coords,
    )
from baderkit.core.utilities.union_find import union, find_root_no_compression, find_root
from baderkit.core.bader.methods.shared_numba import get_edges_w_images, get_best_neighbor
from baderkit.core import Bader
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
def get_unique_neighs(
    i, j, k,
    nx, ny, nz,
    labels,
    images,
    vacuum_mask,
):


    # initialize potential labels
    label0 = -1; image0 = -1
    label1 = -1; image1 = -1
    unique = 0

    # iterate over transforms
    for trans in range(INT_TO_IMAGE.shape[0]):
        # if we've found more than two neighbors, immediately break
        if unique == 3:
            break
        
        if trans == 13:
            continue
        
        # get shifts
        si = INT_TO_IMAGE[trans, 0]
        sj = INT_TO_IMAGE[trans, 1]
        sk = INT_TO_IMAGE[trans, 2]

        # wrap around periodic edges and store shift
        ii, jj, kk, ssi, ssj, ssk = wrap_point_w_shift(
            i+si, j+sj, k+sk, nx, ny, nz
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
                
                elif unique_min == 2 and unique_max == 2:
                    # this is the intersection of the 2-manifolds of 1-saddles
                    # and 2-saddles.
                    # This is a 1-manifold connecting two saddles
                    label = 8
                
                elif unique_min < 2 and unique_max > 2:
                    # this is a bounding edge of the descending 2-manifold of
                    # a 2-saddle. it is a 1-manifold between a minimum and 1-saddle
                    label = 5
                
                elif unique_min > 2 and unique_max < 2:
                    # this is a bounding edge of the ascending 2-manifold of
                    # a 1-saddle. it is a 1-manifold between a maximum and 2-saddle
                    label = 6
                
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
    return manifold_labels

@njit(cache=True)
def group_saddles(
    manifold_labels,
        ):
    # TODO: I need to figure out how crossing periodic boundaries works for this
    # when I need to get two points on either side of the dividing plane.
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
        children = []
        saddle_type = saddle_types[root_idx]
        for saddle_idx, saddle_root in enumerate(roots):
            if saddle_root == root_idx:
                children.append(saddle_indices_3d[saddle_idx])
        if saddle_type == 1:
            saddle1_groups.append(children)
        else:
            saddle2_groups.append(children)
    return saddle1_groups, saddle2_groups

# @njit
def seed_groups(
    manifold_labels,
    critical_groups,
    max_val,
        ):
    nx, ny, nz = manifold_labels.shape
    
    group_idx = 0
    all_new_groups = []
    for groups in critical_groups:
        new_groups = []
        for group in groups:
            new_group = []
            # iterate over the cells making up this group
            all_voxels = []
            max_neighs = 0
            for i,j,k in group:
                # get the parity
                pi = i & 1
                pj = j & 1
                pk = k & 1
                parity = PARITY_TO_INT[pi, pj, pk]
                # get transforms to neighboring voxels
                transforms = PARITY_VOXELS[parity]
                
                # iterate over each neighboring point and assign it as part of this
                # group
                for trans in range(transforms.shape[0]):
                    # get shifts
                    si = transforms[trans, 0]
                    sj = transforms[trans, 1]
                    sk = transforms[trans, 2]

                    # move to original grid (same as (i+si)/2)
                    ii = (i + si) >> 1
                    jj = (j + sj) >> 1
                    kk = (k + sk) >> 1
                    # get this point
                    ni, nj, nk = wrap_point(ii, jj, kk, nx, ny, nz)
                    # if it already has this groups idx, add one
                    label = manifold_labels[ni,nj,nk]
                    if label >= group_idx and label < max_val:
                        manifold_labels[ni,nj,nk] += 1
                        max_neighs = max(label+1-group_idx, max_neighs)
                        continue
                    # otherwise label it and add it to our list
                    manifold_labels[ni,nj,nk] = group_idx
                    all_voxels.append((ni,nj,nk))
            # iterate over all neighboring voxels and only include those that
            # have the maximum number of neighbors
            for i,j,k in all_voxels:
                neigh_num = manifold_labels[i,j,k] - group_idx
                if neigh_num != max_neighs:
                    manifold_labels[i,j,k] = max_val
                else:
                    manifold_labels[i,j,k] = group_idx
                    new_group.append(np.array((i,j,k), dtype=np.uint16))
            group_idx += 1
            
            new_groups.append(np.array(new_group, dtype=np.uint16))
        
        all_new_groups.append(new_groups)
            
    return manifold_labels, all_new_groups

# @njit
def get_saddle_1d_starts(
    saddle_group,
    extrema_labels,
    extrema_images,
    manifold_labels,
    vacuum_mask,
    edge_mask,
    data,
    descend=False,
        ):
    nx,ny,nz = extrema_labels.shape
    ny_nz = ny*nz
    
    # get the lowest point for each bordering basin
    label0 = -1; image0 = -1
    label1 = -1; image1 = -1
    index0 = -1
    index1 = -1
    
    value0 = -1
    value1 = -1
    labels_found = 0
    for i, j, k in saddle_group:
        # get image
        image_idx = extrema_images[i,j,k]
        im1,im2,im3 = INT_TO_IMAGE[image_idx]
        # move point such that its gradient would assign to the center cell
        i1 = i-im1
        j1 = j-im2
        k1 = k-im3
        # check each neighbor
        for trans in range(INT_TO_IMAGE.shape[0]):
            # get shifts
            si = INT_TO_IMAGE[trans, 0]
            sj = INT_TO_IMAGE[trans, 1]
            sk = INT_TO_IMAGE[trans, 2]

            # get this point
            ni, nj, nk, ssi, ssj, ssk = wrap_point_w_shift(i1+si, j1+sj, k1+sk, nx, ny, nz)
            # skip vacuum and non-edge points
            if vacuum_mask[ni,nj,nk] or not edge_mask[ni,nj,nk]:
                continue
            
            # get the label and image of this point
            neigh_label = extrema_labels[ni, nj, nk]
            neigh_image = extrema_images[ni, nj, nk]
            neigh_value = data[ni,nj,nk]
            neigh_index = coords_to_flat(ni, nj, nk, ny_nz, nz)

            # update image to be relative to the current points transformation
            si1 = INT_TO_IMAGE[neigh_image, 0] + ssi
            sj1 = INT_TO_IMAGE[neigh_image, 1] + ssj
            sk1 = INT_TO_IMAGE[neigh_image, 2] + ssk
            neigh_image = IMAGE_TO_INT[si1, sj1, sk1]
            
            override0 = False
            override1 = False
            if labels_found == 0:
                override0 = True
                labels_found = 1
                label0 = neigh_label
                image0 = neigh_image
                
            elif labels_found == 1:
                if neigh_label != label0 or neigh_image != image0:
                    override1 = True
                    labels_found = 2
                    label1 = neigh_label
                    image1 = neigh_image
                elif neigh_label == label0 and neigh_image == image0:
                    if descend and neigh_value < value0:
                        override0 = True
                    elif not descend and neigh_value > value0:
                        override0 = True
                else:
                    raise Exception
                        
            elif labels_found == 2:
                # get matching label and update if value improves
                if neigh_label == label0 and neigh_image == image0:
                    if descend and neigh_value < value0:
                        override0 = True
                    elif not descend and neigh_value > value0:
                        override0 = True
                elif neigh_label == label1 and neigh_image == image1:
                    if descend and neigh_value < value1:
                        override1 = True
                    elif not descend and neigh_value > value1:
                        override1 = True
                else:
                    breakpoint()
                    raise Exception
            if override0:
                value0 = neigh_value
                index0 = neigh_index
            elif override1:
                value1 = neigh_value
                index1 = neigh_index
    return index0, index1

@njit#(cache=True, inline="always")
def get_best_basin_neighbor(
    data: NDArray[np.float64],
    labels: NDArray[np.int64],
    images: NDArray[np.int64],
    i: np.int64,
    j: np.int64,
    k: np.int64,
    neighbor_transforms: NDArray[np.int64],
    neighbor_dists: NDArray[np.int64],
    use_min: bool = False,
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
    # get central label
    label = labels[i,j,k]
    image = images[i,j,k]
    # For each neighbor get the difference in value and if its better
    # than any previous, replace the current best
    for (si, sj, sk), dist in zip(neighbor_transforms, neighbor_dists):
        # loop
        ii, jj, kk, ssi, ssj, ssk = wrap_point_w_shift(i + si, j + sj, k + sk, nx, ny, nz)
        # get neighbors label
        neigh_label = labels[ii,jj,kk]
        # get the label and image of this neighbor
        neigh_label = labels[ii, jj, kk]
        neigh_image = images[ii, jj, kk]

        # update image to be relative to the current points transformation
        si1 = INT_TO_IMAGE[neigh_image, 0] + ssi
        sj1 = INT_TO_IMAGE[neigh_image, 1] + ssj
        sk1 = INT_TO_IMAGE[neigh_image, 2] + ssk
        neigh_image = IMAGE_TO_INT[si1, sj1, sk1]
        if label != neigh_label or image != neigh_image:
            continue
        # calculate the difference in value taking into account distance
        if use_min:
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

# @njit
def get_gradient_1d_path(
    point,
    data,
    extrema_labels,
    manifold_labels,
    neighbor_transforms,
    neighbor_dists,
    max_val,
    use_min = False,
        ):
    nx,ny,nz = data.shape
    ny_nz = ny*nz
    path = [point]
    prev_idx = point
    
    i,j,k = flat_to_coords(path, ny_nz, nz)
    found_extreme = False
    while not found_extreme:
        ni, nj, nk = get_best_basin_neighbor(
            data,
            extrema_labels,
            i,
            j,
            k,
            neighbor_transforms = neighbor_transforms,
            neighbor_dists = neighbor_dists,
            use_min=use_min,
            )
        if manifold_labels[ni,nj,nk] != max_val:
            found_extreme = True
        # add to path and  update
        flat_idx = coords_to_flat(ni,nj,nk, ny_nz, nz)
        if flat_idx == prev_idx:
            break
        prev_idx = flat_idx
        path.append(flat_idx)
        i = ni
        j = nj
        k = nk
    return path

# @njit
def get_saddle_1d_paths(
    saddle_group,
    extrema_labels,
    extrema_images,
    manifold_labels,
    vacuum_mask,
    edge_mask,
    data,
    neighbor_transforms,
    neighbor_dists,
    descend=False,
        ):
    # get the initial points on either side
    point0, point1 = get_saddle_1d_starts(
        saddle_group,
        extrema_labels,
        extrema_images,
        manifold_labels,
        vacuum_mask,
        edge_mask,
        data,
        descend,
        )
    breakpoint()
    path0 = get_gradient_1d_path(
        point0, 
        data, 
        extrema_labels, 
        manifold_labels, 
        neighbor_transforms, 
        neighbor_dists, 
        max_val,
        descend,
        )
    path1 = get_gradient_1d_path(
        point1, 
        data, 
        extrema_labels, 
        manifold_labels, 
        neighbor_transforms, 
        neighbor_dists, 
        max_val,
        descend,
        )
    return path0, path1
            
            
        

# @njit(parallel=True)
# def get_manifold_groups(
#     manifold_labels,
#     maxima_labels,
#     minima_labels,
#     minima_groups,
#     saddle1_groups,
#     saddle2_groups,
#     maxima_groups,
#         ):
#     nx,ny,nz = manifold_labels.shape
#     ny_nz = ny*nz
    
#     # We already have the stable/unstable manifolds for maxima and minima as these
#     # correspond to their basins and the points themselves. Now we need to find
#     # the stable/unstable manifolds for the saddle points. We will not label the
#     # entire 2-manifolds, only the lines connecting saddles
#     for crit_label, critical_groups, allowed_types in zip(
#             (1,2), 
#             (saddle1_groups, saddle2_groups),
#             ((5,8), (6))
#             ):
#     # for crit_label, critical_groups in enumerate((minima_groups, saddle1_groups, saddle2_groups)):
#         # first we create seeds at each point bordering our current group
#         paths, path_types, path_starts, path_ends = seed_manifolds(
#             manifold_labels,
#             critical_groups,
#             crit_label,
#             nx, ny, nz,
#             group_labels[crit_label],
#             allowed_types,
#             )
#         # add path starts/types
#         all_manifold_types.extend(path_types)
#         all_manifold_starts.extend(path_starts)
#         # Each path represents a manifold. We perform breadth-first searches for
#         # each
#         for path_idx in prange(len(paths)):
#             path_points = paths[path_idx]
#             path_type = path_types[path_idx]
            
#             # create a tracker for end points of this manifold
#             end_point = max_value
            
#             # create a queue
#             new_points = [i for i in path_points]
            
#             # start iterating
#             maxima_found = False
#             cycles = 0
#             while len(new_points) != 0 and not maxima_found:
#                 # create new list for next iteration
#                 next_points = []
#                 # check if any of our current points border a critical point
#                 for i,j,k in new_points:
#                     for trans_idx, (si, sj, sk) in enumerate(INT_TO_IMAGE):
#                         ni, nj, nk = wrap_point(i+si, j+sj, k+sk, nx, ny, nz)
#                         label = manifold_labels[ni,nj,nk]
#                         # if this is a critical point, add it to our list of end
#                         # points
#                         if label < 4 and cycles != 0:
#                             flat_idx = coords_to_flat(ni,nj,nk,ny_nz,nz)
#                             end_point = flat_idx
#                             maxima_found = True
#                             break
#                     if maxima_found:
#                         break
#                 # if no maxima was found, get the new points surrounding the
#                 # current ones
#                 for i,j,k in new_points:
#                     # iterate over neighbors
#                     for trans_idx, (si, sj, sk) in enumerate(FACE_TRANSFORMS):
#                         ni, nj, nk = wrap_point(i+si, j+sj, k+sk, nx, ny, nz)
#                         # get the label at this point
#                         label = manifold_labels[ni,nj,nk]
#                         # skip already visited points
#                         if label > 19:
#                             continue
#                         # skip points that are not of the same manifold type
#                         if label != path_type:
#                             continue
#                         # add the point to our queue and full path
#                         point = np.array((ni,nj,nk), dtype=np.uint16)
#                         path_points.append(point)
#                         next_points.append(point)
#                         manifold_labels[ni,nj,nk] += 20
#                 new_points = next_points
#                 cycles += 1
#             # get the unique end points and update our end points list
#             paths[path_idx] = path_points
#             path_ends[path_idx] = end_point
        
#         # right now our path ends are mislabeled with their voxel index rather
#         # than their critical index. We want to go through and relabel them
#         new_end_points = []
#         for end_vox_idx in path_ends:
#             if end_vox_idx == max_value:
#                 new_end_points.append(max_value)
#                 continue
#             for crit_vox_idx, crit_idx in zip(crit_indices, crit_index_labels):
#                 if crit_vox_idx == end_vox_idx:
#                     new_end_points.append(crit_idx)
#                     break
                                
        
#         # append paths and endpoints
#         all_manifolds.extend(paths)
#         all_manifold_ends.extend(new_end_points)
#     return all_manifolds, all_manifold_types, all_manifold_starts, all_manifold_ends
        

bader = Bader.from_vasp("CHGCAR", method="neargrid", persistence_tol=0.01)

# get basins
maxima_labels = bader.maxima_basin_labels
maxima_images = bader.maxima_basin_images

minima_labels = bader.minima_basin_labels
minima_images = bader.minima_basin_images


neighbor_transforms, neighbor_dists = bader.reference_grid.point_neighbor_transforms

maxima_edges = get_edges_w_images(
    maxima_labels,
    maxima_images,
    neighbor_transforms,
    bader.vacuum_mask,
    )
minima_edges = get_edges_w_images(
    minima_labels,
    minima_images,
    neighbor_transforms,
    bader.vacuum_mask,
    )

t0 = time.time()
saddle_labels = get_manifold_labels(
    maxima_labels, 
    minima_labels, 
    maxima_images, 
    minima_images, 
    vacuum_mask=bader.vacuum_mask
    )
t1 = time.time()
print(f"Manifold labeling: {t1-t0}")
saddle1_groups, saddle2_groups = group_saddles(saddle_labels)
t2=time.time()
print(f"Saddle grouping: {t2-t1}")

# create a manifold label
max_val = np.iinfo(np.uint16).max
manifold_labels = np.full_like(maxima_labels, max_val, dtype=np.uint16)
# get maxima/minima groups
maxima_groups, minima_groups = bader.get_persistence_groups()
# seed the groups
manifold_labels, new_groups = seed_groups(
    manifold_labels,
    (minima_groups, saddle1_groups, saddle2_groups, maxima_groups),
    max_val
    )

group = new_groups[1][0]


get_saddle_1d_paths(
    group,
    maxima_labels,
    maxima_images,
    manifold_labels,
    bader.vacuum_mask,
    maxima_edges,
    bader.reference_grid.total,
    neighbor_transforms,
    neighbor_dists,
    descend=False,
        )


# test_grid = bader.reference_grid.copy()
# test_grid.total = manifold_labels != max_val
# test_grid.write("ELFCAR_test_allsaddles")
# for i in range(11):
#     test_grid.total = manifold_labels==i
#     test_grid.write_vasp(f"ELFCAR_test_{i}")