# -*- coding: utf-8 -*-

from baderkit.core import Bader
from baderkit.core.utilities.basic import wrap_point, wrap_point_w_shift
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

@njit(parallel=True)
def find_saddles(
    edge_mask,
    data: NDArray[np.float64],
    neighbor_transforms: NDArray[np.int64],
    use_minima: bool = False,
):

    nx, ny, nz = data.shape
    # create 3D array to store maxima
    maxima = np.zeros_like(data, dtype=np.bool_)
    # loop over each voxel in parallel
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # if this voxel is part of the vacuum, continue
                if not edge_mask[i, j, k]:
                    continue
                # get this voxels value
                value = data[i, j, k]
                is_max = True
                # iterate over the neighboring voxels
                for si, sj, sk in neighbor_transforms:
                    # wrap points
                    ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
                    if not edge_mask[ii, jj, kk]:
                        continue

                    if not use_minima:
                        if data[ii, jj, kk] > value:
                            is_max = False
                            break
                    else:
                        if data[ii, jj, kk] < value:
                            is_max = False
                            break
                if is_max:
                    maxima[i, j, k] = True
    return maxima

@njit(parallel=True)
def label_edges(
    labels,
    neighbor_transforms,
    edge_mask,
        ):
    nx, ny, nz = labels.shape
    # create 3D array to store combined labels
    edge_labels = np.empty_like(labels, dtype=np.uint32)
    max_val = np.iinfo(np.uint16).max
    # loop over each voxel in parallel
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # if this voxel is part of the vacuum, continue
                if not edge_mask[i, j, k]:
                    continue
                # get this voxels label
                label = labels[i,j,k]

                # iterate over the neighboring voxels
                num_neighs = 0
                first_neigh = -1
                for si, sj, sk in neighbor_transforms:
                    # wrap points
                    ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
                    
                    # skip points that aren't edges
                    if not edge_mask[ii, jj, kk]:
                        continue
                    
                    # get neighbor label
                    neigh_label = labels[ii,jj,kk]
                    
                    # skip same label neighs
                    if label == neigh_label:
                        continue
                    
                    # set this as neighboring label
                    if neigh_label != first_neigh:
                        first_neigh = neigh_label
                        num_neighs += 1
                        if num_neighs > 1:
                            break
                if num_neighs == 1:
                    lower = min(label, first_neigh)
                    upper = max(label, first_neigh)
                    new_label = szudzik_pair(lower, upper)
                    edge_labels[i,j,k] = new_label
                else:
                    edge_labels[i,j,k] = max_val
    return edge_labels
                    



@njit(parallel=True)
def get_edges_thin(
        data,
    labeled_array: NDArray[np.int64],
    neighbor_transforms: NDArray[np.int64],
    vacuum_mask: NDArray[np.bool_],
    use_minima: bool = False
):
    """
    In a 3D array of labeled voxels, finds the voxels that neighbor at
    least one voxel with a different label.

    Parameters
    ----------
    labeled_array : NDArray[np.int64]
        A 3D array where each entry represents the basin label of the point.
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
    nx, ny, nz = labeled_array.shape
    # create 3D array to store edges
    edges = np.zeros_like(labeled_array, dtype=np.bool_)
    # loop over each voxel in parallel
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # if this voxel is part of the vacuum, continue
                if vacuum_mask[i, j, k]:
                    continue
                # get this voxels label
                label = labeled_array[i, j, k]
                value = data[i,j,k]
                # iterate over the neighboring voxels
                for si, sj, sk in neighbor_transforms:
                    # wrap points
                    ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
                    # skip vacuum
                    if vacuum_mask[ii,jj,kk]:
                        continue
                    # get neighbors label
                    neigh_label = labeled_array[ii, jj, kk]
                    # if the neighbor shares a label, we continue
                    if neigh_label == label:
                        continue
                    # if the neighbor has a lower/higher value when in maxima/minima
                    # mode, we don't count this voxel as an edge
                    if not use_minima:
                        if data[ii,jj,kk] < value:
                            break    
                    else:
                        if data[ii,jj,kk] > value:
                            break  
                    # otherwise, this is an edge
                    edges[i,j,k] = True
                    break

    return edges

@njit(parallel=True)
def get_edges_periodic(
    labeled_array: NDArray[np.int64],
    images: NDArray[np.int64],
    neighbor_transforms: NDArray[np.int64],
    vacuum_mask: NDArray[np.bool_],
):
    """
    In a 3D array of labeled voxels, finds the voxels that neighbor at
    least one voxel with a different label.

    Parameters
    ----------
    labeled_array : NDArray[np.int64]
        A 3D array where each entry represents the basin label of the point.
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
    nx, ny, nz = labeled_array.shape

    # create 3D array to store edges
    edges = np.zeros_like(labeled_array, dtype=np.bool_)
    # loop over each voxel in parallel
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # if this voxel is part of the vacuum, continue
                if vacuum_mask[i, j, k]:
                    continue
                # get this voxels label and image
                label = labeled_array[i, j, k]
                image = images[i, j, k]

                # iterate over the neighboring voxels
                for si, sj, sk in neighbor_transforms:
                    # wrap points
                    ii, jj, kk, ssi, ssj, ssk = wrap_point_w_shift(i + si, j + sj, k + sk, nx, ny, nz)
                    # skip vacuum
                    if vacuum_mask[ii,jj,kk]:
                        continue
                    # get neighbors label and image
                    neigh_label = labeled_array[ii, jj, kk]
                    neigh_image = images[ii, jj, kk]
                    # if the neighbor has a different label, this is an edge
                    if neigh_label != label:
                        edges[i,j,k] = True
                        break
                    # if the neighbor has a different image it may be an edge.
                    # We don't count this if the neighbor wrapped around
                    if image == neigh_image:
                        continue
                    if ssi != 0 or ssj != 0 or ssk != 0:
                        continue
                    edges[i,j,k] = True
                    break

    return edges

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
    [-1, -1, 1],
    [1, -1, 1],
    [1, 1, 1],
    [-1, 1, 1],
]
POLY_VERTICES = [
    [-1, -1],
    [1, -1],
    [1, 1],
    [-1, 1],
]
EDGE_VERTICES = [
    [-1],
    [1],
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

@njit(parallel=True)
def get_edges_double_grid(
    labeled_array: NDArray[np.int64],
    images: NDArray[np.int64],
    neighbor_transforms: NDArray[np.int64],
    vacuum_mask: NDArray[np.bool_],
):

    nx, ny, nz = labeled_array.shape
    nx2 = nx*2
    ny2 = ny*2
    nz2 = nz*2


    # create 3D array to store edges. This will be twice the size of the original
    # grids
    edges = np.zeros((nx2, ny2, nz2), dtype=np.bool_)
    # loop over each voxel in parallel
    for i in prange(nx2):
        for j in range(ny2):
            for k in range(nz2):
                # get parity
                pi = i&1
                pj = j&1
                pk = k&1
                parity = PARITY_TO_INT[pi,pj,pk]
                dim = PARITY_DIMS[parity]
                # skip vertices and voxels
                if dim != 3:
                    continue

                # get transforms
                transforms = PARITY_VERTICES[parity]
                current_label = -1
                current_image = -1
                for si, sj, sk in transforms:
                    # get the vertex labels and images
                    ii, jj, kk, si, sj, sk = wrap_point_w_shift(int((i + si)/2), int((j + sj)/2), int((k + sk)/2), nx, ny, nz)
                                   
                    # skip if this vertex is part of the vacuum
                    if vacuum_mask[ii,jj,kk]:
                        continue
                    
                    # get the label and image
                    label = labeled_array[ii,jj,kk]
                    image = images[ii,jj,kk]
                    
                    # if we haven't set a label yet, do so
                    if current_label == -1:
                        current_label = label
                        current_image = image
                        continue
    
                    # if the neighbor has a different label, this is an edge
                    if label != current_label:
                        edges[i,j,k] = True
                        break
                    # if the neighbor has a different image it may be an edge.
                    # We don't count this if the neighbor wrapped around
                    if image == current_image:
                        continue
                    
                    nonzero = False
                    for idx in (si, sj, sk):
                        if idx !=0:
                            nonzero=True
                            break
                    if nonzero:
                        continue
    
                    edges[i,j,k] = True
                    break

    return edges

@njit(parallel=True)
def get_edges_voxels(
    labeled_array: NDArray[np.int64],
    images: NDArray[np.int64],
    image_to_int: NDArray[np.int64],
    int_to_image: NDArray[np.int64],
    neighbor_transforms: NDArray[np.int64],
    vacuum_mask: NDArray[np.bool_],
):

    nx, ny, nz = labeled_array.shape
    nx2 = nx*2
    ny2 = ny*2
    nz2 = nz*2

    # create 3D array to store edges. This will be twice the size of the original
    # grids
    edges = np.zeros((nx2, ny2, nz2), dtype=np.bool_)
    transforms = PARITY_VERTICES[-1]
    max_val = np.iinfo(np.uint32).max
    # loop over each voxel in parallel
    for i in prange(nx2):
        for j in range(ny2):
            for k in range(nz2):
                # get parity
                pi = i&1
                pj = j&1
                pk = k&1
                parity = PARITY_TO_INT[pi,pj,pk]
                dim = PARITY_DIMS[parity]
                # skip vertices and voxels
                if dim != 3:
                    continue

                unique_labels = np.full(8, max_val, dtype=np.uint32)

                for trans_idx, (si, sj, sk) in enumerate(transforms):
                    # get the vertex labels and images
                    ii, jj, kk, si, sj, sk = wrap_point_w_shift(int((i + si)/2), int((j + sj)/2), int((k + sk)/2), nx, ny, nz)
                                   
                    # skip if this vertex is part of the vacuum
                    if vacuum_mask[ii,jj,kk]:
                        continue
                    
                    # get the label and image
                    label = labeled_array[ii,jj,kk]
                    image = images[ii,jj,kk]
                    
                    # adjust the image if we've wrapped around a cell
                    si1, sj1, sk1 = int_to_image[image]
                    si1 += si
                    sj1 += sj
                    sk1 += sk
                    image = image_to_int[si1,sj1,sk1]
                    
                    # set label
                    unique_labels[trans_idx] = szudzik_pair(label, image)
                    
                # get the number of unique labels
                unique_labels = np.unique(unique_labels)
                num_unique = len(unique_labels)
                # if we had any vacuum points, we subtract 1
                if unique_labels[-1] == max_val:
                    num_unique -= 1
                
                # if we have 2 unique labels, this is part of a 2d manifold
                if num_unique == 2:
                    edges[i,j,k] = 1
                # if we have more than 2 unique labels this is part of a 1d manifold
                elif num_unique > 2:
                    edges[i,j,k] = 2
                # otherwise, we don't consider this an edge of any type and we
                # continue

    return edges


@njit(parallel=True)
def get_edges_periodic_all(
    labeled_array: NDArray[np.int64],
    images: NDArray[np.int64],
    image_to_int: NDArray[np.int64],
    int_to_image: NDArray[np.int64],
    neighbor_transforms: NDArray[np.int64],
    vacuum_mask: NDArray[np.bool_],
):
    """
    In a 3D array of labeled voxels, finds the voxels that neighbor at
    least one voxel with a different label.

    Parameters
    ----------
    labeled_array : NDArray[np.int64]
        A 3D array where each entry represents the basin label of the point.
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
    nx, ny, nz = labeled_array.shape

    # create 3D array to store edges
    edges = np.zeros_like(labeled_array, dtype=np.uint8)
    # loop over each voxel in parallel
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # if this voxel is part of the vacuum, continue
                if vacuum_mask[i, j, k]:
                    continue

                # get this voxels label and image
                label = labeled_array[i, j, k]
                image = images[i, j, k]

                # iterate over the neighboring voxels
                current_label = -1
                current_image = -1
                unique = 0
                for si, sj, sk in neighbor_transforms:
                    if unique == 2:
                        break
                    # wrap points
                    ii, jj, kk, ssi, ssj, ssk = wrap_point_w_shift(i + si, j + sj, k + sk, nx, ny, nz)
                    # skip vacuum
                    if vacuum_mask[ii,jj,kk]:
                        continue
                    # get neighbors label and image
                    neigh_label = labeled_array[ii, jj, kk]
                    neigh_image = images[ii, jj, kk]
                    
                    # adjust neigh image
                    si1, sj1, sk1 = int_to_image[neigh_image]
                    si1 += ssi
                    sj1 += ssj
                    sk1 += ssk
                    neigh_image = image_to_int[si1,sj1,sk1]
                    
                    # if the neighbor has a different label, this is an edge
                    if neigh_label != label:
                        if current_label != neigh_label:
                            current_label = neigh_label
                            current_image = neigh_image
                            unique +=1
                            continue

                    # if the neighbor has a different image it may be an edge.
                    # We don't count this if the neighbor wrapped around
                    if image == neigh_image or neigh_image == current_image:
                        continue
                    if ssi != 0 or ssj != 0 or ssk != 0:
                        continue

                    current_label = neigh_label
                    current_image = neigh_image
                    unique += 1
                edges[i,j,k] = unique
    return edges

@njit(parallel=True)
def get_saddles(
    minima_manifolds,
    maxima_manifolds,
):

    nx, ny, nz = minima_manifolds.shape

    # create 3D array to store saddles
    edges = np.zeros_like(minima_manifolds, dtype=np.uint8)
    # loop over each voxel in parallel
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                min_val = minima_manifolds[i,j,k]
                max_val = maxima_manifolds[i,j,k]
                if min_val == 0 or max_val == 0:
                    continue
                
                # saddle 1-manifolds?
                if min_val == 1 and max_val == 1:
                    edges[i,j,k] = 1
                
                # type1 saddle
                elif min_val == 1 and max_val == 2:
                    edges[i,j,k] = 2
                
                # type2 saddle
                elif min_val == 2 and max_val == 1:
                    edges[i,j,k] = 3

    return edges


@njit
def get_edge_type(
    i,j,k,
    nx,ny,nz,
    transforms,
    labels,
    images,
    int_to_image,
    image_to_int,
    vacuum_mask,
        ):
    # get this voxels label and image
    label = labels[i, j, k]
    image = images[i, j, k]

    # iterate over the neighboring voxels
    current_label = -1
    current_image = -1
    unique = 0
    for si, sj, sk in transforms:
        if unique == 2:
            break
        # wrap points
        ii, jj, kk, ssi, ssj, ssk = wrap_point_w_shift(i + si, j + sj, k + sk, nx, ny, nz)
        # skip vacuum
        if vacuum_mask[ii,jj,kk]:
            continue
        # get neighbors label and image
        neigh_label = labels[ii, jj, kk]
        neigh_image = images[ii, jj, kk]
        
        # adjust neigh image
        si1, sj1, sk1 = int_to_image[neigh_image]
        si1 += ssi
        sj1 += ssj
        sk1 += ssk
        neigh_image = image_to_int[si1,sj1,sk1]
        
        # if the neighbor has a different label, this is an edge
        if neigh_label != label:
            if current_label != neigh_label:
                current_label = neigh_label
                current_image = neigh_image
                unique +=1
                continue

        # if the neighbor has a different image it may be an edge.
        # We don't count this if the neighbor wrapped around
        if image == neigh_image or neigh_image == current_image:
            continue
        if ssi != 0 or ssj != 0 or ssk != 0:
            continue

        current_label = neigh_label
        current_image = neigh_image
        unique += 1
    return unique

@njit(parallel=True)
def get_manifold_labels(
    maxima_labels: NDArray[np.int64],
    minima_labels: NDArray[np.int64],
    maxima_images: NDArray[np.int64],
    minima_images: NDArray[np.int64],
    image_to_int: NDArray[np.int64],
    int_to_image: NDArray[np.int64],
    neighbor_transforms: NDArray[np.int64],
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
    9: blank
    """
    
    nx, ny, nz = maxima_labels.shape

    # create 3D array to store labels
    manifold_labels = np.empty_like(maxima_labels, dtype=np.uint8)
    # loop over each voxel in parallel
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # if this voxel is part of the vacuum, continue
                if vacuum_mask[i, j, k]:
                    manifold_labels[i,j,k] = 10
                    continue
                
                # get the number of neighbors with different labels
                unique_max = get_edge_type(
                    i, j, k, 
                    nx, ny, nz, 
                    neighbor_transforms, 
                    maxima_labels, 
                    maxima_images, 
                    int_to_image, 
                    image_to_int, 
                    vacuum_mask)
                unique_min = get_edge_type(
                    i, j, k, 
                    nx, ny, nz, 
                    neighbor_transforms, 
                    minima_labels, 
                    minima_images, 
                    int_to_image, 
                    image_to_int, 
                    vacuum_mask)

                if unique_min == 0 and unique_max == 0:
                    # this is not part of a 1 or 2 manifold
                    label = 9
                
                elif unique_min == 0 and unique_max == 1:
                    # this is a separating plane between two maxima 3-manifolds.
                    # It represents the descending 2-manifold from a 2-saddle
                    label = 7
                
                elif unique_min == 0 and unique_max > 1:
                    # this is a bounding edge of the descending 2-manifold of
                    # a 2-saddle. it is a 1-manifold between a minimum and 1-saddle
                    label = 5
                
                elif unique_min == 1 and unique_max == 0:
                    # this is a separating plane between two minima 3-manifolds.
                    # It represents the ascending 2-manifold from a 1-saddle
                    label = 4
                
                elif unique_min > 1 and unique_max == 0:
                    # this is a bounding edge of the ascending 2-manifold of
                    # a 1-saddle. it is a 1-manifold between a maximum and 2-saddle
                    label = 6
                
                elif unique_min == 1 and unique_max == 1:
                    # this is a 1-manifold connecting two saddles
                    label = 8
                
                elif unique_min == 1 and unique_max > 1:
                    # this is a 1-saddle
                    label = 2
                
                elif unique_min > 1 and unique_max == 1:
                    # this is a 2-saddle
                    label = 3
                
                # is both > 2 possible?
                manifold_labels[i,j,k] = label
    return manifold_labels

bader = Bader.from_vasp("CHGCAR")

# get basins
maxima_labels = bader.basin_labels
maxima_images = bader.basin_images

minima_labels = bader.minima_basin_labels
minima_images = bader.minima_basin_images

image_to_int = np.empty([3,3,3], dtype=np.int64)
int_to_image = np.array(list(itertools.product((-1,0,1), repeat=3)))
for shift_idx, (i,j,k) in enumerate(int_to_image):
    image_to_int[i,j,k] = shift_idx
    
neighbor_transforms, _ = bader.reference_grid.point_neighbor_transforms

test = get_manifold_labels(
    maxima_labels, 
    minima_labels, 
    maxima_images, 
    minima_images, 
    image_to_int, 
    int_to_image, 
    neighbor_transforms, 
    vacuum_mask=bader.vacuum_mask
    )

maxima = bader.basin_maxima_vox
minima = bader.minima_vox

# mask_up = get_edges_periodic_all(
#     labeled_array=maxima_labels,
#     images=maxima_images,
#     image_to_int=image_to_int,
#     int_to_image=int_to_image,
#     neighbor_transforms=neighbor_transforms,
#     vacuum_mask=bader.vacuum_mask
#     )
# mask_down = get_edges_periodic_all(
#     labeled_array=minima_labels,
#     images=minima_images,
#     image_to_int=image_to_int,
#     int_to_image=int_to_image,
#     neighbor_transforms=neighbor_transforms,
#     vacuum_mask=bader.vacuum_mask,
#     )

# saddle_mask = get_saddles(mask_down, mask_up)
# # mask_up = bader_up.basin_edges
# # mask_down = bader_down.basin_edges
# mask_total = mask_up & mask_down


