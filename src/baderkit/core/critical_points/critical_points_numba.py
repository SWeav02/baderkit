from baderkit.core.utilities.basic import wrap_point_w_shift
import numpy as np
from numpy.typing import NDArray
from numba import njit, prange

# Things to do:
    # get grid with labels for each type of critical point and manifold
    # Label ascending/descending manifolds
    # create connection map between critical points
    # determine what else is useful to make easily available. 
        # - integral lines?
        # - bader bonded atoms
        # - etc.

@njit(parallel=True)
def get_manifolds(
    labeled_array: NDArray[np.int64],
    images: NDArray[np.int64],
    image_to_int: NDArray[np.int64],
    int_to_image: NDArray[np.int64],
    neighbor_transforms: NDArray[np.int64],
    vacuum_mask: NDArray[np.bool_],
):
    """
    Finds the edges of maxima/minima 3-manifolds. Points that have 1 different
    neighbor lie on the dividing surface of the 3-manifold and correspond to
    2-manifolds. Points that have more than 1 different neighbor lie on the bounding
    edges of these 2-manifolds making up 1-manifolds. The complete description
    is as follows:
        maxima 3-manifolds:
            bounding 2-manifolds are descending connections from type-2 saddles
            to type-1 saddles
            bounding 1-manifolds are descending connections from type-1 saddles
            to minima
        minima 3-manifolds:
            bounding 2-manifolds are ascending connections from type-1 saddles
            to type-2 saddles
            bounding 1-manifolds are ascending connections from type-2 saddles
            to maxima

    Parameters
    ----------
    labeled_array : NDArray[np.int64]
        A 3D array where each entry represents the basin label of the point.
    images : NDArray[np.int64]
        A 3D array where each entry represents the periodic image of the maximum
        or minimum this point belongs to
    image_to_int : NDArray[np.int64]
        A mapping from the 3 integer shifts to each periodic image to a 1 integer
        representation
    int_to_image : NDArray[np.int64]
        A mapping from the 1 integer representation of a periodic image to the
        3 integer shift
    neighbor_transforms : NDArray[np.int64]
        The transformations from each voxel to its neighbors.
    vacuum_mask : NDArray[np.bool_]
        A 3D array representing the location of the vacuum

    Returns
    -------
    edges : NDArray[uint8]
        An array with the same shape as the input array that is 1 at 2-manifolds
        2 at 1-manifolds and 0 elsewhere

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
    """
    Uses the 

    Parameters
    ----------
    minima_manifolds : TYPE
        DESCRIPTION.
    maxima_manifolds : TYPE
        DESCRIPTION.

    Returns
    -------
    edges : TYPE
        DESCRIPTION.

    """

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
    manifold_labels = np.empty(maxima_labels, dtype=np.uint8)
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