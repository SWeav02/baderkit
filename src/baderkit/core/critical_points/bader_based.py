# -*- coding: utf-8 -*-

from baderkit.core import Bader
from baderkit.core.utilities.basic import wrap_point, wrap_point_w_shift
import numpy as np
from numpy.typing import NDArray
from numba import njit, prange

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


bader = Bader.from_vasp("CHGCAR")

# get basins
labels_up = bader.basin_labels
images_up = bader.basin_images

labels_down = bader.minima_basin_labels
images_down = bader.minima_basin_images


neighbor_transforms, _ = bader.reference_grid.point_neighbor_transforms

mask_up = get_edges_periodic(
    labeled_array=labels_up,
    images=images_up,
    neighbor_transforms=neighbor_transforms,
    vacuum_mask=bader.vacuum_mask
    )
mask_down = get_edges_periodic(
    labeled_array=labels_down,
    images=images_down,
    neighbor_transforms=neighbor_transforms,
    vacuum_mask=bader.vacuum_mask,
    )

# mask_up = bader_up.basin_edges
# mask_down = bader_down.basin_edges
mask_total = mask_up & mask_down



# test = find_saddles(
#     edge_mask=mask_total,
#     data=bader.reference_grid.total,
#     neighbor_transforms=neighbor_transforms,
#     use_minima=False,
# )
# test1 = find_saddles(
#     edge_mask=mask_total,
#     data=bader.reference_grid.total,
#     neighbor_transforms=neighbor_transforms,
#     use_minima=True,
# )

# edge_labels = label_edges(
#     labels_up,
#     neighbor_transforms=neighbor_transforms,
#     edge_mask=mask_up,
#     )

# unique_labels = np.unique(edge_labels[mask_up])
# unique_labels = unique_labels[unique_labels!=65535]

# pairs = szudzik_unpair_array(unique_labels)
