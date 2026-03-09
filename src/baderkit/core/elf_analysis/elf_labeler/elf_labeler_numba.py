# -*- coding: utf-8 -*-


import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from baderkit.core.utilities.basic import wrap_point


@njit(parallel=True, cache=True)
def get_feature_edges(
    labeled_array: NDArray[np.int64],
    feature_map: NDArray[np.int64],
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
    feature_map : NDArray[np.int64]
        A 1D array mapping basin labels to feature labels
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
    edges = np.zeros((nx, ny, nz), dtype=np.bool_)
    # loop over each voxel in parallel
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # if this voxel is part of the vacuum, continue
                if vacuum_mask[i, j, k]:
                    continue
                # get this voxels feature
                basin = labeled_array[i, j, k]
                feature_label = feature_map[basin]
                # iterate over the neighboring voxels
                for si, sj, sk in neighbor_transforms:
                    # wrap points
                    ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
                    # get neighbors feature label
                    neigh_basin = labeled_array[ii, jj, kk]
                    neigh_feature_label = feature_map[neigh_basin]
                    # if any label is different, the current voxel is an edge.
                    # Note this in our edge array and break
                    # NOTE: we also check that the neighbor is not part of the
                    # vacuum
                    if (
                        neigh_feature_label != feature_label
                        and not vacuum_mask[ii, jj, kk]
                    ):
                        edges[i, j, k] = True
                        break
    return edges


@njit(cache=True, fastmath=True)
def get_min_avg_feat_surface_dists(
    labels,
    feature_map,
    frac_coords,
    edge_mask,
    matrix,
    max_value,
):
    nx, ny, nz = labels.shape
    # create array to store best dists, sums, and counts
    dists = np.full(len(frac_coords), max_value, dtype=np.float64)
    dist_sums = np.zeros(len(frac_coords), dtype=np.float64)
    edge_totals = np.zeros(len(frac_coords), dtype=np.uint32)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # skip outside edges
                if not edge_mask[i, j, k]:
                    continue
                # get feature label at edge
                feature_label = feature_map[labels[i, j, k]]
                # add to our count
                edge_totals[feature_label] += 1
                # convert from voxel indices to frac
                fi = i / nx
                fj = j / ny
                fk = k / nz
                # calculate the distance to the appropriate frac coord
                ni, nj, nk = frac_coords[feature_label]
                # get differences between each index
                di = ni - fi
                dj = nj - fj
                dk = nk - fk
                # wrap at edges to be as close as possible
                di -= round(di)
                dj -= round(dj)
                dk -= round(dk)
                # convert to cartesian coordinates
                ci = di * matrix[0, 0] + dj * matrix[1, 0] + dk * matrix[2, 0]
                cj = di * matrix[0, 1] + dj * matrix[1, 1] + dk * matrix[2, 1]
                ck = di * matrix[0, 2] + dj * matrix[1, 2] + dk * matrix[2, 2]
                # calculate distance
                dist = np.linalg.norm(np.array((ci, cj, ck), dtype=np.float64))
                # add to our total
                dist_sums[feature_label] += dist
                # if this is the lowest distance, update radius
                if dist < dists[feature_label]:
                    dists[feature_label] = dist
    # get average dists
    average_dists = dist_sums / edge_totals
    return dists, average_dists
