# # -*- coding: utf-8 -*-

# import numpy as np
# from numba import njit, prange
# from numpy.typing import NDArray

# from baderkit.core.methods.shared_numba import (
#     get_best_neighbor,
#     get_gradient_simple,
#     wrap_point,
# )


# @njit(cache=True, parallel=True)
# def get_ongrid_and_rgrads(
#     data: NDArray[np.float64],
#     dir2lat: NDArray[np.float64],
#     neighbor_transforms: NDArray[np.int64],
#     neighbor_dists: NDArray[np.float64],
#     vacuum_mask: NDArray[np.bool_],
# ):
#     """
#     Calculates the ongrid steps and delta r at each point in the grid

#     Parameters
#     ----------
#     data : NDArray[np.float64]
#         A 3D grid of values for each point.
#     car2lat : NDArray[np.float64]
#         A matrix that converts a coordinate in cartesian space to fractional
#         space.
#     neighbor_transforms : NDArray[np.int64]
#         The transformations from each voxel to its neighbors.
#     neighbor_dists : NDArray[np.float64]
#         The distance to each neighboring voxel.
#     vacuum_mask : NDArray[np.bool_]
#         A 3D array representing the location of the vacuum.

#     Returns
#     -------
#     highest_neighbors : NDArray[np.int64]
#         A 4D array where highest_neighbors[i,j,k] returns the steepest neighbor at
#         point (i,j,k)
#     all_drs : NDArray[np.float64]
#         A 4D array where all_drs[i,j,k] returns the delta r between the true
#         gradient and ongrid step at point (i,j,k)
#     maxima_mask : NDArray[np.bool_]
#         A 3D array that is True at maxima

#     """
#     nx, ny, nz = data.shape
#     # create array for storing maxima
#     maxima_mask = np.zeros(data.shape, dtype=np.bool_)
#     # Create a new array for storing pointers
#     highest_neighbors = np.zeros((nx, ny, nz, 3), dtype=np.int64)
#     # Create a new array for storing rgrads
#     # Each (i, j, k) index gives the rgrad [x, y, z]
#     all_drs = np.zeros((nx, ny, nz, 3), dtype=np.float64)
#     gradients = np.zeros((nx, ny, nz, 3), dtype=np.float32)
#     # loop over each grid point in parallel
#     for i in prange(nx):
#         for j in range(ny):
#             for k in range(nz):
#                 # check if this point is part of the vacuum. If it is, we can
#                 # ignore this point.
#                 if vacuum_mask[i, j, k]:
#                     continue
#                 voxel_coord = np.array([i, j, k], dtype=np.int64)
#                 # get gradient
#                 gi, gj, gk = get_gradient_simple(
#                     data=data,
#                     voxel_coord=voxel_coord,
#                     dir2lat=dir2lat,
#                 )
#                 max_grad = 0.0
#                 for x in (gi, gj, gk):
#                     ax = abs(x)
#                     if ax > max_grad:
#                         max_grad = ax
#                 if max_grad < 1e-30:
#                     # we have no gradient so we reset the total delta r
#                     # Check if this is a maximum and if not step ongrid
#                     shift, neigh, is_max = get_best_neighbor(
#                         data=data,
#                         i=i,
#                         j=j,
#                         k=k,
#                         neighbor_transforms=neighbor_transforms,
#                         neighbor_dists=neighbor_dists,
#                     )
#                     # set pointer
#                     highest_neighbors[i, j, k] = neigh
#                     # set dr to 0 because we used an ongrid step
#                     all_drs[i, j, k] = (0.0, 0.0, 0.0)
#                     if is_max:
#                         maxima_mask[i, j, k] = True
#                     continue
#                 # Normalize
#                 gi /= max_grad
#                 gj /= max_grad
#                 gk /= max_grad
#                 # get pointer
#                 pi, pj, pk = round(gi), round(gj), round(gk)
#                 # get dr
#                 di = gi - pi
#                 dj = gj - pj
#                 dk = gk - pk
#                 # get neighbor. Don't bother wrapping because we will do this later
#                 ni = i + pi
#                 nj = j + pj
#                 nk = k + pk
#                 # save neighbor and dr
#                 gradients[i, j, k] = (gi, gj, gk)
#                 highest_neighbors[i, j, k] = (ni, nj, nk)
#                 all_drs[i, j, k] = (di, dj, dk)
#     return highest_neighbors, all_drs, gradients, maxima_mask


# @njit(fastmath=True, cache=True)
# def get_neargrid_labels(
#     data: NDArray[np.float64],
#     highest_neighbors: NDArray[np.int64],
#     all_drs: NDArray[np.float64],
#     maxima_mask: NDArray[np.bool_],
#     vacuum_mask: NDArray[np.bool_],
#     neighbor_transforms: NDArray[np.int64],
#     neighbor_dists: NDArray[np.float64],
# ) -> NDArray[np.bool_]:
#     """
#     Assigns each point to a basin using the neargrid method.

#     Parameters
#     ----------
#     data : NDArray[np.float64]
#         A 3D grid of values for each point.
#     highest_neighbors : NDArray[np.int64]
#         A 4D array where highest_neighbors[i,j,k] returns the steepest neighbor at
#         point (i,j,k)
#     all_drs : NDArray[np.float64]
#         A 4D array where all_drs[i,j,k] returns the delta r between the true
#         gradient and ongrid step at point (i,j,k)
#     maxima_mask : NDArray[np.bool_]
#         A 3D array that is True at maxima
#     vacuum_mask : NDArray[np.bool_]
#         A 3D array representing the location of the vacuum.
#     neighbor_transforms : NDArray[np.int64]
#         The transformations from each voxel to its neighbors.
#     neighbor_dists : NDArray[np.float64]
#         The distance to each neighboring voxel.

#     Returns
#     -------
#     labels : NDArray[np.int64]
#         The assignment for each point on the grid.

#     """
#     nx, ny, nz = data.shape
#     # define an array to assign to
#     labels = np.zeros(data.shape, dtype=np.int64)
#     # create a scratch array for our path
#     path = np.empty((nx * ny * nz, 3), dtype=np.int64)
#     # create a count of basins
#     maxima_num = 1
#     # create a scratch value for delta r
#     total_delta_r = np.zeros(3, dtype=np.float64)
#     current_coord = np.zeros(3, dtype=np.int64)
#     # loop over all voxels
#     for i in range(nx):
#         for j in range(ny):
#             for k in range(nz):
#                 # check if this point is part of the vacuum. If so, we don't
#                 # need to relabel it, so we continue.
#                 if vacuum_mask[i, j, k]:
#                     continue
#                 # check if we've already assigned this point
#                 if labels[i, j, k] != 0:
#                     continue
#                 # reset our delta_r
#                 total_delta_r[:] = 0.0
#                 # reset count for the length of the path
#                 pnum = 0
#                 # start climbing
#                 current_coord[:] = (i, j, k)

#                 while True:
#                     ii, jj, kk = current_coord
#                     # It shouldn't be possible to have entered the vacuum because
#                     # it will alwasy be lower than valid points
#                     assert not vacuum_mask[ii, jj, kk]

#                     # check if we've hit another label
#                     current_label = labels[ii, jj, kk]
#                     if current_label != 0:
#                         # relabel everything in our path and move to the next
#                         # voxel
#                         for p in range(pnum):
#                             x, y, z = path[p]
#                             # relabel to the this neighbors value
#                             labels[x, y, z] = current_label
#                         break  # move to next voxel
#                     # check if we've hit a maximum
#                     if maxima_mask[ii, jj, kk]:
#                         # keep the path labeled as is, and update the current
#                         # point to the same label, then increment the maxima count
#                         labels[ii, jj, kk] = maxima_num
#                         maxima_num += 1
#                         break

#                     # We have an unlabeled, non-max point and need to continue
#                     # our climb
#                     # Assign this point to the current maximum.
#                     # NOTE: We must relabel this as part of the vacuum later
#                     labels[ii, jj, kk] = maxima_num
#                     # add it to our path
#                     path[pnum] = (ii, jj, kk)
#                     pnum = pnum + 1
#                     # make a neargrid step
#                     # 1. get pointer and delta r
#                     new_coord = highest_neighbors[ii, jj, kk].copy()
#                     delta_r = all_drs[ii, jj, kk]
#                     # 2. sum delta r
#                     total_delta_r += delta_r
#                     # 3. update new coord and total delta r
#                     new_coord += np.rint(total_delta_r).astype(np.int64)
#                     total_delta_r -= np.round(total_delta_r)
#                     # 4. wrap coord
#                     new_coord[:] = wrap_point(
#                         new_coord[0], new_coord[1], new_coord[2], nx, ny, nz
#                     )
#                     new_label = labels[new_coord[0], new_coord[1], new_coord[2]]
#                     is_vac = vacuum_mask[new_coord[0], new_coord[1], new_coord[2]]
#                     # Check if the new coord is already on the path
#                     if new_label == maxima_num or is_vac:
#                         # we need to make an ongrid step to avoid repeating steps
#                         # or wandering into the vacuum
#                         _, new_coord, _ = get_best_neighbor(
#                             data=data,
#                             i=ii,
#                             j=jj,
#                             k=kk,
#                             neighbor_transforms=neighbor_transforms,
#                             neighbor_dists=neighbor_dists,
#                         )
#                     # update the coord
#                     current_coord = new_coord
#     return labels

# # @njit(cache=True)
# # def refine_neargrid(
# #     data: NDArray[np.float64],
# #     labels: NDArray[np.int64],
# #     refinement_indices: NDArray[np.int64],
# #     refinement_mask: NDArray[np.bool_],
# #     checked_mask: NDArray[np.bool_],
# #     maxima_mask: NDArray[np.bool_],
# #     highest_neighbors: NDArray[np.int64],
# #     all_drs: NDArray[np.float64],
# #     neighbor_transforms: NDArray[np.int64],
# #     neighbor_dists: NDArray[np.float64],
# #     vacuum_mask: NDArray[np.bool_],
# # ) -> tuple[NDArray[np.int64], np.int64, NDArray[np.bool_], NDArray[np.bool_]]:
# #     """
# #     Refines the provided voxels by running the neargrid method until a maximum
# #     is found for each.

# #     Parameters
# #     ----------
# #     data : NDArray[np.float64]
# #         A 3D grid of values for each point.
# #     labels : NDArray[np.int64]
# #         A 3D grid of labels representing current voxel assignments.
# #     refinement_indices : NDArray[np.int64]
# #         A Nx3 array of voxel indices to perform the refinement on.
# #     refinement_mask : NDArray[np.bool_]
# #         A 3D mask that is true at the voxel indices to be refined.
# #     checked_mask : NDArray[np.bool_]
# #         A 3D mask that is true at voxels that have already been refined.
# #     maxima_mask : NDArray[np.bool_]
# #         A 3D mask that is true at maxima.
# #     highest_neighbors : NDArray[np.int64]
# #         A 4D array where highest_neighbors[i,j,k] returns the steepest neighbor at
# #         point (i,j,k)
# #     all_drs : NDArray[np.float64]
# #         A 4D array where all_drs[i,j,k] returns the delta r between the true
# #         gradient and ongrid step at point (i,j,k)
# #     neighbor_transforms : NDArray[np.int64]
# #         The transformations from each voxel to its neighbors.
# #     neighbor_dists : NDArray[np.float64]
# #         The distance to each neighboring voxel.
# #     vacuum_mask : NDArray[np.bool_]
# #         A 3D array representing the location of the vacuum.

# #     Returns
# #     -------
# #     new_labels : NDArray[np.int64]
# #         The updated assignment for each point on the grid.
# #     reassignments : np.int64
# #         The number of points that were reassigned.
# #     refinement_mask : NDArray[np.bool_]
# #         The updated mask of points that need to be refined
# #     checked_mask : NDArray[np.bool_]
# #         The updated mask of points that have been checked.

# #     """
# #     # create an array for new labels
# #     new_labels = labels.copy()
# #     # get shape
# #     nx, ny, nz = data.shape
# #     # create scratch total_delta_r
# #     total_delta_r = np.zeros(3, dtype=np.float64)
# #     current_coord = np.empty(3, dtype=np.int64)
# #     # create scratch path
# #     path = np.empty((nx * ny * nz, 3), dtype=np.int64)
# #     # now we reassign any voxel in our refinement mask
# #     reassignments = 0
# #     for i, j, k in refinement_indices:
# #         # get our initial label for comparison
# #         label = labels[i, j, k]
# #         # reset the delta r tracker
# #         total_delta_r[:] = 0.0
# #         # create a count for the length of the path
# #         pnum = 0
# #         # set the initial coord
# #         current_coord[:] = (i, j, k)
# #         # start climbing
# #         while True:
# #             ii, jj, kk = current_coord
# #             # check if we've hit a maximum
# #             if maxima_mask[ii, jj, kk]:
# #                 # add this point to our checked list. We use this to make sure
# #                 # this point doesn't get re-added to our list later in the
# #                 # process.
# #                 checked_mask[i, j, k] = True
# #                 # remove it from the refinement list
# #                 refinement_mask[i, j, k] = False
# #                 # remove all points from the path
# #                 for p in range(pnum):
# #                     x, y, z = path[p]
# #                     labels[x, y, z] = abs(labels[x, y, z])
# #                 # We've hit a maximum.
# #                 current_label = labels[ii, jj, kk]
# #                 # Check if this is a reassignment
# #                 if label != current_label:
# #                     reassignments += 1
# #                     # add neighbors to our refinement mask for the next iteration
# #                     for shift in neighbor_transforms:
# #                         # get the new neighbor
# #                         ni = i + shift[0]
# #                         nj = j + shift[1]
# #                         nk = k + shift[2]
# #                         # loop
# #                         ni, nj, nk = wrap_point(ni, nj, nk, nx, ny, nz)
# #                         # If we haven't already checked this point, add it.
# #                         # NOTE: vacuum points are stored in the mask by default
# #                         if not checked_mask[ni, nj, nk]:
# #                             refinement_mask[ni, nj, nk] = True
# #                 # relabel just this voxel then stop the loop
# #                 new_labels[i, j, k] = current_label
# #                 break

# #             # Otherwise, we have not reached a maximum and want to continue
# #             # climbine
# #             # add this label to our path by marking it as negative.
# #             labels[ii, jj, kk] = -labels[ii, jj, kk]
# #             path[pnum] = (ii, jj, kk)
# #             pnum = pnum + 1
# #             # make a neargrid step
# #             # 1. get pointer and delta r
# #             new_coord = highest_neighbors[ii, jj, kk].copy()
# #             delta_r = all_drs[ii, jj, kk]
# #             # 2. sum delta r
# #             total_delta_r += delta_r
# #             # 3. update new coord and total delta r
# #             new_coord += np.rint(total_delta_r).astype(np.int64)
# #             total_delta_r -= np.round(total_delta_r)
# #             # 4. wrap coord
# #             new_coord[:] = wrap_point(
# #                 new_coord[0], new_coord[1], new_coord[2], nx, ny, nz
# #             )
# #             # check if the new coord is already in our path or belongs to the
# #             # vacuum
# #             temp_label = labels[new_coord[0], new_coord[1], new_coord[2]]
# #             is_vac = vacuum_mask[new_coord[0], new_coord[1], new_coord[2]]
# #             if temp_label < 0 or is_vac:
# #                 # we default back to an ongrid step to avoid repeating steps
# #                 _, new_coord, _ = get_best_neighbor(
# #                     data=data,
# #                     i=ii,
# #                     j=jj,
# #                     k=kk,
# #                     neighbor_transforms=neighbor_transforms,
# #                     neighbor_dists=neighbor_dists,
# #                 )
# #                 # reset delta r to avoid further loops
# #                 total_delta_r[:] = 0.0
# #                 # check again if we're still in the same path. If so, cancel
# #                 # the loop and don't write anything
# #                 if labels[new_coord[0], new_coord[1], new_coord[2]] < 0:
# #                     for p in range(pnum):
# #                         x, y, z = path[p]
# #                         labels[x, y, z] = abs(labels[x, y, z])
# #                         break
# #             # update the current coord
# #             current_coord = new_coord

# #     return new_labels, reassignments, refinement_mask, checked_mask
