# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from baderkit.core.utilities.basic import (
    coords_to_flat,
    flat_to_coords,
    wrap_point,
    wrap_point_w_shift,
)
from baderkit.core.utilities.basins import get_best_neighbor


@njit(parallel=True, cache=True)
def get_weight_assignments(
    reference_data,
    labels,
    images,
    charge_data,
    sorted_indices,
    neighbor_transforms: NDArray[np.int64],
    neighbor_alpha: NDArray[np.float64],
    all_neighbor_transforms,
    all_neighbor_dists,
    extrema_mask,
    extrema_indices,
    use_minima: bool = False,
):
    nx, ny, nz = reference_data.shape
    ny_nz = ny * nz
    num_coords = len(sorted_indices)
    full_num_coords = nx * ny * nz
    # create arrays to store neighs. Don't store flux yet
    num_transforms = len(neighbor_transforms)
    neigh_array = np.empty((num_coords, num_transforms), dtype=np.uint32)
    neigh_nums = np.empty(num_coords, dtype=np.uint8)
    # Create 1D arrays to store flattened charge
    flat_charge = np.empty(full_num_coords, dtype=np.float64)
    # Create arrays to store basin charges/volumes
    charges = np.zeros(len(extrema_indices), dtype=np.float64)
    volumes = np.zeros(len(extrema_indices), dtype=np.float64)
    # create array to store the highest contributing neighbor. We will use this
    # for periodic shift tracking
    highest_neighs = np.empty(num_coords, dtype=np.uint8)
    unlabeled_value = np.iinfo(labels.dtype).max
    vacuum_label = unlabeled_value - 1

    # create map from shifts to index
    shift_to_int = np.empty((3, 3, 3), dtype=np.int64)
    int_to_shift = np.empty((27, 3), dtype=np.int64)
    idx = 0
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                shift_to_int[i, j, k] = idx
                int_to_shift[idx] = (i, j, k)
                idx += 1

    ###########################################################################
    # Get neighbors
    ###########################################################################

    # loop over points in parallel and calculate neighbors
    for sorted_idx in prange(num_coords):
        idx = sorted_indices[sorted_idx]
        # get 3D coords
        i, j, k = flat_to_coords(idx, ny_nz, nz)
        # get the reference and charge data
        base_value = reference_data[i, j, k]
        # set flat charge
        flat_charge[idx] = charge_data[i, j, k]

        # BUGFIX: In rare cases, the voronoi neighbors might include a neighbor
        # more than 1 voxel away. In this case it's possible we labeled a maximum
        # earlier that wouldn't be found as a maximum here. We default to the
        # extrema found earlier as we used interpolation to confirm or reject
        # them. Here we need to check for these first to make sure they get
        # assigned properly
        if extrema_mask[i, j, k]:
            # Note this is a maximum
            neigh_nums[sorted_idx] = 0
            # assign the first value to the current label. This will allow us
            # to check if the maximum is the root max in the next section
            neigh_array[sorted_idx, 0] = labels[idx]
            continue

        # get more extreme neighbors at each point
        neigh_num = 0
        highest_neigh = -1
        if not use_minima:
            highest_val = -np.inf
        else:
            highest_val = np.inf
        for si, sj, sk in neighbor_transforms:
            # get neighbor and wrap around periodic boundary
            ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
            # get the neighbors value
            neigh_value = reference_data[ii, jj, kk]

            # if this value is below the current points value, continue
            if (
                not use_minima
                and neigh_value <= base_value
                or use_minima
                and neigh_value >= base_value
            ):
                continue

            # get this neighbors index and add it to our array
            neigh_idx = coords_to_flat(ii, jj, kk, ny_nz, nz)
            # if this point is part of the vacuum, we just continue
            if labels[neigh_idx] == vacuum_label:
                continue
            neigh_array[sorted_idx, neigh_num] = neigh_idx
            neigh_num += 1
            if (
                not use_minima
                and neigh_value > highest_val
                or use_minima
                and neigh_value < highest_val
            ):
                highest_neigh = shift_to_int[si, sj, sk]
                highest_val = neigh_value

        # Check if we had any higher neighbors
        if neigh_num == 0:
            # this is not a real maximum. Assign it to the highest neighbor
            (si, sj, sk), (ni, nj, nk) = get_best_neighbor(
                data=reference_data,
                i=i,
                j=j,
                k=k,
                neighbor_transforms=all_neighbor_transforms,
                neighbor_dists=all_neighbor_dists,
                use_minima=use_minima,
            )
            neigh_idx = coords_to_flat(ni, nj, nk, ny_nz, nz)
            neigh_nums[sorted_idx] = 1  # note a single neighbor
            neigh_array[sorted_idx, 0] = neigh_idx
            highest_neigh = shift_to_int[si, sj, sk]
        else:
            neigh_nums[sorted_idx] = neigh_num
        highest_neighs[sorted_idx] = highest_neigh

    ###########################################################################
    # Assign interior
    ###########################################################################
    # create list to store edge indices
    edge_sorted_indices = []
    added_extrema = []

    # Now we have the neighbors for each point. Loop over them from highest to
    # lowest and assign single basin points
    for sorted_idx, (idx, neighs, neigh_num) in enumerate(
        zip(sorted_indices, neigh_array, neigh_nums)
    ):
        if neigh_num > 0:
            # This is not a maximum. Check if interior point (single basin)
            best_label = unlabeled_value
            is_vac = False
            for neigh_idx, neigh in enumerate(neighs):
                if neigh_idx == neigh_num:
                    break
                label = labels[neigh]
                if label == vacuum_label:
                    # this is part of the vacuum. We want to note this, but not
                    # assign flux to it. We will only assign to it if there is
                    # no other valid neighbor
                    is_vac = True
                    continue
                if label == unlabeled_value:
                    # This neighbor is not an interior and this one can't be either
                    best_label = unlabeled_value
                    break
                elif label != best_label and best_label != unlabeled_value:
                    # We have two different basin assignments and this is not an
                    # interior
                    best_label = unlabeled_value
                    break
                best_label = label

            # If the best label is assigned, this is an interior point and we assign
            if best_label != unlabeled_value:

                labels[idx] = best_label

                charges[best_label] += flat_charge[idx]
                volumes[best_label] += 1.0
                # get the shift to the nearest neighbor
                si, sj, sk = int_to_shift[highest_neighs[sorted_idx]]

                i, j, k = flat_to_coords(idx, ny_nz, nz)
                ni, nj, nk, si, sj, sk = wrap_point_w_shift(
                    i + si, j + sj, k + sk, nx, ny, nz
                )
                # combine neighbors shift
                neigh_idx = coords_to_flat(ni, nj, nk, ny_nz, nz)
                nsi, nsj, nsk = images[neigh_idx]
                images[idx, 0] += si + nsi
                images[idx, 1] += sj + nsj
                images[idx, 2] += sk + nsk
            # if we have no labels and border a vacuum, we assign to vacuum
            elif is_vac:
                labels[idx] = vacuum_label
                # get the shift to the nearest neighbor
                si, sj, sk = int_to_shift[highest_neighs[sorted_idx]]

                i, j, k = flat_to_coords(idx, ny_nz, nz)
                ni, nj, nk, si, sj, sk = wrap_point_w_shift(
                    i + si, j + sj, k + sk, nx, ny, nz
                )
                # combine neighbors shift
                neigh_idx = coords_to_flat(ni, nj, nk, ny_nz, nz)
                nsi, nsj, nsk = images[neigh_idx]
                images[idx, 0] += si + nsi
                images[idx, 1] += sj + nsj
                images[idx, 2] += sk + nsk
            # Otherwise, this point is an exterior point that is partially assigned
            # to multiple basins. We add it to our list.
            else:
                edge_sorted_indices.append(sorted_idx)

        else:

            # Skip if this maximum was already processed
            if idx in added_extrema:
                continue

            # get this extremas current label
            label = labels[idx]

            # Determine the root maximum
            root_idx = label if label != idx else idx

            # check if this is a root
            is_root = idx == root_idx

            # If this root maximum hasn't been added yet, add it
            if root_idx not in added_extrema:
                added_extrema.append(root_idx)
                max_idx = np.searchsorted(extrema_indices, root_idx)
                labels[root_idx] = max_idx
                charges[max_idx] += flat_charge[root_idx]
                volumes[max_idx] += 1.0

            else:
                max_idx = labels[root_idx]

            if not is_root:
                # Assign this point to the correct maximum
                labels[idx] = max_idx
                charges[max_idx] += flat_charge[idx]
                volumes[max_idx] += 1.0

    ###########################################################################
    # Fluxes
    ###########################################################################
    # We only need to calculate the flux for each exterior point. Create an array
    # to store these.
    num_edges = len(edge_sorted_indices)
    flux_array = np.empty((num_edges, num_transforms), dtype=np.float64)
    neigh_array = np.empty_like(flux_array, dtype=np.int64)
    neigh_nums = np.empty(num_edges, dtype=np.uint8)
    # create an array to store pointers from idx to edge idx
    idx_to_edge = np.empty(full_num_coords, dtype=np.uint32)
    # calculate fluxes in parallel. If possible, we will immediately calculate the
    # weight as well
    for edge_idx in prange(len(edge_sorted_indices)):
        sorted_idx = edge_sorted_indices[edge_idx]
        idx = sorted_indices[sorted_idx]
        # set idx to edge value
        idx_to_edge[idx] = edge_idx
        # loop over neighs and get their label. If all of them have labels, we
        # can immediately calculate weights
        # get 3D coords
        i, j, k = flat_to_coords(idx, ny_nz, nz)
        # get the reference and charge data
        base_value = reference_data[i, j, k]
        # set flat charge
        flat_charge[idx] = charge_data[i, j, k]
        # get higher neighbors at each point
        total_flux = 0.0
        best_flux = 0.0
        best_neigh = -1
        neigh_labels = neigh_array[edge_idx]
        neigh_fluxes = flux_array[edge_idx]
        neigh_num = 0
        for (si, sj, sk), alpha in zip(neighbor_transforms, neighbor_alpha):
            # get neighbor and wrap around periodic boundary
            ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
            # get the neighbors value
            neigh_value = reference_data[ii, jj, kk]
            # if this value is below the current points value, continue
            if (
                not use_minima
                and neigh_value <= base_value
                or use_minima
                and neigh_value >= base_value
            ):
                continue
            # get this neighbors index
            neigh_idx = coords_to_flat(ii, jj, kk, ny_nz, nz)
            # get this neighbors label
            neigh_label = labels[neigh_idx]

            if neigh_label == vacuum_label:
                continue

            # calculate the flux flowing to this voxel
            flux = abs(neigh_value - base_value) * alpha
            if flux > best_flux:
                best_flux = flux
                best_neigh = shift_to_int[si, sj, sk]
            total_flux += flux

            # if the neighbor hasn't been assigned, assign flux to this neighbor
            if neigh_label == unlabeled_value:
                # at least one neighbor is also an exterior point
                neigh_fluxes[neigh_num] = flux
                neigh_labels[neigh_num] = -neigh_idx - 1
                neigh_num += 1
                # no_exterior_neighs = False
                continue
            # otherwise, check if this label already exists in our neighbors
            found = False
            for nidx, nlabel in enumerate(neigh_labels):
                if nidx == neigh_num:
                    # we've reached the end of our assigned labels so we break
                    break
                if label == nlabel:
                    neigh_fluxes[nidx] += flux
                    found = True
                    break
            if not found:
                neigh_fluxes[neigh_num] = flux
                neigh_labels[neigh_num] = neigh_label
                neigh_num += 1

        # BUG-FIX
        # in rare cases, we may find no neighbors. This means we found a false
        # maximum earlier and assigned it to an ongrid neighbor that itself
        # ended up being an edge point or another false maximum. To correct for
        # this, we can assign a full flux of 1 to the best ongrid neighbor
        if neigh_num == 0:
            (si, sj, sk), (ni, nj, nk) = get_best_neighbor(
                data=reference_data,
                i=i,
                j=j,
                k=k,
                neighbor_transforms=all_neighbor_transforms,
                neighbor_dists=all_neighbor_dists,
                use_minima=use_minima,
            )
            neigh_idx = coords_to_flat(ni, nj, nk, ny_nz, nz)
            neigh_label = labels[neigh_idx]
            neigh_fluxes[0] = 1.0
            best_neigh = shift_to_int[si, sj, sk]

            # If the neighbor belongs to a basin, assign to the same one. Otherwise,
            # it's an edge and we note the connections.
            if neigh_label >= 0 and neigh_label != unlabeled_value:
                neigh_labels[0] = neigh_label
            else:
                neigh_labels[0] = -neigh_idx - 1
            total_flux = 1.0
            neigh_num = 1

        neigh_nums[edge_idx] = neigh_num
        # normalize fluxes
        neigh_fluxes /= total_flux
        # update the highest neighbor to be the highest flux neighbor
        highest_neighs[sorted_idx] = best_neigh

    ###########################################################################
    # Edge assignments
    ###########################################################################
    # Now we have the fluxes (and some weights) at each edge point. We loop over
    # them from high to low and assign charges, volumes, and labels
    scratch_weights = np.zeros(len(charges), dtype=np.float64)
    weight_mask = np.zeros(len(charges), dtype=np.bool_)
    approx_charges = charges.copy()

    all_weights = []
    all_labels = []
    for edge_idx, (fluxes, neighs, neigh_num) in enumerate(
        zip(flux_array, neigh_array, neigh_nums)
    ):
        sorted_idx = edge_sorted_indices[edge_idx]
        idx = sorted_indices[sorted_idx]
        charge = flat_charge[idx]

        current_labels = []
        # Loop over neighbors and calculate weights for this point
        for neigh_idx, (flux, label) in enumerate(zip(fluxes, neighs)):
            if neigh_idx == neigh_num:
                break
            # NOTE: I was looping over neigh_num here, but sometimes this caused a
            # crash. Maybe numba is trying to us prange even though I didn't ask it to?

            if label >= 0:
                # This is a basin rather than another edge index.
                if not weight_mask[label]:
                    current_labels.append(label)
                    weight_mask[label] = True

                scratch_weights[label] += flux
                continue

            # otherwise, this is another edge index. Get its weight
            label = -label - 1  # convert back to actual neighbor index
            neigh_edge_idx = idx_to_edge[label]

            neigh_labels = all_labels[neigh_edge_idx]
            neigh_weights = all_weights[neigh_edge_idx]
            # loop over neighbors weights and add the portion assigned to this
            # voxel
            for label, weight in zip(neigh_labels, neigh_weights):
                # if there is no weight at this label yet, its new. Add it to our list
                if not weight_mask[label]:
                    current_labels.append(label)
                    weight_mask[label] = True
                scratch_weights[label] += flux * weight

        current_labels = np.array(current_labels, dtype=np.int64)
        current_labels = np.sort(current_labels)

        # Now loop over each label and assign charges, volumes, images, and labels
        best_label = -1
        best_weight = 0.0
        total_weight = 0.0
        tied_labels = False
        tol = 1e-6  # for floating point errors
        # BUGFIX: remove values below a cutoff to help with memory
        reduced_labels = []
        reduced_weights = []

        for label in current_labels:
            weight = scratch_weights[label]
            scratch_weights[label] = 0.0  # reset scratch
            weight_mask[label] = False
            charges[label] += charge * weight
            volumes[label] += weight
            # skip if our weight is below a very small tolerance
            if weight < 1e-30:
                continue

            if weight > best_weight + tol:  # greater than with a tolerance
                best_label = label
                best_weight = weight
                tied_labels = False
            elif weight > best_weight - tol:  # equal to with a tolerance
                tied_labels = True
            # add weight to current weights
            reduced_weights.append(weight)
            reduced_labels.append(label)
            total_weight += weight

        # BUGFIX: cap the lists at a reasonable size to avoid memory explosions
        if len(reduced_labels) > 26:
            weight_array = np.array(reduced_weights, dtype=np.float64)
            ordered_weights = np.argsort(weight_array)[-26:]
            total_weight = weight_array[ordered_weights].sum()
            reduced_weights = [reduced_weights[i] for i in ordered_weights]
            reduced_labels = [reduced_labels[i] for i in ordered_weights]

        # renormalize weights in case we removed any
        reduced_weights = [i / total_weight for i in reduced_weights]

        # add weights/labels for this point to our list
        all_weights.append(reduced_weights)
        all_labels.append(reduced_labels)

        # Now we want to assign our label. If there wasn't a tie in our labels,
        # we assign to highest weight
        if not tied_labels:
            labels[idx] = best_label
            approx_charges[best_label] += charge
        else:
            # we have a tie. We assign to the basin where the added charge will
            # most improve the approximate charge
            best_improvement = -1.0
            for label, weight in zip(reduced_labels, reduced_weights):
                if weight < best_weight - tol:
                    continue
                # calculate the difference from the current charge before and
                # after adding this point
                diff = approx_charges[label] - charges[label]
                before = abs(diff)
                after = abs(diff + charge)
                improvement = (before - after) / charges[label]
                if improvement > best_improvement + tol:
                    best_improvement = improvement
                    best_label = label
            labels[idx] = best_label
            approx_charges[best_label] += charge

        # update shift

        # get the shift to the nearest neighbor
        si, sj, sk = int_to_shift[highest_neighs[sorted_idx]]
        i, j, k = flat_to_coords(idx, ny_nz, nz)
        ni, nj, nk, si, sj, sk = wrap_point_w_shift(i + si, j + sj, k + sk, nx, ny, nz)

        # combine neighbors shift
        neigh_idx = coords_to_flat(ni, nj, nk, ny_nz, nz)

        if labels[neigh_idx] != best_label:
            # back up to the highest neighbor with the same label
            highest_neigh = -1
            if not use_minima:
                highest_val = -np.inf
            else:
                highest_val = np.inf
            for ti, tj, tk in neighbor_transforms:
                # get neighbor and wrap around periodic boundary
                ii, jj, kk, ssi, ssj, ssk = wrap_point_w_shift(
                    i + ti, j + tj, k + tk, nx, ny, nz
                )
                # get the neighbors value
                neigh_value = reference_data[ii, jj, kk]
                # if this value is below the current points value, continue
                if (
                    not use_minima
                    and neigh_value > highest_val
                    or use_minima
                    and neigh_value < highest_val
                ):
                    neigh_idx = coords_to_flat(ii, jj, kk, ny_nz, nz)
                    if labels[neigh_idx] != best_label:
                        continue
                    highest_neigh = neigh_idx
                    highest_val = neigh_value
                    si = ssi
                    sj = ssj
                    sk = ssk
            neigh_idx = highest_neigh

        nsi, nsj, nsk = images[neigh_idx]
        images[idx, 0] += si + nsi
        images[idx, 1] += sj + nsj
        images[idx, 2] += sk + nsk

    return (
        labels,
        images,
        charges,
        volumes,
    )


@njit(parallel=True, cache=True)
def sort_extrema_frac(
    extrema_vox,
    grid_shape,
):
    nx, ny, nz = grid_shape
    ny_nz = ny * nz

    flat_indices = np.zeros(len(extrema_vox), dtype=np.int64)
    for idx in prange(len(flat_indices)):
        i, j, k = extrema_vox[idx]
        flat_indices[idx] = coords_to_flat(i, j, k, ny_nz, nz)

    # sort flat indices from low to high
    sorted_indices = np.argsort(flat_indices)
    # sort extrema from lowest index to highest
    return (
        extrema_vox[sorted_indices],
        flat_indices[sorted_indices],
    )


###############################################################################
# Tests for better labeling. The label assignments never converged well so I've
# given this up for now.
###############################################################################

# @njit(fastmath=True)
# def get_labels_fine(
#     label_array,
#     flat_grid_indices,
#     neigh_pointers,
#     neigh_fluxes,
#     neigh_numbers,
#     volumes,
#     charges,
#     sorted_coords,
#     sorted_charge,
#         ):
#     max_idx = len(sorted_coords) - 1
#     # create an array to store approximate volumes
#     # approx_volumes = np.zeros(len(volumes), dtype=np.int64)
#     # Flip the true volumes/charges so that they are in order from highest to
#     # lowest coord
#     volumes = np.flip(volumes)
#     # charges = np.flip(charges)
#     # multiply charges by 2 so we can avoid a lot of divisions later
#     # charges *= 2
#     # Create an array to store the difference from the ideal volume
#     volume_diff = np.ones(len(volumes), dtype=np.float64)
#     # charge_diff = np.ones(len(charges), dtype=np.float64)
#     # diffs = np.ones(len(volumes), dtype=np.float64)
#     # Create an array to store the ratio by which the volume_diff changes when
#     # a new voxel is added to the corresponding basin
#     volume_ratios = 1.0 / volumes
#     # create a list to store neighbor labels
#     all_neighbor_labels = []
#     # split_voxels = np.zeros(len(pointers), dtype=np.bool_)
#     # loop over points from high to low
#     extrema_num = 0
#     for idx in np.arange(max_idx, -1, -1):
#         # get the charge and position
#         # charge = sorted_charge[idx]
#         i,j,k = sorted_coords[idx]
#         # If there are neighs, this is a maximum. We assign a new basin
#         neighbor_num = neigh_numbers[idx]
#         if neighbor_num == 0:
#             # label the voxel
#             label_array[i,j,k] = extrema_num
#             all_neighbor_labels.append([extrema_num])
#             # update the volume/charge diffs
#             volume_diff[extrema_num] -= volume_ratios[extrema_num]
#             # charge_diff[extrema_num] -= charge / charges[extrema_num]
#             # diffs[extrema_num] -= (volume_ratios[extrema_num] + charge / charges[extrema_num]) # divide by 2 is done earlier
#             extrema_num += 1
#             continue

#         # otherwise, we are not at a maximum
#         # get the pointers/flux
#         pointers = neigh_pointers[idx]
#         # fluxes = neigh_fluxes[idx]

#         # tol = (1/neighbor_num) - 1e-12
#         # reduce to labels/weights
#         labels = []
#         # weights = []
#         # for pointer, flux in zip(pointers, fluxes):
#         for pointer in pointers:
#             # if the pointer is -1 we've reached the end of our list
#             if pointer == -1:
#                 break
#             # if the flux is less than our tolerance, we don't consider this neighbor
#             # if flux < tol:
#             #     continue
#             # otherwise, get the labels at this point
#             neigh_labels = all_neighbor_labels[max_idx-pointer]
#             for label in neigh_labels:
#                 if not label in labels:
#                     labels.append(label)
#             # # otherwise, get the label at this point
#             # ni, nj, nk = sorted_coords[pointer]
#             # label = label_array[ni,nj,nk]
#             # # check if the label exists. If not, add it
#             # found = False
#             # for lidx, rlabel in enumerate(labels):
#             #     if label == rlabel:
#             #         found = True
#             #         # weights[lidx] += flux
#             # if not found:
#             #     # add the new label/weight
#             #     labels.append(label)
#             #     # weights.append(flux)


#         # If there is 1 label, assign this label
#         if len(labels) == 1:
#             label = labels[0]
#             label_array[i,j,k] = label
#             # update volume/charge diffs
#             volume_diff[label] -= volume_ratios[label]
#             # charge_diff[label] -= charge / charges[label]
#             # diffs[label] -= (volume_ratios[label] + charge / charges[label])
#         # if there is more than 1 label, we have a split voxel. As an approximation,
#         # we check how far from the true volume each possible basin is and add
#         # the voxel to the farthest one.
#         else:
#             best_label = -1
#             best_diff = -1.0
#             for label in labels:
#                 # if diffs[label] > best_diff:
#                 #     best_label = label
#                 #     best_diff = diffs[label]
#                 if volume_diff[label] > best_diff:
#                     best_label = label
#                     best_diff = volume_diff[label]
#                 # if charge_diff[label] > best_diff:
#                 #     best_label = label
#                 #     best_diff = charge_diff[label]
#             # update label
#             label_array[i,j,k] = best_label
#             # update diff
#             volume_diff[best_label] -= volume_ratios[best_label]
#             # charge_diff[best_label] -= charge / charges[best_label]
#             # diffs[best_label] -= (volume_ratios[best_label] + charge / charges[best_label])

#         all_neighbor_labels.append(labels)

#     return label_array

###############################################################################
# Parallel attempt. Doesn't scale linearly
###############################################################################

# @njit(parallel=True, cache=True)
# def get_weight_assignments(
#     data,
#     labels,
#     flat_charge,
#     neigh_fluxes,
#     neigh_pointers,
#     weight_extrema_mask,
#     all_neighbor_transforms,
#     all_neighbor_dists,
# ):
#     nx,ny,nz = data.shape
#     # Get the indices corresponding to extrema
#     extrema_indices = np.where(weight_extrema_mask)[0]
#     extrema_num = len(extrema_indices)
#     # We are going to reuse the extrema mask as a mask noting which points don't
#     # need to be checked anymore
#     finished_points = weight_extrema_mask
#     finished_extrema = np.zeros(extrema_num, dtype=np.bool_)
#     # create arrays to store charges, volumes, and pointers
#     charges = flat_charge[extrema_indices]
#     volumes = np.ones(extrema_num, dtype=np.float64)
#     # create array to store the true maximum each local extrema belongs to. This
#     # is used to reduce false weight extrema
#     extrema_map = np.empty(extrema_num, dtype=np.int64)
#     # create array representing total volume
#     flat_volume = np.ones(len(flat_charge), dtype=np.float64)
#     # create secondary arrays to store flow of charge/volume
#     flat_volume1 = np.zeros(len(flat_charge), dtype=np.float64)
#     flat_charge1 = np.zeros(len(flat_charge), dtype=np.float64)
#     # create array to store number of lower neighbors at each point
#     neigh_nums = np.zeros(len(flat_charge), dtype=np.int8)
#     # create counter for if we are on an even/odd loop
#     loop_count = 0

#     # Now we begin our while loop
#     while True:
#         # get the indices to loop over
#         current_indices = np.where(~finished_points)[0]
#         current_extrema = np.where(~finished_extrema)[0]
#         num_current = len(current_indices)
#         extrema_current = len(current_extrema)
#         if num_current == 0 and extrema_current == 0:
#             break
#         # get the charge and volume arrays that were accumulated into last cycle
#         # and the ones to accumulate into this cycle
#         if loop_count % 2 == 0:
#             charge_store = flat_charge
#             volume_store = flat_volume
#             charge_new = flat_charge1
#             volume_new = flat_volume1
#         else:
#             charge_store = flat_charge1
#             volume_store = flat_volume1
#             charge_new = flat_charge
#             volume_new = flat_volume

#         # loop over extrema and sum their neighbors current accumulated charge
#         for max_idx in prange(extrema_num):
#             if finished_extrema[max_idx]:
#                 continue
#             max_pointer = extrema_indices[max_idx]
#             pointers = neigh_pointers[max_pointer]
#             fluxes = neigh_fluxes[max_pointer]
#             # sum each charge
#             new_charge = 0.0
#             new_volume = 0.0
#             for neigh_idx, (pointer, flux) in enumerate(zip(pointers, fluxes)):
#                 # skip neighbors with no charge
#                 if pointer == -1:
#                     continue
#                 # If charge is 0, remove this neighbor
#                 charge = charge_store[pointer]
#                 if charge == 0.0:
#                     pointers[neigh_idx] = -1
#                 new_charge += charge * flux
#                 new_volume += volume_store[pointer] * flux
#             # If no charge was added, we're done with this maximum
#             if new_charge == 0.0:
#                 finished_extrema[max_idx] = True
#                 # Check if this is a true maximum
#                 i,j,k = flat_to_coords(max_pointer, nx, ny, nz)
#                 mi, mj, mk = climb_to_max(data, i, j, k, all_neighbor_transforms, all_neighbor_dists)
#                 # update extrema map and labels
#                 pointer = coords_to_flat(mi,mj,mk,nx,ny,nz)
#                 labels[i,j,k] = pointer
#                 extrema_map[max_idx] = pointer

#             # add charge/volume to total
#             charges[max_idx] += new_charge
#             volumes[max_idx] += new_volume

#         # loop over other points, sum their neighbors, reset charge/volume accumulation
#         for point_idx in prange(num_current):
#             point_pointer = current_indices[point_idx]
#             pointers = neigh_pointers[point_pointer]
#             fluxes = neigh_fluxes[point_pointer]
#             # if this is our first cycle, we want to get the number of neighbors
#             # for each point and reorder our pointers/fluxes for faster iteration
#             if loop_count == 0:
#                 n_neighs = 0
#                 for neigh_idx, pointer in enumerate(pointers):
#                     # skip empty neighbors
#                     if pointer == -1:
#                         continue
#                     # move pointer/flux to farthest left point
#                     pointers[n_neighs] = pointer
#                     fluxes[n_neighs] = fluxes[neigh_idx]
#                     n_neighs += 1
#                 neigh_nums[point_pointer] = n_neighs

#             # otherwise, sum charge/volume as usual
#             n_neighs = neigh_nums[point_pointer]
#             new_charge = 0.0
#             new_volume = 0.0
#             for neigh_idx in range(n_neighs):
#                 neigh_pointer = pointers[neigh_idx]
#                 if neigh_pointer == -1:
#                     continue
#                 charge = charge_store[neigh_pointer]
#                 # if the charge is 0, we no longer need to accumulate charge
#                 # from this point.
#                 if charge == 0.0:
#                     pointers[neigh_idx] = -1
#                     continue
#                 new_charge += charge_store[neigh_pointer] * fluxes[neigh_idx]
#                 new_volume += volume_store[neigh_pointer] * fluxes[neigh_idx]
#             # set new charge and volume
#             charge_new[point_pointer] = new_charge
#             volume_new[point_pointer] = new_volume
#             # if charge was 0 mark this point as not important
#             if new_charge == 0.0:
#                 finished_points[point_pointer] = True

#         loop_count += 1

#     # reduce to true extrema
#     true_extrema = np.unique(extrema_map)
#     reduced_charges = np.zeros(len(true_extrema), dtype=np.float64)
#     reduced_volumes = np.zeros(len(true_extrema), dtype=np.float64)
#     for old_idx, max_label in enumerate(extrema_map):
#         for max_idx, true_max in enumerate(true_extrema):
#             if max_label == true_max:
#                 reduced_charges[max_idx] += charges[old_idx]
#                 reduced_volumes[max_idx] += volumes[old_idx]

#     return reduced_charges, reduced_volumes, labels, true_extrema