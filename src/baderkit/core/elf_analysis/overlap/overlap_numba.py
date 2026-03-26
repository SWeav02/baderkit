# -*- coding: utf-8 -*-

from numba import njit, prange, types

import numpy as np
from numpy.typing import NDArray

from baderkit.core.utilities.transforms import INT_TO_IMAGE, IMAGE_TO_INT

@njit(cache=True, parallel=True)
def get_label_image_map(
    labels,
    images,
    num_labels,
        ):
    nx, ny, nz = labels.shape
    labels_w_images = np.zeros((num_labels, 27), dtype=np.bool_)
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                label = labels[i,j,k]
                # skip vacuum
                if label == num_labels:
                    continue

                # get the image
                image = images[i,j,k]

                labels_w_images[label, image] = True
    # construct label map
    pairs = np.argwhere(labels_w_images)
    label_map = np.empty_like(labels_w_images, dtype=np.int16)
    for idx in prange(len(pairs)):
        i,j = pairs[idx]
        label_map[i,j] = idx
    return pairs, label_map

@njit(cache=True)
def get_unique_overlaps(
    atom_labels,
    local_labels,
    atom_images,
    local_images,
    atom_image_map,
    local_image_map,
    num_atoms,
    num_local,
):
    nx, ny, nz = atom_labels.shape

    # create set to store pairs
    atom_local_pairs = set()

    # collect unique pairs
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                atom_label = atom_labels[i, j, k]
                local_label = local_labels[i, j, k]

                # skip vacuum
                if atom_label == num_atoms or local_label == num_local:
                    continue

                atom_image = atom_images[i, j, k]
                local_image = local_images[i, j, k]

                atom_pair = atom_image_map[atom_label, atom_image]
                local_pair = local_image_map[local_label, local_image]

                pair_val = szudzik_pair(atom_pair, local_pair)
                atom_local_pairs.add(int(pair_val))

    # convert set to array
    out = np.empty(len(atom_local_pairs), dtype=np.int64)

    idx = 0
    for pair in atom_local_pairs:
        out[idx] = pair
        idx += 1

    return out

@njit(parallel=True, cache=True)
def get_overlap_table(
        atom_labels,
        atom_images,
        local_labels,
        local_images,
        num_atoms,
        num_local,
        ):
    nx, ny, nz = local_labels.shape
    # first we construct maps to and from each label/image pair to a
    # single index. This reduces the memory needed to find overlaps
    atom_image_pairs, atom_image_map = get_label_image_map(
        labels=atom_labels,
        images=atom_images,
        num_labels=num_atoms,
        )
    local_image_pairs, local_image_map = get_label_image_map(
        labels=local_labels,
        images=local_images,
        num_labels=num_local,
        )
    atom_local_pairs = get_unique_overlaps(
        atom_labels,
        local_labels,
        atom_images,
        local_images,
        atom_image_map,
        local_image_map,
        num_atoms,
        num_local,
        )
    # get the unique sets of charge/local pairs
    unique_overlaps = np.empty((len(atom_local_pairs), 4), dtype=np.int64)
    for idx in prange(len(atom_local_pairs)):
        pair_idx = atom_local_pairs[idx]
        atom_pair, local_pair = szudzik_reverse(pair_idx)
        atom_label, atom_image = atom_image_pairs[atom_pair]
        local_label, local_image = local_image_pairs[local_pair]
        unique_overlaps[idx] = (atom_label, atom_image, local_label, local_image)
    return unique_overlaps

@njit(parallel=True, cache=True)
def get_overlap_charge_volume(
        unique_overlaps,
        atom_labels,
        atom_images,
        local_labels,
        local_images,
        charge_data,
        cell_volume,
        ):
    nx, ny, nz = local_labels.shape
    # create arrays to store charges/volumes and labels
    pair_charges = np.zeros(len(unique_overlaps), dtype=np.float64)
    pair_volumes = np.zeros(len(unique_overlaps), dtype=np.float64)
    pair_labels = np.full(local_labels.shape, len(unique_overlaps), dtype=np.uint32)
    for idx in prange(len(unique_overlaps)):
        atom_label, atom_image, local_label, local_image = unique_overlaps[idx]

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if atom_labels[i,j,k] != atom_label:
                        continue
                    if local_labels[i,j,k] != local_label:
                        continue
                    if atom_images[i,j,k] != atom_image:
                        continue
                    if local_images[i,j,k] != local_image:
                        continue
                    pair_charges[idx] += charge_data[i,j,k]
                    pair_volumes[idx] += 1
                    pair_labels[i,j,k] = idx

    pair_charges /= nx*ny*nz
    pair_volumes = cell_volume * (pair_volumes / pair_volumes.sum())
    return pair_charges, pair_volumes, pair_labels

@njit(parallel=True, cache=True)
def get_basin_fractions(
    overlap_table,
    overlap_charges,
    num_basins,
    tol,
        ):
    scratch = np.empty((0,0),dtype=np.float64)
    basin_frac = []
    for i in range(num_basins):
        basin_frac.append(scratch.copy())

    for idx in prange(num_basins):
        # get overlap basins that include this local basin
        indices = np.where(overlap_table[:,2] == idx)[0]

        # get the unique neighbors
        basin_neighs = set()
        for overlap_idx in indices:
            neigh_label, neigh_image, _, basin_image = overlap_table[overlap_idx]
            # shift to the label that is in the cell
            mi,mj,mk = INT_TO_IMAGE[neigh_image] - INT_TO_IMAGE[basin_image]
            image = IMAGE_TO_INT[mi,mj,mk]
            pair = szudzik_pair(neigh_label, image)
            basin_neighs.add(int(pair))
        # convert to array
        neighs = np.empty(len(basin_neighs), dtype=np.int64)
        neigh_idx = 0
        for i in basin_neighs:
            neighs[neigh_idx] = i
            neigh_idx += 1
        neighs = np.sort(neighs)

        # now get the total counts for each
        counts = np.zeros(len(neighs), dtype=np.float64)
        for overlap_idx in indices:
            neigh_label, neigh_image, _, basin_image = overlap_table[overlap_idx]
            # shift to the label that is in the cell
            mi,mj,mk = INT_TO_IMAGE[neigh_image] - INT_TO_IMAGE[basin_image]
            image = IMAGE_TO_INT[mi,mj,mk]
            pair = int(szudzik_pair(neigh_label, image))
            pair_idx = np.searchsorted(neighs, pair)
            counts[pair_idx] += overlap_charges[overlap_idx]

        fracs = counts / counts.sum()
        # remove fracs below cutoff
        high_fracs = np.where(fracs > tol)[0]
        fracs = fracs[high_fracs]
        neighs = neighs[high_fracs]
        # convert neighs from szudzik pairs
        neigh_pairs = np.empty((len(neighs),2), dtype=np.int64)
        for neigh_idx in range(len(neigh_pairs)):
            i, j = szudzik_reverse(neighs[neigh_idx])
            neigh_pairs[neigh_idx,0] = int(i)
            neigh_pairs[neigh_idx,1] = int(j)
        # sort from high to low
        sorted_indices = np.flip(np.argsort(fracs))
        basin_frac[idx] = np.column_stack((neigh_pairs[sorted_indices], fracs[sorted_indices]))
    return basin_frac

@njit(parallel=True, cache=True)
def get_qtaim_groups(
    local_fractions,
    num_atoms,
        ):
    scratch = np.empty((0,0),dtype=np.float64)
    atom_frac = []
    for i in range(num_atoms):
        atom_frac.append(scratch.copy())

    for idx in prange(num_atoms):
        group = []
        # loop over local basin fractions
        for local_idx, (local_fracs) in enumerate(local_fractions):
            for atom_idx, atom_image, frac in local_fracs:
                # skip if this isn't the current atom
                if int(atom_idx) != idx:
                    continue
                # otherwise we flip the image and add to our group
                mi, mj, mk = -INT_TO_IMAGE[int(atom_image)]
                image = IMAGE_TO_INT[mi, mj, mk]
                group.append((float(local_idx), float(image), frac))
        # convert group to an array
        group_array = np.empty((len(group),3), dtype=np.float64)
        for group_idx, entry in enumerate(group):
            group_array[group_idx] = entry
        atom_frac[idx] = group_array
        
    return atom_frac

@njit(cache=True)
def get_overlap_fractions(
    overlap_table,
    overlap_charges,
    num_local,
    num_atoms,
    tol=0.001,
        ):

    local_frac = get_basin_fractions(
        overlap_table,
        overlap_charges,
        num_basins=num_local,
        tol=tol,
        )
    atom_groups = get_qtaim_groups(
        local_fractions=local_frac,
        num_atoms=num_atoms,
        )
    return local_frac, atom_groups

@njit(cache=True)
def get_atom_shell_groups(
    atom_local_groups,
    atom_frac_coords,
    local_frac_coords,
    matrix,
    tol=0.2
        ):
    coord_groups = []
    basin_dists = []
    for atom_idx in range(len(atom_frac_coords)):
        atom_frac = atom_frac_coords[atom_idx]
        atom_cart = atom_frac @ matrix
        local_group = atom_local_groups[atom_idx]
        neigh_dists = np.zeros(len(local_group), dtype=np.float64)
        for local_idx, (label, image, _) in enumerate(local_group):
            image1 = INT_TO_IMAGE[int(image)]
            local_frac = local_frac_coords[int(label)] + image1
            local_cart = local_frac @ matrix
            offset = local_cart - atom_cart
            neigh_dists[local_idx] = np.linalg.norm(offset)

        # group by distance
        basin_dists.append(neigh_dists)
        neigh_indices = np.argsort(neigh_dists)
        groups = []
        current_val = neigh_dists[neigh_indices[0]]
        current_group = [neigh_indices[0]]
        for idx in neigh_indices[1:]:
            dist = neigh_dists[idx]
            diff = (dist - current_val) / dist < tol
            current_val = dist
            if dist == 0 or diff:
                current_group.append(idx)
            else:
                group_array = np.array(current_group, dtype=np.int64)
                groups.append(group_array)
                current_group = [idx]
        groups.append(np.array(current_group, dtype=np.int64))

        coord_groups.append(groups)
    return coord_groups, basin_dists

@njit(parallel=True, cache=True)
def get_atom_charge_claims(
        access_sets,
        bond_fractions,
        local_basin_charges,
        equiv_species,
        num_atoms,
        num_local,
        ):
    unique_species = np.unique(equiv_species)
    # create array to store charge claims
    charge_claims = []
    for i in range(num_atoms):
        charge_claims.append(np.empty((0,0), dtype=np.float64))
    
    # create array for the access numbers
    access_numbers = np.empty(num_atoms, dtype=np.float64)
    species_nums = np.empty(num_atoms, dtype=np.float64)
    
    # create array to store connection indices
    connection_indices = np.empty(num_atoms, dtype=np.float64)

    for idx in prange(num_atoms):
        # get the access set for this atom
        access_set = access_sets[idx]
        # create tracker for the total access number
        access_num = 0.0          

        # now find each atoms access claim
        atom_claims = np.zeros((num_atoms, 27), dtype=np.float64)
        for local_idx, local_image in access_set:
            image = INT_TO_IMAGE[int(local_image)]
            charge = local_basin_charges[local_idx]
            access_num += charge
            # loop over the atomic basins overlaped with this label and add their
            # claims
            local_overlap = bond_fractions[local_idx]
            for atom_idx, atom_image, frac in local_overlap:
                # get atom shift
                image1 = INT_TO_IMAGE[int(atom_image)]
                mi, mj, mk = image + image1
                actual_image = IMAGE_TO_INT[mi,mj,mk]
                atom_claims[int(atom_idx), actual_image] += charge*frac
        
        # reduce to only atoms with claims
        atoms = np.argwhere(atom_claims>0)
        flat_atom_claims = np.empty((len(atoms)), dtype=np.float64)
        for i, (x,y) in enumerate(atoms):
            flat_atom_claims[i] = atom_claims[x,y]
        atom_claims = flat_atom_claims
        
        # normalize to access num
        atom_claims = atom_claims / access_num
        access_numbers[idx] = access_num

        charge_claims[idx] = np.column_stack((
            atoms.astype(np.float64),
            atom_claims,
             ))

        # calculate connection index. First we condense down to unique species
        species_claims = np.zeros(len(unique_species)+1, dtype=np.float64)
        
        for i in range(len(atoms)):
            atom_idx, atom_image = atoms[i]
            frac = atom_claims[i]
            # if this is the current atom, add to the first entry
            if atom_idx == idx and atom_image == 13:
                species_claims[0] += frac
                continue
            # otherwise, get the equivalent species
            spec_idx = equiv_species[atom_idx] + 1
            species_claims[spec_idx] += frac

        # calculate atom connection index
        num_spec = len(species_claims)
        species_nums[idx] = num_spec
        index = 0.0
        for i in range(num_spec):
            frac = species_claims[i]
            for j in range(i+1, num_spec):
                frac1 = species_claims[j]
                index += frac * frac1
        nonzero = len(np.nonzero(species_claims)[0])
        index *= 2*nonzero / (nonzero-1)
        connection_indices[idx] = index

    return charge_claims, access_numbers, connection_indices, species_nums

# @njit(cache=True, parallel=True)
# def get_unique_basins_w_images(
#     atom_labels,
#     atom_images,
#     local_labels,
#     local_images,
#     num_atoms,
#     num_local,
#     local_frac,
#     charge_frac,
#         ):
#     nx, ny, nz = atom_labels.shape
#     labels_w_images = np.zeros((num_atoms, 27), dtype=np.bool_)
#     for i in prange(nx):
#         for j in range(ny):
#             for k in range(nz):
#                 atom_label = atom_labels[i,j,k]
#                 local_label = local_labels[i,j,k]
#                 # skip vacuum
#                 if atom_label == num_atoms or local_label == num_local:
#                     continue

#                 # get the shift required to move the charge maximum into the
#                 # same image as the local maximum
#                 shift = np.round(local_frac[local_label] - charge_frac[atom_label]).astype(np.int64)

#                 # get charge image relative to local basin
#                 atom_image = INT_TO_IMAGE[atom_images[i,j,k]]
#                 local_image = INT_TO_IMAGE[local_images[i,j,k]] + shift
#                 mi, mj, mk = local_image - atom_image

#                 image = IMAGE_TO_INT[mi, mj, mk]

#                 labels_w_images[atom_label, image] = True
#     # construct label map
#     pairs = np.argwhere(labels_w_images)
#     label_map = np.empty_like(labels_w_images, dtype=np.int16)
#     for idx in prange(len(pairs)):
#         i,j = pairs[idx]
#         label_map[i,j] = idx
#     return pairs, label_map


# @njit(cache=True)
# def get_overlap_counts(
#     atom_labels: NDArray[np.int64],
#     atom_images: NDArray[np.int64],
#     local_labels: NDArray[np.int64],
#     local_images: NDArray[np.int64],
#     charge_data: NDArray[np.float64],
#     local_frac: NDArray[np.float64],
#     charge_frac: NDArray[np.float64],
#     num_atoms: int,
#     num_local: int,
#         ):
#     nx, ny, nz = local_labels.shape

#     # get the total unique labels/images
#     label_image_pairs, label_image_map = get_unique_basins_w_images(
#         atom_labels=atom_labels,
#         atom_images=atom_images,
#         local_labels=local_labels,
#         local_images=local_images,
#         num_atoms=num_atoms,
#         num_local=num_local,
#         local_frac=local_frac,
#         charge_frac=charge_frac,
#         )

#     # create array to track total populations
#     overlap_counts = np.zeros((len(label_image_pairs), num_local), dtype=np.float64)

#     # What we need:
#     # Overlap labels (distinct types of overlap between charge/local)
#     # Atoms overlapped with each local basin
#     # Counts for atoms overlapped with each local basin

#     # loop over each voxel and count the number of overlaps
#     for i in range(nx):
#         for j in range(ny):
#             for k in range(nz):
#                 # get the labels at this point
#                 atom_label = atom_labels[i, j, k]
#                 local_label = local_labels[i, j, k]

#                 # skip points in vacuum
#                 if atom_label == num_atoms or local_label == num_local:
#                     continue

#                 # get the shift required to move the charge maximum into the
#                 # same image as the local maximum
#                 shift = np.round(local_frac[local_label] - charge_frac[atom_label]).astype(np.int64)

#                 # get charge image relative to local basin
#                 atom_image = INT_TO_IMAGE[atom_images[i,j,k]]
#                 local_image = INT_TO_IMAGE[local_images[i,j,k]] + shift
#                 mi, mj, mk = local_image - atom_image

#                 image = IMAGE_TO_INT[mi, mj, mk]

#                 # add to our count
#                 atom_pair = label_image_map[atom_label, image]
#                 overlap_counts[atom_pair, local_label] += charge_data[i,j,k]

#     return overlap_counts, label_image_pairs, label_image_map



# TODO:
    # add images into this as well
    # calculate connection index
    # calculate nearest neighbor sharing
    # test with total charge densities

@njit(cache=True)
def szudzik_pair(a: int, b: int):
    if a >= b:
        return a * a + a + b
    elif a < b:
        return b * b + a


@njit(cache=True)
def szudzik_reverse(z):
    k = (z ** (1 / 2)) // 1
    kk = k * k

    if z - kk < k:
        a = z - kk
        b = k
    else:
        a = k
        b = z - kk - k
    return int(a), int(b)