# -*- coding: utf-8 -*-

import math

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from baderkit.core.utilities.interpolation import interp_nearest, interp_spline


@njit(parallel=True, cache=True)
def get_elf_radius(
    data,
    feature_labels,
    atom_idx,
    atom_coords,
    neigh_coords,
    covalent_labels,
    bond_dist,
    line_res: int = 20,
):
    # get the number of points to interpolate
    num_points = int(round(bond_dist * line_res))
    # I want this to always be odd because it is common for the exact midpoint
    # to be the correct fraction. This isn't required, but results in clean
    # values in these cases
    if num_points % 2 == 1:
        num_points += 1

    # get the vector pointing from each point along the line to the next
    step_vec = (neigh_coords - atom_coords) / (num_points - 1)

    # create arrays to store the values and labels along the bond
    values = np.empty(num_points, dtype=np.float64)
    labels = np.empty(num_points, dtype=np.int64)
    # calculate the positions, values, and labels along the line in parallel
    for point_idx in prange(num_points):
        x, y, z = atom_coords + float(point_idx) * step_vec
        values[point_idx] = interp_spline(x, y, z, data)
        labels[point_idx] = int(interp_nearest(x, y, z, feature_labels))

    # get the unique labels
    unique_labels = np.unique(labels)

    # SITUATION 1:
    # The atom's nearest neighbor is a translation of itself. The radius
    # must always be halfway between the two and the bond must be covalent
    if len(unique_labels) == 1:
        return 0.5 * bond_dist, True

    # SITUATION 2:
    # The atom has a covalent or metallic bond with its nearest neighbor, the 
    # radius is the closest local maximum to the center of the bond
    covalent = False
    for label in unique_labels:
        if covalent_labels[label]:
            covalent = True
            break
    if covalent:
        use_maximum = True
        # create placeholders for best maxima
        radius_index = -1
        maxima_dist = 1.0e6
        # create a tracker for the last point that belongs to this site
        last_idx = 0
        # get local maxima that are covalent
        midpoint = (len(values) - 1) / 2
        for i, (value, label) in enumerate(zip(values, labels)):
            # skip points that aren't part of the covalent bond
            if not covalent_labels[label]:
                continue
            # if this point is assigned to the current atom, update our idx
            if label == atom_idx:
                last_idx = i
            # check if the point is a maximum
            if ((i == 0) or (values[i - 1] <= value)) and (
                (i == len(values) - 1) or (value > values[i + 1])
            ):
                # if the maximum is closer to the midpoint than previous points,
                # update our best distance
                dist = abs(i - midpoint)
                if dist < maxima_dist:
                    maxima_dist = dist
                    radius_index = i
        # make sure we found a maximum. If not, default to the last point that
        # belongs to the current atom
        if radius_index == -1:
            radius_index = last_idx
            use_maximum = False

    # SITUATION 3:
    # The atom is ionically bonded to its nearest neighbor. The radius is
    # at the first point that does not belong to this atom.
    # NOTE: If we have not labeled metals/electrides yet, this also captures
    # the situation where we treat metal features as separate from the atom
    else:
        use_maximum = False
        midpoint = -1
        # find the first point that doesn't belong to the current atom
        for i, label in enumerate(labels):
            if label != atom_idx:
                midpoint = i
                break
        radius_index = -1
        # get the minimum along the line closest to this point
        minima_dist = 1.0e6
        for i, (value, label) in enumerate(zip(values, labels)):
            # check if the point is a minimum
            if ((i == 0) or (values[i - 1] >= value)) and (
                (i == len(values) - 1) or (value < values[i + 1])
            ):
                # if the minimum is closer to the midpoint than previous points,
                # update our best distance
                dist = abs(i - midpoint)
                if dist < minima_dist:
                    minima_dist = dist
                    radius_index = i

    # Now we want to refine the radius. First, we get the coordinate of the
    # current radius
    current_coord = atom_coords + radius_index * step_vec
    ci, cj, ck = current_coord
    current_value = interp_spline(ci, cj, ck, data)
    # This point must be within one index of the true radius. We iteratively
    # move a step closer to the true value, dividing the step by two after each
    # iteration
    # calculate number of steps needed to reach required resolution
    # res = 1/line_res * 0.5^n
    resolution = 1e-8  # angstroms
    n = round(math.log(resolution * line_res) / math.log(1 / 2))
    step_mult = 1.0
    for i in range(n):
        step_mult /= 2.0
        step = step_vec * step_mult
        # get value above and below
        ucoord = current_coord + step
        ui, uj, uk = ucoord
        dcoord = current_coord - step
        di, dj, dk = dcoord
        up_val = interp_spline(ui, uj, uk, data)
        down_val = interp_spline(di, dj, dk, data)
        if use_maximum:
            # check which value is the highest and adjust our coord
            if up_val > current_value and up_val >= down_val:
                current_value = up_val
                # update coord
                current_coord = ucoord
                # update index
                radius_index += step_mult
            elif down_val > current_value:
                current_value = down_val
                current_coord = dcoord
                radius_index -= step_mult
        else:
            # check which value is the lowest and adjust our coord
            if up_val < current_value and up_val <= down_val:
                current_value = up_val
                # update coord
                current_coord = ucoord
                # update index
                radius_index += step_mult
            elif down_val < current_value:
                current_value = down_val
                current_coord = dcoord
                radius_index -= step_mult

    # We now have a refined radius. Calculate the actual bond distance
    bond_frac = radius_index / (num_points - 1)

    return bond_frac * bond_dist, covalent


@njit(cache=True)
def get_elf_radii(
    equivalent_atoms,
    data,
    feature_labels,
    atom_frac_coords,
    neighbor_indices,
    neighbor_dists,
    neighbor_images,
    covalent_labels: NDArray,
):
    # get the unique atoms we need to calculate radii for
    unique_atoms = np.unique(equivalent_atoms)

    # create array to store radii and their type
    atomic_radii = np.empty(len(atom_frac_coords), dtype=np.float64)
    radius_is_covalent = np.empty(len(atom_frac_coords), dtype=np.bool_)

    # get the radius for each atom. NOTE: We don't do this in parallel because
    # we want the interpolation to be done in parallel instead
    for atom_idx in unique_atoms:
        atom_coords = atom_frac_coords[atom_idx]
        neigh_idx = neighbor_indices[atom_idx]
        bond_dist = neighbor_dists[atom_idx]
        neigh_image = neighbor_images[atom_idx]

        # get the neighbors frac coords
        neigh_coords = atom_frac_coords[neigh_idx] + neigh_image

        # get the radius for this atom
        radius, is_covalent = get_elf_radius(
            data,
            feature_labels,
            atom_idx,
            atom_coords,
            neigh_coords,
            covalent_labels,
            bond_dist,
        )
        atomic_radii[atom_idx] = radius
        radius_is_covalent[atom_idx] = is_covalent

    # update values for equivalent atoms
    for atom_idx in range(len(atomic_radii)):
        equiv_atom = equivalent_atoms[atom_idx]
        atomic_radii[atom_idx] = atomic_radii[equiv_atom]
        radius_is_covalent[atom_idx] = radius_is_covalent[equiv_atom]

    return atomic_radii, radius_is_covalent


@njit(cache=True)
def get_all_atom_elf_radii(
    site_indices,
    neigh_indices,
    site_frac_coords,
    neigh_frac_coords,
    neigh_dists,
    equivalent_bonds,
    data,
    feature_labels,
    covalent_labels,
):

    # get the unique bonds
    unique_bonds = np.unique(equivalent_bonds)

    # create array to store radii
    atomic_radii = np.empty(len(site_indices), dtype=np.float64)
    radius_is_covalent = np.empty(len(site_indices), dtype=np.bool_)

    # get the radius for each unique bond
    for pair_idx in unique_bonds:
        site_idx = site_indices[pair_idx]
        site_frac = site_frac_coords[site_idx]
        neigh_frac = neigh_frac_coords[pair_idx]
        bond_dist = neigh_dists[pair_idx]
        radius, is_covalent = get_elf_radius(
            data,
            feature_labels,
            site_idx,
            site_frac,
            neigh_frac,
            covalent_labels,
            bond_dist,
        )
        atomic_radii[pair_idx] = radius
        radius_is_covalent[pair_idx] = is_covalent

    # update values for equivalent bonds
    for pair_idx in range(len(atomic_radii)):
        equiv_bond = equivalent_bonds[pair_idx]
        atomic_radii[pair_idx] = atomic_radii[equiv_bond]
        radius_is_covalent[pair_idx] = radius_is_covalent[equiv_bond]

    return atomic_radii, radius_is_covalent
