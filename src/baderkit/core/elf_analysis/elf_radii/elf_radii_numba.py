# -*- coding: utf-8 -*-

import math

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from baderkit.core.utilities.interpolation import interp_nearest, interp_spline


@njit(
    parallel=True,
    cache=True,
)
def get_elf_radius_frac(
    data,
    feature_labels,
    atom_idx,
    atom_coords,
    neigh_coords,
    all_frac_coords,
    covalent_labels,
    bond_dist,
    line_res: int = 20,
):
    """
    Calculates the fraction along a bond between two atoms that corresponds
    to the bond radius. Uses ELF data and an array representing
    which atom/nna a grid point belongs to using zero-flux data

    Parameters
    ----------
    data : NDArray
        The ELF data.
    feature_labels : NDArray
        The labeled grid.
    atom_idx : int
        The index of the first atom in the bond.
    atom_coords : NDArray
        The fractional coordinates of the first atom in the bond.
    neigh_coords : NDArray
        The fractional coordinates of the second atom in the bond.
    all_frac_coords : NDArray
        The fractional coordinates of all atoms/dummy atoms in the structure
    covalent_labels : NDArray
        The indices in the feature_labels array that should be considered covalent.
    bond_dist : float
        The lenght of the bond.
    line_res : int, optional
        The number of points per angstrom to interpolate. The default is 20.

    Returns
    -------
    tuple
        The radius, whether or not the bond is covalent, and whether or not no
        radius was able to be found.

    """
    num_atoms = len(all_frac_coords)

    # get the number of points to interpolate
    num_points = int(round(bond_dist * line_res))
    # I want this to always be odd because it is common for the exact midpoint
    # to be the correct fraction. This isn't required, but results in clean
    # values in these cases
    if num_points % 2 == 0:
        num_points += 1

    # get the vector pointing from each point along the line to the next
    step_vec = (neigh_coords - atom_coords) / (num_points - 1)

    # create arrays to store the values and labels along the bond
    values = np.empty(num_points, dtype=np.float64)
    labels = np.empty(num_points, dtype=np.int64)
    # calculate the positions, values, and labels along the line in parallel
    for point_idx in prange(num_points):
        point = atom_coords + float(point_idx) * step_vec
        x, y, z = point
        values[point_idx] = interp_spline(x, y, z, data)
        label = int(interp_nearest(x, y, z, feature_labels))
        # For small cells its possible for this label to be incorrect.
        # In poarticular it may assign to a translation of the actual atom. To
        # test for this, we check if the vector from the point to the site is
        # less than half a unit cell in any direction. This assumes the atom is
        # fairly spherical (which is also an assumption of badelf itself).

        vec = all_frac_coords[label] - point
        if np.max(np.abs(vec)) > 0.5:
            label += num_atoms

        labels[point_idx] = label
    # get the unique labels
    unique_labels = np.unique(labels)
    # TODO: If passes through vacuum, set to last point?

    # SITUATION 1:
    # The atom's nearest neighbor is a translation of itself. The radius
    # must always be halfway between the two and the bond must be covalent
    if len(unique_labels) == 1:
        return 0.5, True, False

    # SITUATION 2:
    # The bond passes through labels that are not the current site or neighbor.
    # This means we either pass through a covalent/metallic bond or we pass through
    # another atom. In the later case, this probably isn't a valid radius and
    # we just return 0.5. In the former case, we return the maximum in the bond.

    covalent = False
    other_atoms = False
    site_idx = labels[0]
    neigh_idx = labels[-1]
    for label in unique_labels:
        equiv_label = label % num_atoms
        if covalent_labels[equiv_label]:
            covalent = True
        elif label != site_idx and label != neigh_idx:
            other_atoms = True
            break
    if other_atoms:
        return 0.5, True, False

    if covalent:
        use_maximum = True
        # create placeholders for best maxima
        radius_index = -1
        maxima_dist = 1.0e6
        # get local maxima that are covalent
        midpoint = (len(values) - 1) / 2
        for i, (value, label) in enumerate(zip(values, labels)):
            # skip points that aren't part of the covalent bond
            equiv_label = label % num_atoms
            if not covalent_labels[equiv_label]:
                continue
            # check if the point is a maximum
            # BUGFIX: We don't allow the first or last point to be considered
            # maxima. If we allow that, the fraction will be refined to a point
            # outside the bond range
            if i == 0 or i == num_points - 1:
                continue
            if values[i - 1] <= value and value > values[i + 1]:
                # if the maximum is closer to the midpoint than previous points,
                # update our best distance
                dist = abs(i - midpoint)
                if dist < maxima_dist:
                    maxima_dist = dist
                    radius_index = i
        # make sure we found a maximum. If not, default to the third method which
        # finds the minimum closest to the last label belonging to the atom along
        # this line
        if radius_index == -1:
            covalent = False

    # SITUATION 3:
    # The atom is ionically bonded to its nearest neighbor. The radius is
    # at the first point that does not belong to this atom.
    # NOTE: If we have not labeled metals/electrides yet, this also captures
    # the situation where we treat metal features as separate from the atom
    if not covalent:
        use_maximum = False
        midpoint = -1
        # find the first point that doesn't belong to the current atom
        for i, label in enumerate(labels):
            # skip points with no label. Doesn't typically happen but might if
            # the ELF is very low around an atom center
            if label == -1:
                continue
            if label != atom_idx:
                midpoint = i
                break
        radius_index = -1
        # get the minimum along the line closest to this point
        minima_dist = 1.0e6
        for i, (value, label) in enumerate(zip(values, labels)):
            # BUGFIX: We don't allow the first or last point to be considered
            # minima. If we allow that, the minimum will be refined to a point
            # outside the bond ranged.
            if i == 0 or i == num_points - 1:
                continue
            # check if the point is a minimum
            if values[i - 1] >= value and value < values[i + 1]:
                # if the minimum is closer to the midpoint than previous points,
                # update our best distance
                dist = abs(i - midpoint)
                if dist < minima_dist:
                    minima_dist = dist
                    radius_index = i

    # Situation 4: If we've still failed to find a radius, we are likely using
    # too few valence electrons and have no core/shell around our atom. In this
    # case we default to a radius halfway between the atoms which in many cases is
    # unreasonable
    if radius_index == -1 or radius_index == 0:
        return 0.5, True, True

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
    return bond_frac, covalent, False


@njit(cache=True)
def get_elf_radius(
    data,
    feature_labels,
    atom_idx,
    atom_coords,
    neigh_coords,
    all_frac_coords,
    covalent_labels,
    bond_dist,
    line_res: int = 20,
):
    """
    Calculates the radius between two atoms using ELF data and an array representing
    which atom/nna a grid point belongs to using zero-flux data

    Parameters
    ----------
    data : NDArray
        The ELF data.
    feature_labels : NDArray
        The labeled grid.
    atom_idx : int
        The index of the first atom in the bond.
    atom_coords : NDArray
        The fractional coordinates of the first atom in the bond.
    neigh_coords : NDArray
        The fractional coordinates of the second atom in the bond.
    covalent_labels : NDArray
        The indices in the feature_labels array that should be considered covalent.
    bond_dist : float
        The lenght of the bond.
    line_res : int, optional
        The number of points per angstrom to interpolate. The default is 20.

    Returns
    -------
    tuple
        The radius, whether or not the bond is covalent, and whether or not no
        radius was able to be found.

    """
    # get frac
    bond_frac, covalent, failed = get_elf_radius_frac(
        data,
        feature_labels,
        atom_idx,
        atom_coords,
        neigh_coords,
        all_frac_coords,
        covalent_labels,
        bond_dist,
        line_res,
    )
    return bond_dist * bond_frac, bond_frac, covalent, failed


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
    bond_fracs = np.empty(len(atom_frac_coords), dtype=np.float64)
    radius_is_covalent = np.empty(len(atom_frac_coords), dtype=np.bool_)

    some_failed = False
    # get the radius for each atom. NOTE: We don't do this in parallel because
    # we want the interpolation to be done in parallel instead
    for unique_idx in prange(len(unique_atoms)):
        atom_idx = unique_atoms[unique_idx]
        atom_coords = atom_frac_coords[atom_idx]
        neigh_idx = neighbor_indices[atom_idx]
        bond_dist = neighbor_dists[atom_idx]
        neigh_image = neighbor_images[atom_idx]

        # get the neighbors frac coords
        neigh_coords = atom_frac_coords[neigh_idx] + neigh_image

        # get the radius for this atom
        radius, frac, is_covalent, failed = get_elf_radius(
            data,
            feature_labels,
            atom_idx,
            atom_coords,
            neigh_coords,
            atom_frac_coords,
            covalent_labels,
            bond_dist,
        )
        if failed:
            some_failed = True

        atomic_radii[atom_idx] = radius
        bond_fracs[atom_idx] = frac
        radius_is_covalent[atom_idx] = is_covalent

    if some_failed:
        print(
            """At least one atoms radius could not be calculated. This is usually
              due to the atom having no core electrons, likely caused by the use
              of too small of a pseudopotential. The radius will default to 1/2 the bond length"""
        )

    # update values for equivalent atoms
    for atom_idx in prange(len(atomic_radii)):
        equiv_atom = equivalent_atoms[atom_idx]
        atomic_radii[atom_idx] = atomic_radii[equiv_atom]
        bond_fracs[atom_idx] = bond_fracs[equiv_atom]
        radius_is_covalent[atom_idx] = radius_is_covalent[equiv_atom]

    return atomic_radii, bond_fracs, radius_is_covalent


@njit(cache=True, parallel=True)
def get_all_atom_elf_radii(
    site_indices,
    neigh_indices,
    site_frac_coords,
    neigh_frac_coords,
    neigh_dists,
    reversed_bonds,
    data,
    feature_labels,
    covalent_labels,
    equivalent_atoms,
):

    # create array to store radii
    atomic_radii = np.empty(len(site_indices), dtype=np.float64)
    bond_fracs = np.empty(len(site_indices), dtype=np.float64)
    radius_is_covalent = np.empty(len(site_indices), dtype=np.bool_)

    some_failed = False
    # get the radius for each bond
    for pair_idx in prange(len(site_indices)):
        site_idx = site_indices[pair_idx]
        neigh_idx = neigh_indices[pair_idx]
        bond_dist = neigh_dists[pair_idx]
        # if these are equivalent atoms we can skip the process entirely and
        # return a bond exactly halfway
        if equivalent_atoms[site_idx] == equivalent_atoms[neigh_idx]:
            atomic_radii[pair_idx] = bond_dist * 0.5
            bond_fracs[pair_idx] = 0.5
            radius_is_covalent[pair_idx] = True
            continue

        site_frac = site_frac_coords[
            site_idx
        ]  # taken from structure to avoid larger array
        neigh_frac = neigh_frac_coords[pair_idx]
        radius, frac, is_covalent, failed = get_elf_radius(
            data,
            feature_labels,
            site_idx,
            site_frac,
            neigh_frac,
            site_frac_coords,
            covalent_labels,
            bond_dist,
        )
        if failed:
            some_failed = True
        if reversed_bonds[pair_idx]:
            radius = bond_dist - radius
            frac = 1 - frac
        atomic_radii[pair_idx] = radius
        bond_fracs[pair_idx] = frac
        radius_is_covalent[pair_idx] = is_covalent

    if some_failed:
        print(
            """At least one atoms radius could not be calculated. This is usually
              due to the atom having no core electrons, likely caused by the use
              of too small of a pseudopotential. The radius will default to 1/2 the bond length"""
        )

    assert (
        bond_fracs.min() >= 0.0 and bond_fracs.max() <= 1.0
    ), "Major failure while finding bond fraction"

    return atomic_radii, bond_fracs, radius_is_covalent
