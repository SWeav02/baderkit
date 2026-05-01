# -*- coding: utf-8 -*-

from numba import njit, prange, types
from math import erf
from scipy.special import erf as scipy_erf

import numpy as np
from numpy.typing import NDArray

from baderkit.core.utilities.transforms import INT_TO_IMAGE
from baderkit.core.utilities.interpolation import linear_slice



@njit(cache=True, parallel=True)
def get_core_dist_ratios(
    labels,
    basin_frac_coords,
    atom_frac_coords,
    matrix,
    nna_indices,
    core_basins,
    volume_bond_fracs,
        ):

    basin_fracs = np.zeros(len(nna_indices), dtype=np.float64)

    for nna_idx in prange(len(nna_indices)):
        # skip cores
        local_idx = nna_indices[nna_idx]
        local_coords = basin_frac_coords[local_idx]
        local_bond_frac = volume_bond_fracs[local_idx]
        total_basin_frac = 0.0

        total_frac = 0.0
        for atom_idx, atom_image, frac in local_bond_frac:
            if atom_idx >= len(atom_frac_coords):
                # this is an nna in the charge density and we don't want to include
                # it.
                continue
            # TODO: Also skip anions?
            atom_coords = atom_frac_coords[int(atom_idx)] + INT_TO_IMAGE[int(atom_image)]
            # labels between the coords
            label_line = linear_slice(labels, atom_coords, local_coords, method="nearest")
            # get the last point that is part of the core
            for idx, i in enumerate(label_line):
                if core_basins[int(i)] == -1:
                    break
            # we found no core and we skip this point
            if idx == 0:
                continue
            total_frac += frac
            # get fraction of bond belonging to the nna
            atom_frac = idx / (len(label_line)-1)
            nna_frac = 1-atom_frac

            # add fraction making up bond
            total_basin_frac += nna_frac * frac

        # adjust for any fractions that had no cores
        if total_frac == 0:
            continue
        frac_mult = 1/total_frac
        # update arrays
        basin_fracs[nna_idx] = total_basin_frac * frac_mult

    return basin_fracs

@njit(cache=True, parallel=True)
def get_core_dists(
    labels,
    basin_frac_coords,
    atom_frac_coords,
    matrix,
    nna_indices,
    core_basins,
    volume_bond_fracs,
        ):

    basin_dists = np.zeros(len(nna_indices), dtype=np.float64)

    for nna_idx in prange(len(nna_indices)):
        # skip cores
        local_idx = nna_indices[nna_idx]
        local_coords = basin_frac_coords[local_idx]
        local_cart_coords = local_coords @ matrix
        local_bond_frac = volume_bond_fracs[local_idx]
        weighted_dist = 0.0

        total_frac = 0.0
        for atom_idx, atom_image, frac in local_bond_frac:
            if atom_idx >= len(atom_frac_coords):
                # this is an nna in the charge density and we don't want to include
                # it.
                continue
            atom_coords = atom_frac_coords[int(atom_idx)] + INT_TO_IMAGE[int(atom_image)]
            total_frac += frac
            # get distance to atom
            atom_cart_coords = atom_coords @ matrix
            dist = np.linalg.norm(atom_cart_coords - local_cart_coords)
            # add this neighbors portion of the fraction
            weighted_dist += dist * frac

        # adjust for any fractions that had no cores
        if total_frac == 0:
            continue
        frac_mult = 1/total_frac
        # update arrays
        basin_dists[nna_idx] = weighted_dist * frac_mult

    return basin_dists

