# -*- coding: utf-8 -*-

from numba import njit, prange, types

import numpy as np
from numpy.typing import NDArray

from baderkit.core.utilities.transforms import INT_TO_IMAGE, IMAGE_TO_INT
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
    
    basin_core_dist_ratios = np.zeros(len(nna_indices), dtype=np.float64)
    basin_dists = np.zeros(len(nna_indices), dtype=np.float64)
    basin_fracs = np.zeros(len(nna_indices), dtype=np.float64)
    
    for nna_idx in prange(len(nna_indices)):
        # skip cores
        local_idx = nna_indices[nna_idx]
        local_coords = basin_frac_coords[local_idx]
        local_cart_coords = local_coords @ matrix
        local_bond_frac = volume_bond_fracs[local_idx]
        weighted_dist = 0.0
        weighted_ratio = 0.0
        total_frac = 0.0
        total_basin_frac = 0.0
        for atom_idx, atom_image, frac in local_bond_frac:
            if atom_idx > len(atom_frac_coords):
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
            
            # add this neighbors portion of the ratio
            weighted_ratio += nna_frac * frac / atom_frac
            
            # add fraction making up bond
            total_basin_frac += nna_frac * frac

            # get distance beyond atom
            atom_cart_coords = atom_coords @ matrix
            dist = np.linalg.norm(atom_cart_coords - local_cart_coords)
            # add this neighbors portion of the fraction
            weighted_dist += nna_frac * dist * frac
        
        # adjust for any fractions that had no cores
        if total_frac == 0:
            continue
        frac_mult = 1/total_frac
        # update array
        basin_core_dist_ratios[nna_idx] = weighted_ratio * frac_mult
        basin_dists[nna_idx] = weighted_dist * frac_mult
        basin_fracs[nna_idx] = total_basin_frac * frac_mult
    return basin_core_dist_ratios, basin_dists, basin_fracs
           
@njit(cache=True)
def get_zeff_nna(
    atom_charges,
    charge_bond_fracs,
    basin_charges,
    nna_indices,
        ):
    
    zeff = atom_charges.copy()
    
    for nna_idx in range(len(nna_indices)):
        # skip cores
        local_idx = nna_indices[nna_idx]
        local_charge = basin_charges[local_idx]
        for atom_idx, atom_image, frac in charge_bond_fracs[local_idx]:
            zeff[int(atom_idx)] -= frac*local_charge
    return zeff
        
@njit(cache=True, parallel=True)
def get_approx_coulomb_potential(
    zeff_charges,
    basin_charges,
    volume_bond_fracs,
    basin_frac_coords,
    atom_frac_coords,
    matrix,
    nna_indices,
    core_basins,
        ):
    
    basin_coulomb_attractions = np.zeros(len(nna_indices), dtype=np.float64)
    
    for nna_idx in prange(len(nna_indices)):
        # skip cores
        local_idx = nna_indices[nna_idx]
        local_coords = basin_frac_coords[local_idx]
        local_cart_coords = local_coords @ matrix
        local_bond_frac = volume_bond_fracs[local_idx]
        
        local_charge = basin_charges[local_idx]
        total_potential = 0.0
        for atom_idx, atom_image, frac in local_bond_frac:
            if atom_idx > len(atom_frac_coords):
                # this is an nna in the charge density and we don't want to include
                # it.
                continue

            # get effective charge of atom
            atom_charge = zeff_charges[int(atom_idx)]
            
            # get distance to the atom
            atom_coords = atom_frac_coords[int(atom_idx)] + INT_TO_IMAGE[int(atom_image)]
            atom_cart_coords = atom_coords @ matrix
            dist = np.linalg.norm(atom_cart_coords - local_cart_coords)
            
            # calculate coulomb potential
            coulomb_pot = local_charge * atom_charge / (dist)
            
            total_potential += coulomb_pot

        # update array
        basin_coulomb_attractions[nna_idx] = total_potential
    return basin_coulomb_attractions