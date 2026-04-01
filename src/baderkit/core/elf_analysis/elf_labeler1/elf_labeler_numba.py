# -*- coding: utf-8 -*-

from numba import njit, prange, types
from math import erf

import numpy as np
from numpy.typing import NDArray

from baderkit.core.utilities.transforms import INT_TO_IMAGE, IMAGE_TO_INT
from baderkit.core.utilities.interpolation import linear_slice

# @njit(cache=True)
def get_core_gaussian_fit(
    total_charge_data,
    atom_labels,
    atom_images,
    basin_labels,
    core_mask,
    atom_frac_coords,
    matrix,
        ):
    shape = np.array(total_charge_data.shape, dtype=np.int64)
    nx, ny, nz = shape
    num_points = nx*ny*nz
    
    num_atoms = len(atom_frac_coords)
    num_basins = len(core_mask)
    
    atom_Q = np.zeros(num_atoms, dtype=np.float64)
    atom_M2 = np.zeros(num_atoms, dtype=np.float64)
    total_electrons = np.zeros(num_atoms, dtype=np.float64)
    basin_electrons = np.zeros(len(core_mask), dtype=np.float64)
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                atom_label = atom_labels[i,j,k]
                basin_label = basin_labels[i,j,k]
                # skip vacuum/dummy atoms
                if atom_label >= num_atoms or basin_label >= num_basins:
                    continue
                # add charge
                charge = total_charge_data[i,j,k] / num_points
                total_electrons[atom_label] += charge
                basin_electrons[basin_label] += charge
                
                # skip points that are not cores
                if core_mask[basin_label] == -1:
                    continue
                
                # get coordinates
                point_frac = np.array((i,j,k), dtype=np.float64) / shape
                point_cart = point_frac @ matrix
                
                atom_frac = atom_frac_coords[atom_label] + INT_TO_IMAGE[atom_images[i,j,k]]
                atom_cart = atom_frac @ matrix
                
                diff = point_cart - atom_cart
                r2 = diff[0]**2 + diff[1]**2 + diff[2]**2

                atom_Q[atom_label] += charge
                atom_M2[atom_label] += charge*r2
    atom_sigmas = np.zeros_like(atom_Q)
    for a in range(num_atoms):
        if atom_Q[a] > 1e-12:
            atom_sigmas[a] = np.sqrt(atom_M2[a] / (3.0 * atom_Q[a]))
        else:
            atom_sigmas[a] = 0.0
    total_electrons = total_electrons
    basin_electrons = basin_electrons
    return atom_sigmas, total_electrons, basin_electrons
                
                

# @njit(cache=True, parallel=True)
def get_basin_potential_energies(
    basin_labels,
    atom_frac_coords,
    matrix,
    nna_indices,
    charge_bond_fracs,
    total_charge_data,
    atom_sigmas,
    nuclei_charges,
    electron_charges,
    basin_electrons,
        ):
    shape = np.array(total_charge_data.shape, dtype=np.int64)
    nx, ny, nz = shape
    

    num_points = nx*ny*nz

    # get the total potential for each basin
    potential_energies = np.zeros(len(nna_indices), dtype=np.float64)

    for nna_idx in prange(len(nna_indices)):
        local_idx = nna_indices[nna_idx]
        # update the atom oxidation states as if this basins were not part of it
        charge_bond_frac = charge_bond_fracs[local_idx]
        
        local_charge = basin_electrons[local_idx]
        # remove charge related to the current basin to prevent double counting
        zeff = nuclei_charges.copy()
        for atom_idx, _, frac in charge_bond_frac:
            if atom_idx >= len(atom_frac_coords):
                continue
            zeff[int(atom_idx)] -= frac*local_charge
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # get the basin this point belongs to
                    basin_idx = basin_labels[i,j,k]
                    # skip non-nna points
                    if basin_idx != local_idx:
                        continue
                    
                    point_charge = -total_charge_data[i,j,k] / num_points
                    
                    # get this points cart coords
                    point_frac = np.array((i,j,k), dtype=np.float64) / shape
                    point_cart = point_frac @ matrix
                    
                    potential = 0.0
    
                    for idx, (atom_idx, atom_image, _) in enumerate(charge_bond_frac):

                        # skip dummy atoms at nnas in the charge density
                        if atom_idx >= len(atom_frac_coords):
                            continue
                        
                        # get this atoms frac coord
                        atom_idx = int(atom_idx)
                        atom_frac = atom_frac_coords[atom_idx] + INT_TO_IMAGE[int(atom_image)]

                        # calculate distance
                        atom_cart = atom_frac @ matrix
                        dist = np.linalg.norm(point_cart - atom_cart)
                        
                        # get effective nucleus charge
                        Z = nuclei_charges[atom_idx]
                        Q = zeff[atom_idx]
                        sigma = atom_sigmas[atom_idx]
                        
                        # calculate contribution to the potential
                        eps = 1e-8
                        if dist < eps:
                            V_nuc = Z / eps
                            V_gauss = Q * np.sqrt(2.0/np.pi) / sigma
                        else:
                            inv_r = 1.0 / dist
                            x = dist / (np.sqrt(2.0) * sigma)
                            V_nuc = Z * inv_r
                            V_gauss = Q * erf(x) * inv_r
                        
                        potential += V_nuc - V_gauss
                        
                    # add to the total potential energy for this basin
                    potential_energies[nna_idx] += potential * point_charge
        breakpoint()

    # normalize
    potential_energies *= 14.3996  # convert to potential in eV
    
    return potential_energies

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
    
    basin_dists = np.zeros(len(nna_indices), dtype=np.float64)
    basin_fracs = np.zeros(len(nna_indices), dtype=np.float64)
    
    for nna_idx in prange(len(nna_indices)):
        # skip cores
        local_idx = nna_indices[nna_idx]
        local_coords = basin_frac_coords[local_idx]
        local_cart_coords = local_coords @ matrix
        local_bond_frac = volume_bond_fracs[local_idx]
        weighted_dist = 0.0
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
        basin_fracs[nna_idx] = total_basin_frac * frac_mult
    return basin_dists, basin_fracs
           
@njit(cache=True)
def get_zeff_nna(
    atom_charges,
    atom_volumes,
    charge_bond_fracs,
    volume_bond_fracs,
    basin_charges,
    basin_volumes,
    core_basins,
        ):
    
    zeff = atom_charges.copy()
    veff = atom_volumes.copy()
    
    for local_idx in range(len(core_basins)):
        # skip cores
        if core_basins[local_idx] != -1:
            continue
        
        local_charge = basin_charges[local_idx]
        local_volume = basin_volumes[local_idx]
        for (atom_idx, atom_image, charge_frac), (_,_,volume_frac) in zip(charge_bond_fracs[local_idx], volume_bond_fracs[local_idx]):
            zeff[int(atom_idx)] -= charge_frac*local_charge
            veff[int(atom_idx)] -= volume_frac*local_volume
    return zeff, veff
        
@njit(cache=True, parallel=True)
def get_approx_coulomb_potential(
    zeff_charges,
    zeff_volumes,
    volume_bond_fracs,
    charge_bond_fracs,
    core_basins,
        ):
    
    zeffs = np.zeros(len(core_basins), dtype=np.float64)
    veffs = np.zeros(len(core_basins), dtype=np.float64)
    
    for local_idx in prange(len(core_basins)):
        # skip cores
        if core_basins[local_idx] != -1:
            continue
        volume_bond_frac = volume_bond_fracs[local_idx]
        charge_bond_frac = charge_bond_fracs[local_idx]
        
        total_charge = 0.0
        total_volume = 0.0
        
        total_charge_frac = 0.0
        total_volume_frac = 0.0
        for (atom_idx, atom_image, volume_frac), (_, _, charge_frac) in zip(volume_bond_frac, charge_bond_frac):
            if atom_idx >= len(zeff_charges):
                # this is an nna in the charge density and we don't want to include
                # it.
                continue

            # get effective charge of atom
            atom_charge = zeff_charges[int(atom_idx)]
            atom_volume = zeff_volumes[int(atom_idx)]
            
            # add fractional contribution
            total_charge += atom_charge * charge_frac
            total_charge_frac += charge_frac
            
            total_volume += atom_volume * volume_frac
            total_volume_frac += volume_frac

        # update array
        zeffs[local_idx] = total_charge / total_charge_frac
        veffs[local_idx] = total_volume / total_volume_frac
    return zeffs, veffs