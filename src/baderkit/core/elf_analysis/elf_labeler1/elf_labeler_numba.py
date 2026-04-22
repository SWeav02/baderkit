# -*- coding: utf-8 -*-

from numba import njit, prange, types
from math import erf
from scipy.special import erf as scipy_erf

import numpy as np
from numpy.typing import NDArray

from baderkit.core.utilities.transforms import INT_TO_IMAGE, IMAGE_TO_INT
from baderkit.core.utilities.interpolation import linear_slice

@njit(cache=True)
def get_valence_potentials(
    charge_data,
    potential_data,
    basin_labels,
    num_basins,
        ):
    shape = np.array(charge_data.shape, dtype=np.int64)
    nx, ny, nz = shape
    num_points = nx*ny*nz
    basin_potentials = np.zeros(num_basins, dtype=np.float64)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                label = basin_labels[i,j,k]
                if label >= num_basins:
                    continue
                # add potential in this voxel
                # (e*num_points) * eV = eV * num_points
                basin_potentials[label] += 0.5 * charge_data[i,j,k] * potential_data[i,j,k]
    # get the weighted potential felt by this basin
    basin_potentials /= num_points
    return basin_potentials

@njit(cache=True)
def get_avg_potentials(
    potential_data,
    basin_labels,
    num_basins,
        ):
    shape = np.array(potential_data.shape, dtype=np.int64)
    nx, ny, nz = shape
    basin_potentials = np.zeros(num_basins, dtype=np.float64)
    basin_counts = np.zeros(num_basins, dtype=np.int64)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                label = basin_labels[i,j,k]
                if label >= num_basins:
                    continue
                # add potential in this voxel
                # (e*num_points) * eV = eV * num_points
                basin_potentials[label] += potential_data[i,j,k]
                basin_counts[label] += 1
    # get the weighted potential felt by this basin
    basin_potentials /= basin_counts
    return basin_potentials


@njit(cache=True)
def get_test(
    potential_data,
    charge_data,
    elf_data,
    basin_labels,
    atom_frac_coords,
    bond_fractions,
    nna_mask,
    matrix,
    num_atoms
        ):
    atom_coords = []
    for basin, fracs in enumerate(bond_fractions):
        coords = []
        idx = 0
        for _,(atom, image, frac) in enumerate(fracs):
            # skip nnas in the charge density
            if atom>=num_atoms:
                continue
            frac_coord = atom_frac_coords[int(atom)] + INT_TO_IMAGE[int(image)]
            cart_coord = frac_coord @ matrix
            coords.append(cart_coord)
        coords_array = np.empty((len(coords), 3), dtype=np.float64)
        for idx, coord in enumerate(coords):
            coords_array[idx] = coord
        atom_coords.append(coords_array)

    num_basins = len(atom_coords)


    shape = np.array(potential_data.shape, dtype=np.int64)
    nx, ny, nz = shape
    num_points = nx*ny*nz
    test_vals = np.zeros(num_basins, dtype=np.float64)
    test_norms = np.zeros(num_basins, dtype=np.float64)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                label = basin_labels[i,j,k]
                if not nna_mask[label]:
                    continue

                coords = np.array((i,j,k),dtype=np.float64)/shape
                cart = coords @ matrix
                charge = charge_data[i,j,k]/num_points
                elf = elf_data[i,j,k]
                # elec_pot = potential_data[i,j,k]
                for atom_cart in atom_coords[label]:
                    dist = np.linalg.norm(cart-atom_cart)
                    potential = elf/dist
                    test_vals[label] += potential
                    test_norms[label] += elf

    test_norms[np.where(test_norms==0)[0]] = 1
    # get the weighted potential felt by this basin
    test_vals /= test_norms
    return test_vals

def solve_poisson(
    data,
    matrix,
    nuclei_positions,
    nuclei_charges,
    sigma=0.1,
):
    """
    Solve Poisson's equation on a periodic lattice using FFT.
    Works for a general parallelepiped cell.

    Assumes CHGCAR-style input:
    - data initially stores (rho * Vcell)
    - dividing by N gives electrons per voxel

    Returns electrostatic potential in eV.
    """

    # --- Lattice vectors ---
    a1, a2, a3 = matrix
    nx, ny, nz = data.shape
    N = nx * ny * nz

    # --- Cell volume ---
    volume = np.dot(a1, np.cross(a2, a3))

    # --- Voxel volume ---
    dV = volume / N

    # --- Grid spacing (approx, for sigma safety only) ---
    # dx = np.linalg.norm(a1) / nx
    # dy = np.linalg.norm(a2) / ny
    # dz = np.linalg.norm(a3) / nz
    # sigma = max(sigma, 2.5 * max(dx, dy, dz))

    # =========================
    # Charge density (CORRECT)
    # =========================

    # CHGCAR → electrons per voxel → divide by voxel volume → e/Å^3
    rho = (data / N) / dV

    # --- Lattice transforms ---
    A = matrix.T
    Ainv = np.linalg.inv(A)

    # --- Fractional grid ---
    x = np.linspace(0, 1, nx, endpoint=False)
    y = np.linspace(0, 1, ny, endpoint=False)
    z = np.linspace(0, 1, nz, endpoint=False)

    xg, yg, zg = np.meshgrid(x, y, z, indexing='ij')
    r_frac = np.stack([xg, yg, zg], axis=-1)

    # =========================
    # Add Gaussian nuclei (CORRECT normalization)
    # =========================
    for q, R_cart in zip(nuclei_charges, nuclei_positions):

        R_frac = Ainv @ R_cart

        dr_frac = r_frac - R_frac
        dr_frac -= np.round(dr_frac)

        dr_cart = dr_frac @ matrix
        r2 = np.sum(dr_cart**2, axis=-1)

        gaussian = np.exp(-r2 / (2 * sigma**2))

        # Normalize so integral = 1
        gaussian /= gaussian.sum() * dV

        # Add nuclear charge
        rho -= q * gaussian

    # =========================
    # FFT solve
    # =========================

    rho_k = np.fft.fftn(rho)

    # --- Reciprocal lattice ---
    b1 = 2 * np.pi * np.cross(a2, a3) / volume
    b2 = 2 * np.pi * np.cross(a3, a1) / volume
    b3 = 2 * np.pi * np.cross(a1, a2) / volume

    # --- Correct k-grid (integer harmonics) ---
    hx = np.fft.fftfreq(nx, d=1.0 / nx)
    hy = np.fft.fftfreq(ny, d=1.0 / ny)
    hz = np.fft.fftfreq(nz, d=1.0 / nz)

    hxg, hyg, hzg = np.meshgrid(hx, hy, hz, indexing='ij')
    kvecs = hxg[..., None] * b1 + hyg[..., None] * b2 + hzg[..., None] * b3

    k2 = np.sum(kvecs**2, axis=-1)
    k2[0, 0, 0] = 1.0  # avoid division by zero

    # --- Poisson equation in reciprocal space ---
    phi_k = 4 * np.pi * rho_k / k2
    phi_k[0, 0, 0] = 0.0  # zero-average potential

    # --- Back transform ---
    phi = np.fft.ifftn(phi_k).real

    # --- Convert to eV ---
    return phi * 14.3996


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