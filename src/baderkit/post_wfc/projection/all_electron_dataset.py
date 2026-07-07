# -*- coding: utf-8 -*-

from pathlib import Path
import json

from pymatgen.core import Element
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

from baderkit.post_wfc.pseudopotentials.paw_dataset import PAWSpecies


@dataclass
class AESpecies:
    """
    Standardized, code-agnostic data representation of atomic core reconstruction 
    parameters. Overcomplete basis channels are canonically orthogonalized and sorted 
    by energy on initialization to yield clear, independent radial charge densities.
    """
    # --- Fields WITHOUT default values first ---
    name: str
    """Name of this all electron reference"""
    
    element: str
    """Chemical element symbol (e.g., 'Ca')."""
    
    Z: float
    """Total electrons in this pseudopotential"""
    
    basis: str
    """The basis set used to generate the reference"""
    
    functional: str
    """The XC functional used to generate the reference"""

    unrestricted: bool
    """Whether or not this reference is unrestricted (spin-polarized)"""
    
    primitives: dict
    """The primitive basis functions grouped by angular momentum"""
    
    radial_grid: NDArray
    """1D array containing the radial coordinate mesh grid points, r."""
    
    # --- Fields WITH default values second ---
    paw_species: PAWSpecies | None = None
    """The pseudopotential this species maps onto"""
    
    angular_momenta: NDArray = field(default_factory=lambda: np.empty(0, dtype=np.int_))
    """1D array of orbital angular momentum quantum numbers, l, for each active channel."""
    
    magnetic_quantum_numbers: NDArray = field(default_factory=lambda: np.empty(0, dtype=np.int_))
    """1D array of magnetic quantum numbers, m, for each active channel."""
    
    eigenvalues: NDArray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    """1D array of atomic reference state energy eigenvalues for each channel."""
    
    reference_occupations: NDArray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    """1D array of reference atomic occupations for each channel."""

    spin_channels: NDArray = field(default_factory=lambda: np.empty(0, dtype=np.int8))
    """1D array mapping spin channel projections (0 for alpha/restricted, 1 for beta)."""

    state_vectors: NDArray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    """2D array of shape (states, flat_basis) containing pre-diagonalized linear combination coefficients."""

    l_slices: dict = field(default_factory=dict)
    """Dictionary mapping angular momentum l values to their index slices within state_vectors."""

    # --- Fields initialized during post_init ---
    radial_rho: NDArray = field(init=False)
    """2D array of shape (channels, grid) containing radial charge density."""
    
    radial_tau: NDArray = field(init=False)
    """2D array of shape (channels, grid) containing radial kinetic energy density."""
    
    def __post_init__(self):
        """
        Generates the radial rho and kinetic energy density matrices on initialization.
        """
        # Final evaluation pass mapping fields onto the active grid space
        self.radial_rho, self.radial_tau = self._compute_radial_densities(alpha=1.0)
        
    def _compute_radial_densities(self, alpha: float) -> tuple[NDArray, NDArray]:
        """
        Evaluates the radial charge density (rho) and kinetic energy density (tau) 
        profiles for a given primitive exponent scaling multiplier (alpha).
        """
        BOHR_TO_ANGSTROM = 0.529177210903
        r_bohr = self.radial_grid / BOHR_TO_ANGSTROM
        
        num_states = len(self.eigenvalues)
        num_grid = len(self.radial_grid)
        radial_rho = np.zeros((num_states, num_grid), dtype=np.float64)
        radial_tau = np.zeros((num_states, num_grid), dtype=np.float64)
        
        for idx in range(num_states):
            l = self.angular_momenta[idx]
            if l in self.primitives:
                c_data = self.primitives[l]
                exps = alpha * c_data["exps"] 
                coeffs = c_data["coeffs"]
                offsets = c_data["offsets"]
                
                # Directly slice the linear state coefficients out of the flat database row
                start_l, end_l = self.l_slices[l]
                c_state = self.state_vectors[idx, start_l:end_l]
                
                dim = len(c_state)
                phi = np.zeros((dim, num_grid), dtype=np.float64)
                phi_prime = np.zeros((dim, num_grid), dtype=np.float64)
                
                r_pow_l = r_bohr ** l
                r_pow_l_plus_1 = r_bohr ** (l + 1)
                
                for p in range(dim):
                    start = offsets[p]
                    end = offsets[p+1]
                    
                    sum_0 = np.zeros(num_grid, dtype=np.float64)
                    sum_1 = np.zeros(num_grid, dtype=np.float64)
                    for k in range(start, end):
                        exp_factor = np.exp(-exps[k] * (r_bohr ** 2))
                        sum_0 += coeffs[k] * exp_factor
                        sum_1 += coeffs[k] * exps[k] * exp_factor
                    
                    phi[p, :] = r_pow_l * sum_0
                    
                    if l == 0:
                        phi_prime[p, :] = -2.0 * r_bohr * sum_1
                    else:
                        phi_prime[p, :] = (l / r_bohr) * phi[p, :] - 2.0 * r_pow_l_plus_1 * sum_1
                
                # Pure state vector transformations (Replaces matrix einsums)
                phi_state = np.dot(c_state, phi)
                phi_prime_state = np.dot(c_state, phi_prime)
                
                rho_final = (phi_state * phi_state) / (4.0 * np.pi * (BOHR_TO_ANGSTROM ** 3))
                
                angular_term = 0.0 if l == 0 else (l * (l + 1) / (r_bohr ** 2))
                raw_tau = 0.5 * ((phi_prime_state * phi_prime_state) + (phi_state * phi_state) * angular_term)
                tau_final = raw_tau / (4.0 * np.pi * (BOHR_TO_ANGSTROM ** 5))
                
                integrated_charge = np.trapezoid(4.0 * np.pi * (self.radial_grid ** 2) * rho_final, self.radial_grid)
                
                if integrated_charge > 1e-6:
                    rho_final /= integrated_charge
                    tau_final /= integrated_charge
                
                radial_rho[idx, :] = rho_final
                radial_tau[idx, :] = tau_final
                
        return radial_rho, radial_tau
    
    @property
    def max_occupations(self) -> NDArray:
        """
        1D array containing the maximum allowed electron occupation for each channel.
        Since channels track explicit m projections, an individual state holds 1.0 electron 
        if spin-unrestricted, and 2.0 electrons if spin-restricted.
        """
        if self.unrestricted:
            return np.ones_like(self.angular_momenta, dtype=np.float64)
        return 2.0 * np.ones_like(self.angular_momenta, dtype=np.float64)

    def get_valence_dataset(self, paw_species: PAWSpecies) -> "AESpecies":
        """
        Extracts a subset of the dataset containing only valence and virtual states, 
        dropping any underlying core states. 
        """
        Z = paw_species.Z
        occupied_indices = np.where(self.reference_occupations > 1e-4)[0]
        sorted_occupied = occupied_indices[np.argsort(self.eigenvalues[occupied_indices])[::-1]]
        
        valence_indices = []
        accumulated_charge = 0.0
        
        for idx in sorted_occupied:
            valence_indices.append(idx)
            accumulated_charge += self.reference_occupations[idx]
            if accumulated_charge >= Z - 1e-4:
                break
                
        virtual_indices = np.where(self.reference_occupations <= 1e-4)[0]
        keep_indices = sorted(list(valence_indices) + list(virtual_indices), key=lambda idx: self.eigenvalues[idx])
        
        return AESpecies(
            name=f"{self.name}_valence",
            element=self.element,
            Z=float(Z),
            basis=self.basis,
            functional=self.functional,
            unrestricted=self.unrestricted,
            primitives=self.primitives,
            radial_grid=self.radial_grid,
            paw_species=paw_species,
            angular_momenta=self.angular_momenta[keep_indices],
            magnetic_quantum_numbers=self.magnetic_quantum_numbers[keep_indices],
            eigenvalues=self.eigenvalues[keep_indices],
            reference_occupations=self.reference_occupations[keep_indices],
            spin_channels=self.spin_channels[keep_indices],
            state_vectors=self.state_vectors[keep_indices, :],
            l_slices=self.l_slices
        )
    
    def get_occupancies(self, min_electrons: float, max_electrons: float) -> NDArray:
        """
        Generates a 1D occupancy array of shape (n_channels,) containing the 
        number of electrons allocated to each channel within the specified 
        electron window [min_electrons, max_electrons] using the Aufbau principle.
        """
        if min_electrons > max_electrons:
            raise ValueError("min_electrons cannot be greater than max_electrons.")
        
        n_channels = len(self.eigenvalues)
        max_occ = self.max_occupations
        occs = np.zeros(n_channels, dtype=np.float64)
        
        prev_count = 0.0
        for idx in range(n_channels):
            current_count = prev_count + max_occ[idx]
            overlap = max(0.0, min(current_count, max_electrons) - max(prev_count, min_electrons))
            occs[idx] = overlap
            prev_count = current_count
                
        return occs

    def get_total_magnetic_moment(self, min_electrons: float, max_electrons: float) -> float:
        """
        Calculates the net integrated atomic magnetic moment (spin polarization: N_alpha - N_beta)
        allocated inside the given Aufbau electron boundaries.
        """
        if not self.unrestricted:
            return 0.0
        
        occupancies = self.get_occupancies(min_electrons, max_electrons)
        spin_factors = np.where(self.spin_channels == 0, 1.0, -1.0)
        return float(np.sum(occupancies * spin_factors))

    def get_radial_magnetic_density(self, min_electrons: float, max_electrons: float) -> NDArray:
        """
        Computes the radial magnetic moment density profile (spin-up density minus spin-down density)
        across the active coordinate grid mesh space.
        """
        if not self.unrestricted:
            return np.zeros(len(self.radial_grid), dtype=np.float64)
            
        occupancies = self.get_occupancies(min_electrons, max_electrons)
        spin_factors = np.where(self.spin_channels == 0, 1.0, -1.0)
        total_mag_rho = np.sum((occupancies * spin_factors) * self.radial_rho.T, axis=1)
        return total_mag_rho

    def get_radial_charge_density(self, min_electrons: float, max_electrons: float) -> NDArray:
        """
        Computes the total radial charge density profiles across a range of total electron counts.
        """
        occupancies = self.get_occupancies(min_electrons, max_electrons)
        total_rho = np.sum(occupancies * self.radial_rho.T, axis=1)
        return total_rho

    def get_radial_kinetic_energy_density(self, min_electrons: float, max_electrons: float) -> NDArray:
        """
        Computes the total radial positive-definite kinetic energy density (KED) profiles 
        across a range of total electron counts.
        """
        occupancies = self.get_occupancies(min_electrons, max_electrons)
        total_tau = np.sum(occupancies * self.radial_tau.T, axis=1)
        return total_tau

    @classmethod
    def from_file(
            cls,
            filename: str | Path,
            paw_species: PAWSpecies = None,
            cutoff_radius: float = 10.0,
            grid_points: int = 2000):
        """
        Parses compressed linear state-vector references generated by compress_checkpoint_to_radial_basis.
        """
        file_path = Path(filename)
        if not file_path.exists():
            raise FileNotFoundError(f"Missing analytical basis binary for element: {file_path}")
            
        data = np.load(file_path)
        metadata = json.loads(str(data["metadata"]))
        
        element = metadata["element"]
        element_py = Element(element)
        matrix_dims = metadata["matrix_layout_dimensions"]
        
        # Precompute the slicing masks mapping global flat array segments back to specific l-shells
        l_slices = {}
        curr_offset = 0
        for l_str in sorted(matrix_dims.keys(), key=int):
            l_val = int(l_str)
            dim = matrix_dims[l_str]
            l_slices[l_val] = (curr_offset, curr_offset + dim)
            curr_offset += dim

        # Set up grid geometry context
        if paw_species is not None:
            paw_grid = paw_species.radial_grid
            if cutoff_radius <= paw_grid[-1]:
                radial_grid = paw_grid.copy()
            else:
                diff1 = paw_grid[-1] - paw_grid[-2]
                ratio1 = paw_grid[-1] / paw_grid[-2] if paw_grid[-2] != 0.0 else 1.0
                ratio2 = paw_grid[-2] / paw_grid[-3] if paw_grid[-3] != 0.0 else 1.0
                
                extended_points = list(paw_grid)
                current_r = paw_grid[-1]
                
                if abs(ratio1 - ratio2) / ratio1 < 1e-4:
                    while current_r < cutoff_radius:
                        current_r *= ratio1
                        extended_points.append(current_r)
                else:
                    while current_r < cutoff_radius:
                        current_r += diff1
                        extended_points.append(current_r)
                        
                radial_grid = np.array(extended_points, dtype=np.float64)
        else:
            radial_grid = np.geomspace(1e-10, cutoff_radius, grid_points)
        
        # Instantiate instance directly from structured rows
        result = cls(
            name=f"{element}_{metadata['functional']}",
            element=element,
            Z=float(element_py.Z),
            basis=metadata.get("basis", None),
            functional=metadata["functional"],
            unrestricted=int(np.max(data["spin_channels"])) > 0,
            primitives=cls._flatten_basis_primitives(metadata["basis_primitives"]),
            radial_grid=radial_grid,
            paw_species=None,
            angular_momenta=data["angular_momenta"],
            magnetic_quantum_numbers=data["magnetic_quantum_numbers"],
            eigenvalues=data["energies"],
            reference_occupations=data["occupancies"],
            spin_channels=data["spin_channels"],
            state_vectors=data["packed_state_vectors"],
            l_slices=l_slices
        )
        
        if paw_species is not None:
            return result.get_valence_dataset(paw_species)
        return result

    @staticmethod
    def _flatten_basis_primitives(basis_primitives):
        """Flattens nested primitive structures into flat continuous 1D arrays and offset pointers."""
        contractions_by_l = {}
        for l_str, bas_list in basis_primitives.items():
            l = int(l_str)
            exps_list = []
            coeffs_list = []
            g_coeffs_list = []
            offsets = [0]
            
            for bas_data in bas_list:
                exps = np.array(bas_data["exponents"], dtype=np.float64)
                coeffs_mat = np.array(bas_data["coefficients"], dtype=np.float64)
                g_coeffs_mat = np.array(bas_data["g_coefficients"], dtype=np.float64)
                nctr = coeffs_mat.shape[1]
                
                for c in range(nctr):
                    exps_list.extend(exps)
                    coeffs_list.extend(coeffs_mat[:, c])
                    g_coeffs_list.extend(g_coeffs_mat[:, c])
                    offsets.append(offsets[-1] + len(exps))
                    
            contractions_by_l[l] = {
                "exps": np.array(exps_list, dtype=np.float64),
                "coeffs": np.array(coeffs_list, dtype=np.float64),
                "g_coeffs": np.array(g_coeffs_list, dtype=np.float64),
                "offsets": np.array(offsets, dtype=np.int32)
            }
        return contractions_by_l