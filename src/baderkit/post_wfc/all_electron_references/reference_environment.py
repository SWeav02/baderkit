#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import numpy as np

class AtomicReferenceEnvironment:
    """
    Manages the non-bonding atomic reference states by parsing compressed analytical 
    basis binaries with pre-applied primitive normalization constants.
    """
    # Standard CODATA conversion factor used by major DFT codes (VASP/QE)
    BOHR_TO_ANGSTROM = 0.5291772109

    def __init__(
            self, 
            structure, 
            valence_counts_map, 
            basis_dir=None,
            ):
        """
        Parameters:
        -----------
        structure : pymatgen.core.Structure
            Crystallographic structure defining cell dimensions and positions.
        valence_counts_map : dict
            The neutral valence electron count (Z_val) for each element's pseudopotential.
        basis_dir : str or Path, optional
            Directory containing the generated {Element}.npz basis files.
        """
        self.structure = structure
        self.z_val_map = valence_counts_map
        self.basis_dir = Path(basis_dir) if basis_dir is not None else Path(__file__).parent
        
        self.missing_map = self.z_val_map
        self.lattice_matrix = self.structure.lattice.matrix
        self.total_neutral_charge = sum(self.z_val_map[site.specie.symbol] for site in self.structure)
        self.total_cell_missing_weight = sum(self.missing_map[site.specie.symbol] for site in self.structure)
        
        self.atomic_basis_headers = {}
        self.valence_pools = {}
        self.unrestricted_flags = {}
        
        self._load_and_initialize_basis_pool()

    def _load_and_initialize_basis_pool(self):
        """Parses NPZ binaries for unique elements and isolates the valence pools."""
        unique_elements = set(site.specie.symbol for site in self.structure)
        
        for element in unique_elements:
            file_path = self.basis_dir / f"{element}.npz"
            if not file_path.exists():
                raise FileNotFoundError(f"Missing analytical basis binary for element: {file_path}")
                
            data = np.load(file_path)
            metadata = json.loads(str(data["metadata"]))
            
            energies = data["energies"]
            occupancies = data["occupancies"]
            spin_channels = data["spin_channels"]
            packed_d = data["packed_d_matrices"]
            
            is_unrestricted = int(np.max(spin_channels)) > 0
            self.unrestricted_flags[element] = is_unrestricted
            self.atomic_basis_headers[element] = self._flatten_basis_primitives(metadata["basis_primitives"])
            
            Z_val = self.z_val_map[element]
            occupied_idx = np.where(occupancies > 1e-4)[0]
            occupied_idx = occupied_idx[np.argsort(energies[occupied_idx])[::-1]]
            
            valence_indices = []
            accumulated_charge = 0.0
            for idx in occupied_idx:
                valence_indices.append(idx)
                accumulated_charge += occupancies[idx]
                if accumulated_charge >= Z_val - 1e-4:
                    break
                    
            virtual_indices = np.where(occupancies <= 1e-4)[0]
            full_pool_indices = list(valence_indices) + list(virtual_indices)
            full_pool_indices = sorted(full_pool_indices, key=lambda idx: energies[idx])
            
            pool_states = []
            matrix_dims = metadata["matrix_layout_dimensions"]
            
            for idx in full_pool_indices:
                pool_states.append({
                    "energy": energies[idx],
                    "spin": spin_channels[idx],
                    "d_blocks": self._unpack_triu_matrices(packed_d[idx], matrix_dims)
                })
                
            self.valence_pools[element] = pool_states

    def _unpack_triu_matrices(self, packed_vector, matrix_dims):
        """Reconstructs full symmetric density matrices from packed upper-triangles."""
        d_blocks = {}
        current_idx = 0
        for l_str in sorted(matrix_dims.keys(), key=int):
            l = int(l_str)
            dim = matrix_dims[l_str]
            size = (dim * (dim + 1)) // 2
            vec = packed_vector[current_idx:current_idx + size]
            current_idx += size
            
            mat = np.zeros((dim, dim))
            iu = np.triu_indices(dim)
            mat[iu] = vec
            mat = mat + mat.T - np.diag(np.diag(mat))
            d_blocks[l] = mat
        return d_blocks

    def _flatten_basis_primitives(self, basis_primitives):
        """Flattens nested primitive structures into quick-eval array tuples grouped by l."""
        contractions_by_l = {}
        for l_str, bas_list in basis_primitives.items():
            l = int(l_str)
            contractions_by_l[l] = []
            for bas_data in bas_list:
                exps = np.array(bas_data["exponents"])
                coeffs_mat = np.array(bas_data["coefficients"])
                nctr = coeffs_mat.shape[1]
                for c in range(nctr):
                    contractions_by_l[l].append((exps, coeffs_mat[:, c]))
        return contractions_by_l

    def _fill_valence_shell_smeared(self, element, target_charge):
        """Fills orbitals up to target_charge with 0-smearing, maintaining degeneracy symmetry."""
        pool = self.valence_pools[element]
        is_unrestricted = self.unrestricted_flags[element]
        max_capacity_per_orbital = 1.0 if is_unrestricted else 2.0
        
        unique_energies = []
        energy_groups = []
        for idx, state in enumerate(pool):
            E = state["energy"]
            found = False
            for u_idx, u_E in enumerate(unique_energies):
                if abs(u_E - E) < 1e-4:
                    energy_groups[u_idx].append(idx)
                    found = True
                    break
            if not found:
                unique_energies.append(E)
                energy_groups.append([idx])
                
        remaining_charge = target_charge
        assigned_occupancies = np.zeros(len(pool))
        
        for group in energy_groups:
            num_orbitals = len(group)
            group_max_capacity = num_orbitals * max_capacity_per_orbital
            
            fill_amount = min(remaining_charge, group_max_capacity)
            fill_per_orbital = fill_amount / num_orbitals
            
            for idx in group:
                assigned_occupancies[idx] = fill_per_orbital
                
            remaining_charge -= fill_amount
            if remaining_charge <= 0:
                break
                
        return assigned_occupancies

    def _evaluate_radial_density(self, element, occupancies, r):
        """Evaluates the analytical spherical electron density at distance r (input in Angstroms)."""
        if r < 1e-8:
            r = 1e-8
            
        r_bohr = r / self.BOHR_TO_ANGSTROM
        contractions = self.atomic_basis_headers[element]
        pool = self.valence_pools[element]
        
        # 1. Evaluate radial profiles (coefficients are already pre-normalized)
        R_vals = {}
        for l, basis_list in contractions.items():
            R_l = np.zeros(len(basis_list))
            for p, (exps, coeffs) in enumerate(basis_list):
                R_l[p] = (r_bohr**l) * np.sum(coeffs * np.exp(-exps * (r_bohr**2)))
            R_vals[l] = R_l
            
        # 2. Contract density blocks
        total_rho_atomic_units = 0.0
        for idx, state in enumerate(pool):
            occ = occupancies[idx]
            if np.abs(occ) <= 1e-5:
                continue
                
            for l, D_l in state["d_blocks"].items():
                R_l = R_vals[l]
                state_rho_l = np.dot(R_l, np.dot(D_l, R_l))
                total_rho_atomic_units += occ * state_rho_l
                
        # CRITICAL FIX: Divide by 4 * pi to convert solid-angle integrated density to point density
        return total_rho_atomic_units / (4 * np.pi * (self.BOHR_TO_ANGSTROM ** 3))
    
    def get_maximum_cell_charge_capacity(self):
        """Calculates the absolute maximum cell valence charge limit before orbital saturation."""
        max_cell_charges = []
        unique_elements = set(site.specie.symbol for site in self.structure)
        
        for element in unique_elements:
            pool = self.valence_pools[element]
            is_unrestricted = self.unrestricted_flags[element]
            max_capacity_per_orbital = 1.0 if is_unrestricted else 2.0
            
            total_element_capacity = len(pool) * max_capacity_per_orbital
            weight_ratio = self.missing_map[element] / self.total_cell_missing_weight
            
            if weight_ratio > 1e-10:
                max_delta_q_cell = (total_element_capacity - self.z_val_map[element]) / weight_ratio
                max_cell_charges.append(self.total_neutral_charge + max_delta_q_cell)
                
        return min(max_cell_charges) if max_cell_charges else np.inf

    def calculate_non_bonding_density(self, frac_coord, total_valence_charge, r_cut=8.0):
        """Calculates reference non-bonding spatial density at a given fractional coordinate."""
        delta_q_cell = total_valence_charge - self.total_neutral_charge
        
        element_occupancy_profiles = {}
        unique_elements = set(site.specie.symbol for site in self.structure)
        
        for element in unique_elements:
            allocated_atom_charge = self.z_val_map[element] + delta_q_cell * (self.missing_map[element] / self.total_cell_missing_weight)
            element_occupancy_profiles[element] = self._fill_valence_shell_smeared(element, allocated_atom_charge)

        total_density = 0.0
        target_cart = np.dot(frac_coord, self.lattice_matrix)
        
        box_limit = int(np.ceil(r_cut / np.min(np.linalg.norm(self.lattice_matrix, axis=1))))
        search_range = range(-box_limit, box_limit + 1)
        
        for site in self.structure:
            element = site.specie.symbol
            occs = element_occupancy_profiles[element]
            atom_base_frac = site.frac_coords
            
            for dx in search_range:
                for dy in search_range:
                    for dz in search_range:
                        image_shift = np.array([dx, dy, dz])
                        image_frac = atom_base_frac + image_shift
                        
                        dr_cart = np.dot(image_frac, self.lattice_matrix) - target_cart
                        r = np.linalg.norm(dr_cart)
                        
                        if r < r_cut:
                            total_density += self._evaluate_radial_density(element, occs, r)
                            
        return total_density
    
    def calculate_sequential_reference_deltas(self, frac_coord, q_bounds, r_cut=8.0):
        """Calculates step-by-step non-bonding reference density changes tracking q_bounds."""
        num_steps = len(q_bounds) - 1
        delta_q_intervals = q_bounds - self.total_neutral_charge
        
        unique_elements = set(site.specie.symbol for site in self.structure)
        
        element_delta_occs = {}
        for element in unique_elements:
            pool_size = len(self.valence_pools[element])
            occs_matrix = np.zeros((num_steps + 1, pool_size))
            weight_ratio = self.missing_map[element] / self.total_cell_missing_weight
            
            for i in range(num_steps + 1):
                alloc_q = self.z_val_map[element] + delta_q_intervals[i] * weight_ratio
                occs_matrix[i, :] = self._fill_valence_shell_smeared(element, alloc_q)
                
            element_delta_occs[element] = occs_matrix[1:, :] - occs_matrix[:-1, :]
            
        total_delta_rhos_atomic_units = np.zeros(num_steps)
        target_cart = np.dot(frac_coord, self.lattice_matrix)
        
        box_limit = int(np.ceil(r_cut / np.min(np.linalg.norm(self.lattice_matrix, axis=1))))
        search_range = range(-box_limit, box_limit + 1)
        
        for site in self.structure:
            element = site.specie.symbol
            delta_occs = element_delta_occs[element]
            atom_base_frac = site.frac_coords
            contractions = self.atomic_basis_headers[element]
            pool = self.valence_pools[element]
            
            for dx in search_range:
                for dy in search_range:
                    for dz in search_range:
                        image_shift = np.array([dx, dy, dz])
                        image_frac = atom_base_frac + image_shift
                        
                        dr_cart = np.dot(image_frac, self.lattice_matrix) - target_cart
                        r = np.linalg.norm(dr_cart)
                        
                        if r < r_cut:
                            r_eval = max(r, 1e-8)
                            r_bohr = r_eval / self.BOHR_TO_ANGSTROM
                            
                            R_vals = {}
                            for l, basis_list in contractions.items():
                                R_l = np.zeros(len(basis_list))
                                for p, (exps, coeffs) in enumerate(basis_list):
                                    R_l[p] = (r_bohr**l) * np.sum(coeffs * np.exp(-exps * (r_bohr**2)))
                                R_vals[l] = R_l
                                
                            orbital_profiles = np.zeros(len(pool))
                            for idx, state in enumerate(pool):
                                state_rho = 0.0
                                for l, D_l in state["d_blocks"].items():
                                    R_l = R_vals[l]
                                    state_rho += np.dot(R_l, np.dot(D_l, R_l))
                                orbital_profiles[idx] = state_rho
                                
                            total_delta_rhos_atomic_units += np.dot(delta_occs, orbital_profiles)
                            
        # CRITICAL FIX: Divide by 4 * pi to convert solid-angle integrated density to point density
        return total_delta_rhos_atomic_units / (4 * np.pi * (self.BOHR_TO_ANGSTROM ** 3))