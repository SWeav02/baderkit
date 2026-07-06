# -*- coding: utf-8 -*-

from pathlib import Path
from functools import cached_property
import numpy as np
from numpy.typing import NDArray
from .reference_dataset import AESpecies
from .reference_numba import (
    find_active_periodic_atoms,
    find_all_voxels_parallel,
    broadcast_atoms_to_grid,
)


class AtomicReferenceEnvironment:
    """
    Manages the non-bonding atomic reference states by parsing compressed analytical 
    basis binaries with pre-applied primitive normalization constants. Organized 
    strictly around a valence-only pseudopotential architecture.
    """
    # Standard CODATA conversion factor used by major DFT codes (VASP/QE)
    BOHR_TO_ANGSTROM = 0.5291772109

    def __init__(
            self, 
            structure, 
            valence_counts, 
            pdos_data,
            cutoff_radius=8.0,
            basis_dir=None,
            ):
        """
        Parameters:
        -----------
        structure : pymatgen.core.Structure
            Crystallographic structure defining cell dimensions and positions.
        valence_counts : dict
            The neutral valence electron count (Z_val) for each element's pseudopotential.
        pdos_data : dict
            Dictionary containing "energy_grid" and "projections" for individual atoms.
        cutoff_radius : float
            The interaction radius boundary in Angstroms. Default: 8.0
        basis_dir : str or Path, optional
            Directory containing the generated {Element}.npz basis files.
        """
        self.structure = structure
        self.valence_counts = valence_counts
        self.cutoff_radius = cutoff_radius
        self.basis_dir = Path(basis_dir) if basis_dir is not None else Path(__file__).parent
        
        self.lattice_matrix = self.structure.lattice.matrix
        self.total_charge = sum(self.valence_counts[site.specie.symbol] for site in self.structure)
        
        self.atomic_basis_headers = {}
        self.valence_pools = {}
        self.unrestricted_flags = {}
        self.subshell_labels = {}
        
        # Dictionary-based multidimensional caching framework
        self._cache_voxel_footprints = {}
        
        self._load_and_initialize_basis_pool()
        
        # Precompute the normalized charge allocation profiles from the PDOS input
        self.norm_total_integral, self.atom_integrals = self._process_pdos(pdos_data)

    ###########################################################################
    # INITIALIZATION MODULES
    ###########################################################################

    def _load_and_initialize_basis_pool(self):
        """
        Parses NPZ binaries, determines absolute spectroscopic labels from the full 
        untruncated basis, and drops core states to maintain a pure pseudopotential pool.
        """
        unique_elements = set(site.specie.symbol for site in self.structure)
        atom_bases = {}
        
        for element in unique_elements:
            file_path = self.basis_dir / f"{element}.npz"
            if not file_path.exists():
                raise FileNotFoundError(f"Missing analytical basis binary for element: {file_path}")
            basis = AESpecies.from_file(file_path)
            basis = basis.get_valence_dataset(self.valence_counts[element])
            atom_bases[element] = basis
        self.atom_bases = atom_bases

    @cached_property
    def semi_periodic_atoms(self):
        """Identifies and caches every Cartesian image position inside the cutoff."""
        if not hasattr(self, '_cache_semi_periodic_atoms'):
            unique_elements = list(self.structure.symbol_set)
            element_to_idx = {elem: i for i, elem in enumerate(unique_elements)}
            
            num_atoms = len(self.structure)
            base_frac_coords = np.zeros((num_atoms, 3), dtype=np.float64)
            atom_types = np.zeros(num_atoms, dtype=np.int32)
            
            for i, site in enumerate(self.structure):
                base_frac_coords[i] = site.frac_coords
                atom_types[i] = element_to_idx[site.specie.symbol]
                
            cart_coords, element_indices, base_indices = find_active_periodic_atoms(
                self.lattice_matrix, base_frac_coords, atom_types, self.cutoff_radius
            )
            
            self._cache_semi_periodic_atoms = {
                "cart_coords": cart_coords,
                "element_indices": element_indices,
                "base_indices": base_indices,
                "element_mapping": unique_elements
            }
        return self._cache_semi_periodic_atoms

    def get_voxel_footprints(self, grid_dims):
        """
        Calculates and caches the voxel indices and scalar distances for all 
        periodic atoms based on the requested grid dimension layout.
        """
        grid_key = tuple(grid_dims)
        
        if grid_key not in self._cache_voxel_footprints:
            spa = self.semi_periodic_atoms
            atom_carts = spa["cart_coords"]
            g_dims = np.array(grid_dims, dtype=np.int64)
            
            all_indices, all_distances = find_all_voxels_parallel(
                atom_carts, self.lattice_matrix, g_dims, self.cutoff_radius
            )
            self._cache_voxel_footprints[grid_key] = (all_indices, all_distances)
            
        return self._cache_voxel_footprints[grid_key]

    def _process_pdos(self, pdos_data: dict):
        """
        Normalizes individual atom PDOS arrays so the occupied states integrate exactly
        to each atom's valence count, then computes the cumulative total cell charge profile.
        """
        energies = pdos_data["energy_grid"]
        
        dx = np.diff(energies)
        
        def cumulative_integrate(y):
            avg_y = 0.5 * (y[:-1] + y[1:])
            integral = np.zeros_like(y)
            integral[1:] = np.cumsum(avg_y * dx)
            return integral

        # Compute unnormalized total cell DOS
        raw_total_dos = np.zeros_like(energies)
        for key, p_sub in pdos_data.items():
            try:
                int(key)
            except:
                continue
            raw_total_dos += p_sub
            
        # Locate the neutral Fermi level
        raw_total_integral = cumulative_integrate(raw_total_dos)
        E_F = np.interp(self.total_charge, raw_total_integral, energies)
        
        atom_integrals = {}
        norm_total_dos = np.zeros_like(energies)
        
        # Normalize each atom's trajectory individually
        for i_atom, p_sub in pdos_data.items():
            try:
                int(i_atom)
            except:
                continue
            symbol = self.structure[i_atom].specie.symbol
            z_val = self.valence_counts[symbol]
            
            raw_atom_integral = cumulative_integrate(p_sub)
            raw_occ = np.interp(E_F, energies, raw_atom_integral)
            
            norm_factor = z_val / raw_occ if raw_occ > 1e-12 else 1.0
            p_norm = p_sub * norm_factor
            
            norm_total_dos += p_norm
            atom_integrals[i_atom] = cumulative_integrate(p_norm)
            
        norm_total_integral = cumulative_integrate(norm_total_dos)
        return norm_total_integral, atom_integrals
    
    def get_partial_radial_charge_densities(
            self, 
            min_charge: float, 
            max_charge: float,
            grid_spacing: float = 0.05,
            ) -> dict[int, NDArray]:
        """
        Computes the total radial charge density profiles across the PDOS-resolved local 
        electron ranges mapped to a specific global cell charge range.
        """
        min_charge, max_charge = self._clean_ranges(min_charge, max_charge)
        
        partial_densities = {}
        for i_atom, alloc_integral in self.atom_integrals.items():
            atom_qs = np.interp([min_charge, max_charge], self.norm_total_integral, alloc_integral)
            symbol = self.structure[i_atom].specie.symbol
            basis = self.atom_bases[symbol]
            
            partial_densities[i_atom] = basis.get_radial_charge_density(
                min_electrons=atom_qs[0], 
                max_electrons=atom_qs[1], 
            )
        return partial_densities
    
    def get_partial_radial_kinetic_energy_densities(
            self, 
            min_charge: float, 
            max_charge: float,
            ) -> dict[int, NDArray]:
        """
        Computes the positive-definite radial kinetic energy density profiles 
        across the PDOS-resolved local electron ranges.
        """
        min_charge, max_charge = self._clean_ranges(min_charge, max_charge)
        
        partial_keds = {}
        for i_atom, alloc_integral in self.atom_integrals.items():
            atom_qs = np.interp([min_charge, max_charge], self.norm_total_integral, alloc_integral)
            symbol = self.structure[i_atom].specie.symbol
            basis = self.atom_bases[symbol]
            
            partial_keds[i_atom] = basis.get_radial_kinetic_energy_density(
                min_electrons=atom_qs[0], 
                max_electrons=atom_qs[1], 
            )
        return partial_keds
            
    ###########################################################################
    # CORE TRACKING & COORDINATE PASSES
    ###########################################################################

    def calculate_non_bonding_density_at_point(
            self, 
            frac_coord, 
            min_charge=None, 
            max_charge=None,
            ):
        """
        Calculates non-bonding reference valence density at a specific coordinate location
        using atom-resolved PDOS charge mapping tracking rules.
        """
        min_charge, max_charge = self._clean_ranges(min_charge, max_charge)
        
        spa = self.semi_periodic_atoms
        atom_carts = spa["cart_coords"]
        atom_bases_indices = spa["base_indices"]
        
        target_cart = np.array(frac_coord, dtype=np.float64) @ self.lattice_matrix
        
        delta_vectors = atom_carts - target_cart
        distances = np.sqrt(np.sum(delta_vectors**2, axis=1))
        
        mask = distances < self.cutoff_radius
        filtered_distances = distances[mask]
        filtered_base_indices = atom_bases_indices[mask]
        
        partial_charges = self.get_partial_radial_charge_densities(min_charge, max_charge)

        any_atom = list(partial_charges.keys())[0]
        n_configurations = partial_charges[any_atom].shape[0]
        total = np.zeros(n_configurations, dtype=np.float64)
        
        for dist, i_atom in zip(filtered_distances, filtered_base_indices):
            symbol = self.structure[i_atom].specie.symbol
            r_grid = self.atom_bases[symbol].radial_grid
            rho_matrix = partial_charges[i_atom]  # shape: (n_configurations, n_grid)
            
            for c in range(n_configurations):
                total[c] += np.interp(dist, r_grid, rho_matrix[c, :])
        
        return total
    
    def calculate_density_at_point_vs_charge(
        self, 
        frac_coord: NDArray, 
        min_charge: float, 
        max_charge: float, 
        num_points: int = 2000,
    ) -> NDArray:
        """Calculates the charge density at a point for a range of total cell charges."""
        min_charge, max_charge = self._clean_ranges(min_charge, max_charge)
        target_charges = np.linspace(min_charge, max_charge, num_points)
        
        density_values = self.calculate_non_bonding_density_at_point(
            frac_coord, 
            min_charge=min_charge, 
            max_charge=max_charge
        )
        
        if len(density_values) != num_points:
            original_charges = np.linspace(min_charge, max_charge, len(density_values))
            density_values = np.interp(target_charges, original_charges, density_values)
            
        return np.column_stack((target_charges, density_values))
    
    def calculate_density_at_point_vs_energy(
        self, 
        frac_coord: NDArray, 
        energy_charge_array: NDArray, 
        num_interp_points: int = 2000,
    ) -> NDArray:
        """Maps input cell energies to their corresponding non-bonding charge densities."""
        energies = energy_charge_array[:, 0]
        charges = energy_charge_array[:, 1]
        
        min_charge = np.min(charges)
        max_charge = np.max(charges)
        
        if np.abs(max_charge - min_charge) < 1e-6:
            density_vs_charge = self.calculate_density_at_point_vs_charge(
                frac_coord, min_charge=min_charge, max_charge=max_charge, num_points=1
            )
            constant_density = density_vs_charge[0, 1]
            densities = np.full(len(energies), constant_density, dtype=np.float64)
        else:
            density_vs_charge = self.calculate_density_at_point_vs_charge(
                frac_coord, 
                min_charge=min_charge, 
                max_charge=max_charge, 
                num_points=num_interp_points
            )
            densities = np.interp(charges, density_vs_charge[:, 0], density_vs_charge[:, 1])
            
        return np.column_stack((energies, densities))

    def generate_charge_density_grid(
                self, 
                grid_dims, 
                min_charge=None, 
                max_charge=None,
                energy_cutoff=None,
                ):
        """Generates the total non-bonding reference charge density grid using custom PDOS weights."""
        min_charge, max_charge = self._clean_ranges(min_charge, max_charge)

        spa = self.semi_periodic_atoms
        # Redirect Numba unique identity maps from global elements to distinct structural atom indexes
        atom_types = spa["base_indices"]
        
        all_indices, all_distances = self.get_voxel_footprints(grid_dims)
        partial_charges = self.get_partial_radial_charge_densities(min_charge, max_charge)

        rho_matrices_list = []
        r_grids_list = []               
        
        # Build 1D lists matching the exact length of the structural atoms count
        for i_atom in range(len(self.structure)):
            symbol = self.structure[i_atom].specie.symbol
            r_grids_list.append(self.atom_bases[symbol].radial_grid)
            rho_matrices_list.append(partial_charges[i_atom])
        
        g_dims = np.array(grid_dims, dtype=np.int64)
        
        rho_3d = broadcast_atoms_to_grid(
            g_dims,
            atom_types,
            all_indices,
            all_distances,
            rho_matrices_list,
            r_grids_list
        )
        
        if energy_cutoff is not None:
            ngx, ngy, ngz = grid_dims
            rho_G = np.fft.fftn(rho_3d)
            recip_lattice = self.structure.lattice.reciprocal_lattice.matrix
            
            h = np.fft.fftfreq(ngx) * ngx
            k = np.fft.fftfreq(ngy) * ngy
            l = np.fft.fftfreq(ngz) * ngz
            H, K, L = np.meshgrid(h, k, l, indexing='ij')
            
            G_vectors = np.stack([H, K, L], axis=-1) @ recip_lattice
            G_magnitudes = np.linalg.norm(G_vectors, axis=-1)
            
            BOHR_TO_ANGSTROM = 0.529177210903
            g_cutoff = np.sqrt(2.0 * energy_cutoff / (27.211386 * BOHR_TO_ANGSTROM**2))

            rho_G[G_magnitudes > g_cutoff] = 0.0
            rho_3d = np.fft.ifftn(rho_G).real
            
        return rho_3d
    
    def _clean_ranges(self, min_charge, max_charge):
        if min_charge is None or min_charge == -np.inf:
            min_charge = 0.0
        if max_charge is None or max_charge == np.inf:
            max_charge = self.total_charge
            
        min_charge = max(min_charge, 0)
        max_charge = min(max_charge, self.total_charge)
        return min_charge, max_charge