# -*- coding: utf-8 -*-

from pathlib import Path
from functools import cached_property
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import RegularGridInterpolator
from rich.progress import track
# from concurrent.futures import ThreadPoolExecutor, as_completed

from rich import print as rprint

from baderkit.post_wfc.projection.all_electron_dataset import AESpecies
from baderkit.post_wfc.wfc_numba import (
    find_active_periodic_atoms,
    find_all_voxels_parallel,
    evaluate_real_harmonics_multi
)

from baderkit.post_wfc.base import BaseWavefunctionEnvironment

# TODO:
    # 1. Update paw environment by removing redundant augmentation environment, etc.
    # 2. Make base dataset class and ensure both datasets are solid
    # 2. Combine numba files
    # 3. parallelize
    # 4. General cleanup
    # 5. Add QE workfunction reading

class AtomicProjectionEnvironment(BaseWavefunctionEnvironment):
    """
    Manages the non-bonding atomic reference states by parsing compressed analytical 
    basis binaries with pre-applied primitive normalization constants.
    """
    BOHR_TO_ANGSTROM = 0.5291772109

    def __init__(
        self, 
        post_wfc,
        basis_dir=None,
        **kwargs
    ):
        # Register post_wfc as the reference state context link
        super().__init__(reference_env=post_wfc, **kwargs)

        self.post_wfc = post_wfc
        self.basis_dir = Path(basis_dir) if basis_dir is not None else (Path(__file__).parent / "bases" / "dyall")
        
        # Pulls structure and valence_counts directly from master post_wfc context smoothly
        self.lattice_matrix = self.structure.lattice.matrix
        self.total_charge = sum(self.valence_counts[site.specie.symbol] for site in self.structure)
        
        self._cache_voxel_footprints = {}
        
        # Initialize basis structures and run heavy orbital projections
        self._load_bases()
        self._project_system()


    ###########################################################################
    # Convenient Properties
    ###########################################################################
    
    @property
    def projection_coefficients(self):
        if getattr(self,"_projection_coefficients",None) is None:
            self._project_system()
        return self._projection_coefficients
    
    @property
    def atom_contributions(self):
        if getattr(self,"_atom_contributions",None) is None:
            self._process_pdos()
        return self._atom_contributions
    
    @property
    def atom_pdos(self):
        if getattr(self, "_atom_pdos", None) is None:
            self._atom_pdos = self.get_atom_projected_density_of_states()
        return self._atom_pdos
    
    @property
    def orbital_pdos(self):
        if getattr(self, "_orbital_pdos", None) is None:
            self._orbital_pdos = self.get_atom_projected_density_of_states()
        return self._orbital_pdos
    
    @property
    def basis_map(self):
        if getattr(self, "_basis_map", None) is None:
            basis_map = []
            for i_atom, site in enumerate(self.structure):
                elem = site.species_string
                basis_map.append(self.atom_bases[elem])
            self._basis_map = basis_map
        return self._basis_map
    
    @property
    def all_bases(self):
        if getattr(self, "_all_bases", None) is None:
            all_bases = []
            atom_to_basis_indices = {i: [] for i in range(len(self.structure))}
            global_basis_idx = 0
            
            for atom_idx, basis in enumerate(self.basis_map):
                for basis_idx in range(len(basis.angular_momenta)):
                    all_bases.append({
                        "atom_idx": atom_idx,
                        "l": basis.angular_momenta[basis_idx],
                        "m": basis.magnetic_quantum_numbers[basis_idx],
                        "q_radial_spline": basis.q_radial_splines[basis_idx],
                    })
                    atom_to_basis_indices[atom_idx].append(global_basis_idx)
                    global_basis_idx += 1
            self._all_bases = all_bases
            self._atom_to_basis_indices = atom_to_basis_indices
        return self._all_bases
    
    @property
    def atom_to_basis_indices(self):
        if getattr(self, "_atom_to_basis_indices", None) is None:
            self.all_bases
        return self._atom_to_basis_indices
    
    @property
    def core_correction_overlaps(self):
        if getattr(self, "_core_correction_overlaps", None) is None:
            self._core_correction_overlaps = [i.core_correction_overlaps for i in self.basis_map]
        return self._core_correction_overlaps
    
    ###########################################################################
    # PDOS Methods
    ###########################################################################
    def get_atom_projected_density_of_states(
        self, 
        spin_channel=-1, 
        use_occupancies=False,
        return_plot=False,
        plot_range=None,
    ) -> dict:
        """
        Computes the atom-resolved Projected Density of States (PDOS) by applying
        the designated smearing method to the orbital projection projection_coefficients.
        """
        coeffs = self._projection_coefficients  # Maps to our updated spin-first array                 
        basis_map = self.all_bases
        num_atoms = len(self.structure)

        # Pre-allocate array for channel intensities: shape (num_atoms, nspin, nkpoints, nbands)
        atom_weights = np.zeros((num_atoms, self.nspin, self.nkpoints, self.nbands), dtype=np.float64)
        
        atom_indices = np.array([b['atom_idx'] for b in basis_map], dtype=np.int32)
        proj_intensity = np.abs(coeffs) ** 2       

        for i_atom in range(num_atoms):
            mask = (atom_indices == i_atom)
            if np.any(mask):
                atom_weights[i_atom] = np.sum(proj_intensity[:, :, mask, :], axis=2)

        # Delegate execution context to the unified smearing helper
        smeared_data = self._compute_smeared_channels(
            channel_weights=atom_weights,
            spin_channel=spin_channel,
            use_occupancies=use_occupancies
        )

        energy_grid = self.energy_grid
        pdos_data = {
            "total": np.zeros(len(energy_grid), dtype=np.float64)
        }
        for i_atom in range(num_atoms):
            pdos_data[i_atom] = smeared_data[i_atom]
            pdos_data["total"] += smeared_data[i_atom]

        if return_plot:
            plot_curves = {}
            for i_atom in range(num_atoms):
                symbol = self.structure[i_atom].specie.symbol
                plot_curves[f"Atom {i_atom} ({symbol})"] = pdos_data[i_atom]
            return self._generate_dos_plot(
                total_dos=pdos_data["total"],
                plot_curves=plot_curves,
                plot_range=plot_range
            )

        return pdos_data

    def get_orbital_character_projected_density_of_states(
        self,
        spin_channel=-1,
        use_occupancies=False,
        return_plot=False,
        plot_range=None,
    ) -> dict:
        """
        Computes the orbital-resolved Projected Density of States (PDOS) decomposed
        by both atomic site location index and angular momentum character (s, p, d, f).
        """
        coeffs = self._projection_coefficients  # Maps to our updated spin-first array
        basis_map = self.all_bases
        
        l_symbols = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
        
        # Identify active combination tracks (atom_idx, l) present across the system basis pool
        channels = sorted(list(set((b['atom_idx'], b['l']) for b in basis_map)))
        num_channels = len(channels)
        
        channel_weights = np.zeros((num_channels, self.nspin, self.nkpoints, self.nbands), dtype=np.float64)
        
        atom_indices = np.array([b['atom_idx'] for b in basis_map], dtype=np.int32)
        l_indices = np.array([b['l'] for b in basis_map], dtype=np.int32)
        proj_intensity = np.abs(coeffs) ** 2
        
        for idx, (i_atom, l_val) in enumerate(channels):
            mask = (atom_indices == i_atom) & (l_indices == l_val)
            if np.any(mask):
                # FIXED: Slice axis 2 explicitly and sum over axis 2 (the basis dimension)
                channel_weights[idx] = np.sum(proj_intensity[:, :, mask, :], axis=2)

        # Delegate execution context to the unified smearing helper
        smeared_data = self._compute_smeared_channels(
            channel_weights=channel_weights,
            spin_channel=spin_channel,
            use_occupancies=use_occupancies
        )
        
        energy_grid = self.energy_grid
        pdos_data = {
            "total": np.zeros(len(energy_grid), dtype=np.float64)
        }
        
        plot_curves = {}
        for idx, (i_atom, l_val) in enumerate(channels):
            l_char = l_symbols.get(l_val, f"l={l_val}")
            key = f"atom_{i_atom}_{l_char}"
            
            pdos_data[key] = smeared_data[idx]
            pdos_data["total"] += smeared_data[idx]
            
            symbol = self.structure[i_atom].specie.symbol
            plot_curves[f"Atom {i_atom} ({symbol}) - {l_char}"] = smeared_data[idx]
            
        if return_plot:
            return self._generate_dos_plot(
                total_dos=pdos_data["total"],
                plot_curves=plot_curves,
                plot_range=plot_range
            )
            
        return pdos_data

    ###########################################################################
    # Promolecular Reconstruction
    ###########################################################################

    @cached_property
    def semi_periodic_atoms(self):
        """
        Identifies, transforms, and caches all atomic Cartesian coordinate image positions
        lying within the designated cutoff interaction radius boundary across periodic boundary limits.
        
        Returns:
        --------
        dict
            A metadata storage dictionary mapping real-space Cartesian coordinates, site types, 
            original asymmetric cell indexes, and unique element symbols.
        """
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

    def _get_voxel_footprints(self, grid_shape):
        """
        Calculates and caches the voxel coordinate index mappings and scalar radial distances 
        for all translationally extended periodic atoms relative to a specified discrete 3D grid layout.
        """
        grid_key = tuple(grid_shape)
        
        if grid_key not in self._cache_voxel_footprints:
            spa = self.semi_periodic_atoms
            atom_carts = spa["cart_coords"]
            
            all_indices, all_distances = find_all_voxels_parallel(
                atom_carts, self.lattice_matrix, grid_shape, self.cutoff_radius
            )
            self._cache_voxel_footprints[grid_key] = (all_indices, all_distances)
            
        return self._cache_voxel_footprints[grid_key]

    def get_promolecular_densities_at_points(
        self, 
        frac_coords, 
        # spin_channel: int = -1,
        energy_range=(-np.inf, np.inf),
        use_partial_occ: bool = True,
    ) -> tuple[float | NDArray, float | NDArray]:
        """
        Calculates the non-bonding promolecular reference valence charge and positive-definite
        kinetic energy density at one or multiple arbitrary continuous fractional coordinate points.
        """
        # get cleaned energy range
        e_min, e_max = self._clean_energy_ranges(energy_range)
        
        # get max occupied point
        max_charge = self.maximum_electrons
        
        # get coordinates as cartesian
        cart_coords = np.atleast_2d(frac_coords) @ self.lattice_matrix  
        
        # create arrays to store results
        rho_total = np.zeros(len(cart_coords))
        tau_total = np.zeros(len(cart_coords))
        
        # get partial charges and tau vs. total energy/charge
        atom_data = self.atom_contributions
        
        if use_partial_occ:
            # Map the max_charge back to find its corresponding maximum occupied energy level
            max_occ_energy = np.interp(max_charge, self.total_charge_grid, self.energy_grid)
            e_max = min(e_max, max_occ_energy)
        
        # build interpolator to get partial atomic charges from the total energy
        # range
        energy_grid = self.energy_grid
        atom_electron_counts = atom_data["partial_charge"]
        atom_indices = np.arange(len(self.structure))
        data_interp = RegularGridInterpolator((atom_indices, energy_grid), atom_electron_counts, method='linear')
        
        # get coordinates to interpolate over
        min_coords = np.c_[atom_indices, np.full_like(atom_indices, e_min, dtype=np.float64)]
        max_coords = np.c_[atom_indices, np.full_like(atom_indices, e_max, dtype=np.float64)]
        
        # interpolate atomic charges
        min_charges = data_interp(min_coords)
        max_charges = data_interp(max_coords)
        min_charges[min_charges<1e-12] = 0.0 # set to exact 0
        
        extended_structure = self.semi_periodic_atoms
        atom_coords = extended_structure["cart_coords"]
        element_indices = extended_structure["element_indices"]
        mapping = extended_structure["element_mapping"]
        base_indices = extended_structure["base_indices"]
        
        for atom_idx in range(len(atom_coords)):
            atom_cart = atom_coords[atom_idx]
            element_idx = element_indices[atom_idx]
            element = mapping[element_idx]
            base_idx = base_indices[atom_idx]
            
            # get distances to each point
            dists = np.sqrt((cart_coords[:, 0] - atom_cart[0]) ** 2 + \
                            (cart_coords[:, 1] - atom_cart[1]) ** 2 + \
                            (cart_coords[:, 2] - atom_cart[2]) ** 2)
            valid_mask = dists < self.cutoff_radius
            valid_dists = dists[valid_mask]
            
            if not len(valid_dists):
                continue
                
            # get basis
            basis = self.atom_bases[element]
            min_count = min_charges[base_idx]
            max_count = max_charges[base_idx]
            
            # interpolate rho from atom at (r, e) where r is the distance and e is
            # the atom's partial electrons. Do for both minimum and maximum and get
            # difference
            min_coords = np.column_stack((np.full_like(valid_dists, min_count, dtype=np.float64), valid_dists))
            max_coords = np.column_stack((np.full_like(valid_dists, max_count, dtype=np.float64), valid_dists))
            
            # rho
            min_rhos = basis.radial_rho_interpolator(min_coords)
            max_rhos = basis.radial_rho_interpolator(max_coords)
            rho_total[valid_mask] += max_rhos - min_rhos
            
            # tau
            min_taus = basis.radial_tau_interpolator(min_coords)
            max_taus = basis.radial_tau_interpolator(max_coords)
            tau_total[valid_mask] += max_taus - min_taus
        
        # Dynamically check original input dimensionality to preserve expected return shape
        if np.ndim(frac_coords) == 1:
            return rho_total[0], tau_total[0]
            
        return rho_total, tau_total
        
    def get_promolecular_densities_vs_charge(
        self, 
        frac_coords: NDArray, 
        min_charge: float, 
        max_charge: float, 
        # spin_channel: int = -1,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """
        Calculates the differential promolecular density charge derivatives (d_rho/dQ and d_tau/dQ) 
        at a specific point coordinate via a central-difference cumulative gradient approach.
        """
        

        # get coordinates as cartesian
        cart_coords = np.atleast_2d(frac_coords) @ self.lattice_matrix  
        
        # get partial charges and tau vs. total energy/charge
        atom_data = self.atom_contributions
        
        # Remove regions where charge grid and partial charges do not change
        charge_grid = np.insert(self.total_charge_grid,0,0)
        valid = np.where((charge_grid[1:]-charge_grid[:-1])>0)[0]
        charge_grid = charge_grid[valid+1]
        
        # clean ranges
        min_charge = max(min_charge, charge_grid[0])
        max_charge = min(max_charge, charge_grid[-1])
        
        # get total charge values we want to collect over
        target_charges = self.total_charge_grid
        
        # FIX: Swapped len(frac_coords) for len(cart_coords) so 1D inputs don't break allocation
        cumulative_rho = np.zeros((len(cart_coords), len(target_charges)), dtype=np.float64)
        cumulative_tau = np.zeros((len(cart_coords), len(target_charges)), dtype=np.float64)
        
        
        # build interpolator to get partial atomic charges from the total energy
        # range
        atom_electron_counts = atom_data["partial_charge"][:,valid]
        atom_indices = np.arange(len(self.structure))
        data_interp = RegularGridInterpolator((atom_indices, charge_grid), atom_electron_counts, method='linear')
        
        extended_structure = self.semi_periodic_atoms
        atom_coords = extended_structure["cart_coords"]
        element_indices = extended_structure["element_indices"]
        mapping = extended_structure["element_mapping"]
        base_indices = extended_structure["base_indices"]
        
        for atom_idx in range(len(atom_coords)):
            atom_cart = atom_coords[atom_idx]
            element_idx = element_indices[atom_idx]
            element = mapping[element_idx]
            base_idx = base_indices[atom_idx]
            
            # get distances to each coord
            dists = np.sqrt((cart_coords[:, 0] - atom_cart[0]) ** 2 + \
                            (cart_coords[:, 1] - atom_cart[1]) ** 2 + \
                            (cart_coords[:, 2] - atom_cart[2]) ** 2)
            valid_mask = np.where(dists < self.cutoff_radius)[0]
            valid_dists = dists[valid_mask]
            
            if not len(valid_dists):
                continue
                
            # get partial charges we need to sum over for this atom
            charge_coords = np.column_stack((np.full_like(target_charges, base_idx, dtype=float), target_charges))
            
            partial_charges = data_interp(charge_coords)
            
            # build coords to interpolate over (electron count, radius)
            extended_dists = np.repeat(valid_dists, len(partial_charges))
            extended_charges = np.tile(partial_charges, len(valid_dists))
            charge_count_coords = np.column_stack((extended_charges, extended_dists))
            
            # get basis
            basis = self.atom_bases[element]
            
            # interpolate rho from atom at (e, r) where r is the distance and e is
            # the atom's partial electrons. Do for both minimum and maximum and get
            # difference
            # rho
            rhos = basis.radial_rho_interpolator(charge_count_coords).reshape(len(valid_dists), len(partial_charges))
            cumulative_rho[valid_mask] += rhos
            
            # tau
            taus = basis.radial_tau_interpolator(charge_count_coords).reshape(len(valid_dists), len(partial_charges))
            cumulative_tau[valid_mask] += taus
            
        drho_dQ = np.gradient(cumulative_rho, target_charges, axis=1)
        dtau_dQ = np.gradient(cumulative_tau, target_charges, axis=1)
        
        # Check original shape to determine return type format
        if np.ndim(frac_coords) == 1:
            return drho_dQ[0], dtau_dQ[0]
            
        return  drho_dQ, dtau_dQ
    
    def get_promolecular_densities_vs_energy(
        self, 
        frac_coord: NDArray, 
        # spin_channel: int = -1,
        energy_range=None,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """
        Maps input cell energies directly to their corresponding non-bonding differential 
        energy densities (d_rho / d_E and d_tau / d_E) via chain-rule derivatives.
        """
        energies = self.energy_grid
        
        # get coordinates as cartesian
        cart_coords = np.atleast_2d(frac_coord) @ self.lattice_matrix
        
        # get partial charges and tau vs. total energy/charge
        atom_data = self.atom_contributions
        
        extended_structure = self.semi_periodic_atoms
        
        # CACHED LOOKUP: Instantaneous response for matching parameter sweeps
        # build interpolator to get partial atomic charges from the total energy range
        atom_electron_counts = atom_data["partial_charge"]
        atom_indices = np.arange(len(self.structure))
        data_interp = RegularGridInterpolator((atom_indices, energies), atom_electron_counts, method='linear')
        
        cumulative_rho = np.zeros((len(cart_coords), len(energies)), dtype=np.float64)
        cumulative_tau = np.zeros((len(cart_coords), len(energies)), dtype=np.float64)
        
        atom_coords = extended_structure["cart_coords"]
        element_indices = extended_structure["element_indices"]
        base_indices = extended_structure["base_indices"]
        mapping = extended_structure["element_mapping"]
        
        for atom_idx in range(len(atom_coords)):
            atom_cart = atom_coords[atom_idx]
            element_idx = element_indices[atom_idx]
            base_idx = base_indices[atom_idx]
            element = mapping[element_idx]
            # get distances to each coord
            dists = np.sqrt((cart_coords[:, 0] - atom_cart[0]) ** 2 + \
                            (cart_coords[:, 1] - atom_cart[1]) ** 2 + \
                            (cart_coords[:, 2] - atom_cart[2]) ** 2)
            valid_mask = np.where(dists < self.cutoff_radius)[0]
            valid_dists = dists[valid_mask]
            
            if not len(valid_dists):
                continue
            
            # get partial charges we need to sum over for this atom
            energy_coords = np.column_stack((np.full_like(energies, base_idx, dtype=float), energies))
            
            partial_charges = data_interp(energy_coords)
            
            # build coords to interpolate over (electron count, radius)
            extended_dists = np.repeat(valid_dists, len(partial_charges))
            extended_charges = np.tile(partial_charges, len(valid_dists))
            charge_count_coords = np.column_stack((extended_charges, extended_dists))
            
            # get basis
            basis = self.atom_bases[element]

            # interpolate rho from atom at (e, r) where r is the distance and e is
            # the atom's partial electrons. Do for both minimum and maximum and get
            # difference
            # rho
            rhos = basis.radial_rho_interpolator(charge_count_coords).reshape(len(valid_dists), len(partial_charges))
            cumulative_rho[valid_mask] += rhos
            
            # tau
            taus = basis.radial_tau_interpolator(charge_count_coords).reshape(len(valid_dists), len(partial_charges))
            cumulative_tau[valid_mask] += taus
            
        drho_dE = np.gradient(cumulative_rho, energies, axis=1)
        dtau_dE = np.gradient(cumulative_tau, energies, axis=1)
        if np.ndim(frac_coord) == 1:
            return drho_dE[0], dtau_dE[0]
            
        return drho_dE, dtau_dE

    def get_promolecular_densities(
        self, 
        grid_shape, 
        # spin_channel: int = -1,
        energy_range=(-np.inf, np.inf),
        use_partial_occ: bool = True,
    ) -> tuple[NDArray, NDArray]:
        """
        Generates continuous 3D promolecular charge and kinetic energy density reference grids
        across the unit cell, evaluated using the requested energy range bounds.
        """
        # CACHED LOOKUP: Leverages the common calibration array cache hit layer
        # get cleaned energy range
        e_min, e_max = self._clean_energy_ranges(energy_range)
        
        # get max occupied point
        max_charge = self.maximum_electrons
        
        # get partial charges and tau vs. total energy/charge
        atom_data = self.atom_contributions
        
        if use_partial_occ:
            # Map the max_charge back to find its corresponding maximum occupied energy level
            max_occ_energy = np.interp(max_charge, self.total_charge_grid, self.energy_grid)
            e_max = min(e_max, max_occ_energy)
        
        # build interpolator to get partial atomic charges from the total energy
        # range
        energy_grid = self.energy_grid
        atom_electron_counts = atom_data["partial_charge"]
        atom_indices = np.arange(len(self.structure))
        data_interp = RegularGridInterpolator((atom_indices, energy_grid), atom_electron_counts, method='linear')
        
        # get 1-d coordinates to interpolate over
        min_coords = np.c_[atom_indices, np.full_like(atom_indices, e_min, dtype=np.float64)]
        max_coords = np.c_[atom_indices, np.full_like(atom_indices, e_max, dtype=np.float64)]
        
        # interpolate atomic charges
        min_charges = data_interp(min_coords)
        max_charges = data_interp(max_coords)
        min_charges[min_charges<1e-12] = 0.0 # set to exact 0
        
        # get voxel indices and distances
        all_indices, all_distances = self._get_voxel_footprints(grid_shape)
        
        # create 1d arrays to store charge density and ked
        rho_data = np.zeros(np.prod(grid_shape), dtype=np.float64)
        tau_data = np.zeros(np.prod(grid_shape), dtype=np.float64)

        num_atoms_cell = len(self.semi_periodic_atoms)
        
        for i_atom in range(num_atoms_cell):
            symbol = self.structure[i_atom].specie.symbol
            basis = self.atom_bases[symbol]
            
            # get charge range for this atom
            min_count = min_charges[i_atom]
            max_count = max_charges[i_atom]
            
            # interpolate radial values based on distances
            grid_indices = all_indices[i_atom]
            grid_dists = all_indices[i_atom]
            
            min_coords_atom = np.column_stack((np.full_like(grid_dists, min_count, dtype=np.float64), grid_dists))
            max_coords_atom = np.column_stack((np.full_like(grid_dists, max_count, dtype=np.float64), grid_dists))
            
            # rho
            min_rhos = basis.radial_rho_interpolator(min_coords_atom)
            max_rhos = basis.radial_rho_interpolator(max_coords_atom)
            
            # tau
            min_taus = basis.radial_tau_interpolator(min_coords_atom)
            max_taus = basis.radial_tau_interpolator(max_coords_atom)
            
            # add to results
            rho_data[grid_indices] += max_rhos - min_rhos
            tau_data[grid_indices] += max_taus - min_taus
        
        rho_3d = rho_data.reshape(grid_shape)
        tau_3d = tau_data.reshape(grid_shape)
        
        return rho_3d, tau_3d
    
    ###########################################################################
    # Deformation methods
    ###########################################################################
    
    def get_deformation_densities(
        self, 
        grid_shape=None, 
        # spin_channel: int = -1, 
        energy_range=(-np.inf, np.inf),
        use_partial_occ: bool = True,
    ) -> tuple[NDArray, NDArray]:
        """
        Calculates the real-space 3D deformation grids inside the unit cell.
        Defined as: Deformation Field = Interacting Density - Promolecular Reference.
        """
        # 1. Evaluate the fully interacting crystalline state density arrays
        rho_int, tau_int = self.get_densities(
            grid_shape=grid_shape,
            # spin_channel=spin_channel,
            energy_range=energy_range,
            use_partial_occ=use_partial_occ,
        )
        
        # 2. Generate matching non-bonding overlapping atomic reference grids
        rho_pro, tau_pro = self.get_promolecular_densities(
            grid_shape=rho_int.shape,
            # spin_channel=spin_channel,
            energy_range=energy_range,
            use_partial_occ=use_partial_occ,
        )
        
        return rho_int - rho_pro, tau_int - tau_pro
    
    def get_deformation_densities_at_points(
        self,
        frac_coords,
        # spin_channel: int = -1,
        energy_range=(-np.inf, np.inf),
    ) -> tuple[float | NDArray, float | NDArray]:
        """
        Calculates the exact deformation charge and positive-definite kinetic energy profiles
        at one or multiple discrete fractional coordinates.
        """
        # Extract points from the fully interacting continuous wavefunction representation backend
        rho_int, tau_int = self.get_densities_at_points(
            frac_coord=frac_coords,
            # spin_channel=spin_channel,
            energy_range=energy_range,
        )

        # Extract matching reference profiles at the exact same coordinate points
        rho_pro, tau_pro = self.get_promolecular_densities_at_points(
            frac_coords=frac_coords,
            # spin_channel=spin_channel,
            energy_range=energy_range,
        )
        
        return rho_int - rho_pro, tau_int - tau_pro
    
    def get_deformation_densities_vs_energy(
        self,
        frac_coord: NDArray,
        # spin_channel: int = -1,
        cumulative: bool = False,
        return_plot: bool = False,
    ) -> tuple:
        """
        Calculates the energy-resolved deformation density spectral curves at a given coordinate.
        Supports both narrow differential energy slices and full cumulative accumulation modes.
        """
        # 1. Fetch interacting densities across the energy grid continuum
        int_res = self.get_densities_vs_energy(
            frac_coord=frac_coord,
            # spin_channel=spin_channel,
            cumulative=False,
            return_plot=False,
        )
        int_rho, int_tau = int_res[0], int_res[1]
        
        # 2. Fetch the corresponding non-bonding promolecular reference densities
        pro_rho, pro_tau = self.get_promolecular_densities_vs_energy(
            frac_coord=frac_coord,
            # spin_channel=spin_channel,
        )
        energies = self.energy_grid
        
        # 3. Integrate the promolecular vectors if cumulative tracking is enabled
        if cumulative:
            pro_rho_eval = cumulative_trapezoid(pro_rho, energies, initial=0)
            pro_tau_eval = cumulative_trapezoid(pro_tau, energies, initial=0)
        else:
            pro_rho_eval = pro_rho
            pro_tau_eval = pro_tau
            
        # 5. Extract the net deformation difference curves
        def_rho = int_rho - pro_rho_eval
        def_tau = int_tau - pro_tau_eval
        
        if return_plot:
            mode_prefix = "Integrated " if cumulative else "Differential "
            x_label = "Accumulated Value Change" if cumulative else "Deformation Intensity (per eV)"
            plot_curves = {
                f"{mode_prefix}Deformation $\\Delta\\rho$": def_rho,
                f"{mode_prefix}Deformation $\\Delta\\tau$": def_tau
            }
            return self._generate_property_plot(
                plot_curves=plot_curves,
                x_label=x_label,
            )
            
        return def_rho, def_tau
    
    ###########################################################################
    # Helper Functions
    ###########################################################################
    def _process_pdos(self):
        """
        Normalizes individual atom PDOS arrays so the occupied states integrate exactly
        to each atom's valence count, then computes the cumulative total cell charge profile.
        """
        # Get PDOS
        pdos_data = self.atom_pdos
        
        # get energies and energy vs. total charge
        normalized_data = {}
        
        total = pdos_data["total"]
        total_cum = cumulative_trapezoid(total, initial=0)
        nonzero = total_cum > 0.0
        
        normalized = []
        charge_data = []
        # Get normalized pdos for each atom
        for i in range(len(self.structure)):
            spectrum = pdos_data[i]
            spectrum_cum = cumulative_trapezoid(spectrum, initial=0)
            # get fraction of total
            spectrum_norm = np.zeros_like(spectrum_cum)
            spectrum_norm[nonzero] = spectrum_cum[nonzero] / total_cum[nonzero]
            normalized.append(spectrum_norm)
            charge_data.append(spectrum_norm * self.total_charge_grid)
            
        normalized_data["normalized"] = np.vstack(normalized)
        normalized_data["partial_charge"] = np.vstack(charge_data)

        self._atom_contributions = normalized_data        
        
    def _load_bases(self):
        """Parses NPZ binaries and filters out target elements matching cell contents."""
        unique_elements = set(site.specie.symbol for site in self.structure)
        atom_bases = {}
        
        for element in unique_elements:
            file_path = self.basis_dir / f"{element}.npz"
            if not file_path.exists():
                raise FileNotFoundError(f"Missing analytical basis binary for element: {file_path}")
            basis = AESpecies.from_file(
                file_path, 
                paw_species=self.paw_datasets[element],
                cutoff_radius=self.cutoff_radius,
                g_cutoff_radius=self.g_cutoff_radius,
                energy_range=self.unsmeared_energy_range,
                energy_tol=0.1
                )
            atom_bases[element] = basis
        self.atom_bases = atom_bases
        
    def _construct_coefficients(self, ispin: int, ikpt: int, active_bands: list) -> np.ndarray:
        """
        Evaluates localized orbital atomic shapes to build coherent G-space wavefunctions.
        Dynamically detects coefficient array layout to remain immune to axis orientation bugs.
        """
        # Retrieve pre-calculated caches
        local_basis_matrix = self._local_basis_matrix[ikpt]
        
        # Slice raw projection_coefficients at the selected k-point & spin
        # shape: (n_spin, n_kpoints, n_basis, nbands)
        coeffs_lcao = self.projection_coefficients[ispin, ikpt, :, active_bands] 
        
        # Synthesize wavefunction: (len(active_bands), n_basis) @ (n_basis, n_qvecs) -> (len(active_bands), n_qvecs)
        return np.dot(coeffs_lcao, local_basis_matrix)
    
    def _project_system(self):
        """
        Production Projector Augmented Wave (PAW) projection engine.
        Deconstructed into small, highly granular sub-steps for targeted debugging.
        Includes in-place unweighted and occupancy-weighted spillage calculation.
        """
        # Get class properties
        structure = self.structure
        atom_positions = structure.cart_coords
        nspin = self.nspin
        nkpoints = self.nkpoints
        nbands = self.nbands
        volume = structure.volume
        
        correction_overlaps = self.core_correction_overlaps
        
        # --- Display Projection Engine Header ---
        rprint("\n" + "="*80)
        rprint("[bold green]          PROJECTION ENGINE          [/bold green]")
        rprint("="*80)
        rprint(f"[bold white]System Dimensions:[/bold white] Spin={nspin}, k-points={nkpoints}, Bands={nbands}")
        rprint(f"[bold white]Unit Cell Volume (Omega):[/bold white] {volume:.6f} Å^3")
        
        # count total number of local basis funcitons and projectors
        n_basis = 0
        n_projectors = 0
        for i in self.basis_map:
            n_basis += len(i.angular_momenta)
            paw = i.paw_species
            n_projectors += len(paw.q_projectors)
        
        #######################################################################
        # Smooth Projection, PAW Projection, and Spillage Setup
        #######################################################################
        rprint("\n" + "="*80)
        rprint("[bold blue]INFO: Executing Reciprocal Projections & PAW Augmentation Loop[/bold blue]")
        rprint("="*80)
        
        # Initialize projection_coefficients array: shape (nspin, nkpoints, n_basis, nbands)
        c_final = np.zeros((nspin, nkpoints, n_basis, nbands), dtype=np.complex128)
    
        # Initialize spillage accumulators
        sum_unweighted_spillage = 0.0
        weighted_spillage_numerator = 0.0
        total_occupancy_denominator = 0.0
        
        # Initialize cache containers to avoid heavy G-space recalculations downstream
        self._local_basis_matrix = [None] * nkpoints
        # self._overlap_matrix_cache = [None] * nkpoints
    
        for ikpt_idx in track(range(nkpoints), description="[bold blue]Mapping Reciprocal Projections...[/]"):
            
            # get reciprocal mesh at k-point
            G_basis_cart = self.get_g_vectors_cart(ikpt_idx)
            
            k_cart = self.kpoints_cart[ikpt_idx]
            
            # get q vectors (k+G) and their norms
            q_vecs = G_basis_cart + k_cart[np.newaxis, :]
            q_norms = np.linalg.norm(q_vecs, axis=1)
            n_qvecs = len(q_norms)
            
            ###################################################################
            # Chi_k and p_k Construction
            ###################################################################
            # Construct local basis matrix (chi) and projector matrices (p) at this k point
            local_basis_matrix = np.zeros((n_basis, n_qvecs), dtype=np.complex128)
            paw_projector_matrices = []
            for atom_idx, local_basis in enumerate(self.basis_map):
                # get paw basis and overlap matrix
                paw_basis = local_basis.paw_species
                
                # calculate phase
                spatial_phase = np.exp(-1j * np.dot(q_vecs, atom_positions[atom_idx]))
                
                # loop over local basis
                for loc_idx in range(n_basis):
                    spline = local_basis.q_radial_splines[loc_idx]
                    l = local_basis.angular_momenta[loc_idx]
                    m = local_basis.magnetic_quantum_numbers[loc_idx]
                    
                    # evaluate radial part
                    chi_r = spline(q_norms)
                    
                    # evaluate angular part
                    chi_a = evaluate_real_harmonics_multi(l, m, q_vecs)
                    
                    local_basis_matrix[loc_idx] = spatial_phase * chi_r * chi_a
                
                paw_projector_matrices.append(paw_basis.build_g_space_projectors(
                    q_vecs,
                    spatial_phase,
                    ))
                
                    
            # Build the overlap matrix S(k) for spillage
            # shape: (n_basis, n_basis)
            overlap_matrix = (local_basis_matrix.conj() @ local_basis_matrix.T) / volume
            
            # Cache the evaluated representations for reconstruction
            self._local_basis_matrix[ikpt_idx] = local_basis_matrix
            # self._overlap_matrix_cache[ikpt_idx] = overlap_matrix
                    
            for ispin in range(nspin):
                # get pseudo projection_coefficients for all bands at this kpoint
                # wfc_coeffs shape: (nbands, n_qvecs)
                wfc_coeffs = self._wf_reader.read_coefficients_batch(ispin, ikpt_idx, np.arange(nbands)).T
                
                # SMOOTH PSEUDO PROJECTION
                smooth_spin = local_basis_matrix.conj() @ wfc_coeffs
    
                # AE PROJECTION
                aug_spin = np.zeros((n_basis, nbands), dtype=np.complex128())
                for atom_idx in range(len(structure)):
                    correction_overlap = correction_overlaps[atom_idx]
                    projector_matrix_conj = paw_projector_matrices[atom_idx].conj()
                    aug_spin += correction_overlap @ (projector_matrix_conj @ wfc_coeffs) 
                    
                # FULL PROJECTION
                coeffs_spin = (smooth_spin + aug_spin) / np.sqrt(volume)
                
                # correct self overlap
                # C_lcao = overlap_matrix^-1 * C_active
                coeffs_lcao = np.linalg.solve(overlap_matrix, coeffs_spin)
                c_final[ispin, ikpt_idx] = coeffs_lcao
                
                # SPILLAGE
                f_n = self.occupancies[ispin, ikpt_idx]
                
                # Solve overlap_matrix * X = coeffs_spin
                X = np.linalg.solve(overlap_matrix, coeffs_spin)
                
                # Compute diagonal of C_dagger * S^-1 * C
                diag_captured = np.real(np.sum(coeffs_spin.conj() * X, axis=0))
                diag_captured = np.clip(diag_captured, 0.0, 1.0)
                spillage_n = 1.0 - diag_captured
                
                # Accumulate values
                sum_unweighted_spillage += np.sum(spillage_n)
                weighted_spillage_numerator += np.sum(f_n * spillage_n)
                total_occupancy_denominator += np.sum(f_n)
                
        # Save projection_coefficients back to the class
        self._projection_coefficients = c_final
        
        # calculate spillage
        total_states_count = nkpoints * nspin * nbands
        all_bands_spillage = (sum_unweighted_spillage / total_states_count) * 100.0
        charge_spillage = (weighted_spillage_numerator / total_occupancy_denominator) * 100.0
        
        # Save metrics to class instance
        self.band_spillage = all_bands_spillage
        self.charge_spillage = charge_spillage
        
        rprint("\n" + "="*80)
        rprint("[bold yellow]            SPILLAGE METRICS REPORT            [/bold yellow]")
        rprint("="*80)
        rprint(f"  -> [bold white]All-Bands Spillage (S):[/bold white]      {all_bands_spillage:.4f} %")
        rprint(f"  -> [bold white]Charge (Occupancy) Spillage:[/bold white]  {charge_spillage:.4f} %")
        rprint("="*80 + "\n")
        
    @classmethod
    def from_directory(
        cls, 
        directory: Path | str = Path("."), 
        fmt: str = "vasp", 
        scipy_workers: int = -1, 
        **kwargs,
    ):
        from baderkit.post_wfc.paw_environment import PAWEnvironment
        post_wfc = PAWEnvironment.from_directory(
            directory=directory, 
            fmt=fmt, 
            scipy_workers=scipy_workers, 
            **kwargs,
        )
        return post_wfc.projection_environment