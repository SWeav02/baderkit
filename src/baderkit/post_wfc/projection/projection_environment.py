# -*- coding: utf-8 -*-

from pathlib import Path
from functools import cached_property
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import RegularGridInterpolator
from rich.progress import Progress
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from rich import print as rprint

from baderkit.post_wfc.projection.all_electron_dataset import AESpecies
from baderkit.post_wfc.wfc_numba import (
    find_active_periodic_atoms,
    find_all_voxels_parallel,
    evaluate_real_harmonics_multi
)

from baderkit.post_wfc.base_env import BaseWavefunctionEnvironment

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
    
    @property
    def voxels_near_atoms(self):
        if getattr(self, "_voxels_near_atoms", None) is None:
            self._voxels_near_atoms = self._get_voxel_footprints()
        return self._voxels_near_atoms
    
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
                
            frac_coords, cart_coords, element_indices, base_indices = find_active_periodic_atoms(
                self.lattice_matrix, base_frac_coords, atom_types, self.cutoff_radius
            )
            
            self._cache_semi_periodic_atoms = {
                "frac_coords": frac_coords,
                "cart_coords": cart_coords,
                "element_indices": element_indices,
                "base_indices": base_indices,
                "element_mapping": unique_elements
            }
        return self._cache_semi_periodic_atoms

    def _get_voxel_footprints(self):
        """
        Calculates and caches the voxel coordinate index mappings and scalar radial distances 
        for all translationally extended periodic atoms relative to a specified discrete 3D grid layout.
        """
        
        spa = self.semi_periodic_atoms
        frac_coords = spa["frac_coords"]
        
        all_indices, all_distances, _ = find_all_voxels_parallel(
            frac_coords, 
            self.lattice_matrix, 
            self.fft_grid_shape, 
            [self.cutoff_radius for _ in range(len(frac_coords))],
        )
            
        return all_indices, all_distances

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
        # spin_channel: int = -1,
        energy_range=(-np.inf, np.inf),
        use_partial_occ: bool = True,
    ) -> tuple[NDArray, NDArray]:
        """
        Generates continuous 3D promolecular charge and kinetic energy density reference grids
        across the unit cell, evaluated using the requested energy range bounds.
        """
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
        
        # build interpolator to get partial atomic charges from the total energy range
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
        all_indices, all_distances = self.voxels_near_atoms
        
        # create 1d arrays to store charge density and ked
        total_voxels = np.prod(self.fft_grid_shape)
        rho_data = np.zeros(total_voxels, dtype=np.float64)
        tau_data = np.zeros(total_voxels, dtype=np.float64)

        # Extract periodic image attributes cleanly from the dictionary context
        extended_structure = self.semi_periodic_atoms
        atom_coords = extended_structure["cart_coords"]
        element_indices = extended_structure["element_indices"]
        mapping = extended_structure["element_mapping"]
        base_indices = extended_structure["base_indices"]
        
        num_atoms_extended = len(atom_coords)
        
        # Loop over every single periodic image in the cutoff range
        for i_atom in range(num_atoms_extended):
            base_idx = base_indices[i_atom]
            element = mapping[element_indices[i_atom]]
            basis = self.atom_bases[element]
            
            # FIXED: Reference the primary cell base atom index for partial charges
            min_count = min_charges[base_idx]
            max_count = max_charges[base_idx]
            
            # extract voxel indices and real physical distances (in Angstroms)
            grid_indices = all_indices[i_atom]
            grid_dists = all_distances[i_atom]
            
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
        
        rho_3d = rho_data.reshape(self.fft_grid_shape)
        tau_3d = tau_data.reshape(self.fft_grid_shape)
        
        return rho_3d, tau_3d
    
    ###########################################################################
    # Deformation methods
    ###########################################################################
    
    def get_deformation_densities(
        self, 
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
            # spin_channel=spin_channel,
            energy_range=energy_range,
            use_partial_occ=use_partial_occ,
        )
        
        # 2. Generate matching non-bonding overlapping atomic reference grids
        rho_pro, tau_pro = self.get_promolecular_densities(
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
        # shape: (len(active_bands), nbasis)
        coeffs_lcao = self.projection_coefficients[ispin, ikpt, :, active_bands]
        
        # Synthesize wavefunction: (len(active_bands), nbasis) @ (nbasis, n_Kvecs)
        return np.dot(coeffs_lcao, local_basis_matrix)# / np.sqrt(self.structure.volume)
    
    def _project_system(self):
        """
        Production Projector Augmented Wave (PAW) projection engine.
        Executed sequentially to leverage the native multi-threaded BLAS/LAPACK backend.
        Utilizes a robust least-squares truncated SVD solver to handle overcomplete
        basis sets and prevent unphysical coefficient explosions.
        """
        structure = self.structure
        atom_positions = structure.cart_coords
        nspin = self.nspin
        nkpoints = self.nkpoints
        nbands = self.nbands
        volume = structure.volume
        correction_overlaps = self.core_correction_overlaps
        
        rprint("\n" + "="*80)
        rprint("[bold green]         PROJECTION ENGINE          [/bold green]")
        rprint("="*80)
        rprint(f"[bold white]System Dimensions:[/bold white] Spin={nspin}, k-points={nkpoints}, Bands={nbands}")
        rprint(f"[bold white]Unit Cell Volume (Omega):[/bold white] {volume:.6f} Å^3")
        
        # Count total number of local basis functions
        nbasis = sum(len(i.angular_momenta) for i in self.basis_map)
        
        rprint("\n" + "="*80)
        rprint("[bold blue]INFO: Executing Reciprocal Projections & PAW Augmentation[/bold blue]")
        rprint("="*80)
        
        # Initialize containers to store final projection coefficient results
        c_final = np.zeros((nspin, nkpoints, nbasis, nbands), dtype=np.complex128)
        self._local_basis_matrix = [None] * nkpoints
        
        # create arrays to store spillage per band
        band_spillage = np.zeros(nbands)
        occupied_band_spillage  = np.zeros(nbands)
        occupied_band_spillage_norm = np.zeros(nbands)
        
        # Loop over k points
        with Progress() as progress:
            task = progress.add_task("[bold blue]Mapping Reciprocal Projections...", total=nkpoints)
            
            for ikpt_idx in range(nkpoints):
                # get K vectors (k + G)
                G_basis_cart = self.get_g_vectors_cart(ikpt_idx)
                k_cart = self.kpoints_cart[ikpt_idx]
                
                K_vecs = G_basis_cart + k_cart[np.newaxis, :]
                K_norms = np.linalg.norm(K_vecs, axis=1)
                K_norms = np.where(K_norms < 1e-14, 1e-14, K_norms)
                n_Kvecs = len(K_norms)
                
                # Normalize K_vecs for spherical harmonics
                K_hat = K_vecs / K_norms[:, np.newaxis]
                
                k_weight = self.kpoint_weights[ikpt_idx]
                
                # create arrays/lists to store basis matrices, Chi_alpha, and
                # projector matrices, p.
                local_basis_matrix = np.zeros((nbasis, n_Kvecs), dtype=np.complex128)
                paw_projector_matrices = []
                
                # collect projector matrices
                global_basis_idx = 0
                for atom_idx, local_basis in enumerate(self.basis_map):
                    paw_basis = local_basis.paw_species
                    # get phase
                    spatial_phase = np.exp(-1j * np.dot(K_vecs, atom_positions[atom_idx]))
                    
                    # Loop over local orbitals and assign them using a global index tracker
                    for orb_idx in range(len(local_basis.angular_momenta)):
                        # get quantum numbers
                        l = local_basis.angular_momenta[orb_idx]
                        m = local_basis.magnetic_quantum_numbers[orb_idx]
                        
                        # cubic spline interpolation of radial part
                        chi_r = local_basis.q_radial_splines[orb_idx](K_norms)
                        # exact angular part
                        chi_a = evaluate_real_harmonics_multi(l, m, K_hat)
                        # save basis matrix
                        local_basis_matrix[global_basis_idx] = spatial_phase * chi_r * chi_a
                        global_basis_idx += 1
                    
                    # get projector matrix (already conjugate)
                    paw_projector_matrices.append(
                        paw_basis.build_g_space_conj_projectors(K_vecs, spatial_phase)
                    )
                
                # Read coefficients for each spin
                coeffs_spins = [
                    self._wf_reader.read_coefficients_batch(ispin, ikpt_idx, np.arange(nbands)).T 
                    for ispin in range(nspin)
                ]
                coeff_spins = np.hstack(coeffs_spins)
                
                # !!! SMOOTH PART !!!
                # Calculate <Chi|Psi_smooth>
                smooth_part = local_basis_matrix.conj() @ coeff_spins
                overlap_matrix_smooth = ((local_basis_matrix.conj() @ local_basis_matrix.T) / volume)
                
                # !!! AUG PART !!!
                # Calculate Sum_r,i( <p|Psi_smooth> * (<Chi|Phi_ae>-<Chi|Phi_ps>))
                aug_part = np.zeros((nbasis, nspin * nbands), dtype=np.complex128)
                overlap_matrix_aug = np.zeros_like(overlap_matrix_smooth)
                for atom_idx in range(len(structure)):
                    # read precalculated correction overlaps (<Chi|Phi_ae>-<Chi|Phi_ps>)
                    correction_overlap = correction_overlaps[atom_idx]
                    # Get projector matrix (already conjugate)
                    projector_matrix_conj = paw_projector_matrices[atom_idx]
                    # Project and add to augmentation part
                    aug_part += correction_overlap @ (projector_matrix_conj @ coeff_spins)
                    
                # !!! TOTAL !!!
                # Get total part
                coeff_spins = (smooth_part + aug_part) / np.sqrt(volume)
                overlap_matrix = overlap_matrix_smooth + overlap_matrix_aug
                
                # Solve for LCAO coefficients
                # rcond=1e-4 drops singular values below 1e-4 * max_singular_value
                coeffs_lcao, residuals, rank, s_solver = np.linalg.lstsq(
                    overlap_matrix, 
                    coeff_spins, 
                    rcond=1e-6
                )
                
                # !!! NORMALIZE !!!
                # The coefficients must be normalized in the same manor as PAW
                # codes, that is for the total system, not just the smooth part
                
                # Compute the norm squared of each reconstructed band: diag(C^dagger @ S @ C)
                S_C = overlap_matrix @ coeffs_lcao
                
                O_nn = np.sum(coeffs_lcao.conj() * S_C, axis=0)
                band_norms = np.sqrt(O_nn.real)
                # ==============================================================================
                # --- DIAGNOSTIC 3: CORE VS VALENCE DISSECTION & G-TRUNCATION (ikpt = 0) ---
                # ==============================================================================
                if ikpt_idx == 0:
                    rprint("\n[bold yellow]" + "="*80)
                    rprint("      DIAGNOSTIC 3: CORE VS VALENCE DISSECTION & G-TRUNCATION (ikpt = 0)")
                    rprint("="*80)
                    
                    # 1. Local Basis Function G-Space Norms (S_ii)
                    S_diag = np.real(np.diag(overlap_matrix))
                    rprint("\n[bold white]1. Local Basis Function G-Space Norms (S_ii):[/bold white]")
                    rprint("  (Values < 0.8 indicate severe reciprocal-space truncation at current ENCUT)")
                    
                    global_b_idx = 0
                    for atom_idx, local_basis in enumerate(self.basis_map):
                        for orb_idx in range(len(local_basis.angular_momenta)):
                            l = local_basis.angular_momenta[orb_idx]
                            m = local_basis.magnetic_quantum_numbers[orb_idx]
                            s_val = S_diag[global_b_idx]
                            rprint(f"  Atom {atom_idx} | Orb {orb_idx:2d} (l={l}, m={m:2d}) : S_ii = {s_val:10.6f}")
                            global_b_idx += 1
                
                    # 2. Smooth vs Augmentation Decomposition across Bands
                    rprint("\n[bold white]2. Smooth vs. Augmentation Power Ratio per Band:[/bold white]")
                    rprint("  Band  |  ||Smooth||^2  |   ||Aug||^2   | Aug/Total Ratio |  Spillage (1 - O_nn)")
                    rprint("  -------------------------------------------------------------------------")
                    
                    smooth_norms_sq = np.sum(np.abs(smooth_part)**2, axis=0) / volume
                    aug_norms_sq    = np.sum(np.abs(aug_part)**2, axis=0) / volume
                    
                    for b in range(nbands):
                        sm_pwr = smooth_norms_sq[b]
                        aug_pwr = aug_norms_sq[b]
                        tot_pwr = sm_pwr + aug_pwr
                        aug_ratio = aug_pwr / tot_pwr if tot_pwr > 1e-12 else 0.0
                        o_val = O_nn[b].real
                        
                        rprint(f"  {b+1:4d}  | {sm_pwr:12.6f} | {aug_pwr:12.6f} | {aug_ratio*100:13.2f}% | {1.0 - o_val:+10.6f}")
                
                    rprint("[bold yellow]" + "="*80 + "\n[/bold yellow]")
                
                # Safeguard against division-by-zero for uncaptured/empty states
                safe_norms = np.where(band_norms < 1e-8, 1.0, band_norms)
                
                # Normalize the LCAO coefficients
                coeffs_lcao = coeffs_lcao / safe_norms[np.newaxis, :]
                
                # Unpack and store results directly into target arrays
                self._local_basis_matrix[ikpt_idx] = local_basis_matrix
                c_final[:, ikpt_idx] = coeffs_lcao.reshape(nbasis, nspin, nbands).transpose(1, 0, 2)
                
                # !!! Spillage !!!
                # calculate spillage
                spillage = k_weight * np.abs(1-O_nn)
                
                # reshape by spin
                spillage = spillage.reshape(nspin, nbands).T
                
                # calculate spillage from occupied states
                occs = self.occupancies[:, ikpt_idx, :].T
                occ_sum = occs.sum()
                if occ_sum < 1e-12:
                    occ_spillage = np.zeros(nbands)
                else:
                    occ_spillage = np.mean(spillage*occs, 1)
                # calculate spillage from all states
                spillage = np.mean(spillage, 1)
                
                # save per band
                band_spillage += spillage
                occupied_band_spillage += occ_spillage
                occupied_band_spillage_norm += occ_spillage / occs.sum()
                
                progress.update(task, advance=1)
    
        # Save projection coefficients
        self._projection_coefficients = c_final
        
        # Calculate global spillage parameters
        total_spillage = band_spillage.sum() / nbands * 100
        occupied_total_spillage = occupied_band_spillage_norm.sum() * 100
        
        self.band_spillage = band_spillage
        self.occupied_total_spillage = occupied_band_spillage
        self.spillage = total_spillage
        self.occupied_spillage = occupied_total_spillage

        rprint("\n" + "="*80)
        rprint("[bold yellow]            SPILLAGE METRICS REPORT            [/bold yellow]")
        rprint("="*80)
        rprint(f"  -> [bold white]All-Bands Spillage (S):[/bold white]      {total_spillage:.4f} %")
        rprint(f"  -> [bold white]Charge (Occupancy) Spillage:[/bold white]  {occupied_total_spillage:.4f} %")
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