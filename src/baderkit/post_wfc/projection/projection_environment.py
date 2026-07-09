# -*- coding: utf-8 -*-

from pathlib import Path
import logging
from functools import cached_property
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid
from scipy.special import gamma
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn, TimeRemainingColumn

from .all_electron_dataset import AESpecies
from .projection_numba import (
    find_active_periodic_atoms,
    find_all_voxels_parallel,
    broadcast_atoms_to_grid,
    evaluate_orbital_g_space,
)
from baderkit.post_wfc.wfc_numba import _integrate_tetrahedra_spectral_density_numba
from baderkit.post_wfc.base import BaseWavefunctionEnvironment

# TODO:
    # Make sure basis is loaded in metallic units (or just saved this way)
    # Fix any methods that assum atomic units
    # Fix projection method

class AtomicProjectionEnvironment(BaseWavefunctionEnvironment):
    """
    Manages the non-bonding atomic reference states by parsing compressed analytical 
    basis binaries with pre-applied primitive normalization constants.
    """
    BOHR_TO_ANGSTROM = 0.5291772109

    def __init__(
        self, 
        post_wfc,
        cutoff_radius=8.0,
        basis_dir=None,
        **kwargs
    ):
        # Register post_wfc as the reference state context link
        super().__init__(reference_env=post_wfc, **kwargs)

        self.post_wfc = post_wfc
        self.cutoff_radius = cutoff_radius
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
    def maximum_charge(self) -> float:
        """
        Calculates the total charge capacity of the system if all available electronic 
        states across all atomic basis channels were fully occupied.
        """
        total = 0.0
        for site in self.structure:
            basis = self.atom_bases[site.specie.symbol]
            max_per_state = 1.0 if getattr(basis, "unrestricted", False) else 2.0
            total += max_per_state * len(basis.angular_momenta)
        return total
    
    @property
    def basis_map(self):
        if getattr(self,"_basis_map",None) is None:
            self._project_system()
        return self._basis_map
    
    @property
    def coefficients(self):
        if getattr(self,"_coefficients",None) is None:
            self._project_system()
        return self._coefficients
    
    @property
    def norm_total_integral(self):
        if getattr(self,"_norm_total_integral",None) is None:
            self._process_pdos()
        return self._norm_total_integral
    
    @property
    def atom_integrals(self):
        if getattr(self,"_atom_integrals",None) is None:
            self._process_pdos()
        return self._atom_integrals
    
    ###########################################################################
    # Property Calculations
    ###########################################################################

    def get_rho_tau(
        self, 
        grid_shape=None, 
        spin_channel=-1, 
        energy_range=(-np.inf, np.inf), 
        use_partial_occ=True, 
        return_density_matrices=False,
        use_shrod_tau=False,
        **kwargs
    ):
        """Computes localized LCAO real-space rho and tau densities using the base engine."""
        if grid_shape is None: 
            grid_shape = self._minimum_fft_size * 2
        grid_shape_tuple = tuple(grid_shape)
        
        num_basis_tot = len(self.basis_map)
        global_density_matrix = np.zeros((num_basis_tot, num_basis_tot), dtype=np.complex128)
        
        # Define inline callback to process active state populations into the density matrix
        def dm_accumulation_callback(ispin, ikpt, active_bands, weights):
            coeffs_active = self.coefficients[ispin, ikpt, active_bands, :]
            for b_idx, weight in enumerate(weights):
                global_density_matrix[:] += weight * np.outer(coeffs_active[b_idx], coeffs_active[b_idx].conj())
                
        callback = dm_accumulation_callback if return_density_matrices else None
        
        # Call core transformation engine
        rho, tau = self._execute_core_density_loop(
            grid_shape=grid_shape_tuple,
            spin_channel=spin_channel,
            energy_range=energy_range,
            use_partial_occ=use_partial_occ,
            weight_callback=callback,
            use_shrod_tau=use_shrod_tau,
        )
            
        results = [rho, tau]
        if return_density_matrices:
            results.append(global_density_matrix)
        return tuple(results)
    
    def get_rho_tau_vs_energy(
        self, 
        include_aug=False,
        **kwargs,
    ):
        # Force method to not use augmentations
        return super().get_rho_tau_vs_energy(
            include_aug=False,
            **kwargs,
            )
    
    
    
    ###########################################################################
    # PDOS Methods
    ###########################################################################
    def get_atom_projected_density_of_states(
        self, 
        spin_channel=-1, 
        energy_range=None, 
        resolution=200, 
        method="gaussian", 
        sigma=None, 
        use_occupancies=False,
        return_plot=False,
    ) -> dict:
        """
        Computes the atom-resolved Projected Density of States (PDOS) by applying
        the designated smearing method to the orbital projection coefficients.
        """
        coeffs = self.coefficients                  
        basis_map = self.basis_map
        num_atoms = len(self.structure)

        # Pre-allocate array for channel intensities
        atom_weights = np.zeros((num_atoms, self.nspin, self.nkpoints, self.nbands), dtype=np.float64)
        
        atom_indices = np.array([b['atom_index'] for b in basis_map], dtype=np.int32)
        proj_intensity = np.abs(coeffs) ** 2       

        for i_atom in range(num_atoms):
            mask = (atom_indices == i_atom)
            if np.any(mask):
                atom_weights[i_atom] = np.sum(proj_intensity[..., mask], axis=-1)

        # Delegate execution context to the unified smearing helper
        energy_grid, smeared_data = self._compute_smeared_channels(
            channel_weights=atom_weights,
            spin_channel=spin_channel,
            energy_range=energy_range,
            resolution=resolution,
            method=method,
            sigma=sigma,
            use_occupancies=use_occupancies
        )

        pdos_data = {
            "energy_grid": energy_grid,
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
                energy_grid=energy_grid,
                total_dos=pdos_data["total"],
                plot_curves=plot_curves,
                energy_range=energy_range
            )

        return pdos_data

    def get_orbital_character_projected_density_of_states(
        self,
        spin_channel=-1,
        energy_range=None,
        resolution=200,
        method="gaussian",
        sigma=None,
        use_occupancies=False,
        return_plot=False,
    ) -> dict:
        """
        Computes the orbital-resolved Projected Density of States (PDOS) decomposed
        by both atomic site location index and angular momentum character (s, p, d, f).
        """
        coeffs = self.coefficients
        basis_map = self.basis_map
        
        l_symbols = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
        
        # Identify active combination tracks (atom_index, l) present across the system basis pool
        channels = sorted(list(set((b['atom_index'], b['l']) for b in basis_map)))
        num_channels = len(channels)
        
        channel_weights = np.zeros((num_channels, self.nspin, self.nkpoints, self.nbands), dtype=np.float64)
        
        atom_indices = np.array([b['atom_index'] for b in basis_map], dtype=np.int32)
        l_indices = np.array([b['l'] for b in basis_map], dtype=np.int32)
        proj_intensity = np.abs(coeffs) ** 2
        
        for idx, (i_atom, l_val) in enumerate(channels):
            mask = (atom_indices == i_atom) & (l_indices == l_val)
            if np.any(mask):
                channel_weights[idx] = np.sum(proj_intensity[..., mask], axis=-1)

        # Delegate execution context to the unified smearing helper
        energy_grid, smeared_data = self._compute_smeared_channels(
            channel_weights=channel_weights,
            spin_channel=spin_channel,
            energy_range=energy_range,
            resolution=resolution,
            method=method,
            sigma=sigma,
            use_occupancies=use_occupancies
        )
        
        pdos_data = {
            "energy_grid": energy_grid,
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
                energy_grid=energy_grid,
                total_dos=pdos_data["total"],
                plot_curves=plot_curves,
                energy_range=energy_range
            )
            
        return pdos_data

    def _compute_smeared_channels(
        self,
        channel_weights: NDArray,  # shape: (num_channels, nspin, nkpoints, nbands)
        spin_channel: int = -1,
        energy_range: list | tuple | None = None,
        resolution: int = 200,
        method: str = "gaussian",
        sigma: float | None = None,
        use_occupancies: bool = False,
    ) -> tuple[NDArray, NDArray]:
        """Consolidated pipeline to map state selection masks and execute smearing methods."""
        
        energy_range = self._clean_energy_ranges(energy_range, method, sigma)
        e_min, e_max = energy_range
        
        method, sigma = self._get_default_sigma(method, sigma)
        bands = self.energies
        num_channels = channel_weights.shape[0]
        
        num_points = int(round((e_max - e_min)*resolution))
        energy_grid = np.linspace(e_min, e_max, num_points)
        delta_e = energy_grid[1] - energy_grid[0] if num_points > 1 else 0.0
        
        if spin_channel == 1 and self.nspin == 1:
            spin_channel = 0
    
        spin_all = [spin_channel] if spin_channel != -1 else list(range(self.nspin))
        factor = 2 if (spin_channel == -1 and self.nspin == 1) else 1

        smeared_data = np.zeros((num_channels, num_points), dtype=np.float64)

        # --- Pipeline 1: Analytic Tetrahedron Profile Method ---
        if method == "tetrahedron":
            full_map = self.full_to_irr_map
            eigenvalues = bands[spin_all][:, full_map, :]  
            tetra_indices = self.tetrahedra_indices
            tetra_weight = 1.0 / len(tetra_indices)
            
            if use_occupancies:
                w_t = (self.occupancies[spin_all][:, full_map, :] * factor)[..., np.newaxis]
            else:
                w_t = np.ones_like(eigenvalues)[..., np.newaxis] * factor
                
            band_mask = np.any((eigenvalues >= e_min) & (eigenvalues <= e_max), axis=(0, 1))
            
            if np.any(band_mask):
                eigenvalues_filtered = eigenvalues[:, :, band_mask]
                
                for c in range(num_channels):
                    c_w_full = channel_weights[c][spin_all][:, full_map, :]
                    w_t_c = w_t * c_w_full[..., np.newaxis]
                    w_t_filtered = w_t_c[:, :, band_mask]
                    
                    smeared_data[c] = _integrate_tetrahedra_spectral_density_numba(
                        energy_grid, tetra_indices, eigenvalues_filtered, w_t_filtered, tetra_weight,
                    )[0].sum(axis=0)
                    
        # --- Pipeline 2: Analytic Matrix Broadening Broadcaster ---
        else:
            kpt_weights = self.kpoint_weights
            bands_flat = bands[spin_all].ravel()
            
            if use_occupancies:
                w_t = (self.occupancies[spin_all] * kpt_weights[None, :, None]).ravel() * factor
            else:
                w_t = (np.ones_like(bands[spin_all]) * kpt_weights[None, :, None]).ravel() * factor
                
            mask = (bands_flat >= e_min) & (bands_flat <= e_max)
            bands_filtered = bands_flat[mask]
            
            if len(bands_filtered) > 0:
                if method == "none":
                    smear_matrix = np.zeros((num_points, len(bands_filtered)))
                    closest_idx = np.round((bands_filtered - e_min) / delta_e).astype(int)
                    valid_mask = (closest_idx >= 0) & (closest_idx < num_points)
                    smear_matrix[closest_idx[valid_mask], np.where(valid_mask)[0]] = 1.0 / delta_e
                else:
                    delta_E = energy_grid[:, None] - bands_filtered[None, :]
                    smear_matrix = self._get_smear_matrix(delta_E / sigma, method, sigma)
            
                for c in range(num_channels):
                    c_w_flat = channel_weights[c][spin_all].ravel()
                    w_t_c = w_t * c_w_flat
                    w_t_filtered = w_t_c[mask]
                    smeared_data[c] = np.dot(smear_matrix, w_t_filtered)
                    
        # Apply secondary broadening kernel to the tetrahedron curves if configured
        if method == "tetrahedron" and sigma > 0.0:
            n_kernel = int(np.ceil(4.0 * sigma / delta_e))
            if n_kernel > 0:
                x_kernel = np.arange(-n_kernel, n_kernel + 1) * delta_e
                kernel = np.exp(-0.5 * (x_kernel / sigma)**2)
                kernel /= np.sum(kernel)
                for c in range(num_channels):
                    smeared_data[c] = np.convolve(smeared_data[c], kernel, mode='same')
                    
        return energy_grid, smeared_data
    
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
            g_dims = np.array(grid_shape, dtype=np.int64)
            
            all_indices, all_distances = find_all_voxels_parallel(
                atom_carts, self.lattice_matrix, g_dims, self.cutoff_radius
            )
            self._cache_voxel_footprints[grid_key] = (all_indices, all_distances)
            
        return self._cache_voxel_footprints[grid_key]

    def get_atom_radial_rho_tau(
        self, 
        min_charge: float, 
        max_charge: float,
        spin_channel: int = -1,
        use_partial_occ: bool = True,
    ) -> tuple[dict[int, NDArray], dict[int, NDArray]]:
        """
        Evaluates the non-interacting atomic radial charge and kinetic energy density profiles 
        by mapping the global cell charge window onto local atomic shells using PDOS allocation rules.
        """
        min_charge, max_charge = self._clean_charge_ranges(min_charge, max_charge)
        
        partial_rhos = {}
        partial_taus = {}
        
        for i_atom, alloc_integral in self.atom_integrals.items():
            atom_qs = np.interp([min_charge, max_charge], self.norm_total_integral, alloc_integral)
            symbol = self.structure[i_atom].specie.symbol
            basis = self.atom_bases[symbol]
            
            partial_rhos[i_atom] = basis.get_radial_charge_density(
                min_electrons=atom_qs[0], 
                max_electrons=atom_qs[1], 
            )
            partial_taus[i_atom] = basis.get_radial_kinetic_energy_density(
                min_electrons=atom_qs[0], 
                max_electrons=atom_qs[1], 
            )
            
        return partial_rhos, partial_taus
        
    def get_promolecular_rho_tau_at_points(
        self, 
        frac_coords, 
        spin_channel: int = -1,
        energy_range=(-np.inf, np.inf),
        method="gaussian",
        sigma=None,
        resolution: int = 200,
        use_partial_occ: bool = True,
    ) -> tuple[float | NDArray, float | NDArray]:
        """
        Calculates the non-bonding promolecular reference valence charge and positive-definite
        kinetic energy density at one or multiple arbitrary continuous fractional coordinate points.
        """
        # CACHED LOOKUP: Dynamically resolves or extracts cached integration calibration curves
        energy_charge_array = self._get_total_charge_vs_energy(
            spin_channel=spin_channel, 
            use_partial_occ=use_partial_occ, 
            method=method, 
            sigma=sigma,
            resolution=resolution
        )
        
        e_min, e_max = self._clean_energy_ranges(energy_range, method, sigma)
        
        num_points = int(round((e_max-e_min)*resolution))
        energies = np.linspace(e_min, e_max, num_points)
        charges = np.interp(energies, energy_charge_array[:,0], energy_charge_array[:,1])

        min_charge = np.min(charges)
        max_charge = np.max(charges)
        
        spa = self.semi_periodic_atoms
        atom_carts = spa["cart_coords"]          
        atom_bases_indices = spa["base_indices"]  
        
        frac_coords_arr = np.asarray(frac_coords, dtype=np.float64)
        is_single_point = frac_coords_arr.ndim == 1
        target_cart = np.atleast_2d(frac_coords_arr) @ self.lattice_matrix  
        num_targets = target_cart.shape[0]
        
        rho_total = np.zeros(num_targets, dtype=np.float64)
        tau_total = np.zeros(num_targets, dtype=np.float64)
        
        partial_rhos, partial_taus = self.get_atom_radial_rho_tau(
            min_charge, max_charge, spin_channel=spin_channel, use_partial_occ=use_partial_occ
        )

        for j, i_atom in enumerate(atom_bases_indices):
            atom_pos = atom_carts[j]
            
            dist_sq = (target_cart[:, 0] - atom_pos[0]) ** 2 + \
                      (target_cart[:, 1] - atom_pos[1]) ** 2 + \
                      (target_cart[:, 2] - atom_pos[2]) ** 2
            dists_j = np.sqrt(dist_sq)
            
            mask_j = dists_j < self.cutoff_radius
            if np.any(mask_j):
                symbol = self.structure[i_atom].specie.symbol
                r_grid = self.atom_bases[symbol].radial_grid
                
                rho_total += np.interp(dists_j, r_grid, partial_rhos[i_atom]) * mask_j
                tau_total += np.interp(dists_j, r_grid, partial_taus[i_atom]) * mask_j
        
        if is_single_point:
            return rho_total[0], tau_total[0]
            
        return rho_total, tau_total
    
    def get_promolecular_rho_tau_vs_charge(
        self, 
        frac_coord: NDArray, 
        min_charge: float, 
        max_charge: float, 
        spin_channel: int = -1,
        resolution: int = 200,
        method="gaussian",
        sigma=None,
        use_partial_occ: bool = False,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """
        Calculates the differential promolecular density charge derivatives (d_rho/dQ and d_tau/dQ) 
        at a specific point coordinate via a central-difference cumulative gradient approach.
        """
        # CACHED LOOKUP: Leverages standard lookup cache signature natively
        energy_charge_array = self._get_total_charge_vs_energy(
            spin_channel=spin_channel, 
            use_partial_occ=use_partial_occ, 
            method=method, 
            sigma=sigma,
            resolution=resolution,
        )
        num_points = len(energy_charge_array)
        min_charge, max_charge = self._clean_charge_ranges(min_charge, max_charge)
        target_charges = np.linspace(min_charge, max_charge, num_points)
        
        cumulative_rho = np.empty(len(target_charges), dtype=np.float64)
        cumulative_tau = np.empty(len(target_charges), dtype=np.float64)
        
        charges = np.linspace(min_charge, max_charge, num_points)
        energies = np.interp(charges, energy_charge_array[:,1], energy_charge_array[:,0])
        energy_range = (energies.min(), energies.max())
        
        for idx, charge in enumerate(target_charges):
            r_val, t_val = self.get_promolecular_rho_tau_at_points(
                frac_coord, 
                spin_channel=spin_channel,
                energy_range=energy_range,
                method=method,
                sigma=sigma,
                resolution=resolution,
                use_partial_occ=use_partial_occ,
            )
            cumulative_rho[idx] = r_val
            cumulative_tau[idx] = t_val
            
        if num_points > 1:
            drho_dQ = np.gradient(cumulative_rho, target_charges)
            dtau_dQ = np.gradient(cumulative_tau, target_charges)
        else:
            drho_dQ = np.zeros(len(target_charges), dtype=np.float64)
            dtau_dQ = np.zeros(len(target_charges), dtype=np.float64)
            
        return target_charges, drho_dQ, dtau_dQ
    
    def get_promolecular_rho_tau_vs_energy(
        self, 
        frac_coord: NDArray, 
        spin_channel: int = -1,
        energy_range=None,
        method="gaussian",
        sigma=None,
        resolution: int = 200,
        use_partial_occ: bool = False,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """
        Maps input cell energies directly to their corresponding non-bonding differential 
        energy densities (d_rho / d_E and d_tau / d_E) via chain-rule derivatives.
        """
        e_min, e_max = self._clean_energy_ranges(energy_range, method, sigma)
        num_points = int(round((e_max - e_min) * resolution))
        energies = np.linspace(e_min, e_max, num_points)
        
        drho_dE = np.zeros(num_points, dtype=np.float64)
        dtau_dE = np.zeros(num_points, dtype=np.float64)
        
        spa = self.semi_periodic_atoms
        atom_carts = spa["cart_coords"]
        atom_bases_indices = spa["base_indices"]
        target_cart = np.atleast_2d(frac_coord) @ self.lattice_matrix
        
        # CACHED LOOKUP: Instantaneous response for matching parameter sweeps
        energy_charge_array = self._get_total_charge_vs_energy(
            spin_channel=spin_channel, 
            use_partial_occ=use_partial_occ, 
            method=method, 
            sigma=sigma,
            resolution=resolution
        )
        charges = np.interp(energies, energy_charge_array[:,0], energy_charge_array[:,1])
        
        delta_q = 1e-4
        for idx, energy in enumerate(energies):
            current_charge = charges[idx]
            
            partial_rhos, partial_taus = self.get_atom_radial_rho_tau(
                min_charge=current_charge - delta_q, 
                max_charge=current_charge + delta_q,
                spin_channel=spin_channel,
                use_partial_occ=use_partial_occ,
            )
            
            rho_at_E = 0.0
            tau_at_E = 0.0
            
            for j, i_atom in enumerate(atom_bases_indices):
                dist = np.linalg.norm(target_cart[0] - atom_carts[j])
                if dist < self.cutoff_radius:
                    symbol = self.structure[i_atom].specie.symbol
                    r_grid = self.atom_bases[symbol].radial_grid
                    
                    rho_at_E += np.interp(dist, r_grid, partial_rhos[i_atom]) / (2 * delta_q)
                    tau_at_E += np.interp(dist, r_grid, partial_taus[i_atom]) / (2 * delta_q)
            
            drho_dE[idx] = rho_at_E
            dtau_dE[idx] = tau_at_E

        dQ_dE = np.gradient(charges, energies)
        drho_dE = drho_dE * dQ_dE
        dtau_dE = dtau_dE * dQ_dE
        
        return energies, drho_dE, dtau_dE

    def get_promolecular_rho_tau(
        self, 
        grid_shape, 
        spin_channel: int = -1,
        energy_range=(-np.inf, np.inf),
        method="gaussian",
        sigma=None,
        resolution: int = 200,
        use_partial_occ: bool = True,
    ) -> tuple[NDArray, NDArray]:
        """
        Generates continuous 3D promolecular charge and kinetic energy density reference grids
        across the unit cell, evaluated using the requested energy range bounds.
        """
        # CACHED LOOKUP: Leverages the common calibration array cache hit layer
        energy_charge_array = self._get_total_charge_vs_energy(
            spin_channel=spin_channel, 
            use_partial_occ=use_partial_occ, 
            method=method, 
            sigma=sigma,
            resolution=resolution
        )
        
        e_min, e_max = self._clean_energy_ranges(energy_range, method, sigma)
        num_points = int(round((e_max-e_min)*resolution))
        energies = np.linspace(e_min, e_max, num_points)
        charges = np.interp(energies, energy_charge_array[:,0], energy_charge_array[:,1])

        min_charge = np.min(charges)
        max_charge = np.max(charges)
        
        spa = self.semi_periodic_atoms
        atom_types = spa["base_indices"]
        
        all_indices, all_distances = self._get_voxel_footprints(grid_shape)
        partial_rhos, partial_taus = self.get_atom_radial_rho_tau(
            min_charge, max_charge, spin_channel=spin_channel, use_partial_occ=use_partial_occ
        )

        rho_matrices_list = []
        tau_matrices_list = []
        r_grids_list = []                
        
        num_atoms_cell = len(self.structure)
        paw_r1_list = np.zeros(num_atoms_cell, dtype=np.float64)
        
        for i_atom in range(num_atoms_cell):
            symbol = self.structure[i_atom].specie.symbol
            r_grids_list.append(self.atom_bases[symbol].radial_grid)
            rho_matrices_list.append(partial_rhos[i_atom])
            tau_matrices_list.append(partial_taus[i_atom])
            paw_r1_list[i_atom] = self._aug_environment.paw_datasets[symbol].radial_grid[0]
        
        g_dims = np.array(grid_shape, dtype=np.int64)
        
        rho_3d = broadcast_atoms_to_grid(
            g_dims, atom_types, all_indices, all_distances, rho_matrices_list, r_grids_list, paw_r1_list,
        )
        tau_3d = broadcast_atoms_to_grid(
            g_dims, atom_types, all_indices, all_distances, tau_matrices_list, r_grids_list, paw_r1_list,
        )
            
        return rho_3d, tau_3d
    
    ###########################################################################
    # Deformation methods
    ###########################################################################
    
    def get_deformation_rho_tau(
        self, 
        grid_shape=None, 
        spin_channel: int = -1, 
        energy_range=(-np.inf, np.inf),
        method="gaussian",
        sigma=None,
        use_partial_occ: bool = True,
    ) -> tuple[NDArray, NDArray]:
        """
        Calculates the real-space 3D deformation grids inside the unit cell.
        Defined as: Deformation Field = Interacting Density - Promolecular Reference.
        """
        # 1. Evaluate the fully interacting crystalline state density arrays
        rho_int, tau_int = self.get_rho_tau(
            grid_shape=grid_shape,
            spin_channel=spin_channel,
            energy_range=energy_range,
            use_partial_occ=use_partial_occ,
            method=method,
            sigma=sigma,
        )
        
        # 2. Generate matching non-bonding overlapping atomic reference grids
        rho_pro, tau_pro = self.get_promolecular_rho_tau(
            grid_shape=rho_int.shape,
            spin_channel=spin_channel,
            energy_range=energy_range,
            method=method,
            sigma=sigma,
            use_partial_occ=use_partial_occ,
        )
        
        return rho_int - rho_pro, tau_int - tau_pro
    
    def get_deformation_rho_tau_at_points(
        self,
        frac_coords,
        spin_channel: int = -1,
        energy_range=(-np.inf, np.inf),
        method="gaussian",
        sigma=None,
        use_partial_occ: bool = False,
    ) -> tuple[float | NDArray, float | NDArray]:
        """
        Calculates the exact deformation charge and positive-definite kinetic energy profiles
        at one or multiple discrete fractional coordinates.
        """
        # Extract points from the fully interacting continuous wavefunction representation backend
        rho_int, tau_int = self.get_rho_tau_at_points(
            frac_coord=frac_coords,
            spin_channel=spin_channel,
            energy_range=energy_range,
            use_partial_occ=use_partial_occ,
            method=method,
            sigma=sigma,
        )

        # Extract matching reference profiles at the exact same coordinate points
        rho_pro, tau_pro = self.get_promolecular_rho_tau_at_points(
            frac_coords=frac_coords,
            spin_channel=spin_channel,
            energy_range=energy_range,
            method=method,
            sigma=sigma,
            use_partial_occ=use_partial_occ,
        )
        
        return rho_int - rho_pro, tau_int - tau_pro
    
    def get_deformation_rho_tau_vs_energy(
        self,
        frac_coord: NDArray,
        spin_channel: int = -1,
        energy_range=(-np.inf, np.inf),
        method="gaussian",
        sigma=None,
        resolution: int = 200,
        cumulative: bool = False,
        return_plot: bool = False,
        use_partial_occ: bool = False,
    ) -> tuple:
        """
        Calculates the energy-resolved deformation density spectral curves at a given coordinate.
        Supports both narrow differential energy slices and full cumulative accumulation modes.
        """
        # 1. Fetch interacting densities across the energy grid continuum
        int_res = self.get_rho_tau_vs_energy(
            frac_coord=frac_coord,
            spin_channel=spin_channel,
            energy_range=energy_range,
            resolution=resolution,
            cumulative=False,
            return_plot=False,
            use_partial_occ=use_partial_occ,
            method=method,
            sigma=sigma,
        )
        int_energy, int_rho, int_tau = int_res[0], int_res[1], int_res[2]
        
        # 2. Fetch the corresponding non-bonding promolecular reference densities
        energies, pro_rho, pro_tau = self.get_promolecular_rho_tau_vs_energy(
            frac_coord=frac_coord,
            spin_channel=spin_channel,
            energy_range=energy_range,
            method=method,
            sigma=sigma,
            resolution=resolution,
            use_partial_occ=use_partial_occ,
        )
        
        # 3. Integrate the promolecular vectors if cumulative tracking is enabled
        if cumulative:
            cum_pro_rho = np.zeros_like(energies)
            cum_pro_tau = np.zeros_like(energies)
            if len(energies) > 1:
                cum_pro_rho[1:] = cumulative_trapezoid(pro_rho, energies)
                cum_pro_tau[1:] = cumulative_trapezoid(pro_tau, energies)
                
            pro_rho_eval = cum_pro_rho
            pro_tau_eval = cum_pro_tau
        else:
            pro_rho_eval = pro_rho
            pro_tau_eval = pro_tau
            
        # 4. Interpolate interacting values to align perfectly with the target energy levels
        rho_int_interp = np.interp(energies, int_energy, int_rho)
        tau_int_interp = np.interp(energies, int_energy, int_tau)
        
        # 5. Extract the net deformation difference curves
        def_rho = rho_int_interp - pro_rho_eval
        def_tau = tau_int_interp - pro_tau_eval
        
        if return_plot:
            mode_prefix = "Integrated " if cumulative else "Differential "
            x_label = "Accumulated Value Change" if cumulative else "Deformation Intensity (per eV)"
            plot_curves = {
                f"{mode_prefix}Deformation $\\Delta\\rho$": def_rho,
                f"{mode_prefix}Deformation $\\Delta\\tau$": def_tau
            }
            return self._generate_property_plot(
                energy_grid=energies,
                plot_curves=plot_curves,
                x_label=x_label,
                energy_range=energy_range
            )
            
        return energies, def_rho, def_tau
    
    ###########################################################################
    # Helper Functions
    ###########################################################################
    def _process_pdos(self):
        """
        Normalizes individual atom PDOS arrays so the occupied states integrate exactly
        to each atom's valence count, then computes the cumulative total cell charge profile.
        """
        # Call the high-accuracy analytic tetrahedron method to populate our raw spectral data
        pdos_data = self.get_atom_projected_density_of_states(method="tetrahedron")
        
        energies = pdos_data["energy_grid"]
        dx = np.diff(energies)
        
        def cumulative_integrate(y):
            avg_y = 0.5 * (y[:-1] + y[1:])
            integral = np.zeros_like(y)
            integral[1:] = np.cumsum(avg_y * dx)
            return integral

        raw_total_dos = np.zeros_like(energies)
        for key, p_sub in pdos_data.items():
            try:
                int(key)
            except ValueError:
                continue
            raw_total_dos += p_sub
            
        raw_total_integral = cumulative_integrate(raw_total_dos)
        E_F = np.interp(self.total_charge, raw_total_integral, energies)
        
        atom_integrals = {}
        norm_total_dos = np.zeros_like(energies)
        
        for i_atom, p_sub in pdos_data.items():
            try:
                int(i_atom)
            except ValueError:
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
        self._norm_total_integral = norm_total_integral
        self._atom_integrals = atom_integrals
    
    def _load_bases(self):
        """Parses NPZ binaries and filters out target elements matching cell contents."""
        unique_elements = set(site.specie.symbol for site in self.structure)
        atom_bases = {}
        
        for element in unique_elements:
            file_path = self.basis_dir / f"{element}.npz"
            if not file_path.exists():
                raise FileNotFoundError(f"Missing analytical basis binary for element: {file_path}")
            basis = AESpecies.from_file(file_path, paw_species=self._aug_environment.paw_datasets[element])
            atom_bases[element] = basis
        self.atom_bases = atom_bases
    
    def _construct_coefficients(self, ispin: int, ikpt: int, active_bands: list) -> np.ndarray:
        """Evaluates localized orbital atomic shapes to build coherent G-space wavefunctions."""
        coeffs_active = self.coefficients[ispin, ikpt, active_bands, :]
        num_basis_tot = len(self.basis_map)
        
        k_cart = self.post_wfc.kpoints_cart[ikpt]
        G_basis_cart = self.post_wfc.get_plane_waves_basis_cart_from_idx(ikpt)
        
        q_vecs = G_basis_cart + k_cart[np.newaxis, :]
        q_norms = np.linalg.norm(q_vecs, axis=1)
        
        Chi_q = np.zeros((num_basis_tot, G_basis_cart.shape[0]), dtype=np.complex128)
        for mu, orb in enumerate(self.basis_map):
            spatial_phase = np.exp(-1j * np.dot(q_vecs, self.structure[orb['atom_index']].coords))
            
            Chi_q[mu, :] = spatial_phase * evaluate_orbital_g_space(
                q_vecs, q_norms, orb['l'], orb['m'], orb['alphas'], orb['g_coefficients']
            )
            
        # FIXED: A direct discrete dot product now returns unscaled plane-wave coefficients
        # that perfectly match the raw VASP normalization convention.
        return np.dot(coeffs_active, Chi_q)
    
    def _analyze_basis_leakage(self, basis_map, lattice_matrix, containment_threshold=0.99):
        """
        Dynamically audits basis orbital leakage by comparing the physical charge
        density enclosed within the cell's inscribed sphere against its total 
        infinite-space norm.
        """
        # Calculate the minimum unit cell face-to-face width
        norms = np.linalg.norm(lattice_matrix, axis=1)
        L_min = np.min(norms)
        R_cell = L_min / 2.0  # Maximum radius of an inscribed sphere in the unit cell
        
        classified_basis = []
        
        print("\n" + "="*80)
        print("         DYNAMIC BASIS ETYMOLOGY & INTEGRATED CONTAINMENT ENGINE     ")
        print("="*80)
        print(f"Cell Inscribed Sphere Radius (R_cell): {R_cell:.4f} \u00c5")
        print(f"Target Enclosed Containment Threshold  : {containment_threshold * 100.0:.1f}%")
        print(f" " + "-"*75)
        print(f"{'ID':<4} | {'State':<5} | {'Total Norm':<12} | {'Enclosed Norm':<15} | {'Leakage %':<11} | {'Domain Assignment'}")
        print("-"*80)
        
        # Build radial grids: one up to R_cell, one extending to "infinity" (50 \u00c5)
        r_enclosed = np.linspace(0.0, R_cell, 1000)
        r_infinite = np.linspace(0.0, 50.0, 5000)
        
        for mu, orb in enumerate(basis_map):
            alphas = orb['alphas']
            coeffs = orb['coeffs']
            l = orb['l']
            
            # --- 1. Compute Enclosed Norm up to R_cell ---
            R_enc = np.zeros_like(r_enclosed)
            for k in range(len(alphas)):
                R_enc += coeffs[k] * (r_enclosed ** l) * np.exp(-alphas[k] * (r_enclosed**2))
            prob_enc = (R_enc ** 2) * 4.0 * np.pi * (r_enclosed ** 2)
            enclosed_norm = np.trapezoid(prob_enc, r_enclosed)
            
            # --- 2. Compute Total Analytical Norm (Infinite-Space Approximation) ---
            R_inf = np.zeros_like(r_infinite)
            for k in range(len(alphas)):
                R_inf += coeffs[k] * (r_infinite ** l) * np.exp(-alphas[k] * (r_infinite**2))
            prob_inf = (R_inf ** 2) * 4.0 * np.pi * (r_infinite ** 2)
            total_norm = np.trapezoid(prob_inf, r_infinite)
            
            # Avoid divide-by-zero on unphysical/empty channels
            if total_norm > 1e-8:
                fractional_containment = enclosed_norm / total_norm
            else:
                fractional_containment = 1.0
                
            leakage_percent = (1.0 - fractional_containment) * 100.0
            is_diffuse = fractional_containment < containment_threshold
            
            assignment = "RECIPROCAL GRID (Diffuse)" if is_diffuse else "ANALYTICAL SPACE (Compact)"
            state_label = f"{l}s" if l == 0 else f"{l}p"
            
            print(f" {mu:2d}  | {state_label:<5} | {total_norm:12.6f} | {enclosed_norm:15.6f} | {leakage_percent:10.2f}% | {assignment}")
            
            orb_meta = orb.copy()
            orb_meta['is_diffuse'] = is_diffuse
            orb_meta['containment'] = fractional_containment
            orb_meta['leakage_percent'] = leakage_percent
            classified_basis.append(orb_meta)
            
        print("="*80 + "\n")
        return classified_basis
    
    def _project_system(self):
        """
        Production Projector Augmented Wave (PAW) projection engine.
        Deconstructed into 4 clean mathematical pillars with exact analytical 
        real-space normalization and gauge alignment.
        """
        # TODO:
            # This is very close to working, but the reciprocal space method
            # results in physically incorrect negative spillage.
            # On the other hand, the real space variant is exceptionally slow,
            # at least the most recent version.
            # A couple of thoughts:
                # 1. We might be able to increase the FFT grid size to avoid
                # issues with cusps
                # 2. We ought to be able to use a logarithmic mapping of the
                # real space basis sets to ensure they are well resolved.

        post_wfc = self.post_wfc
        structure = self.structure
        nspin = post_wfc.nspin
        nkpoints = post_wfc.nkpoints
        nbands = post_wfc.nbands
        volume = structure.volume
        num_atoms = len(structure)
        
        # Cartesian positions of all cell sites
        atom_positions = np.array([site.coords for site in structure], dtype=np.float64)
        
        tol = 0.0
        alpha_min = 1e-4
        state_alpha_min = 0.15
        
        print("\n" + "="*80)
        print("         PROJECTOR AUGMENTED WAVE (PAW) PRODUCTION SYSTEM ENGINE        ")
        print("="*80)
        print(f"System Dimensions: Spin={nspin}, k-points={nkpoints}, Bands={nbands}")
        print(f"Unit Cell Volume (Omega): {volume:.6f} Å^3")
        
        #######################################################################
        # STEP 1: Basis Pruning & Exact Analytical Renormalization
        #######################################################################
        # OBJECTIVE: Select, filter, and normalize localized atomic reference 
        # wavefunctions from the all-electron database to construct a clean, 
        # linearly independent orbital space for the crystalline system.
        
        # MECHANICS:
        #   1. Align the database's semi-core states with the crystal's active
        #      DFT energy scale via a rigid reference energy shift.
        #   2. Programmatically identify and include all subshells sitting below 
        #      e_max, plus exactly one additional virtual subshell above it to 
        #      provide complete variational flexibility for high-lying bands.
        #   3. Strip away unphysically sharp or diffuse primitive Gaussians while 
        #      safely protecting physically occupied semi-core and valence states.
        #   4. Execute an exact analytical real-space integration of the truncated 
        #      contracted Gaussians to restore a perfect unit norm (1.000000) for 
        #      every single localized basis function before grid projection.
        
        logging.info("Executing Step 1: Core reference orbital pruning and analytical renormalization.")
        
        unique_elements = set(site.specie.symbol for site in structure)
        species_pruned_bases = {}
        e_min, e_max = self.get_energy_range(method=None)
        
        # Filtering thresholds to discard numerical noise without sacrificing physical completeness
        alpha_min = 1e-4          # Lower bound to eliminate unphysically diffuse primitive tails
        state_alpha_min = 0.01     # Structural ceiling to filter out completely unstable virtual channels
        
        for elem in unique_elements:
            atom_basis = self.atom_bases[elem]
            
            # Align the isolated atom energy spectrum with the periodic crystal baseline by matching 
            # the lowest remaining semi-core state with the lowest bound DFT energy eigenvalue
            atom_energies = atom_basis.eigenvalues.copy()
            atom_energies += e_min - atom_energies[0]
            
            # Programmatically find the energy boundary that spans all states below e_max plus one full subshell
            unique_energies = np.sort(np.unique(atom_energies))
            states_below_max = unique_energies <= e_max
            n_below = np.sum(states_below_max)
            
            if n_below < len(unique_energies):
                # Dynamically capture the next higher unique energy level to encompass the entire next degenerate shell
                cutoff_energy = unique_energies[n_below]
                logging.info(f"Element '{elem}': Crystal valence window maximum = {e_max:.4f} eV. "
                             f"Dynamic cutoff extended to encompass next virtual subshell at {cutoff_energy:.4f} eV.")
            else:
                cutoff_energy = unique_energies[-1]
                logging.info(f"Element '{elem}': Crystal valence window maximum = {e_max:.4f} eV. "
                             f"All available database subshells sit within active window boundaries.")
            
            # Extract indices for all atomic states satisfying our dynamic energy cutoff constraint
            valid_bases = np.where(atom_energies <= (cutoff_energy + 1e-4))[0]
            pruned_orbs_for_elem = []
            
            logging.debug(f"Element '{elem}': Checking {len(valid_bases)} active database channels against structural tail filters.")
            
            for i in valid_bases:
                l = atom_basis.angular_momenta[i]
                m = atom_basis.magnetic_quantum_numbers[i]
                energy = atom_energies[i]
                occ = atom_basis.reference_occupations[i]
                
                # Fetch primitive basis contraction mappings for the current angular momentum l channel
                c_data = atom_basis.primitives[l]
                exps = c_data["exps"]
                coeffs = c_data["coeffs"]
                g_coeffs = c_data["g_coeffs"]
                offsets = c_data["offsets"]
                dim = len(offsets) - 1
                
                # Isolate the JIT-optimized radial state coefficients from the database layout
                start_l, end_l = atom_basis.l_slices[l]
                state_coeffs = atom_basis.state_vectors[i, start_l:end_l]
                
                orb_alphas = []
                orb_coeffs = []
                orb_g_coeffs = []
                
                # Decontract and scale primitives by their state coefficients to parse the active wavefunction profile
                for p in range(dim):
                    start = offsets[p]
                    end = offsets[p+1]
                    c_p = state_coeffs[p]
                    for prim_idx in range(start, end):
                        alpha = exps[prim_idx]
                        if alpha >= alpha_min:
                            orb_alphas.append(alpha)
                            # CRITICAL: Truncation filtering and real-space integrals must scale using real-space coefficients
                            orb_coeffs.append(coeffs[prim_idx] * c_p)
                            orb_g_coeffs.append(g_coeffs[prim_idx] * c_p)
                
                if len(orb_alphas) == 0:
                    continue
                    
                # Guard rails: Protect physically occupied core/valence states from being accidentally dropped by diffuse tail constraints
                if occ < 1e-4 and np.max(orb_alphas) < state_alpha_min:
                    logging.debug(f"  -> State ID {i:2d} (l={l}): Skipped. Unoccupied and too diffuse to sit on active grid safely.")
                    continue
                
                alphas_arr = np.array(orb_alphas, dtype=np.float64)
                coeffs_arr = np.array(orb_coeffs, dtype=np.float64)
                g_coeffs_arr = np.array(orb_g_coeffs, dtype=np.float64)
                
                # Calculate the exact analytical real-space self-overlap integral over the contracted Gaussians.
                # Bypasses grid-undersampling artifacts near the nucleus completely using the standard Gamma function identity:
                # \int_0^\infty r^{2l+2} e^{-(alpha_k + alpha_j)r^2} dr = 0.5 * A^{-(l+1.5)} * \Gamma(l+1.5)
                n_prim = len(alphas_arr)
                trunc_norm = 0.0
                gamma_factor = gamma(l + 1.5)
                
                for k in range(n_prim):
                    for j in range(n_prim):
                        A = alphas_arr[k] + alphas_arr[j]
                        analytical_integral = 0.5 * (A ** -(l + 1.5)) * gamma_factor
                        trunc_norm += coeffs_arr[k] * coeffs_arr[j] * analytical_integral
                
                # Rescale the linear primitive array weights to force a perfect real-space normalization of 1.0
                if trunc_norm > 1e-8:
                    coeffs_arr /= np.sqrt(trunc_norm)
                    g_coeffs_arr /= np.sqrt(trunc_norm)
                
                # VERIFICATION PASS: Re-integrate post-normalization primitive matrices to confirm 1.000000 convergence
                post_norm = 0.0
                for k in range(n_prim):
                    for j in range(n_prim):
                        A = alphas_arr[k] + alphas_arr[j]
                        analytical_integral = 0.5 * (A ** -(l + 1.5)) * gamma_factor
                        post_norm += coeffs_arr[k] * coeffs_arr[j] * analytical_integral
                
                logging.debug(f"  -> State ID {i:2d} (l={l}, m={m}): Shifted Energy = {energy:10.4f} eV | "
                              f"Pre-Norm = {trunc_norm:11.6f} -> Verified Post-Norm = {post_norm:.6f}")
                
                pruned_orbs_for_elem.append({
                    'index': i, 'l': l, 'm': m,
                    'alphas': alphas_arr, 'coeffs': coeffs_arr, 'g_coefficients': g_coeffs_arr,
                    'original_trunc_norm': trunc_norm, 'verified_norm': post_norm
                })
            
            species_pruned_bases[elem] = pruned_orbs_for_elem
            logging.info(f"Element '{elem}': Successfully pruned and verified {len(pruned_orbs_for_elem)} normalized atomic states.")

        # Flatten structural dictionary maps into a unified system-wide basis index array
        basis_map = []
        for i_atom in range(num_atoms):
            elem = structure[i_atom].species_string
            for orb in species_pruned_bases[elem]:
                basis_map.append({
                    'atom_index': i_atom, 'element': elem, 'l': orb['l'], 'm': orb['m'],
                    'alphas': orb['alphas'], 'coeffs': orb['coeffs'], 'g_coefficients': orb['g_coefficients']
                })
        n_basis = len(basis_map)
        logging.info(f"Step 1 Complete: Total System Basis Matrix Size successfully mapped (n_basis = {n_basis}).")
        
        #######################################################################
        # STEP 2: Reciprocal Grid Integration & Smooth Manifold Mapping
        #######################################################################
        # OBJECTIVE: Map the analytically normalized LCAO reference orbitals onto 
        # the discrete periodic plane-wave grid of the unit cell, compute the 
        # reciprocal-space overlap matrix (S_k), and project the smooth DFT valence 
        # eigenstates onto the localized basis functions (P_ps).
        
        # MECHANICS:
        #   1. Construct the 3D Cartesian scattering vectors (q = G + k) for each 
        #      k-point plane-wave grid context.
        #   2. Evaluate the analytical Fourier transforms of the localized orbitals 
        #      in G-space, applying the structural phase factors e^{-i G . R}.
        #   3. Establish a precise, grid-matched normalization conversion scale 
        #      by mapping the maximum valence diagonal element to unity.
        #   4. Assert explicit shape and norm compliance on incoming DFT planewave 
        #      coefficients to catch grid representation mismatches before projection.
        logging.info("Step 2: Projecting localized orbitals onto periodic plane-wave grids.")
        
        S_k = np.zeros((nkpoints, n_basis, n_basis), dtype=np.complex128)
        P_ps = np.zeros((nspin, nkpoints, nbands, n_basis), dtype=np.complex128)
        
        # Initialize atom-centered PAW projector weight matrices
        C_paw = []
        for i_atom in range(num_atoms):
            elem = structure[i_atom].species_string
            dataset = post_wfc._aug_environment.paw_datasets[elem]
            C_paw.append(np.zeros((nspin, nkpoints, nbands, len(dataset.angular_momenta)), dtype=np.complex128))

        # --- Baseline Grid Normalization Pre-Calculation ---
        # FIXED: Use the true physical unit cell volume (Omega) as the absolute scale
        # to ensure plane-wave spaces match the absolute real-space core spheres.
        norm_factor = volume
        logging.info(f"Absolute unit cell volume scaling factor locked: {norm_factor:.6f}")
        ikpt_ref = 0
        gvecs_ref, _ = post_wfc.get_plane_waves_basis_idx(ikpt_ref, grid_shape=post_wfc._minimum_fft_size * 2)
        rgvec_ref = gvecs_ref @ (2 * np.pi * post_wfc.reciprocal_lattice)
        q_vecs_ref = rgvec_ref + post_wfc.kpoints_cart[ikpt_ref][np.newaxis, :]
        q_norms_ref = np.linalg.norm(q_vecs_ref, axis=1)
        
        chi_matrix_ref = np.zeros((n_basis, len(q_norms_ref)), dtype=np.complex128)
        for mu, orb in enumerate(basis_map):
            # Translate 0-indexed database layouts back to physical harmonic conventions [-l, ..., +l]
            m_physical = orb['m'] - orb['l']
            chi_g = evaluate_orbital_g_space(q_vecs_ref, q_norms_ref, orb['l'], m_physical, orb['alphas'], orb['g_coefficients'])
            phase_factor = np.exp(-1j * np.dot(q_vecs_ref, atom_positions[orb['atom_index']]))
            chi_matrix_ref[mu, :] = chi_g * phase_factor

        # Use the maximum valence self-overlap diagonal as our system-wide integration volume baseline
        # norm_factor = np.max((chi_matrix_ref.conj() @ chi_matrix_ref.T).real.diagonal())
        # logging.debug(f"Programmatic G-space grid normalization factor locked: {norm_factor:.6f}")

        # --- Main Reciprocal Space Integration Loop ---
        for ikpt_idx in range(nkpoints):
            # Extract active 3D Cartesian scattering vectors (q = G + k) for the current grid context
            gvecs, _ = post_wfc.get_plane_waves_basis_idx(ikpt_idx, grid_shape=post_wfc._minimum_fft_size * 2)
            rgvec_k = gvecs @ (2 * np.pi * post_wfc.reciprocal_lattice)
            q_vecs_k = rgvec_k + post_wfc.kpoints_cart[ikpt_idx][np.newaxis, :]
            q_norms_k = np.linalg.norm(q_vecs_k, axis=1)
            n_q_k = len(q_norms_k)
            
            # Map the all-electron basis functions onto the current k-point plane-wave grid
            chi_mat_k = np.zeros((n_basis, n_q_k), dtype=np.complex128)
            for mu, orb in enumerate(basis_map):
                m_physical = orb['m'] - orb['l']
                chi_g = evaluate_orbital_g_space(q_vecs_k, q_norms_k, orb['l'], m_physical, orb['alphas'], orb['g_coefficients'])
                phase = np.exp(-1j * np.dot(q_vecs_k, atom_positions[orb['atom_index']]))
                chi_mat_k[mu, :] = chi_g * phase
            
            # Compute the localized overlap matrix scaled by our volume normalization baseline
            S_k[ikpt_idx] = (chi_mat_k.conj() @ chi_mat_k.T) / norm_factor
            
            # Build Cartesian PAW pseudopotential projector matrices for all cell sites
            G_basis_cart = post_wfc.get_plane_waves_basis_cart_from_idx(ikpt_idx)
            P_G_atoms = []
            for i_atom in range(num_atoms):
                elem = structure[i_atom].species_string
                dataset = post_wfc._aug_environment.paw_datasets[elem]
                P_G_atoms.append(dataset.build_g_space_projectors(
                    k_cart=post_wfc.kpoints_cart[ikpt_idx], g_vectors_cart=G_basis_cart,
                    atom_cart_pos=structure[i_atom].coords, cell_volume=volume
                ))
                
            for ispin in range(nspin):
                # Load the raw DFT coefficients from the wavefunction backup
                wfc_coeffs = post_wfc.get_plane_wave_coefficients_batch(ispin, ikpt_idx, np.arange(nbands))
                
                # Hard Validation Guard: Enforce strict row-column plane-wave matching
                if wfc_coeffs.shape[1] != n_q_k:
                    logging.error(f"Critical grid error at k-point {ikpt_idx}: wfc columns ({wfc_coeffs.shape[1]}) "
                                  f"do not match reciprocal mesh count ({n_q_k}).")
                    raise ValueError("Wavefunction storage geometry and active plane-wave mesh are out of sync.")
                
                # Project the smooth valence wavefunctions and extract atom-centered projector weights
                P_ps[ispin, ikpt_idx, :, :] = (wfc_coeffs @ chi_mat_k.conj().T) / np.sqrt(norm_factor)
                for i_atom in range(num_atoms):
                    C_paw[i_atom][ispin, ikpt_idx, :, :] = wfc_coeffs @ P_G_atoms[i_atom].T

        logging.info("Step 2 Complete: Reciprocal manifolds and projector arrays successfully mapped.")
                    
        #######################################################################
        # STEP 3: Core Augmentation Corrections & Phase Gauge Alignment
        #######################################################################
        """
        OBJECTIVE: Compute the real-space core-reconstruction discrepancy integrals 
        (Delta_M) inside the PAW augmentation spheres, accumulate the multi-atom 
        contributions, and verify phase gauge alignment.
        
        MECHANICS:
          1. Correctly reconstruct the real-space radial wavefunction profile R(r)
             over the PAW logarithmic grid using real-space coefficients.
          2. Evaluate the 1D overlap integral between the LCAO basis and the 
             all-electron minus pseudo partial wave difference up to the cutoff radius.
          3. Contract the spatial projector weights (C_paw) with the discrepancy 
             matrix (Delta_M) using an Einstein summation pass.
          4. Audit the phase coherence sign between the smooth valence background 
             and core corrections to prevent destructive manifold cancellation.
        """
        logging.info("Step 3: Resolving real-space core augmentation corrections and phase gauge alignment.")
        
        Delta_M = []
        for i_atom in range(num_atoms):
            elem = structure[i_atom].species_string
            dataset = post_wfc._aug_environment.paw_datasets[elem]
            r_grid = dataset.radial_grid
            u_ae = dataset.all_electron_partial_waves
            u_ps = dataset.pseudo_partial_waves
            
            # Map system basis functions onto localized atomic partial wave channels
            M_atom = np.zeros((n_basis, len(dataset.angular_momenta)), dtype=np.float64)
            
            for mu, orb in enumerate(basis_map):
                # Spatial Guard: Skip evaluations for basis functions centered on other cell sites
                if orb['atom_index'] != i_atom:
                    continue
                    
                l_mu = orb['l']
                # Map 0-indexed database layouts back to standard physical real harmonic conventions [-l, ..., +l]
                m_physical = orb['m'] - l_mu
                
                # Reconstruct the true real-space radial representation using real-space coefficients
                R_mu_r = np.zeros(len(r_grid), dtype=np.float64)
                for k in range(len(orb['alphas'])):
                    R_mu_r += orb['coeffs'][k] * np.exp(-orb['alphas'][k] * (r_grid**2))
                if l_mu > 0:
                    R_mu_r *= (r_grid ** l_mu)
                
                # Integrate across every tabulated atomic projector/partial wave channel
                for i in range(len(dataset.angular_momenta)):
                    # Enforce strict angular momentum and physical magnetic quantum number matching
                    if dataset.angular_momenta[i] != l_mu or dataset.magnetic_nums[i] != m_physical:
                        continue
                        
                    # Target only the region enclosed by the PAW augmentation sphere cutoff radius (r_c)
                    r_c = dataset.cutoff_radii[i]
                    active_mask = r_grid <= r_c
                    
                    # Core Integrand: R_mu(r) * [u_ae(r) - u_ps(r)] * r * dr
                    integrand = R_mu_r * (u_ae[i] - u_ps[i]) * r_grid
                    M_atom[mu, i] = np.trapezoid(integrand[active_mask], r_grid[active_mask])
            
            Delta_M.append(M_atom)

        # Contract spatial projector weights with the core discrepancy matrices
        core_accum = np.zeros_like(P_ps)
        for i_atom in range(num_atoms):
            core_accum += np.einsum('sknp,bp->sknb', C_paw[i_atom], Delta_M[i_atom])
            
        # Audit global directional phase alignment between the smooth background and core shells
        coherence = np.sum(P_ps.conj() * core_accum).real
        logging.debug(f"Cross-manifold phase gauge coherence determined: {coherence:.4e}")
        
        # Apply automatic alignment fallback in case of global dataset inversion
        if coherence < 0:
            logging.warning("Destructive phase interference caught between LCAO and PAW gauges. Aligning sign.")
            core_accum *= -1.0
            
        p_total = P_ps + core_accum
        logging.info("Step 3 Complete: Core augmentation corrections successfully resolved and gauge-aligned.")
        
        #######################################################################
        # PIPELINE SCALE AUDIT SUITE (DIAGNOSTIC)
        #######################################################################
        print(f"\n" + "="*80)
        print(f"            PIPELINE RADIAL & PROJECTOR SCALE AUDIT SUITE              ")
        print("="*80)
        
        # 1. Audit the real-space log-radial grid spacing metric
        print(f"[AUDIT 1] Radial Grid Unit Integrity:")
        for i_atom in range(num_atoms):
            elem = structure[i_atom].species_string
            dataset = post_wfc._aug_environment.paw_datasets[elem]
            r = dataset.radial_grid
            
            # Compute the total integrated volume element of the core sphere: \int_0^{r_c} 4\pi r^2 dr
            r_c = dataset.cutoff_radii[0]
            mask = r <= r_c
            sphere_vol = np.trapezoid(4.0 * np.pi * (r[mask]**2), r[mask])
            print(f"  * Atom {i_atom} ({elem}): Cutoff Radius = {r_c:.4f} | Integrated Sphere Volume = {sphere_vol:.4f}")
            
        # 2. Audit the raw internal partial wave normalization
        print(f"\n[AUDIT 2] PAW Dataset Partial Wave Completeness:")
        for i_atom in range(num_atoms):
            elem = structure[i_atom].species_string
            dataset = post_wfc._aug_environment.paw_datasets[elem]
            u_ae = dataset.all_electron_partial_waves
            u_ps = dataset.pseudo_partial_waves
            r = dataset.radial_grid
            
            for i in range(min(5, len(dataset.angular_momenta))):
                r_c = dataset.cutoff_radii[i]
                mask = r <= r_c
                # Tabulated partial waves in VASP are u(r) = r*R(r), so norm is \int u^2 dr
                norm_ae = np.trapezoid(u_ae[i][mask]**2, r[mask])
                norm_ps = np.trapezoid(u_ps[i][mask]**2, r[mask])
                deficit = norm_ae - norm_ps
                print(f"  * Channel {i} (l={dataset.angular_momenta[i]}): AE Self-Norm = {norm_ae:.4f} | PS Self-Norm = {norm_ps:.4f} | Net Core Deficit = {deficit:.4f}")

        # 3. Audit the Projector Weight Magnitude Relative to Smooth States
        print(f"\n[AUDIT 3] Cross-Subsystem Magnitude Profile (k-point 0):")
        for n in range(nbands):
            raw_pps_norm = np.linalg.norm(P_ps[0, 0, n, :])
            raw_core_norm = np.linalg.norm(core_accum[0, 0, n, :])
            print(f"  * Band {n}: Smooth Projection Vector Norm = {raw_pps_norm:.4f} | Core Correction Vector Norm = {raw_core_norm:.4f}")
        print("="*80 + "\n")
        #######################################################################
        # DIAGNOSTIC: LOGARITHMIC RADIAL SUPERCELL AUDIT SUITE
        #######################################################################
        print("\n" + "="*80)
        print("        LOGARITHMIC RADIAL SUPERCELL PERIODIC INTEGRATION AUDIT       ")
        print("="*80)
        
        lattice_vectors = structure.lattice.matrix
        norms = np.linalg.norm(lattice_vectors, axis=1)
        L_min = np.min(norms)
        
        # 1. Generate a high-fidelity logarithmic radial grid
        # N_log points packed heavily near r_min (near nucleus) out to a large supercell radius
        N_log = 2000
        r_min = 1e-6
        r_max = 25.0 
        log_mesh = np.linspace(np.log(r_min), np.log(r_max), N_log)
        r_log = np.exp(log_mesh)
        dr_log = r_log * (log_mesh[1] - log_mesh[0]) # Jacobian included in differential element
        
        # Angular sampling baseline (Spherical Integration via \u03b8, \u03c6 quadrature)
        # For l=0 cross-talk, a standard angular mesh is perfectly exact
        n_theta, n_phi = 32, 64
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2*np.pi, n_phi)
        THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
        
        # Angular integration weight element
        d_omega = np.sin(THETA) * (theta[1] - theta[0]) * (phi[1] - phi[0])
        
        # Convert angular mesh to unit Cartesian directions
        kx = np.sin(THETA) * np.cos(PHI)
        ky = np.sin(THETA) * np.sin(PHI)
        kz = np.cos(THETA)
        
        print(f"Evaluating exact log-radial overlap matrix over 5x5x5 lattice translations:")
        print(f"{'Matrix Element':<16} | {'Grid-Based S_k':<16} | {'Log-Radial Supercell S_R':<25} | {'Delta'}")
        print("-"*80)
        
        # Audit the target s-block subspace (Basis 0: 3s, Basis 4: 4s)
        s_indices = [0, 4]
        
        for mu in s_indices:
            for nu in s_indices:
                if mu > nu: continue # Symmetric matrix
                
                orb_mu = basis_map[mu]
                orb_nu = basis_map[nu]
                
                total_s_r = 0.0
                
                # Loop over periodic image shells R
                range_mesh = np.arange(-2, 3)
                for nx in range_mesh:
                    for ny in range_mesh:
                        for nz in range_mesh:
                            R_vec = nx * lattice_vectors[0] + ny * lattice_vectors[1] + nz * lattice_vectors[2]
                            
                            # Integrate over the 3D spherical volume elements using our log-radial grid
                            cell_overlap = 0.0
                            
                            for i_r in range(N_log):
                                r = r_log[i_r]
                                # Evaluate central orbital \chi_\mu(\vec{r})
                                # R_\mu(r)
                                R_mu_val = np.sum(orb_mu['coeffs'] * np.exp(-orb_mu['alphas'] * (r**2))) / np.sqrt(4.0 * np.pi)
                                
                                # Evaluate shifted coordinate vectors: \vec{r} - \vec{R}
                                rx_vec = r * kx - R_vec[0]
                                ry_vec = r * ky - R_vec[1]
                                rz_vec = r * kz - R_vec[2]
                                r_shift_sq = rx_vec**2 + ry_vec**2 + rz_vec**2
                                
                                # Evaluate shifted orbital profile \chi_\nu(\vec{r} - \vec{R})
                                R_nu_shifted = np.zeros_like(r_shift_sq)
                                for k in range(len(orb_nu['alphas'])):
                                    R_nu_shifted += orb_nu['coeffs'][k] * np.exp(-orb_nu['alphas'][k] * r_shift_sq)
                                R_nu_shifted /= np.sqrt(4.0 * np.pi)
                                
                                # Angular integration integration over solid angle
                                angular_sum = np.sum(R_nu_shifted * d_omega)
                                
                                # Accumulate radial shell volume element: r^2 * dr * R_mu * \int R_nu d\Omega
                                cell_overlap += (r**2) * dr_log[i_r] * R_mu_val * angular_sum
                                
                            total_s_r += cell_overlap
                            
                grid_val = S_k[0, mu, nu].real
                delta = abs(grid_val - total_s_r)
                print(f"  S_{mu}{nu} ({'3s' if mu==0 else '4s'}-{'3s' if nu==0 else '4s'}) | {grid_val:14.6f} | {total_s_r:23.6f}   | {delta:.6f}")
                
        print("="*80 + "\n")
        #######################################################################
        # STEP 4: Unified Log-Radial Supercell Subspace Assembly
        #######################################################################
        logging.info("Executing Step 4: Constructing unified periodic analytical metric.")
        
        c_final = np.zeros_like(p_total, dtype=np.complex128)
        S_AE = np.zeros_like(S_k)
        
        # 1. Pre-calculate the exact log-radial periodic matrix for this crystal geometry
        # (Using the identical kernel setup verified in your diagnostic audit)
        N_log = 2000
        r_mesh = np.linspace(np.log(1e-6), np.log(25.0), N_log)
        r_log = np.exp(r_mesh)
        dr_log = r_log * (r_mesh[1] - r_mesh[0])
        
        n_theta, n_phi = 32, 64
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2*np.pi, n_phi)
        THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
        d_omega = np.sin(THETA) * (theta[1] - theta[0]) * (phi[1] - phi[0])
        
        kx = np.sin(THETA) * np.cos(PHI)
        ky = np.sin(THETA) * np.sin(PHI)
        kz = np.cos(THETA)
        
        lattice_vectors = structure.lattice.matrix
        range_mesh = np.arange(-2, 3)
        
        S_periodic_base = np.zeros((n_basis, n_basis), dtype=np.complex128)
        
        for mu in range(n_basis):
            for nu in range(n_basis):
                orb_mu = basis_map[mu]
                orb_nu = basis_map[nu]
                total_overlap = 0.0
                
                # Spatial translation loop over neighboring periodic cell shells
                for nx in range_mesh:
                    for ny in range_mesh:
                        for nz in range_mesh:
                            R_vec = nx * lattice_vectors[0] + ny * lattice_vectors[1] + nz * lattice_vectors[2]
                            cell_overlap = 0.0
                            
                            for i_r in range(N_log):
                                r = r_log[i_r]
                                R_mu_val = np.sum(orb_mu['coeffs'] * np.exp(-orb_mu['alphas'] * (r**2))) / np.sqrt(4.0 * np.pi)
                                
                                rx = r * kx - R_vec[0]
                                ry = r * ky - R_vec[1]
                                rz = r * kz - R_vec[2]
                                r_shift_sq = rx**2 + ry**2 + rz**2
                                
                                R_nu_shifted = np.zeros_like(r_shift_sq)
                                for k in range(len(orb_nu['alphas'])):
                                    R_nu_shifted += orb_nu['coeffs'][k] * np.exp(-orb_nu['alphas'][k] * r_shift_sq)
                                R_nu_shifted /= np.sqrt(4.0 * np.pi)
                                
                                cell_overlap += (r**2) * dr_log[i_r] * R_mu_val * np.sum(R_nu_shifted * d_omega)
                                
                            total_overlap += cell_overlap
                S_periodic_base[mu, nu] = total_overlap

        # 2. Map the pre-calculated periodic target matrix across all k-points
        for ikpt_idx in range(nkpoints):
            for mu in range(n_basis):
                for nu in range(n_basis):
                    # Inject the unified log-radial periodic matrix element
                    S_AE[ikpt_idx, mu, nu] = S_periodic_base[mu, nu]

        # 3. Execute Subspace Inversion via Canonical Orthogonalization
        for ikpt_idx in range(nkpoints):
            s_eigenvals, s_eigenvecs = np.linalg.eigh(S_AE[ikpt_idx].real)
            max_ev = np.max(s_eigenvals)
            
            lin_dep_threshold = 1e-4 * max_ev
            keep_indices = np.where(s_eigenvals > lin_dep_threshold)[0]
            
            s_val_reduced = s_eigenvals[keep_indices]
            s_vec_reduced = s_eigenvecs[:, keep_indices]
            inv_sqrt_s = s_vec_reduced / np.sqrt(s_val_reduced)
            s_inverse_k = inv_sqrt_s @ inv_sqrt_s.conj().T
            
            for ispin in range(nspin):
                c_final[ispin, ikpt_idx, :, :] = p_total[ispin, ikpt_idx, :, :] @ s_inverse_k.conj()

        # Finalize and print the clean spillage outputs
        captured_manifold = np.einsum('sknb,sknb->skn', c_final.conj(), p_total).real
        self.band_spillage = 1.0 - captured_manifold
        
        print(f"\n[STEP 4 DIAGNOSTICS] Aligned All-Electron Band Spillage Profile (k-point 0):")
        for n in range(nbands):
            state_spillage = self.band_spillage[0, 0, n]
            print(f"  * Band {n:2d}: Occupancy = {post_wfc.occupancies[0,0,n]:.4f} | Manifold Spillage = {state_spillage:9.6f}")
            
        self.integrated_spillage = np.einsum('k,skn,skn->', post_wfc.kpoint_weights, post_wfc.occupancies, self.band_spillage) / np.sum(post_wfc.occupancies)
        print("\n" + "="*80)
        print(f"PRODUCTION SOLUTION RESOLVED -> Integrated System Spillage: {self.integrated_spillage:.6f}")
        print("="*80 + "\n")
        
        return c_final
        
    @classmethod
    def from_directory(
        cls, 
        directory: Path | str = Path("."), 
        fmt: str = "vasp", 
        scipy_workers: int = -1, 
        **kwargs,
    ):
        from baderkit.post_wfc.post_wfc import PostWFC
        post_wfc = PostWFC.from_directory(
            directory=directory, 
            fmt=fmt, 
            scipy_workers=scipy_workers, 
            **kwargs,
        )
        return post_wfc.projection_environment