# -*- coding: utf-8 -*-

from pathlib import Path
import logging
from functools import cached_property
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid
from scipy.special import gamma
from rich.progress import track

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
    # Try to fix cusp difference. Might help to put atomic bases through an FFT filter of some kind. Might also help projection as well
    # See if we can resove the spillage at higher unoccupied statess. I don't see why these should exist.

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
            
        # FIX: Scaling by 1 / sqrt(Volume) resolves the LCAO-to-Periodic plane wave continuum mapping mismatch
        volume_norm = 1.0 / np.sqrt(self.post_wfc.structure.volume)
        return np.dot(coeffs_active, Chi_q) * volume_norm
    
    def _project_system(self):
        """
        Production Projector Augmented Wave (PAW) projection engine.
        Deconstructed into 4 clean mathematical pillars with exact analytical 
        real-space normalization and gauge alignment.
        """
        post_wfc = self.post_wfc
        structure = self.structure
        nspin = post_wfc.nspin
        nkpoints = post_wfc.nkpoints
        nbands = post_wfc.nbands
        volume = structure.volume
        num_atoms = len(structure)
        l_symbols = {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g'}
        
        print("\n" + "="*80)
        print("          PROJECTOR AUGMENTED WAVE (PAW) PRODUCTION SYSTEM ENGINE        ")
        print("="*80)
        print(f"System Dimensions: Spin={nspin}, k-points={nkpoints}, Bands={nbands}")
        print(f"Unit Cell Volume (Omega): {volume:.6f} Å^3")
        
        # Cartesian positions of all cell sites
        atom_positions = np.array([site.coords for site in structure], dtype=np.float64)
        
        tol = 0.1
        alpha_min = 1e-4
        state_alpha_min = 0.15
        
        # =====================================================================
        # DIAGNOSTIC: RAW UNPRUNED ATOMIC BASE SPECTRUM
        # =====================================================================
        print("\n" + "="*80)
        print("          RAW UNPRUNED ATOMIC DATABASE SPECTRUM (PRE-PRUNING)          ")
        print("="*80)
        unique_elements = set(site.specie.symbol for site in structure)
        for elem in unique_elements:
            if elem not in self.atom_bases:
                print(f"Element {elem} not found in atom_bases library.")
                continue
            atom_basis = self.atom_bases[elem]
            print(f"Element: {elem}")
            print(f"{'Index':<5} | {'True Shell':<10} | {'Energy (eV)':<12} | {'Occupation':<10} | {'Spin':<5}")
            print("-"*60)
            
            for idx in range(len(atom_basis.eigenvalues)):
                l = atom_basis.angular_momenta[idx]
                n = atom_basis.principal_quantum_numbers[idx]
                energy = atom_basis.eigenvalues[idx]
                occ = atom_basis.reference_occupations[idx]
                spin = atom_basis.spin_channels[idx]
                
                l_sym = l_symbols.get(l, f"l={l}")
                shell_label = f"{n}{l_sym}"
                
                print(f" {idx:<5} | {shell_label:<10} | {energy:12.4f} | {occ:10.4f} | {spin:<5}")
        print("="*80 + "\n")
        
        #######################################################################
        # STEP 1: Basis Pruning & Symmetry-Balanced Virtual Recovery
        #######################################################################
        logging.info("Step 1: Core reference orbital pruning and symmetry-balanced recovery.")
        
        unique_elements = set(site.specie.symbol for site in structure)
        e_min, e_max = self.get_energy_range(method=None)
        
        element_counts = {}
        for site in structure:
            sym = site.specie.symbol
            element_counts[sym] = element_counts.get(sym, 0) + 1
            
        alpha_min = 1e-4          
        state_alpha_min = 0.01   
        total_basis_count = 0
        
        for elem in unique_elements:
            atom_basis = self.atom_bases[elem]
            for l_val in atom_basis.angular_momenta:
                total_basis_count += (2 * l_val + 1) * element_counts[elem]
        
        # --- PASS 1: Raw Parsing and Confinement Filtering ---
        candidates_all_elements = {}
        for elem in unique_elements:
            atom_basis = self.atom_bases[elem]
            
            atom_energies = atom_basis.eigenvalues.copy()
            atom_energies += e_min - atom_energies[0]
            
            unique_energies = np.sort(np.unique(atom_energies))
            states_below_max = unique_energies <= e_max
            n_below = np.sum(states_below_max)
            
            cutoff_energy = unique_energies[n_below] if n_below < len(unique_energies) else unique_energies[-1]
            valid_bases = np.where(atom_energies <= (cutoff_energy + (cutoff_energy * tol)))[0]
            
            element_candidates = []
            for i in valid_bases:
                l = atom_basis.angular_momenta[i]
                energy = atom_energies[i]
                occ = atom_basis.reference_occupations[i]
                
                c_data = atom_basis.primitives[l]
                exps = c_data["exps"]
                coeffs = c_data["coeffs"]
                g_coeffs = c_data["g_coeffs"]
                offsets = c_data["offsets"]
                dim = len(offsets) - 1
                
                start_l, end_l = atom_basis.l_slices[l]
                state_coeffs = atom_basis.state_vectors[i, start_l:end_l]
                
                orb_alphas = []
                orb_coeffs = []
                orb_g_coeffs = []
                
                for p in range(dim):
                    start = offsets[p]
                    end = offsets[p+1]
                    c_p = state_coeffs[p]
                    for prim_idx in range(start, end):
                        alpha = exps[prim_idx]
                        if alpha >= alpha_min:
                            orb_alphas.append(alpha)
                            orb_coeffs.append(coeffs[prim_idx] * c_p)
                            orb_g_coeffs.append(g_coeffs[prim_idx] * c_p)
                
                if len(orb_alphas) == 0:
                    continue
                    
                alphas_arr = np.array(orb_alphas, dtype=np.float64)
                coeffs_arr = np.array(orb_coeffs, dtype=np.float64)
                g_coeffs_arr = np.array(orb_g_coeffs, dtype=np.float64)
                
                max_alpha = np.max(alphas_arr)
                min_alpha_val = np.min(alphas_arr)
                is_occupied = occ >= 1e-4
                is_confined = max_alpha >= state_alpha_min
                
                initially_kept = is_occupied or is_confined
                origin_status = "OCCUPIED" if is_occupied else ("CONFINED" if is_confined else "REJECTED VIRTUAL")
                
                element_candidates.append({
                    'index': i, 'l': l, 'energy': energy, 'occ': occ,
                    'alphas': alphas_arr, 'coeffs': coeffs_arr, 'g_coefficients': g_coeffs_arr,
                    'max_alpha': max_alpha, 'min_alpha': min_alpha_val, 'initially_kept': initially_kept,
                    'origin_status': origin_status
                })
                
            candidates_all_elements[elem] = element_candidates

        # --- PASS 2: Dynamic Symmetry-Balanced Round-Robin Safeguard ---
        def calc_total_basis(basis_set_dict):
            total = 0
            for el, orbs in basis_set_dict.items():
                for orb in orbs:
                    total += (2 * orb['l'] + 1) * element_counts[el]
            return total

        species_pruned_bases = {el: [o for o in orbs if o['initially_kept']] for el, orbs in candidates_all_elements.items()}
        current_total_basis = calc_total_basis(species_pruned_bases)
        
        if current_total_basis < nbands:
            logging.info(f"Basis size shortfall detected ({current_total_basis} functions < {nbands} bands). Transitioning to round-robin symmetry recovery.")
            
            pool_by_symmetry = {}
            for elem, orbs in candidates_all_elements.items():
                for orb in orbs:
                    if not orb['initially_kept']:
                        l = orb['l']
                        pool_by_symmetry.setdefault((elem, l), []).append(orb)
            
            for key in pool_by_symmetry:
                pool_by_symmetry[key].sort(key=lambda x: x['max_alpha'], reverse=True)
            
            symmetry_keys = sorted(list(pool_by_symmetry.keys()))
            
            while current_total_basis < nbands and symmetry_keys:
                added_in_round = 0
                for key in list(symmetry_keys):
                    if pool_by_symmetry[key]:
                        orb = pool_by_symmetry[key].pop(0)
                        orb['origin_status'] = f"RECOVERED VIRTUAL ({l_symbols.get(key[1], 'l='+str(key[1]))})"
                        species_pruned_bases[key[0]].append(orb)
                        added_in_round += 1
                        
                        current_total_basis = calc_total_basis(species_pruned_bases)
                        if current_total_basis >= nbands:
                            break
                    else:
                        symmetry_keys.remove(key)
                
                if added_in_round == 0:
                    break

        # --- FINAL PASS: Exact Analytical Renormalization of Retained States ---
        for elem in unique_elements:
            for orb in species_pruned_bases[elem]:
                l = orb['l']
                alphas_arr = orb['alphas']
                coeffs_arr = orb['coeffs']
                g_coeffs_arr = orb['g_coefficients']
                n_prim = len(alphas_arr)
                
                trunc_norm = 0.0
                gamma_factor = gamma(l + 1.5)
                
                for k in range(n_prim):
                    for j in range(n_prim):
                        A = alphas_arr[k] + alphas_arr[j]
                        analytical_integral = 0.5 * (A ** -(l + 1.5)) * gamma_factor
                        trunc_norm += coeffs_arr[k] * coeffs_arr[j] * analytical_integral
                
                if trunc_norm > 1e-8:
                    coeffs_arr /= np.sqrt(trunc_norm)
                    g_coeffs_arr /= np.sqrt(trunc_norm)
                
                post_norm = 0.0
                for k in range(n_prim):
                    for j in range(n_prim):
                        A = alphas_arr[k] + alphas_arr[j]
                        analytical_integral = 0.5 * (A ** -(l + 1.5)) * gamma_factor
                        post_norm += coeffs_arr[k] * coeffs_arr[j] * analytical_integral
                        
                orb['verified_norm'] = post_norm

        basis_map = []
        for i_atom in range(num_atoms):
            elem = structure[i_atom].species_string
            for orb in species_pruned_bases[elem]:
                l = orb['l']
                for m_idx in range(2 * l + 1):
                    m_physical = m_idx - l  
                    
                    basis_map.append({
                        'atom_index': i_atom, 
                        'element': elem, 
                        'l': l, 
                        'm': m_physical, 
                        'alphas': orb['alphas'], 
                        'coeffs': orb['coeffs'], 
                        'g_coefficients': orb['g_coefficients'],
                        'energy': orb['energy'], 
                        'occ': orb['occ'], 
                        'origin_status': orb['origin_status'],
                        'min_alpha': orb['min_alpha'], 
                        'max_alpha': orb['max_alpha']
                    })
        n_basis = len(basis_map)

        #######################################################################
        # DIAGNOSTIC: PRUNED BASIS EXPONENT SPECTRUM REPORT
        #######################################################################
        print("\n" + "="*80)
        print("          PRUNED BASIS SET PRIMITIVE EXPONENT SPECTRUM REPORT          ")
        print("="*80)
        
        lbl_id, lbl_state, lbl_eng, lbl_occ = "ID", "State", "Energy (eV)", "Occupancy"
        lbl_min, lbl_max, lbl_origin = "Min Alpha", "Max Alpha", "Origin Status"
        
        print(f"{lbl_id:<4} | {lbl_state:<8} | {lbl_eng:<11} | {lbl_occ:<9} | {lbl_min:<11} | {lbl_max:<11} | {lbl_origin}")
        print("-"*80)
        
        for mu, orb in enumerate(basis_map):
            l_sym = l_symbols.get(orb['l'], f"l={orb['l']}")
            state_base = f"{orb['l']}{l_sym}"
            m_phys = orb['m']  # FIX: Reads true precalculated physical quantum number directly
            state_label_with_m = f"{state_base}({m_phys:+d})" if orb['l'] > 0 else state_base
            
            print(f" {mu:2d}  | {state_label_with_m:<8} | {orb['energy']:11.4f} | {orb['occ']:9.4f} | {orb['min_alpha']:11.4e} | {orb['max_alpha']:11.4e} | {orb['origin_status']}")
            
        print("="*80 + "\n")
        
        #######################################################################
        # STEP 2: Reciprocal Grid Integration & Smooth Manifold Mapping
        #######################################################################
        logging.info("Step 2: Projecting localized orbitals onto periodic plane-wave grids.")
        
        S_k = np.zeros((nkpoints, n_basis, n_basis), dtype=np.complex128)
        P_ps = np.zeros((nspin, nkpoints, nbands, n_basis), dtype=np.complex128)
        
        C_paw = []
        for i_atom in range(num_atoms):
            elem = structure[i_atom].species_string
            dataset = post_wfc._aug_environment.paw_datasets[elem]
            C_paw.append(np.zeros((nspin, nkpoints, nbands, len(dataset.angular_momenta)), dtype=np.complex128))

        norm_factor = volume
        
        for ikpt_idx in track(range(nkpoints), description="[bold blue]Mapping Reciprocal Projections...[/]"):
            gvecs, _ = post_wfc.get_plane_waves_basis_idx(ikpt_idx, grid_shape=post_wfc._minimum_fft_size * 2)
            rgvec_k = gvecs @ (2 * np.pi * post_wfc.reciprocal_lattice)
            q_vecs_k = rgvec_k + post_wfc.kpoints_cart[ikpt_idx][np.newaxis, :]
            q_norms_k = np.linalg.norm(q_vecs_k, axis=1)
            n_q_k = len(q_norms_k)
            
            chi_mat_k = np.zeros((n_basis, n_q_k), dtype=np.complex128)
            for mu, orb in enumerate(basis_map):
                chi_g = evaluate_orbital_g_space(
                    q_vecs_k, q_norms_k, orb['l'], orb['m'], orb['alphas'], orb['g_coefficients']
                )
                phase = np.exp(-1j * np.dot(q_vecs_k, atom_positions[orb['atom_index']]))
                chi_mat_k[mu, :] = chi_g * phase
            
            S_k[ikpt_idx] = (chi_mat_k.conj() @ chi_mat_k.T) / norm_factor
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
                wfc_coeffs = post_wfc.get_plane_wave_coefficients_batch(ispin, ikpt_idx, np.arange(nbands))
                if wfc_coeffs.shape[1] != n_q_k:
                    raise ValueError("Wavefunction storage geometry and active plane-wave mesh are out of sync.")
                
                P_ps[ispin, ikpt_idx, :, :] = (wfc_coeffs @ chi_mat_k.conj().T) / np.sqrt(norm_factor)
                for i_atom in range(num_atoms):
                    C_paw[i_atom][ispin, ikpt_idx, :, :] = wfc_coeffs @ P_G_atoms[i_atom].T
                    
        #######################################################################
        # STEP 3: Core Augmentation Corrections & Phase Gauge Alignment
        #######################################################################
        logging.info("Step 3: Resolving real-space core augmentation corrections and phase gauge alignment.")
        
        Delta_M = []
        for i_atom in range(num_atoms):
            elem = structure[i_atom].species_string
            dataset = post_wfc._aug_environment.paw_datasets[elem]
            r_grid = dataset.radial_grid
            u_ae = dataset.all_electron_partial_waves
            u_ps = dataset.pseudo_partial_waves
            
            M_atom = np.zeros((n_basis, len(dataset.angular_momenta)), dtype=np.float64)
            
            for mu, orb in enumerate(basis_map):
                if orb['atom_index'] != i_atom:
                    continue
                    
                l_mu = orb['l']
                m_physical = orb['m']  # FIX: Directly extracts clean matching quantum number
                
                R_mu_r = np.zeros(len(r_grid), dtype=np.float64)
                for k in range(len(orb['alphas'])):
                    R_mu_r += orb['coeffs'][k] * np.exp(-orb['alphas'][k] * (r_grid**2))
                if l_mu > 0:
                    R_mu_r *= (r_grid ** l_mu)
                
                for i in range(len(dataset.angular_momenta)):
                    if dataset.angular_momenta[i] != l_mu or dataset.magnetic_nums[i] != m_physical:
                        continue
                        
                    r_c = dataset.cutoff_radii[i]
                    active_mask = r_grid <= r_c
                    integrand = R_mu_r * (u_ae[i] - u_ps[i]) * r_grid
                    M_atom[mu, i] = np.trapezoid(integrand[active_mask], r_grid[active_mask])
            
            Delta_M.append(M_atom)

        core_accum = np.zeros_like(P_ps)
        for i_atom in range(num_atoms):
            core_accum += np.einsum('sknp,bp->sknb', C_paw[i_atom], Delta_M[i_atom])
            
        coherence = np.sum(P_ps.conj() * core_accum).real
        if coherence < 0:
            core_accum *= -1.0
            
        p_total = P_ps + core_accum
        
        #######################################################################
        # STEP 4: Vectorized Reciprocal Augmentation Grid Assembly
        #######################################################################
        logging.info("Step 4: Constructing augmented periodic metric via manual G-grid.")
        
        c_final = np.zeros_like(p_total, dtype=np.complex128)
        S_AE = np.zeros_like(S_k)
        
        augmentation_grid_multiplier = 3
        base_shape = post_wfc._minimum_fft_size
        aug_shape = [int(n * augmentation_grid_multiplier) for n in base_shape]
        
        nx = np.arange(-aug_shape[0]//2, aug_shape[0]//2)
        ny = np.arange(-aug_shape[1]//2, aug_shape[1]//2)
        nz = np.arange(-aug_shape[2]//2, aug_shape[2]//2)
        
        KX, KY, KZ = np.meshgrid(nx, ny, nz, indexing='ij')
        g_integer_matrix = np.stack([KX.flatten(), KY.flatten(), KZ.flatten()], axis=1)
        rgvec_augmented = g_integer_matrix @ (2 * np.pi * post_wfc.reciprocal_lattice)
        n_q_augmented = rgvec_augmented.shape[0]
        
        for ikpt_idx in track(range(nkpoints), description="[bold green]Inverting High-G Subspaces...[/]"):
            q_vecs_k = rgvec_augmented + post_wfc.kpoints_cart[ikpt_idx][np.newaxis, :]
            q_norms_k = np.linalg.norm(q_vecs_k, axis=1)
            
            chi_mat_augmented = np.zeros((n_basis, n_q_augmented), dtype=np.complex128)
            for mu, orb in enumerate(basis_map):
                chi_g = evaluate_orbital_g_space(q_vecs_k, q_norms_k, orb['l'], orb['m'], orb['alphas'], orb['g_coefficients'])
                phase = np.exp(-1j * np.dot(q_vecs_k, atom_positions[orb['atom_index']]))
                chi_mat_augmented[mu, :] = chi_g * phase
            
            S_AE[ikpt_idx] = (chi_mat_augmented.conj() @ chi_mat_augmented.T) / volume
            
            s_eigenvals, s_eigenvecs = np.linalg.eigh(S_AE[ikpt_idx])
            max_ev = np.max(s_eigenvals)
            
            lin_dep_threshold = 5e-4 * max_ev
            keep_indices = np.where(s_eigenvals > lin_dep_threshold)[0]
            
            s_val_reduced = s_eigenvals[keep_indices]
            s_vec_reduced = s_eigenvecs[:, keep_indices]
            inv_sqrt_s = s_vec_reduced / np.sqrt(s_val_reduced)
            s_inverse_k = inv_sqrt_s @ inv_sqrt_s.conj().T
            
            for ispin in range(nspin):
                c_final[ispin, ikpt_idx, :, :] = p_total[ispin, ikpt_idx, :, :] @ s_inverse_k.conj()

        raw_spillage = 1.0 - np.einsum('sknb,sknb->skn', c_final.conj(), p_total).real
        
        neg_spillage_threshold = 1e-4
        most_negative_spillage = np.min(raw_spillage)
        
        if most_negative_spillage < -neg_spillage_threshold:
            import warnings
            spin_idx, kpt_idx, band_idx = np.unravel_index(np.argmin(raw_spillage), raw_spillage.shape)
            error_msg = (
                f"\n[METRIC VIOLATION]: Overlap matrix lost positive-semidefiniteness.\n"
                f"  * Minimum Spillage Value Observed : {most_negative_spillage:.6f}\n"
                f"  * Violation Coordinate Location   : Spin={spin_idx}, k-point={kpt_idx}, Band={band_idx}"
            )
            warnings.warn(error_msg, RuntimeWarning)
            
        self.band_spillage = raw_spillage
        self.integrated_spillage = np.einsum('k,skn,skn->', post_wfc.kpoint_weights, post_wfc.occupancies, self.band_spillage) / np.sum(post_wfc.occupancies)
        max_system_spillage = np.max(self.band_spillage)
        
        #######################################################################
        # FINAL BALANCED TELEMETRY SUMMARY REPORT
        #######################################################################
        print("\n" + "="*80)
        print("                 FINAL SYSTEM BAND MANIFOLD SPILLAGE REPORT                ")
        print("="*80)
        print(f"{'Band index':<12} | {'Occupancy (Γ-point)':<22} | {'Manifold Spillage (Γ-point)':<28} | Status")
        print("-"*80)
        for n in range(nbands):
            state_spillage = self.band_spillage[0, 0, n]
            occupancy_val = post_wfc.occupancies[0, 0, n]
            status = "❌ VIOLATION" if state_spillage < -neg_spillage_threshold else ("⚠️ LEAKAGE" if state_spillage > 0.05 else "✅ FULLY RESOLVED")
            print(f"  Band {n:2d}      | {occupancy_val:22.4f} | {state_spillage:28.6f} | {status}")
        print("-"*80)
        print(f"  * Maximum Observed State Spillage Across System : {max_system_spillage:.6f}")
        print(f"  * Integrated Total Crystal System Spillage     : {self.integrated_spillage:.6f}")
        print("="*80 + "\n")
        
        # =====================================================================
        # STATE ASSIGNMENT & RETURN INTERFACE
        # =====================================================================
        # FIX: Standardized non-prefixed fields to establish an explicit interface link with construct_coefficients
        self._basis_map = basis_map
        self._coefficients = c_final  # Shape: (nspin, nkpoints, nbands, n_basis)
    
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