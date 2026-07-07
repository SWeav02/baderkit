# -*- coding: utf-8 -*-

from pathlib import Path
import logging
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn
from functools import cached_property
import numpy as np
from numpy.typing import NDArray

from baderkit.post_wfc.pseudopotentials.augmentation_numba import compute_reciprocal_projectors
from .all_electron_dataset import AESpecies
from .projection_numba import (
    find_active_periodic_atoms,
    find_all_voxels_parallel,
    broadcast_atoms_to_grid,
    evaluate_orbital_g_space,
)
from baderkit.post_wfc.wfc_numba import _integrate_tetrahedra_spectral_density_numba
from baderkit.post_wfc.base import BaseWavefunctionEnvironment


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
    # Property Calculations (Placeholders)
    ###########################################################################
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
            
        # Coherently reconstruct wavefunctions: C(k) * Chi(G,k)
        return np.dot(coeffs_active, Chi_q)

    def get_rho_tau(
        self, 
        grid_shape=None, 
        spin_channel=-1, 
        energy_range=(-np.inf, np.inf), 
        use_partial_occ=True, 
        return_density_matrices=False,
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
            weight_callback=callback
        )
            
        results = [rho, tau]
        if return_density_matrices:
            results.append(global_density_matrix)
        return tuple(results)
    
    
    
    ###########################################################################
    # PDOS Methods
    ###########################################################################
    def get_atom_projected_density_of_states(
        self, 
        spin_channel=-1, 
        energy_range=None, 
        num_points=2000, 
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
            num_points=num_points,
            method=method,
            sigma=sigma,
            use_occupancies=use_occupancies
        )

        pdos_data = {
            "energy_grid": energy_grid,
            "total": np.zeros(num_points, dtype=np.float64)
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
        num_points=2000,
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
            num_points=num_points,
            method=method,
            sigma=sigma,
            use_occupancies=use_occupancies
        )
        
        pdos_data = {
            "energy_grid": energy_grid,
            "total": np.zeros(num_points, dtype=np.float64)
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
        num_points: int = 2000,
        method: str = "gaussian",
        sigma: float | None = None,
        use_occupancies: bool = False,
    ) -> tuple[NDArray, NDArray]:
        """Consolidated pipeline to map state selection masks and execute smearing methods."""
        method, sigma = self._get_default_sigma(method, sigma)
        bands = self.energies
        num_channels = channel_weights.shape[0]
        
        if energy_range is None:
            e_min, e_max = self.get_energy_range(method, sigma)
        else:
            e_min, e_max = energy_range
            full_e_min, full_e_max = self.get_energy_range(method, sigma)
            if e_min is None or e_min == -np.inf: e_min = full_e_min
            if e_max is None or e_max == np.inf: e_max = full_e_max
                
        energy_grid = np.linspace(e_min, e_max, num_points)
        delta_e = energy_grid[1] - energy_grid[0] if num_points > 1 else 0.0
        
        if spin_channel == 1 and self.nspin == 1:
            spin_channel = 0
    
        spin_all = [spin_channel] if spin_channel != -1 else list(range(self.nspin))
        factor = 2 if (spin_channel == -1 and self.nspin == 1) else 1
            
        if method == "none":
            pad = 0.5 * delta_e if num_points > 1 else 0.0
        elif method == "tetrahedron":
            pad = 4.0 * sigma if sigma > 0.0 else 0.0
        else:
            pad = 5.0 * sigma

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
                
            band_mask = np.any((eigenvalues >= e_min - pad) & (eigenvalues <= e_max + pad), axis=(0, 1))
            
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
                
            mask = (bands_flat >= e_min - pad) & (bands_flat <= e_max + pad)
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

    def _get_voxel_footprints(self, grid_dims):
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

    def get_atom_radial_rho_tau(
        self, 
        min_charge: float, 
        max_charge: float
    ) -> tuple[dict[int, NDArray], dict[int, NDArray]]:
        """
        Unifies charge and kinetic energy radial profile evaluation across the 
        PDOS-resolved local electronic shells mapped to a specific global cell charge window.
        """
        min_charge, max_charge = self._clean_ranges(min_charge, max_charge)
        
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
        min_charge=None, 
        max_charge=None
    ) -> tuple[float | NDArray, float | NDArray]:
        """
        Calculates the non-bonding reference valence charge and kinetic energy density
        at one or multiple continuous fractional coordinate locations using atom-resolved 
        PDOS tracking rules.

        Parameters:
        -----------
        frac_coords : array-like
            Fractional coordinates matching shape (3,) for a single point, 
            or shape (N, 3) for multiple target points.
        """
        min_charge, max_charge = self._clean_ranges(min_charge, max_charge)
        
        spa = self.semi_periodic_atoms
        atom_carts = spa["cart_coords"]          # shape: (M, 3)
        atom_bases_indices = spa["base_indices"]  # shape: (M,)
        
        # Standardize coordinate dimensions into a 2D matrix layout
        frac_coords_arr = np.asarray(frac_coords, dtype=np.float64)
        is_single_point = frac_coords_arr.ndim == 1
        target_cart = np.atleast_2d(frac_coords_arr) @ self.lattice_matrix  # shape: (N, 3)
        num_targets = target_cart.shape[0]
        
        # Pre-allocate zero-filled accumulators for all target positions
        rho_total = np.zeros(num_targets, dtype=np.float64)
        tau_total = np.zeros(num_targets, dtype=np.float64)
        
        # Fetch individual element species radial templates
        partial_rhos, partial_taus = self.get_atom_radial_rho_tau(min_charge, max_charge)

        # Vectorize calculations column-by-column across the periodic atom images
        for j, i_atom in enumerate(atom_bases_indices):
            atom_pos = atom_carts[j]
            
            # Distance from this specific periodic image to ALL target coordinates
            dists_j = np.sqrt(
                (target_cart[:, 0] - atom_pos[0]) ** 2 +
                (target_cart[:, 1] - atom_pos[1]) ** 2 +
                (target_cart[:, 2] - atom_pos[2]) ** 2
            )
            
            # Isolate target coordinates lying within this atom's interaction boundary
            mask_j = dists_j < self.cutoff_radius
            if np.any(mask_j):
                symbol = self.structure[i_atom].specie.symbol
                r_grid = self.atom_bases[symbol].radial_grid
                
                # Highly optimized C-level interpolation executed across the target space array
                rho_total += np.interp(dists_j, r_grid, partial_rhos[i_atom]) * mask_j
                tau_total += np.interp(dists_j, r_grid, partial_taus[i_atom]) * mask_j
        
        # Unpack array metrics to pristine native floats if input was a single point coordinate
        if is_single_point:
            return rho_total[0], tau_total[0]
            
        return rho_total, tau_total
    
    def get_promolecular_rho_tau_vs_charge(
        self, 
        frac_coord: NDArray, 
        min_charge: float, 
        max_charge: float, 
        num_points: int = 2000,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Calculates the differential profiles (d_rho/dQ and d_tau/dQ) at a coordinate via a stable cumulative matrix approach."""
        min_charge, max_charge = self._clean_ranges(min_charge, max_charge)
        target_charges = np.linspace(min_charge, max_charge, num_points)
        
        cumulative_rho = np.empty(len(target_charges), dtype=np.float64)
        cumulative_tau = np.empty(len(target_charges), dtype=np.float64)
        
        for idx, charge in enumerate(target_charges):
            r_val, t_val = self.get_promolecular_rho_tau_at_points(
                frac_coord, 
                min_charge=min_charge, 
                max_charge=charge
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
        energy_charge_array: NDArray = None, 
        num_interp_points: int = 2000,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Maps input cell energies to their corresponding non-bonding differential densities (d_rho / d_E and d_tau / d_E)."""
        if energy_charge_array is None:
            energy_charge_array = self.total_charge_vs_energy

        sorted_indices = np.argsort(energy_charge_array[:, 0])
        energies = energy_charge_array[sorted_indices, 0]
        charges = energy_charge_array[sorted_indices, 1]
        
        min_charge = np.min(charges)
        max_charge = np.max(charges)
        
        if np.abs(max_charge - min_charge) < 1e-6:
            drho_dE_sorted = np.zeros(len(energies), dtype=np.float64)
            dtau_dE_sorted = np.zeros(len(energies), dtype=np.float64)
        else:
            target_charges, drho_dQ, dtau_dQ = self.get_promolecular_rho_tau_vs_charge(
                frac_coord, 
                min_charge=min_charge, 
                max_charge=max_charge, 
                num_points=num_interp_points
            )
            drho_dQ_sorted = np.interp(charges, target_charges, drho_dQ)
            dtau_dQ_sorted = np.interp(charges, target_charges, dtau_dQ)
            
            dQ_dE_sorted = np.gradient(charges, energies)
            drho_dE_sorted = drho_dQ_sorted * dQ_dE_sorted
            dtau_dE_sorted = dtau_dQ_sorted * dQ_dE_sorted
            
        original_order = np.argsort(sorted_indices)
        
        return (
            energy_charge_array[:, 0], 
            drho_dE_sorted[original_order], 
            dtau_dE_sorted[original_order]
        )

    def get_promolecular_rho_tau(
        self, 
        grid_dims, 
        min_charge=None, 
        max_charge=None,
    ) -> tuple[NDArray, NDArray]:
        """Generates both non-bonding reference charge and kinetic energy density 3D grids using custom PDOS weights."""
        min_charge, max_charge = self._clean_ranges(min_charge, max_charge)

        spa = self.semi_periodic_atoms
        atom_types = spa["base_indices"]
        
        all_indices, all_distances = self._get_voxel_footprints(grid_dims)
        partial_rhos, partial_taus = self.get_atom_radial_rho_tau(min_charge, max_charge)

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
        
        g_dims = np.array(grid_dims, dtype=np.int64)
        
        # Sequentially broadcast profiles across the structural voxel maps
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
        spin_channel=-1, 
        energy_range=(-np.inf, np.inf), 
        use_partial_occ=True,
        energy_charge_array=None
    ) -> tuple[NDArray, NDArray]:
        """
        Calculates the real-space 3D deformation charge and kinetic energy density grids
        (Interacting - Promolecular) inside the unit cell.
        """
        if energy_charge_array is None:
            energy_charge_array = self.total_charge_vs_energy

        # 1. Compute fully interacting densities on the real-space grid
        rho_int, tau_int = self.calculate_rho_tau(
            grid_shape=grid_shape,
            spin_channel=spin_channel,
            energy_range=energy_range,
            use_partial_occ=use_partial_occ,
            return_density_matrices=False
        )
        
        # 2. Map target energy boundaries to cell charge ranges if mapping array is supplied
        min_charge, max_charge = None, None
        if energy_charge_array is not None:
            sorted_arr = energy_charge_array[np.argsort(energy_charge_array[:, 0])]
            min_charge = float(np.interp(energy_range[0], sorted_arr[:, 0], sorted_arr[:, 1]))
            max_charge = float(np.interp(energy_range[1], sorted_arr[:, 0], sorted_arr[:, 1]))
            
        # 3. Generate matching non-bonding promolecular reference grids
        rho_pro, tau_pro = self.get_promolecular_rho_tau(
            grid_dims=rho_int.shape,
            min_charge=min_charge,
            max_charge=max_charge
        )
        
        return rho_int - rho_pro, tau_int - tau_pro
    
    def get_deformation_rho_tau_at_points(
        self,
        frac_coords,
        spin_channel=-1,
        energy_range=(-np.inf, np.inf),
        use_partial_occ=True,
        energy_charge_array=None
    ) -> tuple[float | NDArray, float | NDArray]:
        """
        Calculates the exact deformation charge and kinetic energy density profiles
        (Interacting - Promolecular) at one or multiple discrete fractional coordinates.
        """
        if energy_charge_array is None:
            energy_charge_array = self.total_charge_vs_energy

        # 1. Evaluate exact point wavefunctions from the continuous representation backend
        rho_int, tau_int = self.get_rho_tau_at_point(
            frac_coord=frac_coords,
            spin_channel=spin_channel,
            energy_range=energy_range,
            use_partial_occ=use_partial_occ,
            include_aug=False
        )
        
        # 2. Map target energy boundaries to cell charge ranges if mapping array is supplied
        min_charge, max_charge = None, None
        if energy_charge_array is not None:
            sorted_arr = energy_charge_array[np.argsort(energy_charge_array[:, 0])]
            min_charge = float(np.interp(energy_range[0], sorted_arr[:, 0], sorted_arr[:, 1]))
            max_charge = float(np.interp(energy_range[1], sorted_arr[:, 0], sorted_arr[:, 1]))
            
        # 3. Compute the vectorized promolecular reference values at the same positions
        rho_pro, tau_pro = self.get_promolecular_rho_tau_at_points(
            frac_coords=frac_coords,
            min_charge=min_charge,
            max_charge=max_charge
        )
        
        return rho_int - rho_pro, tau_int - tau_pro
    
    def get_deformation_rho_tau_vs_energy(
        self,
        frac_coord: NDArray,
        energy_charge_array: NDArray = None,
        spin_channel: int = -1,
        num_interp_points: int = 2000,
        cumulative: bool = False,
        return_plot: bool = False,
    ) -> tuple:
        """
        Calculates the energy-resolved deformation density spectral curves (Interacting - Promolecular)
        at a specific fractional coordinate. Supporting both differential and cumulative integrated modes.
        """
        from scipy.integrate import cumulative_trapezoid

        if energy_charge_array is None:
            energy_charge_array = self.total_charge_vs_energy

        energies = energy_charge_array[:, 0]
        energy_range = (float(np.min(energies)), float(np.max(energies)))
        
        # 1. Fetch interacting densities across the energy grid continuum
        int_res = self.get_rho_tau_vs_energy(
            frac_coord=frac_coord,
            spin_channel=spin_channel,
            energy_range=energy_range,
            num_points=num_interp_points,
            include_aug=False,
            cumulative=cumulative,
            return_plot=False
        )
        int_energy, int_rho, int_tau = int_res[0], int_res[1], int_res[2]
        
        # 2. Fetch differential non-bonding promolecular reference densities
        pro_energy, pro_rho, pro_tau = self.get_promolecular_rho_tau_vs_energy(
            frac_coord=frac_coord,
            energy_charge_array=energy_charge_array,
            num_interp_points=num_interp_points
        )
        
        # 3. Handle cumulative numerical integration tracking for the promolecular arrays
        if cumulative:
            sort_idx = np.argsort(pro_energy)
            pro_energy_s = pro_energy[sort_idx]
            pro_rho_s = pro_rho[sort_idx]
            pro_tau_s = pro_tau[sort_idx]
            
            cum_pro_rho = np.zeros_like(pro_energy_s)
            cum_pro_tau = np.zeros_like(pro_energy_s)
            if len(pro_energy_s) > 1:
                cum_pro_rho[1:] = cumulative_trapezoid(pro_rho_s, pro_energy_s)
                cum_pro_tau[1:] = cumulative_trapezoid(pro_tau_s, pro_energy_s)
                
            orig_idx = np.argsort(sort_idx)
            pro_rho_eval = cum_pro_rho[orig_idx]
            pro_tau_eval = cum_pro_tau[orig_idx]
        else:
            pro_rho_eval = pro_rho
            pro_tau_eval = pro_tau
            
        # 4. Interpolate interacting properties onto the exact user-supplied energy levels
        rho_int_interp = np.interp(energies, int_energy, int_rho)
        tau_int_interp = np.interp(energies, int_energy, int_tau)
        
        # 5. Extract the net deformation delta curves
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
    
    def _clean_ranges(self, min_charge, max_charge):
        if min_charge is None or min_charge == -np.inf:
            min_charge = 0.0
        if max_charge is None or max_charge == np.inf:
            max_charge = self.maximum_charge
            
        min_charge = max(min_charge, 0)
        max_charge = min(max_charge, self.maximum_charge)
        return min_charge, max_charge
    
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
    
    def _project_system(self):
        post_wfc = self.post_wfc
        structure = self.structure
        nspin = post_wfc.nspin
        nkpoints = post_wfc.nkpoints
        nbands = post_wfc.nbands
        
        logging.info("Initializing crystalline Projector Augmented Wave (PAW) projection pipeline.")
        logging.info("Constructing atomic orbital basis...")
        basis_map = []
        for i_atom in range(len(structure)):
            elem = structure[i_atom].species_string
            atom_basis = self.atom_bases[elem]
            
            for idx, (l, m) in enumerate(zip(atom_basis.angular_momenta, atom_basis.magnetic_quantum_numbers)):
                c_data = atom_basis.primitives[l]
                exps = c_data["exps"]
                coeffs = c_data["coeffs"]
                g_coeffs = c_data["g_coeffs"]
                offsets = c_data["offsets"]
                dim = len(offsets) - 1
                
                start_l, end_l = atom_basis.l_slices[l]
                state_coeffs = atom_basis.state_vectors[idx, start_l:end_l]
                
                orb_alphas = []
                orb_coeffs = []
                orb_g_coeffs = []
                
                for p in range(dim):
                    start = offsets[p]
                    end = offsets[p+1]
                    c_p = state_coeffs[p]
                    
                    orb_alphas.extend(exps[start:end])
                    orb_coeffs.extend(coeffs[start:end] * c_p)
                    orb_g_coeffs.extend(g_coeffs[start:end] * c_p)
                
                basis_map.append({
                    'atom_index': i_atom,
                    'element': elem,
                    'l': l,
                    'm': m,
                    'alphas': np.array(orb_alphas, dtype=np.float64),
                    'coeffs': np.array(orb_coeffs, dtype=np.float64),
                    'g_coefficients': np.array(orb_g_coeffs, dtype=np.float64)
                })
                
        num_basis_tot = len(basis_map)
        logging.info(
            f"Basis generated successfully. Total functions: {num_basis_tot} across {len(structure)} atomic sites."
        )
        
        logging.debug(f"Allocating coefficient projection matrix shape: ({nspin}, {nkpoints}, {nbands}, {num_basis_tot})")
        coefficients = np.zeros((nspin, nkpoints, nbands, num_basis_tot), dtype=np.complex128)
        
        total_states_counted = 0
        accumulated_explained_variance = 0.0
        total_k_tasks = nspin * nkpoints
        logging.info(f"Beginning reciprocal space electronic state projections across {total_k_tasks} irreducible k-points.")
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            transient=True
        ) as progress:
            
            proj_task = progress.add_task(
                "[cyan]Projecting Bloch wavefunctions...", 
                total=total_k_tasks
            )
            
            for ispin in range(nspin):
                for ikpt in range(nkpoints):
                    progress.update(
                        proj_task, 
                        description=f"[cyan]Projecting Bloch waves (Spin {ispin}, k-point {ikpt+1}/{nkpoints})"
                    )
                    
                    k_cart = post_wfc.kpoints_cart[ikpt]
                    G_basis_cart = post_wfc.get_plane_waves_basis_cart_from_idx(ikpt)
                    num_pw = G_basis_cart.shape[0]
                    
                    q_vecs = G_basis_cart + k_cart[np.newaxis, :]
                    q_norms = np.linalg.norm(q_vecs, axis=1)
                    
                    coeffs_all_bands = post_wfc.get_plane_wave_coefficients_batch(ispin, ikpt, range(nbands))
                    
                    Chi_q = np.zeros((num_basis_tot, num_pw), dtype=np.complex128)
                    for mu, orb in enumerate(basis_map):
                        R_atom = structure[orb['atom_index']].coords
                        spatial_phase = np.exp(-1j * np.dot(q_vecs, R_atom))
                        Chi_q[mu, :] = spatial_phase * evaluate_orbital_g_space(
                            q_vecs, 
                            q_norms, 
                            orb['l'], 
                            orb['m'], 
                            orb['alphas'], 
                            orb['g_coefficients'],
                        )
                        
                    S_k = np.dot(Chi_q, Chi_q.conj().T)
                    S_k_reg = S_k + 1e-9 * np.eye(num_basis_tot)
                    T_total = np.dot(Chi_q, coeffs_all_bands.conj().T)
                    
                    for i_atom in range(len(structure)):
                        elem = structure[i_atom].species_string
                        dataset = self._aug_environment.paw_datasets[elem]
                        h = dataset.q_linear_grid[-1]
                        
                        P_G_matrix = compute_reciprocal_projectors(
                            k_cart, G_basis_cart, structure[i_atom].coords, h,
                            len(dataset.q_linear_grid), structure.volume, 
                            dataset.reciprocal_projectors, dataset.angular_momenta, dataset.magnetic_nums
                        )
                        
                        W_proj_strengths = np.dot(P_G_matrix, coeffs_all_bands.T)
                        atom_basis_indices = [mu for mu, b in enumerate(basis_map) if b['atom_index'] == i_atom]
                        
                        delta_t = np.zeros((len(atom_basis_indices), len(dataset.angular_momenta)), dtype=np.float64)
                        r_grid = dataset.radial_grid
                        w_radial = 4.0 * np.pi * (r_grid ** 2)
                        
                        for local_idx, mu in enumerate(atom_basis_indices):
                            orb = basis_map[mu]
                            r_bohr = r_grid / 0.529177210903
                            orb_radial = np.zeros(len(r_grid), dtype=np.float64)
                            for alpha, d in zip(orb['alphas'], orb['coeffs']):
                                orb_radial += d * (r_bohr ** orb['l']) * np.exp(-alpha * (r_bohr ** 2))
                                
                            for i_proj in range(len(dataset.angular_momenta)):
                                if orb['l'] == dataset.angular_momenta[i_proj] and orb['m'] == dataset.magnetic_nums[i_proj]:
                                    phi_diff = dataset.all_electron_partial_waves[i_proj] - dataset.pseudo_partial_waves[i_proj]
                                    integrand = w_radial * orb_radial * phi_diff
                                    delta_t[local_idx, i_proj] = np.trapezoid(integrand, r_grid)
                                    
                        T_total[atom_basis_indices, :] += np.dot(delta_t, W_proj_strengths)
                        
                    C_k = np.linalg.solve(S_k_reg, T_total)
                    coefficients[ispin, ikpt, :, :] = C_k.T
                    
                    explained_variance_k = np.sum(C_k.conj() * T_total, axis=0).real
                    occupied_mask = post_wfc.occupancies[ispin, ikpt, :] > 1e-4
                    accumulated_explained_variance += np.sum(explained_variance_k[occupied_mask])
                    total_states_counted += np.sum(occupied_mask)
                    
                    progress.advance(proj_task)
                    
        global_spillage = 1.0 - (accumulated_explained_variance / max(1, total_states_counted))
        self.spillage = max(0.0, global_spillage)
        
        logging.info(f"Projection matrix assembly complete. Unified Crystalline Spillage: {(self.spillage*100):.6f}%")
        if self.spillage > 0.05:
            logging.warning("Spillage error exceeds 5%")
            
        self._basis_map = basis_map
        self._coefficients = coefficients

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