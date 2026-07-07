# -*- coding: utf-8 -*-

import numpy as np
from scipy.fft import ifftn, set_workers

from baderkit.post_wfc.pseudopotentials.augmentation_numba import (
    compute_reciprocal_projectors,
    enforce_matrix_symmetrization
    )
from .base import BaseWavefunctionEnvironment


class PostWFC(BaseWavefunctionEnvironment):
    """
    Standardized, code-agnostic post-processing driver executing 3D fast Fourier transforms 
    to map discrete reciprocal coefficients into dense localized real-space properties.
    """
    
    ###########################################################################
    # Projection Mapping
    ###########################################################################
    @property
    def projection_environment(self):
        if getattr(self, "_projection_environment", None) is None:
            from .projection.projection_environment import AtomicProjectionEnvironment
            self._projection_environment = AtomicProjectionEnvironment(self)
        return self._projection_environment
        
    ###########################################################################
    # Plane wave methods
    ###########################################################################
    def get_plane_waves_frac(self, grid_shape=None):
        """Generates fractional coordinate arrays for plane waves in standard FFT wrapped frequency order."""
        if grid_shape is None: 
            grid_shape = self._minimum_fft_size * 2
        grid_shape = tuple(grid_shape)
        if grid_shape in self._grid_cache: 
            return self._grid_cache[grid_shape]
            
        Nx, Ny, Nz = grid_shape
        fx = [ii if ii < Nx // 2 + 1 else ii - Nx for ii in range(Nx)]
        fy = [jj if jj < Ny // 2 + 1 else jj - Ny for jj in range(Ny)]
        fz = [kk if kk < Nz // 2 + 1 else kk - Nz for kk in range(Nz)]
        
        gx, gy, gz = np.meshgrid(fx, fy, fz, indexing='ij')
        self._grid_cache[grid_shape] = (gx, gy, gz)
        return gx, gy, gz

    def plane_waves_cart(self, grid_shape=None):
        """Transforms integer fractional meshgrid coordinate axes into explicit Cartesian grid coordinates (in A^-1)."""
        gx, gy, gz = self.get_plane_waves_frac(grid_shape)
        cx, cy, cz = np.tensordot(
            self.reciprocal_lattice * np.pi * 2, [gx, gy, gz], axes=(0, 0))
        return cx, cy, cz
        
    def get_plane_waves_basis_idx(self, ikpt, grid_shape=None, expected_npw=None):
        """Gathers explicit active g-vectors and pre-wrapped grid coordinates from the current wf_reader."""
        if grid_shape is None: 
            grid_shape = self._minimum_fft_size * 2
        gvectors = self._wf_reader.read_gvectors(ikpt)
        return gvectors, (gvectors % np.asarray(grid_shape)[np.newaxis, :]).astype(int)

    def get_plane_waves_basis_cart_from_idx(self, ikpt, grid_shape=None, expected_npw=None):
        """Projects integer Miller vector blocks straight into Cartesian inverse Angstrom coordinates."""
        gvectors, _ = self.get_plane_waves_basis_idx(ikpt, grid_shape, expected_npw)
        return gvectors @ (2 * np.pi * self.reciprocal_lattice)
        
    def get_plane_wave_coefficients(self, ispin, ikpt, iband):
        """Single-band coefficient query acting as a backward-compatible wf_reader bridge layer."""
        return self._wf_reader.read_coefficients(ispin, ikpt, iband)
    
    def get_plane_wave_coefficients_batch(self, ispin, ikpt, bands):
        """Single-band coefficient query acting as a backward-compatible wf_reader bridge layer."""
        return self._wf_reader.read_coefficients_batch(ispin, ikpt, bands)
        
    def get_pseudo_wavefunction(self, ispin=0, ikpt=0, iband=0, grid_shape=None, kr_phase=False, coeffs=None):
        """Obtain the pseudo-wavefunction of the specified KS states in real space."""
        if grid_shape is None: 
            grid_shape = self._minimum_fft_size * 2
        grid_shape = tuple(grid_shape)
        Nx, Ny, Nz = grid_shape
        
        if kr_phase:
            phase = np.exp(1j * np.pi * 2 * np.sum(self.kpoints[ikpt] * (np.mgrid[0:Nx, 0:Ny, 0:Nz].reshape((3, Nx*Ny*Nz)).T / np.array(grid_shape, dtype=float)), axis=1)).reshape(grid_shape)
        else:
            phase = 1.0
            
        gvectors, gvec_wrapped = self.get_plane_waves_basis_idx(ikpt, grid_shape)
        phi_k = np.zeros(grid_shape, dtype=np.complex128)
        if coeffs is None: 
            coeffs = self.get_plane_wave_coefficients(ispin, ikpt, iband)
            
        phi_k[gvec_wrapped[:, 0], gvec_wrapped[:, 1], gvec_wrapped[:, 2]] = coeffs
        
        with set_workers(self.scipy_workers):
            pseudo_wfs = ifftn(phi_k * np.sqrt(Nx * Ny * Nz)) * phase
        return pseudo_wfs
    
    ###########################################################################
    # Augmentation Reconstruction
    ###########################################################################
    def calculate_onsite_occupancy(
        self, 
        energy_range=(-np.inf, np.inf), 
        spin_channel=-1, 
        use_partial_occ=True
    ) -> list[np.ndarray]:
        """
        Calculates the integrated onsite occupancy/density matrix for each atomic site 
        as a drop-in replacement for calculate_onsite_occupancy, keeping the core 
        symmetrization logic intact.
        """
        energies = self.energies
        occupancies = self.occupancies
        kweights = self.kpoint_weights
        num_atoms = len(self.structure)
        
        energy_min, energy_max = energy_range
        spin_indices = [i for i in range(self.nspin)] if spin_channel == -1 else [spin_channel]
        spin_degeneracy = 2.0 if self.nspin == 1 else 1.0

        # Pre-allocate density matrices and gather static atom-specific dataset attributes
        density_matrices = []
        
        for i_atom in range(num_atoms):
            elem = self.structure[i_atom].species_string
            dataset = self._aug_environment.paw_datasets[elem]
            num_projectors = len(dataset.angular_momenta)
            density_matrices.append(np.zeros((num_projectors, num_projectors), dtype=np.complex128))

        for ispin in spin_indices:
            for ikpt in range(self.nkpoints):
                w_k = kweights[ikpt]
                k_energies = energies[ispin, ikpt, :]
                
                # Screen active bands falling inside the target window early
                mask = (k_energies >= energy_min) & (k_energies <= energy_max)
                if use_partial_occ:
                    mask &= (occupancies[ispin, ikpt, :] > 0.0)
                    
                active_bands = np.where(mask)[0]
                if active_bands.size == 0:
                    continue
                    
                k_cart = self.kpoints_cart[ikpt]
                G_basis_cart = self.get_plane_waves_basis_cart_from_idx(ikpt)
                
                for i_atom in range(num_atoms):
                    elem = self.structure[i_atom].species_string
                    dataset = self._aug_environment.paw_datasets[elem]
                    h = dataset.q_linear_grid[-1]
                    num_projectors = len(dataset.angular_momenta)
                    
                    # Compute position-dependent structural projector matrix per site
                    P_G_matrix = compute_reciprocal_projectors(
                        k_cart, 
                        G_basis_cart, 
                        self.structure[i_atom].coords, 
                        h,
                        len(dataset.q_linear_grid), 
                        self.structure.volume, 
                        dataset.reciprocal_projectors, 
                        dataset.angular_momenta, 
                        dataset.magnetic_nums
                    )
                    
                    for band in active_bands:
                        f_nk = occupancies[ispin, ikpt, band] if use_partial_occ else 1.0
                        weight = spin_degeneracy * w_k * f_nk
                        if weight == 0.0:
                            continue
                            
                        C_G = self.get_plane_wave_coefficients(ispin, ikpt, band)
                        c = np.dot(P_G_matrix, C_G)
                        
                        contr = weight * np.outer(c, np.conj(c))
                        density_matrices[i_atom] += contr
        
        for i_atom in range(num_atoms):
            elem = self.structure[i_atom].species_string
            dataset = self._aug_environment.paw_datasets[elem]
            
            density_matrix = density_matrices[i_atom]
            final_density_matrix = enforce_matrix_symmetrization(density_matrix, dataset.angular_momenta, dataset.magnetic_nums)
            density_matrices[i_atom] = final_density_matrix.real        
             
        return density_matrices
    
    ###########################################################################
    # Property Calculations
    ###########################################################################
    
    def _construct_coefficients(self, ispin: int, ikpt: int, active_bands: list) -> np.ndarray:
        """Pulls native raw coefficients directly out of the binary file stream reader."""
        return self._wf_reader.read_coefficients_batch(ispin, ikpt, active_bands)

    def get_rho_tau(
        self, 
        grid_shape=None, 
        spin_channel=-1, 
        energy_range=(-np.inf, np.inf), 
        use_partial_occ=True, 
        include_aug=True,
        return_density_matrices=False,
    ):
        """Computes plane-wave real-space rho and tau densities using the base engine."""
        if grid_shape is None: 
            grid_shape = self._minimum_fft_size * 2
        grid_shape_tuple = tuple(grid_shape)
        
        # Call core transformation engine
        rho, tau = self._execute_core_density_loop(
            grid_shape=grid_shape_tuple,
            spin_channel=spin_channel,
            energy_range=energy_range,
            use_partial_occ=use_partial_occ
        )
        
        if include_aug:
            density_matrices = self.calculate_onsite_occupancy(energy_range, spin_channel, use_partial_occ)
            aug_tau, corr_tau = self._aug_environment.calculate_onsite_ke_densities(grid_shape_tuple, density_matrices)
            aug_rho, corr_rho = self._aug_environment.calculate_onsite_densities(grid_shape_tuple, density_matrices)
            tau += aug_tau - corr_tau
            rho += aug_rho - corr_rho
        else:
            density_matrices = None
            
        results = [rho, tau]
        if return_density_matrices:
            results.append(density_matrices)
        return tuple(results)

    ###########################################################################
    # Spectral Calculations
    ###########################################################################
    def _apply_onsite_augmentation_at_point(
        self, ispin, ikpt, active_bands, coeffs_list, rgvec, weight, frac_coord, rho_bands, tau_bands, include_aug
    ):
        """Implements the full localized PAW sphere reconstructions at a single coordinate location."""
        if not include_aug or len(active_bands) != len(coeffs_list):
            return rho_bands, tau_bands
            
        point_cart = frac_coord @ self._aug_environment.lattice_matrix
        num_atoms = len(self.structure)
        k = self.kpoints_cart[ikpt]
        
        # Gather raw projectivity overlaps for all structural sites
        projections_all_atoms = []
        for i_atom in range(num_atoms):
            elem = self.structure[i_atom].species_string
            dataset = self._aug_environment.paw_datasets[elem]
            h = dataset.q_linear_grid[-1]
            
            P_G_matrix = compute_reciprocal_projectors(
                k, rgvec, self.structure[i_atom].coords, h,
                len(dataset.q_linear_grid), self.structure.volume, 
                dataset.reciprocal_projectors, dataset.angular_momenta, dataset.magnetic_nums
            )
            proj_atom = np.dot(P_G_matrix, coeffs_list.T)
            projections_all_atoms.append(proj_atom)
        
        aug_bands = np.zeros(len(active_bands), dtype=np.float64)
        aug_ke_bands = np.zeros(len(active_bands), dtype=np.float64)
        
        # Compute individual 1D state corrections by establishing local density matrices
        for n_idx in range(len(active_bands)):
            band_density_matrices = []
            for i_atom in range(num_atoms):
                proj = projections_all_atoms[i_atom][:, n_idx]
                dm = np.outer(proj, proj.conj()).real
                band_density_matrices.append(dm)
                
            # Evaluate localized all-electron vs pseudo onsite charge adjustments (AE - PS)
            ae_rho, ps_rho = self._aug_environment.calculate_onsite_densities_at_point(
                point_cart=point_cart, density_matrices=band_density_matrices
            )
            aug_bands[n_idx] = ae_rho - ps_rho
            
            # Evaluate localized all-electron vs pseudo onsite kinetic adjustments (AE - PS)
            ae_tau, ps_tau = self._aug_environment.calculate_onsite_ke_densities_at_point(
                point_cart=point_cart, density_matrices=band_density_matrices
            )
            aug_ke_bands[n_idx] = ae_tau - ps_tau
        
        # Accumulate localized PAW terms onto the continuous background channels
        rho_bands += aug_bands * weight
        tau_bands += aug_ke_bands * weight
        
        return rho_bands, tau_bands
