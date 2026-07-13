# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from pathlib import Path
from functools import cached_property
import logging

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.fft import fftn, ifftn, set_workers
from baderkit.post_wfc.wf_readers.base import HSQDTM

from .wfc_numba import _integrate_tetrahedra_spectral_density_numba, _integrate_tetrahedra_analytic_charge_numba
from .pseudopotentials.augmentation_numba import compute_reciprocal_projectors

class BaseWavefunctionEnvironment(ABC):
    """
    Abstract baseline context managing shared structural parameters, k-point meshes,
    and cell symmetries. Supports master-companion delegation to share identical
    memory spaces between active representations.
    """
    def __init__(
        self, 
        wf_reader=None, 
        aug_environment=None,
        valence_counts=None,
        smearing="tet",
        sigma=None,
        resolution=500,
        scipy_workers: int = -1,
        reference_env=None,
        augmentation_encut=None,
        **kwargs
        ):
        """Initializes state or points straight to a companion reference environment."""
        self._reference_env = reference_env
        
        # If this is a companion instance, skip duplicating baseline state fields entirely
        if reference_env is not None:
            return
            
        # Otherwise, this is the master instance: initialize state variables natively
        self._wf_reader = wf_reader
        self._meta = wf_reader.meta
        self._aug_environment = aug_environment
        self.scipy_workers = scipy_workers
        self._augmentation_encut = augmentation_encut
        
        self._smearing, self._sigma = self._get_default_sigma(smearing, sigma)
        self._resolution = resolution
        
        self._structure = self._meta.structure
        self._lattice = self._structure.lattice.matrix               
        self._reciprocal_lattice = np.linalg.inv(self._lattice).T   
        
        self._meta.energies = self._meta.energies - self._meta.efermi
        self._meta.efermi = 0.0
        
        if valence_counts is None or valence_counts is False:
            valence_counts = {}
        self._valence_counts = valence_counts
        
        lattice_norm = np.linalg.norm(self._lattice, axis=1)
        CUTOFF = np.ceil(np.sqrt(self._meta.energy_cutoff / HSQDTM) / (2 * np.pi / lattice_norm))
        self._minimum_fft_size = np.array(2 * CUTOFF + 1, dtype=int)
        
        self._projector_cache = {}
        self._channel_slices = {}
        total_ch = 0
        for i_atom in range(len(self._structure)):
            elem = self._structure[i_atom].species_string
            dataset = self._aug_environment.paw_datasets[elem]
            num_ch = len(dataset.angular_momenta)
            self._channel_slices[i_atom] = slice(total_ch, total_ch + num_ch)
            total_ch += num_ch
        self._total_projector_channels = total_ch

        self._kpoint_multiplicities = None
        self._kpoint_weights = None
        self._grid_cache = {}  
        self._tetrahedra_indices = None

    # --- FLUENT SWITCHING INTERFACE POOL ---
    @property
    def plane_wave(self):
        """Returns the canonical plane-wave representation environment."""
        if self._reference_env is not None:
            return self._reference_env.plane_wave
        return self

    @property
    def projection(self):
        """Returns the localized atomic orbital projection representation environment."""
        if self._reference_env is not None:
            return self._reference_env.projection
        
        if getattr(self, "_projection_environment", None) is None:
            from .projection_environment import AtomicProjectionEnvironment
            self._projection_environment = AtomicProjectionEnvironment(post_wfc=self)
        return self._projection_environment

    def __getattr__(self, name):
        """Intercepts internal fields, grid caches, and private states on the master."""
        # 1. Check if the attribute is explicitly defined as a descriptor/property 
        # in the class hierarchy but failed internally with an AttributeError
        for cls in self.__class__.__mro__:
            if name in cls.__dict__:
                attr = cls.__dict__[name]
                if hasattr(attr, "__get__"):
                    # Manually invoke the descriptor to surface the real internal traceback
                    return attr.__get__(self, self.__class__)

        # 2. Safe fallback delegation to the master/reference environment
        if self._reference_env is not None:
            try:
                return getattr(self._reference_env, name)
            except AttributeError:
                pass
                
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
    @property
    def smearing(self) -> str:
        return self._smearing
    
    @property
    def sigma(self) -> float | None:
        return self._sigma
    
    @property
    def resolution(self) -> float | None:
        return self._resolution
    
    @property
    def num_points(self) -> int:
        e_min, e_max = self.energy_range
        return int(round((e_max - e_min)*self.resolution))
    
    @property
    def energy_grid(self) -> NDArray:
        if getattr(self, "_energy_grid", None) is None:
            e_min, e_max = self.energy_range
            self._energy_grid = np.linspace(e_min, e_max, self.num_points)
        return self._energy_grid
    
    @property
    def total_charge_grid(self) -> NDArray:
        if getattr(self, "_total_charge_grid", None) is None:
            self._total_charge_grid = self._get_total_charge_vs_energy()
        return self._total_charge_grid
    
    @property
    def smear_matrix(self) -> NDArray:
        if getattr(self, "_smear_matrix", None):
            self._smear_matrix = self._get_smear_matrix()
        return self._smear_matrix
    
    @property
    def tdos(self) -> NDArray:
        if getattr(self, "_tdos", None) is None:
            self._tdos = self.get_density_of_states(-1)
        return self._tdos
    
    @property
    def spin_dos(self) -> dict:
        if getattr(self, "_spin_dos", None) is None:
            if self.nspin == 1:
                self._spin_dos = {
                    0: self.tdos/2,
                    1: self.tdos/2,
                    }
            else:
                spin_dos = {}
                for i in range(2):
                    spin_dos[i] = self.get_density_of_states(i)
                self._spin_dos = spin_dos
        return self._spin_dos
                    
    # --- PROPERTY FORWARDING OVERRIDES ---
    @property
    def valence_counts(self) -> dict | None:
        return self._reference_env.valence_counts if self._reference_env else self._aug_environment.valence_counts

    @property
    def nspin(self) -> int:
        return self._reference_env.nspin if self._reference_env else self._meta.nspin
        
    @property
    def nkpoints(self) -> int:
        return self._reference_env.nkpoints if self._reference_env else self._meta.nkpts
        
    @property
    def nbands(self) -> int:
        return self._reference_env.nbands if self._reference_env else self._meta.nbands
        
    @property
    def occupancies(self):
        return self._reference_env.occupancies if self._reference_env else self._meta.occupancies
        
    @property
    def energies(self):
        return self._reference_env.energies if self._reference_env else self._meta.energies
    
    @property
    def energy_cutoff(self):
        return self._reference_env.energy_cutoff if self._reference_env else self._meta.energy_cutoff
    
    @property
    def energy_range(self) -> tuple[float, float]:
        """
        Returns relative boundary offsets scaled directly against the Fermi level (E - E_f).
        If a smearing method is provided, the range is dynamically padded based on the 
        analytical tail decay rate of the function to prevent truncation artifacts.
        """
        if getattr(self, "_energy_range", None) is None:
            e_min = np.min(self.energies)
            e_max = np.max(self.energies)
            dE = (e_max - e_min) / self.resolution
            
            if self.method == "none":
                pad = 0.5 * dE
            elif self.method in ["gaussian", "methfessel-paxton", "mp"]:
                pad = 6.5 * self.sigma
            elif self.method in ["fermi-dirac", "fd"]:
                pad = 14.0 * self.sigma
            elif self.method == "tetrahedron":
                # FIXED: Reduced to exactly 14.0 * self.sigma to perfectly align the boundaries
                # with the Fermi-Dirac kernel decay envelope, removing the large unnecessary zero buffers.
                pad = 14.0 * self.sigma if self.sigma > 0.0 else 0.0
            else:
                pad = 5.0 * self.sigma
                
            self._energy_range = e_min - pad, e_max + pad
        return self._energy_range
    
    @property
    def unsmeared_energy_range(self) -> tuple[float, float]:
        if getattr(self, "_unsmeared_energy_range", None) is None:
            self._unsmeared_energy_range = np.min(self.energies), np.max(self.energies)
        return self._unsmeared_energy_range
            
    
    @property
    def augmentation_encut(self):
        if self._augmentation_encut is None:
            self._augmentation_encut = self.energy_cutoff * 8
        return self._augmentation_encut
        
    @property
    def structure(self):
        return self._reference_env.structure if self._reference_env else self._structure
        
    @property
    def lattice(self):
        return self._reference_env.lattice if self._reference_env else self._lattice
        
    @property
    def reciprocal_lattice(self):
        return self._reference_env.reciprocal_lattice if self._reference_env else self._reciprocal_lattice
        
    @property
    def efermi(self) -> float:
        return self._reference_env.efermi if self._reference_env else self._meta.efermi
    
    @property
    def kpoints(self):
        return self._reference_env.kpoints if self._reference_env else self._meta.kpoints
    
    @property
    def kpoints_cart(self):
        return np.dot(self.kpoints, 2 * np.pi * self.reciprocal_lattice)

    # --- Symmetry Unfolding Infrastructure ---
    @property
    def kpoint_multiplicities(self):
        if self._kpoint_multiplicities is None:
            recp_symm_ops = self.structure.lattice.get_recp_symmetry_operation()
            recip_rotations = []
            for op in recp_symm_ops:
                R_recip = np.round(op.rotation_matrix).astype(int)
                if not any(np.array_equal(R_recip, R) for R in recip_rotations):
                    recip_rotations.append(R_recip)
            for R in [-R for R in recip_rotations]:
                if not any(np.array_equal(R, ex_R) for ex_R in recip_rotations):
                    recip_rotations.append(R)
                    
            multiplicities = []
            for k in self.kpoints:
                star_kpts = []
                for R in recip_rotations:
                    k_wrapped = np.mod(R @ k, 1.0)
                    is_duplicate = False
                    for eq in star_kpts:
                        diff = np.mod(k_wrapped - eq, 1.0)
                        if np.all(np.minimum(diff, 1.0 - diff) < 1e-5):
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        star_kpts.append(k_wrapped)
                multiplicities.append(len(star_kpts))
            self._kpoint_multiplicities = np.array(multiplicities)
        return self._kpoint_multiplicities

    def _unfold_brillouin_zone(self):
        recp_symm_ops = self.structure.lattice.get_recp_symmetry_operation()
        recip_rotations = []
        for op in recp_symm_ops:
            R_recip = np.round(op.rotation_matrix).astype(int)
            if not any(np.array_equal(R_recip, R) for R in recip_rotations):
                recip_rotations.append(R_recip)
        for R in [-R for R in recip_rotations]:
            if not any(np.array_equal(R, ex_R) for ex_R in recip_rotations):
                recip_rotations.append(R)

        kpts_full, full_to_irr = [], []
        for ikpt, k in enumerate(self.kpoints):
            star_kpts = []
            for R in recip_rotations:
                k_wrapped = np.mod(R @ k, 1.0)
                is_duplicate = False
                for eq in star_kpts:
                    diff = np.mod(k_wrapped - eq, 1.0)
                    if np.all(np.minimum(diff, 1.0 - diff) < 1e-5):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    star_kpts.append(k_wrapped)
            for unique_k in star_kpts:
                kpts_full.append(unique_k)
                full_to_irr.append(ikpt)
                    
        self._kpoints_full = np.array(kpts_full)
        self._full_to_irr_map = np.array(full_to_irr, dtype=int)

    @property
    def kpoint_weights(self):
        if self._kpoint_weights is None:
            self._kpoint_weights = self.kpoint_multiplicities / np.sum(self.kpoint_multiplicities)
        return self._kpoint_weights
        
    @property
    def kpoints_full(self):
        if getattr(self, "_kpoints_full", None) is None:
            self._unfold_brillouin_zone()
        return self._kpoints_full

    @property
    def kpoints_cart_full(self):
        return np.dot(self.kpoints_full, 2 * np.pi * self.reciprocal_lattice)
        
    @property
    def full_to_irr_map(self):
        if getattr(self, "_full_to_irr_map", None) is None:
            self._unfold_brillouin_zone()
        return self._full_to_irr_map
    
    @property
    def tetrahedra_indices(self):
        if self._tetrahedra_indices is None:
            self._tetrahedra_indices = self._get_tetrahedra()
        return self._tetrahedra_indices
    
    @cached_property
    def maximum_electrons(self):
        return self.total_charge_vs_energy[-1,1]
    
    ###########################################################################
    # Property Calculations
    ###########################################################################
    @abstractmethod
    def get_rho_tau(self, *args, **kwargs):
        pass
    
    def get_rho_tau_at_points(
        self,
        frac_coord,
        return_grad_rho_sq: bool = False,
        return_lap_rho: bool = False,
        spin_channel: int = -1,
        energy_range: tuple = (-np.inf, np.inf),
        use_partial_occ: bool = True,
        include_aug: bool = True,
        use_shrod_tau=False,
    ) -> tuple:
        """
        Calculates the interacting electronic charge density (rho) and Schrödinger kinetic 
        energy density (tau) at one or multiple continuous fractional coordinate locations.
        """
        frac_coord_arr = np.asarray(frac_coord, dtype=np.float64)
        is_single_point = frac_coord_arr.ndim == 1
        target_frac = np.atleast_2d(frac_coord_arr)
        num_targets = target_frac.shape[0]
        
        # create tracker for rho/tau
        rho_total = np.zeros(num_targets, dtype=np.complex128)
        tau_total = np.zeros(num_targets, dtype=np.complex128)
        grad_rho_sq_total = np.zeros(num_targets, dtype=np.float64) if return_grad_rho_sq or not use_shrod_tau else None
        lap_rho_total = np.zeros(num_targets, dtype=np.float64) if return_lap_rho else None
        
        norm_factor = 1.0 / np.sqrt(self.structure.volume)
        spin_indices = [i for i in range(self.nspin)] if spin_channel == -1 else [spin_channel]
        
        for ispin in spin_indices:
            for ikpt in range(self.nkpoints):
                active_bands = []
                weights = []
                for iband in range(self.nbands):
                    rel_energy = self.energies[ispin, ikpt, iband]
                    if not (energy_range[0] <= rel_energy <= energy_range[1]):
                        continue
                    rspin = 2.0 if self.nspin == 1 else 1.0
                    weight = rspin * self.kpoint_weights[ikpt] * (self.occupancies[ispin, ikpt, iband] if use_partial_occ else 1.0)
                    if weight > 0:
                        active_bands.append(iband)
                        weights.append(weight)
                        
                if not active_bands:
                    continue
                
                # Calculate rho/tau for smooth pseudo charge density
                coeffs_gspace = self._construct_coefficients(ispin, ikpt, active_bands)
                num_active_bands = coeffs_gspace.shape[0]
                
                gvectors, _ = self.get_plane_waves_basis_idx(
                    ikpt, grid_shape=self._minimum_fft_size * 2, expected_npw=coeffs_gspace.shape[1]
                )
                
                phases = np.exp(2j * np.pi * np.dot(gvectors, target_frac.T))
                
                rgvec = gvectors @ (2 * np.pi * self.reciprocal_lattice)
                k_cart = self.kpoints_cart[ikpt]
                K_cart = rgvec + k_cart[np.newaxis, :]
                gk2 = np.linalg.norm(K_cart, axis=1)**2
                
                phi_at_points = np.dot(coeffs_gspace, phases) * norm_factor
                lap_phi_at_points = np.dot(coeffs_gspace, -gk2[:, np.newaxis] * phases) * norm_factor
                
                active_weights = np.array(weights)
                w_arr = active_weights[:, np.newaxis]
                
                rho_total += np.sum(phi_at_points.conj() * phi_at_points * w_arr, axis=0)
                tau_total += np.sum(-phi_at_points * lap_phi_at_points.conj() * w_arr, axis=0)
                
                # Apply augmentation corrections
                if include_aug:
                    num_atoms = len(self.structure)
                    projections_all_atoms = []
                    for i_atom in range(num_atoms):
                        elem = self.structure[i_atom].species_string
                        dataset = self._aug_environment.paw_datasets[elem]
                        h = dataset.q_linear_grid[-1]
                        
                        P_G_matrix = compute_reciprocal_projectors(
                            k_cart, rgvec, self.structure[i_atom].coords, h,
                            len(dataset.q_linear_grid), self.structure.volume,
                            dataset.reciprocal_projectors, dataset.angular_momenta, dataset.magnetic_nums
                        )
                        proj_atom = np.dot(P_G_matrix, coeffs_gspace.T)
                        projections_all_atoms.append(proj_atom)
                        
                    for n_idx, frac in enumerate(target_frac):
                        point_cart = frac @ self._aug_environment.lattice_matrix
                        
                        aug_bands = np.zeros(num_active_bands, dtype=np.float64)
                        aug_ke_bands = np.zeros(num_active_bands, dtype=np.float64)
                        
                        for b_idx in range(num_active_bands):
                            band_density_matrices = []
                            for i_atom in range(num_atoms):
                                proj = projections_all_atoms[i_atom][:, b_idx]
                                dm = np.outer(proj, proj.conj()).real
                                band_density_matrices.append(dm)
                                
                            ae_rho, ps_rho = self._aug_environment.calculate_onsite_densities_at_point(
                                point_cart=point_cart, density_matrices=band_density_matrices
                            )
                            aug_bands[b_idx] = ae_rho - ps_rho
                            
                            ae_tau, ps_tau = self._aug_environment.calculate_onsite_ke_densities_at_point(
                                point_cart=point_cart, density_matrices=band_density_matrices
                            )
                            aug_ke_bands[b_idx] = ae_tau - ps_tau
                            
                        rho_total[n_idx] += np.sum(aug_bands * active_weights)
                        tau_total[n_idx] += np.sum(aug_ke_bands * active_weights)
                        
                if return_grad_rho_sq or return_lap_rho:
                    grad_phi_at_points = np.zeros((num_active_bands, num_targets, 3), dtype=complex)
                    for idim in range(3):
                        grad_phi_at_points[:, :, idim] = np.dot(coeffs_gspace, 1j * K_cart[:, idim, np.newaxis] * phases) * norm_factor
                        
                if return_grad_rho_sq:
                    grad_rho_tensor = 2.0 * (phi_at_points[:, :, np.newaxis].conj() * grad_phi_at_points).real
                    grad_rho_sq_bands = np.sum(grad_rho_tensor**2, axis=2)  
                    grad_rho_sq_total += np.sum(grad_rho_sq_bands * w_arr, axis=0)
                    
                if return_lap_rho or not use_shrod_tau:
                    grad_psi_sq = np.sum(np.abs(grad_phi_at_points)**2, axis=2)  
                    lap_rho_bands = 2.0 * (grad_psi_sq + (phi_at_points.conj() * lap_phi_at_points).real)
                    if not use_shrod_tau:
                        tau_total += lap_rho_bands / 2
                    if return_lap_rho:
                        lap_rho_total += np.sum(lap_rho_bands * w_arr, axis=0)

        # convert to real space
        rho_total = rho_total.real
        tau_total = tau_total.real

        if is_single_point:
            results = [rho_total[0], tau_total[0]]
            if return_grad_rho_sq: results.append(grad_rho_sq_total[0])
            if return_lap_rho: results.append(lap_rho_total[0])
        else:
            results = [rho_total, tau_total]
            if return_grad_rho_sq: results.append(grad_rho_sq_total)
            if return_lap_rho: results.append(lap_rho_total)
            
        return tuple(results) if len(results) > 2 else (results[0], results[1])

    def get_rho_tau_vs_energy(
        self, 
        frac_coord, 
        return_grad_rho_sq = False,
        return_lap_rho = False,
        spin_channel = -1, 
        include_aug=True,
        cumulative=False,
        return_plot=False,
        plot_range=None,
        use_shrod_tau=False,
    ):
        """
        Calculates exact state-resolved kinetic and charge density metrics at a single point coordinate,
        supporting both differential spectral slices and full cumulative accumulation options.
        """
    
        def point_callback(ispin, ikpt, coeffs_list, gvectors, kx_idx, ky_idx, kz_idx, weight, gshape, norm_factor):
            # Evaluate continuous phase factor exp(2pi * i * G.r) to avoid real-space grid voxel snapping
            phases = np.exp(2j * np.pi * np.dot(gvectors, frac_coord))
            
            # Map reciprocal plane waves out to Cartesian coordinates to construct exact momentum vectors
            rgvec = gvectors @ (2 * np.pi * self.reciprocal_lattice)
            k = self.kpoints_cart[ikpt]             
            K_cart = rgvec + k[np.newaxis, :]             
            gk2 = np.sum(K_cart**2, axis=1)             
            
            # Extract real-space wave amplitudes and laplacians at coordinate via direct dot product
            phi_at_point = np.dot(coeffs_list, phases) * norm_factor
            lap_phi_at_point = np.dot(coeffs_list, -gk2 * phases) * norm_factor
            
            # Construct standard background pseudo-charge and kinetic metric distributions
            rho_bands = phi_at_point.conj() * phi_at_point * weight
            tau_bands = -phi_at_point * lap_phi_at_point.conj() * weight
            
            num_bands_coeffs = coeffs_list.shape[0]
            
            # AUTOMATIC RECONSTRUCTION POLYMORPHIC SAFEGUARD:
            # Exclude Projector Augmented Wave (PAW) onsite modifications if our instance contains 
            # an LCAO projection layer (`_construct_coefficients`). Localized projections represent 
            # reference atomic profiles (like all-electron AESpecies states) that are already fully reconstructed,
            # meaning PAW-sphere grid corrections are physically inapplicable and would cause severe numerical distortion.
            if include_aug and not hasattr(self, '_construct_coefficients'):
                point_cart = frac_coord @ self._aug_environment.lattice_matrix
                num_atoms = len(self.structure)
                
                # Project continuous wave components onto core-region angular momentum channels
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
                
                aug_bands = np.zeros(num_bands_coeffs, dtype=np.float64)
                aug_ke_bands = np.zeros(num_bands_coeffs, dtype=np.float64)
                
                # Accumulate onsite atomic corrections (All Electron charge density minus Pseudo charge density)
                for n_idx in range(num_bands_coeffs):
                    band_density_matrices = []
                    for i_atom in range(num_atoms):
                        proj = projections_all_atoms[i_atom][:, n_idx]
                        dm = np.outer(proj, proj.conj()).real
                        band_density_matrices.append(dm)
                        
                    ae_rho, ps_rho = self._aug_environment.calculate_onsite_densities_at_point(
                        point_cart=point_cart, density_matrices=band_density_matrices
                    )
                    aug_bands[n_idx] = ae_rho - ps_rho
                    
                    ae_tau, ps_tau = self._aug_environment.calculate_onsite_ke_densities_at_point(
                        point_cart=point_cart, density_matrices=band_density_matrices
                    )
                    aug_ke_bands[n_idx] = ae_tau - ps_tau
                
                # Layer core restoration channels atop our continuous pseudo-background matrix
                rho_bands += aug_bands * weight
                tau_bands += aug_ke_bands * weight
            
            metrics = [rho_bands.real, tau_bands.real]
            
            # Optionally compute spatial vector gradient and laplacian parameters via chain-rule derivatives
            if return_grad_rho_sq or return_lap_rho or not use_shrod_tau:
                grad_phi_at_point = np.zeros((num_bands_coeffs, 3), dtype=complex)
                for idim in range(3):
                    grad_phi_at_point[:, idim] = np.dot(coeffs_list, 1j * K_cart[:, idim] * phases) * norm_factor
                
                if return_grad_rho_sq:
                    grad_rho_vec = 2.0 * (phi_at_point[:, np.newaxis].conj() * grad_phi_at_point).real
                    grad_rho_sq_bands = np.sum(grad_rho_vec**2, axis=1) * weight
                    metrics.append(grad_rho_sq_bands)
                else:
                    metrics.append(None)
                    
                if return_lap_rho or not use_shrod_tau:
                    grad_psi_sq = np.sum(np.abs(grad_phi_at_point)**2, axis=1)
                    # Real component evaluation mapping exact Laplacian fields: Re(ψ* ∇²ψ)
                    lap_rho_bands = 2.0 * (grad_psi_sq + (phi_at_point.conj() * lap_phi_at_point).real) * weight
                    
                    # Convert standard kinetic definition to true positive-definite representation
                    if not use_shrod_tau:
                        metrics[1] += lap_rho_bands / 2
                    
                    if return_lap_rho:
                        metrics.append(lap_rho_bands)
                    else:
                        metrics.append(None)
                    
            return metrics
    
        # Allocate required output arrays depending on optional structural flags
        num_metrics = 4 if (return_grad_rho_sq or return_lap_rho) else 2

        # Route variables through our newly updated polymorphic engine
        smeared = self._execute_spectral_engine(
            num_metrics=num_metrics, 
            spin_channel=spin_channel, 
            eval_callback=point_callback
        )
        smeared = [i for i in smeared if i is not None]
        
        # Symmetrically integrate differential curves if cumulative mode is toggled active
        if cumulative:
            for idx in range(len(smeared)):
                cum_array = cumulative_trapezoid(smeared[idx], self.energy_grid, initial=0)
                smeared[idx] = cum_array

        # Construct and route visualization output curves
        if return_plot:
            prefix = "Integrated " if cumulative else ""
            x_label = "Accumulated Integrated Value" if cumulative else "Differential Density Magnitude (per eV)"
            
            plot_curves = {
                f"{prefix}Charge Density $\\rho$": smeared[0],
                f"{prefix}Kinetic Density $\\tau$": smeared[1]
            }
            if return_grad_rho_sq:
                plot_curves[f"{prefix}Gradient $|\\nabla\\rho|^2$"] = smeared[2]
            if return_lap_rho:
                plot_curves[f"{prefix}Laplacian $\\nabla^2\\rho$"] = smeared[3]
                
            return self._generate_property_plot(
                plot_curves=plot_curves,
                x_label=x_label,
                plot_range=plot_range,
            )
            
        return smeared

    def get_localization_function(
            self, 
            grid_shape=None, 
            include_aug=True,
            energy_range=(-np.inf, np.inf), 
            spin_channel=-1, 
            localization_function="elf", 
            savin_correction=True,
            use_partial_occ=True,
            ):
        """Calculates specific localized electron topological indicators (ELF, LOL, or ELI-D)."""
        if grid_shape is None: 
            grid_shape = self._minimum_fft_size * 2
            
        rho, tau = self.get_rho_tau(
            grid_shape=grid_shape, 
            include_aug=include_aug,
            energy_range=energy_range, 
            spin_channel=spin_channel, 
            use_partial_occ=use_partial_occ, 
            use_shrod_tau=False,
            )
        if localization_function == "lol":
            from baderkit.post_wfc.localization_functions import lol
            return lol(rho, tau, savin_correction, spin_channel != -1)
            
        with set_workers(self.scipy_workers): 
            rho_q = fftn(rho, norm='ortho')
        gx, gy, gz = self.calculate_gradient(rho_q, is_reciprocal=True)
        grad_sq = gx**2 + gy**2 + gz**2
        
        if localization_function == "elid":
            from baderkit.post_wfc.localization_functions import elid
            return elid(rho, tau, grad_sq)
        elif localization_function == "elf":
            from baderkit.post_wfc.localization_functions import elf
            return elf(rho, tau, grad_sq, savin_correction, spin_channel != -1)
        
    def get_localization_function_vs_energy(
        self, 
        frac_coord, 
        spin_channel=-1, 
        include_aug=True,
        localization_function="elf", 
        savin_correction=True,
        cumulative=True,
        return_plot=False,
    ) -> tuple:
        """
        Calculates topological electron localization indicators (ELF, LOL, or ELI-D)
        at a single continuous fractional coordinate point as a function of energy.

        Parameters:
        -----------
        frac_coord : array-like
            Fractional coordinates [x, y, z] of the target point.
        cumulative : bool, default=True
            If True, calculates the indicators from the cumulatively integrated state
            fields up to each energy level (highly recommended for topology tracking).
            If False, evaluates the indicators purely on narrow differential energy slices.
        """
        loc_fn_lower = localization_function.lower()
        if loc_fn_lower not in ["elf", "lol", "elid"]:
            raise ValueError(f"Unknown localization function identifier profile: {localization_function}")

        # ELF and ELI-D structurally depend on real-space first and second derivatives
        need_derivatives = loc_fn_lower in ["elf", "elid"]
        
        # Pull raw energy-resolved trajectories from the core point-callback engine
        contributions = self.get_rho_tau_vs_energy(
            frac_coord=frac_coord,
            return_grad_rho_sq=need_derivatives,
            return_lap_rho=False,
            spin_channel=spin_channel,
            include_aug=include_aug,
            cumulative=True,
            return_plot=False,
            use_shrod_tau=False,
        )
        rho = contributions[0]
        tau = contributions[1]
        grad_sq = contributions[2] if need_derivatives else None

        is_spin = spin_channel != -1

        # Evaluate the localized vector profiles along the array axes
        if loc_fn_lower == "lol":
            from baderkit.post_wfc.localization_functions import lol
            loc_data = lol(rho, tau, savin_correction, is_spin)
        elif loc_fn_lower == "elid":
            from baderkit.post_wfc.localization_functions import elid
            loc_data = elid(rho, tau, grad_sq)
        elif loc_fn_lower == "elf":
            from baderkit.post_wfc.localization_functions import elf
            loc_data = elf(rho, tau, grad_sq, savin_correction, is_spin)
            
        # if not cumulative, we get the derivative
        if not cumulative:
            loc_data = np.gradient(loc_data, self.energy_grid)

        if return_plot:
            mode_prefix = "Integrated" if cumulative else "Differential"
            label = f"{mode_prefix} {localization_function.upper()}"
            return self._generate_property_plot(
                plot_curves={label: loc_data},
                x_label="Topological Indicator Value",
            )

        return loc_data
    
    def get_localization_function_at_points(
        self,
        frac_coord,
        spin_channel: int = -1,
        energy_range: tuple = (-np.inf, np.inf),
        use_partial_occ: bool = True,
        include_aug: bool = True,
        localization_function: str = "elf",
        savin_correction: bool = True,
    ) -> float | np.ndarray:
        """
        Calculates topological electron localization indicators (ELF, LOL, or ELI-D)
        directly at one or multiple discrete continuous fractional coordinate locations.
        """
        loc_fn_lower = localization_function.lower()
        if loc_fn_lower not in ["elf", "lol", "elid"]:
            raise ValueError(f"Unknown localization function identifier profile: {localization_function}")

        need_derivatives = loc_fn_lower in ["elf", "elid"]
        
        # Accumulate total integrated point properties directly from the wavefunctions
        contributions = self.get_rho_tau_at_point(
            frac_coord=frac_coord,
            return_grad_rho_sq=need_derivatives,
            return_lap_rho=False,
            spin_channel=spin_channel,
            energy_range=energy_range,
            use_partial_occ=use_partial_occ,
            include_aug=include_aug,
            use_shrod_tau=False,
        )
        
        rho = contributions[0]
        tau = contributions[1]
        grad_sq = contributions[2] if need_derivatives else None
        
        is_spin = spin_channel != -1

        # Evaluate topological expressions over the resulting point arrays
        if loc_fn_lower == "lol":
            from baderkit.post_wfc.localization_functions import lol
            return lol(rho, tau, savin_correction, is_spin)
        elif loc_fn_lower == "elid":
            from baderkit.post_wfc.localization_functions import elid
            return elid(rho, tau, grad_sq)
        elif loc_fn_lower == "elf":
            from baderkit.post_wfc.localization_functions import elf
            return elf(rho, tau, grad_sq, savin_correction, is_spin)
        
    def get_density_of_states(
        self, 
        spin_channel=-1, 
        return_plot=False,
        plot_range=None,
    ):
        """Constructs energy coordinate profiles outlining the Electronic Density of States (DOS)."""
        # Create a single channel representing uniform weights for total system DOS tracking
        total_weights = np.ones((1, self.nspin, self.nkpoints, self.nbands), dtype=np.float64)
        
        # RIGOROUS FIX: Multiplicative normalization correction for the tetrahedron method.
        # Compares the discrete trapezoidal integral of the grid-sampled DOS against the 
        # exact analytical total charge sum rule, rescaling the array to ensure exact conservation.
        smeared_data = self._compute_smeared_channels(
            channel_weights=total_weights,
            spin_channel=spin_channel,
            use_occupancies=False
        )
        
        dos = smeared_data[0]
                    
        if return_plot:
            return self._generate_dos_plot(
                total_dos=dos,
                plot_curves={},
                plot_range=plot_range
            )
        return dos
    
    def _compute_smeared_channels(
        self,
        channel_weights: NDArray,  # shape: (num_channels, nspin, nkpoints, nbands)
        spin_channel: int = -1,
        use_occupancies: bool = False,
    ) -> NDArray:
        """Consolidated pipeline to map state selection masks and execute smearing methods."""
        bands = self.energies
        energy_grid = self.energy_grid
        delta_e = energy_grid[1] - energy_grid[0]
        
        if spin_channel == 1 and self.nspin == 1:
            spin_channel = 0
    
        spin_all = [spin_channel] if spin_channel != -1 else list(range(self.nspin))
        factor = 2 if (spin_channel == -1 and self.nspin == 1) else 1
        num_channels = channel_weights.shape[0]

        smeared_data = np.zeros((num_channels, len(energy_grid)), dtype=np.float64)

        # --- Pipeline 1: Analytic Tetrahedron Profile Method ---
        if self.smearing == "tetrahedron":
            full_map = self.full_to_irr_map
            eigenvalues = bands[spin_all][:, full_map, :]  
            tetra_indices = self.tetrahedra_indices
            tetra_weight = 1.0 / len(tetra_indices)
            
            if use_occupancies:
                w_t = (self.occupancies[spin_all][:, full_map, :] * factor)[..., np.newaxis]
            else:
                w_t = np.ones_like(eigenvalues)[..., np.newaxis] * factor
                
            for c in range(num_channels):
                c_w_full = channel_weights[c][spin_all][:, full_map, :]
                w_t_c = w_t * c_w_full[..., np.newaxis]
                
                smeared_data[c] = _integrate_tetrahedra_spectral_density_numba(
                    energy_grid, tetra_indices, eigenvalues, w_t_c, tetra_weight,
                )[0].sum(axis=0)
                    
        # --- Pipeline 2: Analytic Matrix Broadening Broadcaster ---
        else:
            kpt_weights = self.kpoint_weights
            
            for c in range(num_channels):
                c_w_flat = channel_weights[c][spin_all].ravel()
                if use_occupancies:
                    w_t = (self.occupancies[spin_all] * kpt_weights[None, :, None]).ravel() * factor * c_w_flat
                else:
                    w_t = (np.ones_like(bands[spin_all]) * kpt_weights[None, :, None]).ravel() * factor * c_w_flat
                smeared_data[c] = np.dot(self.smear_matrix, w_t)
                    
        # RIGOROUS FIX: Multiplicative normalization correction for the tetrahedron method.
        # Compares the discrete trapezoidal integral of the grid-sampled DOS against the 
        # exact analytical total charge sum rule, rescaling the array to ensure exact conservation.
        if self.smearing == "tetrahedron":
            if self.sigma > 0.0:
                n_kernel = int(np.ceil(14.0 * self.sigma / delta_e))
                if n_kernel > 0:
                    x_kernel = np.arange(-n_kernel, n_kernel + 1) * delta_e
                    scaled_x = x_kernel / self.sigma
                    exp_term = np.exp(np.clip(scaled_x, -50, 50))
                    kernel = exp_term / (exp_term + 1.0)**2
                    kernel /= np.sum(kernel)
                    for c in range(num_channels):
                        smeared_data[c] = np.convolve(smeared_data[c], kernel, mode='same')
            
            kpt_weights = self.kpoint_weights
            for c in range(num_channels):
                if use_occupancies:
                    exact_total_charge = np.sum(channel_weights[c][spin_all] * self.occupancies[spin_all] * kpt_weights[None, :, None]) * factor
                else:
                    exact_total_charge = np.sum(channel_weights[c][spin_all] * kpt_weights[None, :, None]) * factor
                    
                calculated_total_charge = np.trapezoid(smeared_data[c], energy_grid)
                if calculated_total_charge > 0.0 and exact_total_charge > 0.0:
                    smeared_data[c] *= (exact_total_charge / calculated_total_charge)
                    
        return smeared_data

    

    ###########################################################################
    # Public Helper Functions
    ###########################################################################
    def calculate_laplacian(self, data, is_reciprocal=False):
        """Evaluates second-derivative field Laplacian grid profiles via algebraic Fourier space multiplication."""
        # FIX: Dynamically resolve the Cartesian G-space grids using the incoming data shape
        Gx, Gy, Gz = self.plane_waves_cart(grid_shape=data.shape)
        G2 = Gx**2 + Gy**2 + Gz**2  
        
        if not is_reciprocal:
            with set_workers(self.scipy_workers): 
                recip_data = fftn(data, norm='ortho')
        else: 
            recip_data = data
            
        recip_lap = -G2 * recip_data
        with set_workers(self.scipy_workers): 
            real_lap = ifftn(recip_lap, norm='ortho')
        return real_lap.real
        
    def calculate_gradient(self, data, is_reciprocal=False):
        """Evaluates first-derivative components partial vectors fields using spatial Fourier transforms."""
        if not is_reciprocal:
            with set_workers(self.scipy_workers): 
                recip_data = fftn(data, norm='ortho')
        else: 
            recip_data = data
            
        # FIX: Dynamically resolve the Cartesian G-space grids using the incoming data shape
        Gx, Gy, Gz = self.plane_waves_cart(grid_shape=data.shape)
        with set_workers(self.scipy_workers):
            grad_x = ifftn(1j * Gx * recip_data, norm='ortho')
            grad_y = ifftn(1j * Gy * recip_data, norm='ortho')
            grad_z = ifftn(1j * Gz * recip_data, norm='ortho')
        return grad_x.real, grad_y.real, grad_z.real
    
    def get_electrons_in_energy_range(
            self, 
            energy_range=None, 
            ):
        """Integrates occupied DOS profiles across designated energy limits."""
        
        e_min, e_max = self._clean_energy_ranges(energy_range)
        
        if e_min == e_max:
            return 0.0
        
        energy_grid = self.energy_grid
        charge_grid=self.total_charge_grid
        min_charge, max_charge = np.interp((e_min,e_max), energy_grid, charge_grid)
        return max_charge - min_charge
        
    def find_energy_for_electron_count(
            self, 
            target_electrons, 
            e_min=None, 
            ):
        """Identifies relative energy cutoff limits enclosing specific targeted electron populations."""
        full_e_min, full_e_max = self.energy_range
        
        e_min = e_min or 0.0
        e_max = e_min + target_electrons
        
        return np.interp((e_min, e_max), self.total_charge_grid, self.energy_grid)
        
    ###########################################################################
    # Private Helper functions
    ###########################################################################
    
    @abstractmethod
    def _construct_coefficients(self, ispin: int, ikpt: int, active_bands: list) -> np.ndarray:
        """
        Abstract method to resolve and return plane-wave representation coefficients 
        matching shape (len(active_bands), num_plane_waves) at a specific k-point.
        """
        pass

    def _execute_core_density_loop(
        self,
        grid_shape: tuple,
        spin_channel: int = -1,
        energy_range: tuple = (-np.inf, np.inf),
        use_partial_occ: bool = True,
        weight_callback=None,
        use_shrod_tau=False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Centralized state-space transformation driver. Iterates through system eigenstates,
        applies Schrödinger kinetic operators, and executes dense 3D IFFT conversions to real-space grids.
        """
        
        Nx, Ny, Nz = grid_shape
        # Adjust norm factor to purely scale real-space units based on crystal volume
        normFac = 1.0 / np.sqrt(self.structure.volume)
        
        rho = np.zeros(grid_shape, dtype=np.complex128)
        tau = np.zeros(grid_shape, dtype=np.complex128)
        spin_indices = [i for i in range(self.nspin)] if spin_channel == -1 else [spin_channel]
        
        for ispin in spin_indices:
            for ikpt in range(self.nkpoints):
                active_bands = []
                weights = []
                for iband in range(self.nbands):
                    rel_energy = self.energies[ispin, ikpt, iband]
                    if not (energy_range[0] <= rel_energy <= energy_range[1]): 
                        continue
                    rspin = 2.0 if self.nspin == 1 else 1.0
                    weight = rspin * self.kpoint_weights[ikpt] * (self.occupancies[ispin, ikpt, iband] if use_partial_occ else 1.0)
                    if weight > 0:
                        active_bands.append(iband)
                        weights.append(weight)
                        
                if not active_bands: 
                    continue
                
                if weight_callback is not None:
                    weight_callback(ispin, ikpt, active_bands, weights)
                
                coeffs_gspace = self._construct_coefficients(ispin, ikpt, active_bands)
                
                gvectors, gvec_wrapped = self.get_plane_waves_basis_idx(
                    ikpt, grid_shape, expected_npw=coeffs_gspace.shape[1]
                )
                rgvec = gvectors @ (2 * np.pi * self.reciprocal_lattice)
                k_cart = self.kpoints_cart[ikpt]
                gk2 = np.sum((rgvec + k_cart[np.newaxis, :])**2, axis=1)
                
                phi_k = np.zeros((len(active_bands), Nx, Ny, Nz), dtype=np.complex128)
                lap_phi_k = np.zeros((len(active_bands), Nx, Ny, Nz), dtype=np.complex128)
                
                phi_k[:, gvec_wrapped[:, 0], gvec_wrapped[:, 1], gvec_wrapped[:, 2]] = coeffs_gspace
                lap_phi_k[:, gvec_wrapped[:, 0], gvec_wrapped[:, 1], gvec_wrapped[:, 2]] = -gk2[np.newaxis, :] * coeffs_gspace
                
                # FIX: norm='forward' removes artificial bumps by handling the grid scaling natively
                with set_workers(self.scipy_workers):
                    phi_r = ifftn(phi_k, axes=(1, 2, 3), norm='forward') * normFac
                    lap_phi_r = ifftn(lap_phi_k, axes=(1, 2, 3), norm='forward') * normFac
                    
                w_arr = np.array(weights)[:, np.newaxis, np.newaxis, np.newaxis]
                rho += np.sum(phi_r.conj() * phi_r * w_arr, axis=0)
                tau += np.sum(-phi_r * lap_phi_r.conj() * w_arr, axis=0)
        
        rho = self._symmetrize_3d_grid(rho.real)
        tau =  self._symmetrize_3d_grid(tau.real)
        
        if not use_shrod_tau:
            laplacian = self.calculate_laplacian(rho)
            tau += laplacian/2
                
        return rho, tau
    
    def _get_tetrahedra(self):
        """Identifies uniform k-point grid dimensions and splits each micro-cell into 6 tetrahedra."""
        kpts = np.mod(np.round(self.kpoints_full, 6), 1.0)
        u1, u2, u3 = np.unique(kpts[:, 0]), np.unique(kpts[:, 1]), np.unique(kpts[:, 2])
        nk1, nk2, nk3 = len(u1), len(u2), len(u3)
        
        i = np.argmin(np.abs(kpts[:, 0, None] - u1[None, :]), axis=1)
        j = np.argmin(np.abs(kpts[:, 1, None] - u2[None, :]), axis=1)
        k = np.argmin(np.abs(kpts[:, 2, None] - u3[None, :]), axis=1)
        
        grid_indices = np.zeros((nk1, nk2, nk3), dtype=int)
        grid_indices[i, j, k] = np.arange(len(kpts))
            
        I, J, K = np.ogrid[:nk1, :nk2, :nk3]
        Ip = (I + 1) % nk1
        Jp = (J + 1) % nk2
        Kp = (K + 1) % nk3
        
        c000 = grid_indices[I,  J,  K ][:, :, :, None]
        c100 = grid_indices[Ip, J,  K ][:, :, :, None]
        c010 = grid_indices[I,  Jp, K ][:, :, :, None]
        c110 = grid_indices[Ip, Jp, K ][:, :, :, None]
        c001 = grid_indices[I,  J,  Kp][:, :, :, None]
        c101 = grid_indices[Ip, J,  Kp][:, :, :, None]
        c011 = grid_indices[I,  Jp, Kp][:, :, :, None]
        c111 = grid_indices[Ip, Jp, Kp][:, :, :, None]
        
        t1 = np.concatenate([c000, c100, c110, c111], axis=-1).reshape(-1, 4)
        t2 = np.concatenate([c000, c100, c101, c111], axis=-1).reshape(-1, 4)
        t3 = np.concatenate([c000, c001, c101, c111], axis=-1).reshape(-1, 4)
        t4 = np.concatenate([c000, c010, c110, c111], axis=-1).reshape(-1, 4)
        t5 = np.concatenate([c000, c010, c011, c111], axis=-1).reshape(-1, 4)
        t6 = np.concatenate([c000, c001, c011, c111], axis=-1).reshape(-1, 4)
        
        return np.vstack([t1, t2, t3, t4, t5, t6])
        
    def _get_default_sigma(self, method, sigma):
        if method is None: method = "none"
            
        shorthands = {
            "none": "none",
            "n": "none",
            "g": "gaussian",
            "gauss": "gaussian",
            "gaussian": "gaussian", 
            "methfessel-paxton": "methfessel-paxton",
            "mp": "methfessel-paxton", 
            "fermi-dirac": "fermi-dirac", 
            "fm": "fermi-dirac",
            "tetrahedral": "tetrahedron",
            "tetrahedron": "tetrahedron", 
            "tet": "tetrahedron",
            "tetra": "tetrahedron",
            }
        
        self.method = shorthands.get(method if isinstance(method, str) else method, None)
        if self.method is None:
            raise ValueError(f"Unknown smearing method: '{method}'")
            
        # ensure sigma is greater than 0
        if sigma is not None:
            invalid_sigma = sigma < 0.0
            if invalid_sigma:
                old_sigma = sigma
                sigma = None
        
        if sigma is None:
            default_sigma = {
                "none": 0.0, 
                "gaussian": 0.1, 
                "methfessel-paxton": 0.18,
                "fermi-dirac": 300, # Input in Kelvin 
                "tetrahedron": 300, # Same as fermi-dirac
                }
            sigma = default_sigma[self.method]
            invalid_sigma=False
        
        if invalid_sigma:
            logging.warning(f"Invalid sigma: {old_sigma}. Using {self.method} default: {sigma}.")
            
        if self.method == "fermi-dirac" or self.method == "tetrahedron":
            sigma = 8.617333262e-5 * sigma
        
        return self.method, sigma
        
    def _get_smear_matrix(self):
        """Helper matrix generator parsing customized analytical broadening distributions."""
        
        # Manual construction for None
        if self.smearing == "none":
            e_min, e_max = self.energy_range
            delta_e = self.energy_grid[1] - self.energy_grid[0]
            smear_matrix = np.zeros((self.num_points, len(self.energies)))
            closest_idx = np.round((self.energies - e_min) / delta_e).astype(int)
            valid_mask = (closest_idx >= 0) & (closest_idx < self.num_points)
            smear_matrix[closest_idx[valid_mask], np.where(valid_mask)[0]] = 1.0 / delta_e
            return smear_matrix
            
        delta_E = self.energy_grid[:, None] - self.energies[None, :]
        x = delta_E / self.sigma
        
        if self.smearing == "gaussian":
            return np.exp(-0.5 * x**2) / (self.sigma * np.sqrt(2 * np.pi))
        elif self.smearing in ["methfessel-paxton", "mp"]:
            term_0 = np.exp(-x**2) / np.sqrt(np.pi)
            return ((1.5 - x**2) * term_0) / self.sigma
        elif self.smearing in ["fermi-dirac", "fd"]:
            exp_term = np.exp(np.clip(x, -50, 50))
            return (exp_term / (exp_term + 1.0)**2) / self.sigma
        else:
            raise ValueError(f"Unknown smearing self.smearing: '{self.smearing}'")
            
    def _execute_spectral_engine(
            self,
            num_metrics,
            spin_channel,
            eval_callback
            ):
        """
        Unified high-performance orchestration engine for plane-wave spectral decompositions.
        
        Polymorphically dispatches state collection paths depending on whether the calling
        instance is a raw plane-wave driver (PostWFC) or a localized reference projection 
        environment (AtomicProjectionEnvironment).
        """
        
        # Configure spin channels: index all channels if -1, otherwise isolate requested index
        spin_indices = [i for i in range(self.nspin)] if spin_channel == -1 else [spin_channel]
        
        # Determine real-space FFT grid dimensions and volume normalization scaling
        grid_shape = self._minimum_fft_size * 2
        nx, ny, nz = grid_shape
        norm_factor = 1.0 / np.sqrt(self.structure.volume)
        
        # Allocate continuous block memory for state properties across metrics, spins, k-points, and bands
        raw_data = np.zeros((num_metrics, self.nspin, self.nkpoints, self.nbands), dtype=float)
        delta_e = self.energy_grid[1] - self.energy_grid[0]
        
        # Standardize strings and scale widths (e.g., Kelvin temperature -> eV units)
        kpoint_weights = self.kpoint_weights

        # Main orchestration loop over active spin channels and k-points
        for ispin in spin_indices:
            for ikpt in range(self.nkpoints):
                
                active_bands = [iband for iband in range(self.nbands)]
                
                # Assign Brillouin zone integration weights (handled natively within analytical tetrahedra)
                rspin = 2.0 if self.nspin == 1 else 1.0
                weight = rspin * kpoint_weights[ikpt] if self.smearing != "tetrahedron" else rspin
                
                # POLYMORPHIC COUPLING GATE:
                # If the instance contains an LCAO project-basis mapping layer, retrieve 
                # reconstructed coefficients matrix from memory. Otherwise, execute a raw binary 
                # disk batch-read of plane-wave pseudo-wavefunctions.
                if hasattr(self, '_construct_coefficients'):
                    coeffs_list = self._construct_coefficients(ispin, ikpt, active_bands)
                else:
                    coeffs_list = self._wf_reader.read_coefficients_batch(ispin, ikpt, active_bands)
                
                # Fetch reciprocal grid indices corresponding to the active plane-wave cutoff
                gvectors, _ = self.get_plane_waves_basis_idx(ikpt, grid_shape, expected_npw=coeffs_list.shape[1])
                
                # Wrap reciprocal coordinate indexes to protect FFT grid boundaries
                kx_idx = gvectors[:, 0] % nx
                ky_idx = gvectors[:, 1] % ny
                kz_idx = gvectors[:, 2] % nz
                
                # Fire the callback to evaluate spatial properties at this k-point/spin slice
                metrics_block = eval_callback(
                    ispin, ikpt, coeffs_list, gvectors, kx_idx, ky_idx, kz_idx, 
                    weight, grid_shape, norm_factor
                )
                
                # Map computed properties back into the global state data cache
                for imetric, metric_bands in enumerate(metrics_block):
                    if metric_bands is not None:
                        raw_data[imetric, ispin, ikpt, active_bands] = metric_bands
                        
        # POST-PROCESSING KERNEL EVALUATION:
        # Generate a post-processing convolution array if convolved tetrahedron smearing is active.
        use_convolution = (self.smearing == "tetrahedron" and self.sigma > 0.0)
        if use_convolution:
            # Span kernel up to 14*sigma to capture slow exponential decay tails of Fermi-Dirac distribution
            n_kernel = int(np.ceil(14.0 * self.sigma / delta_e))
            if n_kernel > 0:
                x_kernel = np.arange(-n_kernel, n_kernel + 1) * delta_e
                scaled_x = x_kernel / self.sigma
                exp_term = np.exp(np.clip(scaled_x, -50, 50))
                # Evaluate analytical first derivative of Fermi-Dirac distribution
                kernel = exp_term / (exp_term + 1.0)**2
                kernel /= np.sum(kernel)
            else:
                use_convolution = False

        # Route 1: Execute Numba-accelerated analytical tetrahedral cell integration
        if self.smearing == "tetrahedron":
            full_map = self.full_to_irr_map
            cached_metrics = np.ascontiguousarray(np.transpose(raw_data, (1, 2, 3, 0)))
            eigenvalues = self.energies[:, full_map, :]
            cached_metrics = cached_metrics[:, full_map, :, :]
            
            tetra_indices = self.tetrahedra_indices
            tetra_weight = 1.0 / len(tetra_indices)
        
            smeared_output = _integrate_tetrahedra_spectral_density_numba(
                self.energy_grid, tetra_indices, eigenvalues,
                cached_metrics, tetra_weight,
            )
        
            smeared_results = []
            for imetric in range(num_metrics):
                smeared = np.sum(smeared_output[imetric], axis=0)
                # Symmetrically smooth step features if post-processing smearing is enabled
                if use_convolution:
                    smeared = np.convolve(smeared, kernel, mode='same')
                smeared_results.append(smeared)
            return smeared_results
            
        # Route 2: Continuous matrix multiplication fallback for standard continuous broadening functions
        smear_matrix = self.smear_matrix
            
        smeared_results = []
        for imetric in range(num_metrics):
            vals_all = raw_data[imetric, spin_indices].ravel()
            smeared = np.dot(smear_matrix, vals_all)
            smeared_results.append(smeared)
            
        return smeared_results
    
    def _symmetrize_3d_grid(self, field):
        """Averages real-space property grid profiles over space-group operations."""
        symmetry = self.structure.symmetry_data
        Nx, Ny, Nz = field.shape
        sym_field = np.zeros_like(field)
        
        mx, my, mz = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing='ij')
        coords = np.stack([mx.ravel() / Nx, my.ravel() / Ny, mz.ravel() / Nz], axis=0)
        
        for R, t in zip(symmetry.rotations, symmetry.translations):
            R_inv = np.round(np.linalg.inv(R)).astype(int)
            tc = ((R_inv @ coords - (R_inv @ t)[:, np.newaxis]) % 1.0)
            
            sym_field += field[
                np.round(tc[0] * Nx).astype(int) % Nx, 
                np.round(tc[1] * Ny).astype(int) % Ny, 
                np.round(tc[2] * Nz).astype(int) % Nz
            ].reshape(Nx, Ny, Nz)
            
        return sym_field / len(symmetry.rotations)
    
    def _get_total_charge_vs_energy(
        self, 
    ) -> np.ndarray:
        """
        Computes, caches, and retrieves the total cell charge integrated as a function 
        of energy. Uses a multi-key cache to avoid repeating expensive analytical 
        tetrahedron integrations or high-density DOS convolutions.
        """
        
        # Enforce dynamic unaliased energy ranges using original parameters
        energy_grid = self.energy_grid
        spin_indices = [i for i in range(self.nspin)]
        # Linearly scale sampling density based on the chosen resolution ratio
        delta_e = energy_grid[1] - energy_grid[0]
        
        # Route 1: Direct analytical tetrahedron step integration (Un-smeared)
        if self.method == "tetrahedron" and (self.sigma is None or self.sigma == 0.0):
            tetra_indices = self.tetrahedra_indices
            tetra_weight = 1.0 / len(tetra_indices)
            rspin = 2.0 if self.nspin == 1 else 1.0
            
            
            full_map = self.full_to_irr_map
            eigenvalues_full = self.energies[spin_indices][:, full_map, :]
            total_charge = _integrate_tetrahedra_analytic_charge_numba(
                energy_grid=energy_grid,
                tetra_indices=tetra_indices,
                eigenvalues=eigenvalues_full,
                tetra_weight=tetra_weight,
                rspin=rspin
            )
        
        # Route 2: Convolved Tetrahedron Method (Direct Cumulative Charge Convolution)
        elif self.method == "tetrahedron" and self.sigma > 0.0:
            tetra_indices = self.tetrahedra_indices
            tetra_weight = 1.0 / len(tetra_indices)
            rspin = 2.0 if self.nspin == 1 else 1.0
            
            full_map = self.full_to_irr_map
            eigenvalues_full = self.energies[spin_indices][:, full_map, :]
            
            # 1. Compute the exact analytical un-smeared cumulative charge profile
            total_charge_unsmeared = _integrate_tetrahedra_analytic_charge_numba(
                energy_grid=energy_grid,
                tetra_indices=tetra_indices,
                eigenvalues=eigenvalues_full,
                tetra_weight=tetra_weight,
                rspin=rspin
            )
            
            # 2. Build the exact Fermi-Dirac derivative kernel matching your post-processing environment
            n_kernel = int(np.ceil(14.0 * self.sigma / delta_e))
            if n_kernel > 0:
                x_kernel = np.arange(-n_kernel, n_kernel + 1) * delta_e
                scaled_x = x_kernel / self.sigma
                exp_term = np.exp(np.clip(scaled_x, -50, 50))
                kernel = exp_term / (exp_term + 1.0)**2
                kernel /= np.sum(kernel)
                
                # 3. Replicate edge values to protect plateaus against zero-padding artifacts
                padded_charge = np.pad(total_charge_unsmeared, n_kernel, mode='edge')
                total_charge = np.convolve(padded_charge, kernel, mode='valid')
            else:
                total_charge = total_charge_unsmeared
        
        # Route 3: Fallback path for analytical continuous smearing methods (Gaussian, MP, FD matrix)
        else:
            total_dos = self.tdos
            
            dx = np.diff(energy_grid)
            avg_dos = 0.5 * (total_dos[:-1] + total_dos[1:])
            total_charge = np.zeros_like(energy_grid)
            total_charge[1:] = np.cumsum(avg_dos * dx)
                
        
        return total_charge

    def _clean_charge_ranges(self, min_charge, max_charge):
        if min_charge is None or min_charge == -np.inf:
            min_charge = 0.0
        if max_charge is None or max_charge == np.inf:
            max_charge = self.maximum_charge
            
        min_charge = max(min_charge, 0)
        max_charge = min(max_charge, self.maximum_charge)
        return min_charge, max_charge
    
    def _clean_energy_ranges(self, energy_range):
        if energy_range is None:
            e_min, e_max = self.energy_range
        else:
            e_min, e_max = energy_range
            full_e_min, full_e_max = self.energy_range
            if e_min is None or e_min == -np.inf: e_min = full_e_min
            if e_max is None or e_max == np.inf: e_max = full_e_max
        return e_min, e_max

    ###########################################################################
    # Plotting Helpers
    ###########################################################################
    def _generate_property_plot(self, plot_curves, x_label, plot_range=None):
        """
        Shared visualization module that converts point-resolved physical metrics 
        vs energy levels into a clean, publication-ready Matplotlib figure object.
        """
        import matplotlib.pyplot as plt
        
        # Initialize figure frame with a high-resolution canvas size
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        
        max_val = 1e-6
        # Loop over every property dataset passed in the plot tracking dictionary
        for label, data in plot_curves.items():
            if data is not None:
                # Plot properties on the X-axis and energies on the Y-axis (swapped layout axis)
                ax.plot(data, self.energy_grid, label=label, linewidth=2.5)
                
                # Dynamically determine visible viewport boundaries to prevent over-scaling the X-axis limit
                ymin, ymax = self.energy_grid[0], self.energy_grid[-1]
                if plot_range is not None:
                    if plot_range[0] is not None and plot_range[0] != -np.inf: 
                        ymin = plot_range[0]
                    if plot_range[1] is not None and plot_range[1] != np.inf: 
                        ymax = plot_range[1]
                
                max_val = data.max()
            
        # Enforce explicit axis viewport boundaries matching the calculation limits
        ymin, ymax = self.energy_grid[0], self.energy_grid[-1]
        if plot_range is not None:
            if plot_range[0] is not None and plot_range[0] != -np.inf: 
                ymin = plot_range[0]
            if plot_range[1] is not None and plot_range[1] != np.inf: 
                ymax = plot_range[1]
                
        # Apply a clean 5% padding on the right edge so line paths do not clip the border
        ax.set_xlim(0.0, 1.05 * max_val)
        ax.set_ylim(ymin, ymax)
        
        # Apply minimalist styling parameters reflecting the 'plotly_white' template canvas
        ax.set_facecolor("white")
        ax.grid(True, which="both", color="black", alpha=0.05, linestyle="-")
        
        # Hide top and right outer borders for a modern look
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
            
        # Draw a clear dashed indicator tracking the system Fermi level (Energy = 0.0 eV)
        ax.axhline(0.0, linestyle="--", color="gray", alpha=0.8, linewidth=1.5)
        
        # Position the EF label tracking data space on Y, but viewport space (1% from left border) on X
        ax.text(0.01, 0.0, "$E_F$ ", transform=ax.get_yaxis_transform(), 
                color="gray", va="bottom", ha="left", fontsize=12)
        
        # Solid vertical baseline tracking the zero-density origin boundary
        ax.axvline(0.0, color="black", linewidth=1)
        
        # Apply customized typography elements across axes frames
        ax.set_xlabel(x_label, fontsize=14, family="sans-serif")
        ax.set_ylabel("Energy - $E_F$ (eV)", fontsize=14, family="sans-serif")
        ax.legend(fontsize=12, loc="upper right", frameon=True)
        
        plt.tight_layout()
        return fig
    
    def _generate_dos_plot(self, total_dos, plot_curves, plot_range=None):
        """
        Shared high-performance plotting module that converts raw spectral data matrices 
        into a polished, publication-ready Matplotlib figure object.
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        
        # Plot baseline Total DOS
        ax.plot(total_dos, self.energy_grid, label="total", color="black", linewidth=2.5)
        
        # Plot individual contributing channels
        for label, data in plot_curves.items():
            ax.plot(data, self.energy_grid, label=label, linewidth=2.5)
            
        # Compute exact bounded viewport ranges 
        ymin, ymax = self.energy_grid[0], self.energy_grid[-1]
        if plot_range is not None:
            if plot_range[0] is not None and plot_range[0] != -np.inf: 
                ymin = plot_range[0]
            if plot_range[1] is not None and plot_range[1] != np.inf: 
                ymax = plot_range[1]
        
        ax.set_xlim(0.0, 1.05 * total_dos.max())
        ax.set_ylim(ymin, ymax)
        
        # Styling parameters mirroring the clean plotly_white layout
        ax.set_facecolor("white")
        ax.grid(True, which="both", color="black", alpha=0.05, linestyle="-")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
            
        # Structural reference lines and indicators
        ax.axhline(0.0, linestyle="--", color="gray", alpha=0.8, linewidth=1.5)
        ax.text(0.01, 0.0, "$E_F$ ", transform=ax.get_yaxis_transform(), 
                color="gray", va="bottom", ha="left", fontsize=12)
        ax.axvline(0.0, color="black", linewidth=1)
        
        # Frame text elements
        ax.set_xlabel("DOS (states/eV)", fontsize=14, family="sans-serif")
        ax.set_ylabel("Energy - $E_F$ (eV)", fontsize=14, family="sans-serif")
        ax.legend(fontsize=12, loc="upper right", frameon=True)
        
        plt.tight_layout()
        return fig

    ###########################################################################
    # From methods
    ###########################################################################
    @classmethod
    def from_directory(
            cls, 
            directory: Path | str = Path("."), 
            fmt: str = "vasp", 
            scipy_workers: int = -1, 
            **kwargs,
            ):
        """Dynamic factory mapping disk files to code-specific parsing instances."""
        if fmt == "vasp": 
            from baderkit.post_wfc.wf_readers import VaspReader as wf_reader
        elif fmt == "qe": 
            from baderkit.post_wfc.wf_readers import QeReader as wf_reader
        else: 
            raise ValueError(f"Unknown reader profile format string: {fmt}")
            
        from baderkit.post_wfc.pseudopotentials.augmentation_environment import PAWAugmentationEnvironment
        aug_env = PAWAugmentationEnvironment.from_directory(directory=directory, fmt=fmt)
        
        # Safely pop out environment arguments to insulate reader construction from TypeErrors
        reader_kwargs = kwargs.copy()
        cutoff_radius = reader_kwargs.pop("cutoff_radius", 8.0)
        basis_dir = reader_kwargs.pop("basis_dir", None)
        valence_counts = reader_kwargs.pop("valence_counts", None)
        
        return cls(
            wf_reader=wf_reader(directory=Path(directory), **reader_kwargs), 
            aug_environment=aug_env,
            scipy_workers=scipy_workers,
            cutoff_radius=cutoff_radius,
            basis_dir=basis_dir,
            valence_counts=valence_counts,
            **kwargs
        )