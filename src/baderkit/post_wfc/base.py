# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from pathlib import Path
from functools import cached_property

import numpy as np
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.fft import fftn, ifftn, set_workers
from baderkit.post_wfc.wf_readers.base import HSQDTM

from .wfc_numba import _integrate_tetrahedra_spectral_density_numba

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
        scipy_workers: int = -1,
        reference_env=None,
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

        self._total_charge = None
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
    
    def get_energy_range(self, method=None, sigma=None):
        """
        Returns relative boundary offsets scaled directly against the Fermi level (E - E_f).
        If a smearing method is provided, the range is padded to capture the continuous 
        tails of the smeared distribution.
        """
        e_min = np.min(self.energies)
        e_max = np.max(self.energies)
        
        if method is None:
            return e_min, e_max
            
        formal_method, formal_sigma = self._get_default_sigma(method, sigma)
        
        if formal_method == "none":
            pad = 0.0
        elif formal_method == "tetrahedron":
            pad = 4.0 * formal_sigma if formal_sigma > 0.0 else 0.0
        else:
            pad = 5.0 * formal_sigma
            
        return e_min - pad, e_max + pad
        
    @property
    def energy_cutoff(self):
        return self._reference_env.energy_cutoff if self._reference_env else self._meta.energy_cutoff
        
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
    def total_charge_vs_energy(self) -> np.ndarray:
        """
        Computes and caches the total cell charge integrated as a function of energy 
        across the entire unaliased system valence spectrum. 

        Returns:
        --------
        np.ndarray
            A 2D array of shape (num_points, 2), where column 0 represents the 
            energy grid (eV) and column 1 is the cumulative integrated charge Q(E).
        """
        # A callback that simply returns the state weights to compute total DOS
        def total_dos_callback(ispin, ikpt, coeffs_list, gvectors, kx_idx, ky_idx, kz_idx, weight, gshape, norm_factor):
            n_bands = coeffs_list.shape[0]
            return [np.full(n_bands, weight)]

        # Execute the engine across the full un-windowed range using default settings
        energy_grid, smeared = self._execute_spectral_engine(
            num_metrics=1,
            spin_channel=-1,
            energy_range=None,
            num_points=2000,
            method="tetrahedron",
            sigma=None,
            eval_callback=total_dos_callback
        )
        
        total_dos = smeared[0]
        
        # Numerically integrate the total DOS to get cumulative cell charge Q(E)
        dx = np.diff(energy_grid)
        avg_dos = 0.5 * (total_dos[:-1] + total_dos[1:])
        total_charge = np.zeros_like(energy_grid)
        total_charge[1:] = np.cumsum(avg_dos * dx)
        
        # Column-stacking returns the exact (N, 2) shape expected by the deformation functions
        return np.column_stack((energy_grid, total_charge))
    
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
    ) -> tuple:
        """
        Calculates the interacting electronic charge density (rho) and positive-definite kinetic 
        energy density (tau) at one or multiple continuous fractional coordinate locations.
        Optimized via vectorized plane-wave phase evaluations across all target coordinates.

        Parameters:
        -----------
        frac_coord : array-like
            Fractional coordinates matching shape (3,) for a single point, 
            or shape (N, 3) for multiple target points.
        """
        frac_coord_arr = np.asarray(frac_coord, dtype=np.float64)
        is_single_point = frac_coord_arr.ndim == 1
        target_frac = np.atleast_2d(frac_coord_arr)  # shape: (N, 3)
        num_targets = target_frac.shape[0]
        
        # Pre-allocate accumulators for the point batch
        rho_total = np.zeros(num_targets, dtype=np.float64)
        tau_total = np.zeros(num_targets, dtype=np.float64)
        grad_rho_sq_total = np.zeros(num_targets, dtype=np.float64) if return_grad_rho_sq else None
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
                
                # Dynamic subclass coefficient resolution: shape (len(active_bands), num_pw)
                coeffs_gspace = self._construct_coefficients(ispin, ikpt, active_bands)
                
                # Fetch reciprocal integer tracking configurations
                gvectors, _ = self.get_plane_waves_basis_idx(
                    ikpt, grid_shape=self._minimum_fft_size * 2, expected_npw=coeffs_gspace.shape[1]
                )
                
                # Vectorized evaluation of plane-wave phases across ALL coordinates simultaneously
                phases = np.exp(2j * np.pi * np.dot(gvectors, target_frac.T))
                
                # Construct Cartesian reciprocal momentum space vectors
                rgvec = gvectors @ (2 * np.pi * self.reciprocal_lattice)
                k_cart = self.kpoints_cart[ikpt]
                K_cart = rgvec + k_cart[np.newaxis, :]
                
                # Coherently reconstruct background wavefunctions across the point array
                phi_at_points = np.dot(coeffs_gspace, phases) * norm_factor
                
                # Unconditionally evaluate the 3 Cartesian spatial gradient channels analytically
                tau_bands = np.zeros((len(active_bands), num_targets), dtype=np.float64)
                grad_phi_at_points = np.zeros((len(active_bands), num_targets, 3), dtype=complex)
                for idim in range(3):
                    grad_phi_dim = np.dot(coeffs_gspace, 1j * K_cart[:, idim, np.newaxis] * phases) * norm_factor
                    grad_phi_at_points[:, :, idim] = grad_phi_dim
                    tau_bands += (grad_phi_dim.conj() * grad_phi_dim).real
                
                active_weights = np.array(weights)
                w_arr = active_weights[:, np.newaxis]  # shape: (len(active_bands), 1)
                
                # Apply scaling factor down to state k-weights and degeneracies
                tau_bands_weighted = tau_bands * w_arr
                rho_total += np.sum((phi_at_points.conj() * phi_at_points).real * w_arr, axis=0)
                tau_total += np.sum(tau_bands_weighted, axis=0)
                
                # Apply localized PAW corrections if overridden by child lifecycle hook
                if include_aug and hasattr(self, "_apply_onsite_augmentation_at_point"):
                    for n_idx, frac in enumerate(target_frac):
                        r_b = (phi_at_points[:, n_idx].conj() * phi_at_points[:, n_idx]).real * active_weights
                        t_b = tau_bands_weighted[:, n_idx]
                        
                        # Pass the active weights array so child corrections scale dynamically per state
                        r_b_aug, t_b_aug = self._apply_onsite_augmentation_at_point(
                            ispin=ispin, ikpt=ikpt, active_bands=active_bands, coeffs_list=coeffs_gspace,
                            rgvec=rgvec, weight=active_weights, frac_coord=frac,
                            rho_bands=r_b, tau_bands=t_b, include_aug=True
                        )
                        rho_total[n_idx] += np.sum(r_b_aug - r_b)
                        tau_total[n_idx] += np.sum(t_b_aug - t_b)
                        
                # Resolve complex spatial derivatives if tracking flags are activated
                if return_grad_rho_sq or return_lap_rho:
                    if return_grad_rho_sq:
                        grad_rho_tensor = 2.0 * (phi_at_points[:, :, np.newaxis].conj() * grad_phi_at_points).real
                        grad_rho_sq_bands = np.sum(grad_rho_tensor**2, axis=2)  
                        grad_rho_sq_total += np.sum(grad_rho_sq_bands * w_arr, axis=0)
                        
                    if return_lap_rho:
                        # Laplacian only gets populated if explicitly activated
                        gk2 = np.sum(K_cart**2, axis=1)
                        lap_phi_at_points = np.dot(coeffs_gspace, -gk2[:, np.newaxis] * phases) * norm_factor
                        grad_psi_sq = np.sum(np.abs(grad_phi_at_points)**2, axis=2)  
                        lap_rho_bands = 2.0 * (grad_psi_sq + (phi_at_points.conj() * lap_phi_at_points).real)
                        lap_rho_total += np.sum(lap_rho_bands * w_arr, axis=0)

        # Unpack results array back to pristine floats if a single point coordinate was handed in
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
        energy_range=None, 
        num_points=2000, 
        method = "gaussian", 
        sigma=None,
        grid_shape=None,
        include_aug=True,
        cumulative=False,
        return_plot=False,
    ):
        """
        Calculates exact state-resolved kinetic and charge density metrics at a single point coordinate,
        supporting both differential spectral slices and full cumulative accumulation options.
        """
        from scipy.integrate import cumulative_trapezoid

        # Fallback to standard double unaliased grid dimensions if custom dimensions are omitted
        grid_shape = grid_shape if grid_shape is not None else self._minimum_fft_size * 2
        nx, ny, nz = grid_shape
        
        # Map real continuous fractional coordinates into discrete periodic meshgrid indices
        ix = int(np.round(frac_coord[0] * nx)) % nx
        iy = int(np.round(frac_coord[1] * ny)) % ny
        iz = int(np.round(frac_coord[2] * nz)) % nz
    
        def point_callback(ispin, ikpt, coeffs_list, gvectors, kx_idx, ky_idx, kz_idx, weight, gshape, norm_factor):
            # Calculate structural coordinate phase shifting factors for the explicit point
            phases = np.exp(2j * np.pi * (kx_idx * ix / nx + ky_idx * iy / ny + kz_idx * iz / nz))
            
            # Construct Cartesian reciprocal momentum space vectors for the active block
            rgvec = gvectors @ (2 * np.pi * self.reciprocal_lattice)
            k = self.kpoints_cart[ikpt]             
            K_cart = rgvec + k[np.newaxis, :]             
            
            # Evaluate baseline pseudo-wavefunction value at the target coordinate
            phi_at_point = np.dot(coeffs_list, phases) * norm_factor
            
            # Unconditionally evaluate the 3 Cartesian spatial gradient channels analytically
            grad_phi_at_point = np.zeros((len(coeffs_list), 3), dtype=complex)
            for idim in range(3):
                grad_phi_at_point[:, idim] = np.dot(coeffs_list, 1j * K_cart[:, idim] * phases) * norm_factor
            
            # Extract fields scaled by state k-weights and degeneracies
            rho_bands = (phi_at_point.conj() * phi_at_point).real * weight
            
            # Positive-Definite Kinetic Energy formulation: sum(|grad_alpha|^2)
            tau_bands = np.sum(np.abs(grad_phi_at_point)**2, axis=1) * weight
            
            # Isolate the index configurations of states matching the requested energy boundaries
            active_bands = []
            for iband in range(self.nbands):
                rel_energy = self.energies[ispin, ikpt, iband]
                if energy_range is not None:
                    if not (energy_range[0] <= rel_energy <= energy_range[1]):
                        continue
                active_bands.append(iband)
            
            # Delegate to child class lifecycle hook to apply representation-specific terms
            rho_bands, tau_bands = self._apply_onsite_augmentation_at_point(
                ispin=ispin, ikpt=ikpt, active_bands=active_bands, coeffs_list=coeffs_list,
                rgvec=rgvec, weight=weight, frac_coord=frac_coord, 
                rho_bands=rho_bands, tau_bands=tau_bands, include_aug=include_aug
            )
            
            metrics = [rho_bands, tau_bands]
            
            # Resolve complex spatial derivatives if gradient or laplacian flags are specified
            if return_grad_rho_sq or return_lap_rho:
                if return_grad_rho_sq:
                    grad_rho_vec = 2.0 * (phi_at_point[:, np.newaxis].conj() * grad_phi_at_point).real
                    grad_rho_sq_bands = np.sum(grad_rho_vec**2, axis=1) * weight
                    metrics.append(grad_rho_sq_bands)
                else:
                    metrics.append(None)
                    
                if return_lap_rho:
                    gk2 = np.sum(K_cart**2, axis=1)             
                    lap_phi_at_point = np.dot(coeffs_list, -gk2 * phases) * norm_factor
                    grad_psi_sq = np.sum(np.abs(grad_phi_at_point)**2, axis=1)
                    lap_rho_bands = 2.0 * (grad_psi_sq + (phi_at_point.conj() * lap_phi_at_point.conj()).real) * weight
                    metrics.append(lap_rho_bands)
                else:
                    metrics.append(None)
                    
            return metrics
    
        # Allocate required collector arrays inside the spectral decomposition framework
        num_metrics = 4 if (return_grad_rho_sq or return_lap_rho) else 2

        energy_grid, smeared = self._execute_spectral_engine(
            num_metrics=num_metrics, spin_channel=spin_channel, energy_range=energy_range, 
            num_points=num_points, method=method, sigma=sigma, eval_callback=point_callback
        )
        
        # --- Handle Cumulative Trapezoidal Integrations ---
        if cumulative:
            for idx in range(len(smeared)):
                if smeared[idx] is not None:
                    cum_array = np.zeros(len(energy_grid), dtype=np.float64)
                    if len(energy_grid) > 1:
                        cum_array[1:] = cumulative_trapezoid(smeared[idx], energy_grid)
                    smeared[idx] = cum_array

        # Route directly to the graphing handler if return_plot flag is active
        if return_plot:
            prefix = "Integrated " if cumulative else ""
            x_label = "Accumulated Integrated Value" if cumulative else "Differential Density Magnitude (per eV)"
            
            plot_curves = {
                f"{prefix}Charge Density $\\rho$": smeared[0],
                f"{prefix}Kinetic Density $\\tau$": smeared[1]
            }
            if return_grad_rho_sq and smeared[2] is not None:
                plot_curves[f"{prefix}Gradient $|\\nabla\\rho|^2$"] = smeared[2]
            if return_lap_rho and smeared[3] is not None:
                plot_curves[f"{prefix}Laplacian $\\nabla^2\\rho$"] = smeared[3]
                
            return self._generate_property_plot(
                energy_grid=energy_grid,
                plot_curves=plot_curves,
                x_label=x_label,
                energy_range=energy_range
            )
            
        results = [energy_grid, smeared[0], smeared[1]]
        if return_grad_rho_sq: results.append(smeared[2])
        if return_lap_rho: results.append(smeared[3])
        return tuple(results)

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
            )
        if localization_function == "lol":
            from baderkit.post_wfc.localization_functions import lol
            return lol(rho, tau, savin_correction, spin_channel != -1)
            
        with set_workers(self.scipy_workers): 
            rho_q = fftn(rho, norm='ortho')
        lap_rho = self.calculate_laplacian(rho_q, is_reciprocal=True)
        gx, gy, gz = self.calculate_gradient(rho_q, is_reciprocal=True)
        grad_sq = gx**2 + gy**2 + gz**2
        
        if localization_function == "elid":
            from baderkit.post_wfc.localization_functions import elid
            return elid(rho, tau, lap_rho, grad_sq)
        elif localization_function == "elf":
            from baderkit.post_wfc.localization_functions import elf
            return elf(rho, tau, lap_rho, grad_sq, savin_correction, spin_channel != -1)
        
    def get_localization_function_vs_energy(
        self, 
        frac_coord, 
        spin_channel=-1, 
        energy_range=None, 
        num_points=2000, 
        method="gaussian", 
        sigma=None,
        grid_shape=None,
        include_aug=True,
        localization_function="elf", 
        savin_correction=True,
        cumulative=False,
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
            return_lap_rho=need_derivatives,
            spin_channel=spin_channel,
            energy_range=energy_range,
            num_points=num_points,
            method=method,
            sigma=sigma,
            grid_shape=grid_shape,
            include_aug=include_aug,
            cumulative=cumulative,
            return_plot=False
        )
        
        energy_grid = contributions[0]
        rho = contributions[1]
        tau = contributions[2]
        grad_sq = contributions[3] if need_derivatives else None
        lap_rho = contributions[4] if need_derivatives else None

        is_spin = spin_channel != -1

        # Evaluate the localized vector profiles along the array axes
        if loc_fn_lower == "lol":
            from baderkit.post_wfc.localization_functions import lol
            loc_data = lol(rho, tau, savin_correction, is_spin)
        elif loc_fn_lower == "elid":
            from baderkit.post_wfc.localization_functions import elid
            loc_data = elid(rho, tau, lap_rho, grad_sq)
        elif loc_fn_lower == "elf":
            from baderkit.post_wfc.localization_functions import elf
            loc_data = elf(rho, tau, lap_rho, grad_sq, savin_correction, is_spin)

        if return_plot:
            mode_prefix = "Integrated" if cumulative else "Differential"
            label = f"{mode_prefix} {localization_function.upper()}"
            return self._generate_property_plot(
                energy_grid=energy_grid,
                plot_curves={label: loc_data},
                x_label="Topological Indicator Value",
                energy_range=energy_range
            )

        return energy_grid, loc_data
    
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
            return_lap_rho=need_derivatives,
            spin_channel=spin_channel,
            energy_range=energy_range,
            use_partial_occ=use_partial_occ,
            include_aug=include_aug
        )
        
        rho = contributions[0]
        tau = contributions[1]
        grad_sq = contributions[2] if need_derivatives else None
        lap_rho = contributions[3] if need_derivatives else None
        
        is_spin = spin_channel != -1

        # Evaluate topological expressions over the resulting point arrays
        if loc_fn_lower == "lol":
            from baderkit.post_wfc.localization_functions import lol
            return lol(rho, tau, savin_correction, is_spin)
        elif loc_fn_lower == "elid":
            from baderkit.post_wfc.localization_functions import elid
            return elid(rho, tau, lap_rho, grad_sq)
        elif loc_fn_lower == "elf":
            from baderkit.post_wfc.localization_functions import elf
            return elf(rho, tau, lap_rho, grad_sq, savin_correction, is_spin)
        
    def get_density_of_states(
        self, 
        spin_channel=-1, 
        energy_range=None, 
        num_points=2000, 
        method="gaussian", 
        sigma=None, 
        use_occupancies=False,
        return_plot=False,
    ):
        """Constructs energy coordinate profiles outlining the Electronic Density of States (DOS)."""
        method, sigma = self._get_default_sigma(method, sigma)
        bands = self.energies
        
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
            
            if not np.any(band_mask):
                dos = np.zeros(num_points)
            else:
                eigenvalues_filtered = eigenvalues[:, :, band_mask]
                w_t_filtered = w_t[:, :, band_mask]
                dos = _integrate_tetrahedra_spectral_density_numba(
                    energy_grid, tetra_indices, eigenvalues_filtered, w_t_filtered, tetra_weight,
                )[0].sum(axis=0)
            
        else:
            kpt_weights = self.kpoint_weights
            bands_flat = bands[spin_all].ravel()
            
            if use_occupancies:
                w_t = (self.occupancies[spin_all] * kpt_weights[None, :, None]).ravel() * factor
            else:
                w_t = (np.ones_like(bands[spin_all]) * kpt_weights[None, :, None]).ravel() * factor
                
            mask = (bands_flat >= e_min - pad) & (bands_flat <= e_max + pad)
            bands_filtered = bands_flat[mask]
            w_t_filtered = w_t[mask]
            
            if method == "none":
                smear_matrix = np.zeros((num_points, len(bands_filtered)))
                if len(bands_filtered) > 0:
                    closest_idx = np.round((bands_filtered - e_min) / delta_e).astype(int)
                    valid_mask = (closest_idx >= 0) & (closest_idx < num_points)
                    smear_matrix[closest_idx[valid_mask], np.where(valid_mask)[0]] = 1.0 / delta_e
            else:
                delta_E = energy_grid[:, None] - bands_filtered[None, :]
                smear_matrix = self._get_smear_matrix(delta_E / sigma, method, sigma)
                
            dos = np.dot(smear_matrix, w_t_filtered)
            
        if method == "tetrahedron" and sigma > 0.0:
            n_kernel = int(np.ceil(4.0 * sigma / delta_e))

            if n_kernel > 0:
                x_kernel = np.arange(-n_kernel, n_kernel + 1) * delta_e
                kernel = np.exp(-0.5 * (x_kernel / sigma)**2)
                kernel /= np.sum(kernel)
                dos = np.convolve(dos, kernel, mode='same')
                    
        if return_plot:
            return self._generate_dos_plot(
                energy_grid=energy_grid,
                total_dos=dos,
                plot_curves={},
                energy_range=energy_range
            )
        return {
            "energy_grid": energy_grid,
            "total_dos": dos,
            }

    ###########################################################################
    # Public Helper Functions
    ###########################################################################
    def calculate_laplacian(self, data, is_reciprocal=False):
        """Evaluates second-derivative field Laplacian grid profiles via algebraic Fourier space multiplication."""
        Gx, Gy, Gz = self.plane_waves_cart()
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
            
        Gx, Gy, Gz = self.plane_waves_cart()
        with set_workers(self.scipy_workers):
            grad_x = ifftn(1j * Gx * recip_data, norm='ortho')
            grad_y = ifftn(1j * Gy * recip_data, norm='ortho')
            grad_z = ifftn(1j * Gz * recip_data, norm='ortho')
        return grad_x.real, grad_y.real, grad_z.real
    
    def get_electrons_in_energy_range(self, e_min=None, e_max=None, num_points=5000, method="gaussian", sigma=None):
        """Integrates occupied DOS profiles across designated energy limits."""
        method, sigma = self._get_default_sigma(method, sigma)
        full_e_min, full_e_max = self.get_energy_range(method, sigma)
        
        if e_min is None or e_min == -np.inf: e_min = full_e_min
        if e_max is None or e_max == np.inf: e_max = full_e_max
        if e_min == e_max:
            return 0.0
        egrid, dens = self.get_density_of_states(
            spin_channel=-1, energy_range=(e_min, e_max), num_points=num_points, 
            method=method, sigma=sigma, use_occupancies=True
        )
        return trapezoid(dens, egrid)
        
    def find_energy_for_electron_count(self, target_electrons, e_min=None, assume_full_occupancy=False, num_points=5000, method="gaussian", sigma=None):
        """Identifies relative energy cutoff limits enclosing specific targeted electron populations."""
        method, sigma = self._get_default_sigma(method, sigma)
        full_e_min, full_e_max = self.get_energy_range(method, sigma)
        
        if e_min is None or e_min == -np.inf:
            e_start = full_e_min
        else:
            e_start = e_min
            
        egrid, density = self.get_density_of_states(
            spin_channel=-1, energy_range=(e_start, full_e_max), num_points=num_points, 
            method=method, sigma=sigma, use_occupancies=not assume_full_occupancy
        )
        
        if assume_full_occupancy:
            density = density * np.max(self.occupancies)
            
        cum_charge = np.zeros(len(egrid))
        cum_charge[1:] = cumulative_trapezoid(density, egrid)
        
        if target_electrons > cum_charge[-1]: 
            raise ValueError("Target electron allocation total exceeds evaluated capacity parameters grid envelope limits.")
            
        return np.interp(target_electrons, cum_charge, egrid)
    
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
        weight_callback=None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Centralized state-space transformation driver. Iterates through system eigenstates,
        applies positive-definite gradient operators, and executes dense 3D IFFT conversions 
        to real-space grids.
        """
        from scipy.fft import ifftn, set_workers
        
        Nx, Ny, Nz = grid_shape
        normFac = np.sqrt((Nx * Ny * Nz) / self.structure.volume)
        
        rho = np.zeros(grid_shape, dtype=np.float64)
        tau = np.zeros(grid_shape, dtype=np.float64)
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
                
                # Trigger callback hook for class-specific sub-matrix operations (e.g., LCAO density matrices)
                if weight_callback is not None:
                    weight_callback(ispin, ikpt, active_bands, weights)
                
                # Execute subclass-specific coefficient resolution
                coeffs_gspace = self._construct_coefficients(ispin, ikpt, active_bands)
                
                # --- Resolve Wavefunction Cartesian Gradients in Fourier Space ---
                gvectors, gvec_wrapped = self.get_plane_waves_basis_idx(
                    ikpt, grid_shape, expected_npw=coeffs_gspace.shape[1]
                )
                rgvec = gvectors @ (2 * np.pi * self.reciprocal_lattice)
                k_cart = self.kpoints_cart[ikpt]
                K_cart = rgvec + k_cart[np.newaxis, :]
                
                # Box frequency distributions into 3D grid layout configurations
                phi_k = np.zeros((len(active_bands), Nx, Ny, Nz), dtype=np.complex128)
                grad_x_k = np.zeros((len(active_bands), Nx, Ny, Nz), dtype=np.complex128)
                grad_y_k = np.zeros((len(active_bands), Nx, Ny, Nz), dtype=np.complex128)
                grad_z_k = np.zeros((len(active_bands), Nx, Ny, Nz), dtype=np.complex128)
                
                # Assign baseline wavefunctions and analytic partial derivatives (i * K_cart)
                phi_k[:, gvec_wrapped[:, 0], gvec_wrapped[:, 1], gvec_wrapped[:, 2]] = coeffs_gspace
                grad_x_k[:, gvec_wrapped[:, 0], gvec_wrapped[:, 1], gvec_wrapped[:, 2]] = 1j * K_cart[:, 0] * coeffs_gspace
                grad_y_k[:, gvec_wrapped[:, 0], gvec_wrapped[:, 1], gvec_wrapped[:, 2]] = 1j * K_cart[:, 1] * coeffs_gspace
                grad_z_k[:, gvec_wrapped[:, 0], gvec_wrapped[:, 1], gvec_wrapped[:, 2]] = 1j * K_cart[:, 2] * coeffs_gspace
                
                # Execute high-throughput parallel multidimensional inverse Fourier mappings
                with set_workers(self.scipy_workers):
                    phi_r = ifftn(phi_k * np.sqrt(Nx * Ny * Nz), axes=(1, 2, 3)) * normFac
                    grad_x_r = ifftn(grad_x_k * np.sqrt(Nx * Ny * Nz), axes=(1, 2, 3)) * normFac
                    grad_y_r = ifftn(grad_y_k * np.sqrt(Nx * Ny * Nz), axes=(1, 2, 3)) * normFac
                    grad_z_r = ifftn(grad_z_k * np.sqrt(Nx * Ny * Nz), axes=(1, 2, 3)) * normFac
                    
                w_arr = np.array(weights)[:, np.newaxis, np.newaxis, np.newaxis]
                rho += np.sum((phi_r.conj() * phi_r).real * w_arr, axis=0)
                
                # Accumulate the positive-definite kinetic metric: sum(|grad_alpha|^2)
                tau_state_sq = (grad_x_r.conj() * grad_x_r + 
                                grad_y_r.conj() * grad_y_r + 
                                grad_z_r.conj() * grad_z_r).real
                tau += np.sum(tau_state_sq * w_arr, axis=0)
                
        return self._symmetrize_3d_grid(rho), self._symmetrize_3d_grid(tau)
    
    def _apply_onsite_augmentation_at_point(
        self, ispin, ikpt, active_bands, coeffs_list, rgvec, weight, frac_coord, rho_bands, tau_bands, include_aug
    ):
        """Base lifecycle hook for localized core reconstructions. No-op by default."""
        return rho_bands, tau_bands
    
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
            "none": "none", "gaussian": "gaussian", "methfessel-paxton": "methfessel-paxton",
            "mp": "methfessel-paxton", "fermi-dirac": "fermi-dirac", "fm": "fermi-dirac",
            "tetrahedron": "tetrahedron", "tet": "tetrahedron",
            }
        
        formal_method = shorthands.get(method if isinstance(method, str) else method, None)
        if formal_method is None:
            raise ValueError(f"Unknown smearing method: '{method}'")
        
        if sigma is None:
            default_sigma = {
                "none": 0.0, "gaussian": 0.1, "methfessel-paxton": 0.18,
                "fermi-dirac": 300, "tetrahedron": 0.04,
                }
            sigma = default_sigma[formal_method]
            
        if formal_method == "fermi-dirac":
            sigma = 8.617333262e-5 * sigma
        
        return formal_method, sigma
        
    def _get_smear_matrix(self, x, method, sigma):
        """Helper matrix generator parsing customized analytical broadening distributions."""
        if method == "none":
            raise ValueError("Smearing matrix for 'none' method must be constructed via discrete grid-binning.")
        elif method == "gaussian":
            return np.exp(-0.5 * x**2) / (sigma * np.sqrt(2 * np.pi))
        elif method in ["methfessel-paxton", "mp"]:
            term_0 = np.exp(-x**2) / np.sqrt(np.pi)
            return ((1.5 - x**2) * term_0) / sigma
        elif method in ["fermi-dirac", "fd"]:
            exp_term = np.exp(np.clip(x, -50, 50))
            return (exp_term / (exp_term + 1.0)**2) / sigma
        else:
            raise ValueError(f"Unknown smearing method: '{method}'")
        
    def _execute_spectral_engine(
            self,
            num_metrics,
            spin_channel,
            energy_range,
            num_points,
            method,
            sigma,
            eval_callback
            ):
        """Unified high-performance orchestration engine for plane-wave spectral decompositions."""
        method, sigma = self._get_default_sigma(method, sigma)
        kpoint_weights = self.kpoint_weights
        
        spin_indices = [i for i in range(self.nspin)] if spin_channel == -1 else [spin_channel]
        grid_shape = self._minimum_fft_size * 2
        nx, ny, nz = grid_shape
        norm_factor = 1.0 / np.sqrt(self.structure.volume)
        
        raw_data = np.zeros((num_metrics, self.nspin, self.nkpoints, self.nbands), dtype=float)
        
        if energy_range is None:
            e_min, e_max = self.get_energy_range(method, sigma)
        else:
            e_min, e_max = energy_range
            full_e_min, full_e_max = self.get_energy_range(method, sigma)
            if e_min is None or e_min == -np.inf: e_min = full_e_min
            if e_max is None or e_max == np.inf: e_max = full_e_max
            
        energy_grid = np.linspace(e_min, e_max, num_points)
        delta_e = energy_grid[1] - energy_grid[0] if num_points > 1 else 0.0

        if method == "none":
            pad = 0.5 * delta_e if num_points > 1 else 0.0
        elif method == "tetrahedron":
            pad = 4.0 * sigma if sigma > 0.0 else 0.0
        else:
            pad = 5.0 * sigma

        for ispin in spin_indices:
            for ikpt in range(self.nkpoints):
                energies_ik = self.energies[ispin, ikpt]
                active_bands = [iband for iband in range(self.nbands) 
                                if energies_ik[iband] >= e_min - pad and energies_ik[iband] <= e_max + pad]
                
                if not active_bands:
                    continue
                
                rspin = 2.0 if self.nspin == 1 else 1.0
                weight = rspin * kpoint_weights[ikpt] if method != "tetrahedron" else rspin
                
                coeffs_list = self._wf_reader.read_coefficients_batch(ispin, ikpt, active_bands)
                gvectors, _ = self.get_plane_waves_basis_idx(ikpt, grid_shape, expected_npw=coeffs_list.shape[1])
                
                kx_idx = gvectors[:, 0] % nx
                ky_idx = gvectors[:, 1] % ny
                kz_idx = gvectors[:, 2] % nz
                
                metrics_block = eval_callback(
                    ispin, ikpt, coeffs_list, gvectors, kx_idx, ky_idx, kz_idx, 
                    weight, grid_shape, norm_factor
                )
                
                for imetric, metric_bands in enumerate(metrics_block):
                    if metric_bands is not None:
                        raw_data[imetric, ispin, ikpt, active_bands] = metric_bands
                        
        use_convolution = (method == "tetrahedron" and sigma > 0.0)
        if use_convolution:
            n_kernel = int(np.ceil(4.0 * sigma / delta_e))
            if n_kernel > 0:
                x_kernel = np.arange(-n_kernel, n_kernel + 1) * delta_e
                kernel = np.exp(-0.5 * (x_kernel / sigma)**2)
                kernel /= np.sum(kernel)
            else:
                use_convolution = False

        if method == "tetrahedron":
            full_map = self.full_to_irr_map
            cached_metrics = np.ascontiguousarray(np.transpose(raw_data, (1, 2, 3, 0)))
            eigenvalues = self.energies[:, full_map, :]
            cached_metrics = cached_metrics[:, full_map, :, :]
            
            eigenvalues_spin = eigenvalues[spin_indices]
            cached_metrics_spin = cached_metrics[spin_indices]
            
            band_mask = np.any((eigenvalues_spin >= e_min - pad) & (eigenvalues_spin <= e_max + pad), axis=(0, 1))
            
            if not np.any(band_mask):
                return energy_grid, [np.zeros(num_points) for _ in range(num_metrics)]
                
            eigenvalues_filtered = eigenvalues_spin[:, :, band_mask]
            cached_metrics_filtered = cached_metrics_spin[:, :, band_mask, :]
        
            tetra_indices = self.tetrahedra_indices
            tetra_weight = 1.0 / len(tetra_indices)
        
            smeared_output = _integrate_tetrahedra_spectral_density_numba(
                energy_grid, tetra_indices, eigenvalues_filtered,
                cached_metrics_filtered, tetra_weight,
            )
        
            smeared_results = []
            for imetric in range(num_metrics):
                smeared = np.sum(smeared_output[imetric], axis=0)
                if use_convolution:
                    smeared = np.convolve(smeared, kernel, mode='same')
                smeared_results.append(smeared)
            return energy_grid, smeared_results
            
        bands_all = self.energies[spin_indices]
        bands_flat = bands_all.ravel()
        
        mask = (bands_flat >= e_min - pad) & (bands_flat <= e_max + pad)
        bands_filtered = bands_flat[mask]
        
        if method == "none":
            smear_matrix = np.zeros((num_points, len(bands_filtered)))
            if len(bands_filtered) > 0:
                closest_idx = np.round((bands_filtered - e_min) / delta_e).astype(int)
                valid_mask = (closest_idx >= 0) & (closest_idx < num_points)
                smear_matrix[closest_idx[valid_mask], np.where(valid_mask)[0]] = 1.0 / delta_e
        else:
            delta_E = energy_grid[:, None] - bands_filtered[None, :]
            smear_matrix = self._get_smear_matrix(delta_E / sigma, method, sigma)
            
        smeared_results = []
        for imetric in range(num_metrics):
            vals_all = raw_data[imetric, spin_indices].ravel()
            vals_filtered = vals_all[mask]
            smeared = np.dot(smear_matrix, vals_filtered)
            smeared_results.append(smeared)
            
        return energy_grid, smeared_results
    
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

    ###########################################################################
    # Plotting Helpers
    ###########################################################################
    @staticmethod
    def _generate_property_plot(energy_grid, plot_curves, x_label, energy_range=None):
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
                ax.plot(data, energy_grid, label=label, linewidth=2.5)
                
                # Dynamically determine visible viewport boundaries to prevent over-scaling the X-axis limit
                ymin, ymax = energy_grid[0], energy_grid[-1]
                if energy_range is not None:
                    if energy_range[0] is not None and energy_range[0] != -np.inf: 
                        ymin = energy_range[0]
                    if energy_range[1] is not None and energy_range[1] != np.inf: 
                        ymax = energy_range[1]
                
                # Isolate values falling purely within the visible window bounding mask
                mask = (energy_grid >= ymin) & (energy_grid <= ymax)
                if np.any(mask):
                    max_val = max(max_val, float(np.max(data[mask])))
            
        # Enforce explicit axis viewport boundaries matching the calculation limits
        ymin, ymax = energy_grid[0], energy_grid[-1]
        if energy_range is not None:
            if energy_range[0] is not None and energy_range[0] != -np.inf: 
                ymin = energy_range[0]
            if energy_range[1] is not None and energy_range[1] != np.inf: 
                ymax = energy_range[1]
                
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
    
    @staticmethod
    def _generate_dos_plot(energy_grid, total_dos, plot_curves, energy_range=None):
        """
        Shared high-performance plotting module that converts raw spectral data matrices 
        into a polished, publication-ready Matplotlib figure object.
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        
        # Plot baseline Total DOS
        ax.plot(total_dos, energy_grid, label="total", color="black", linewidth=2.5)
        
        # Plot individual contributing channels
        for label, data in plot_curves.items():
            ax.plot(data, energy_grid, label=label, linewidth=2.5)
            
        # Compute exact bounded viewport ranges 
        ymin, ymax = energy_grid[0], energy_grid[-1]
        if energy_range is not None:
            if energy_range[0] is not None and energy_range[0] != -np.inf: 
                ymin = energy_range[0]
            if energy_range[1] is not None and energy_range[1] != np.inf: 
                ymax = energy_range[1]
                
        mask = (energy_grid >= ymin) & (energy_grid <= ymax)
        dos_max = float(np.max(total_dos[mask])) if np.any(mask) else 1.0
        
        ax.set_xlim(0.0, 1.05 * dos_max)
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