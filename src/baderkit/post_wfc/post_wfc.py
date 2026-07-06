# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
from scipy.fft import fftn, ifftn, set_workers
from scipy.integrate import trapezoid, cumulative_trapezoid

from baderkit.post_wfc.wf_readers.base import HSQDTM
from .all_electron_references.reference_environment import AtomicReferenceEnvironment
from baderkit.post_wfc.pseudopotentials.augmentation_numba import (
    compute_reciprocal_projectors,
    enforce_matrix_symmetrization
    )
from .wfc_numba import _integrate_tetrahedra_spectral_density_numba


class PostWFC:
    """
    Standardized, code-agnostic post-processing driver executing 3D fast Fourier transforms 
    to map discrete reciprocal coefficients into dense localized real-space properties.
    """
    def __init__(
        self, 
        wf_reader, 
        aug_environment,
        valence_counts,
        scipy_workers: int = -1,
        **kwargs
        ):
        """Initializes properties and calculates necessary minimum FFT grid boundaries."""
        self._wf_reader = wf_reader
        self._meta = wf_reader.meta
        self._aug_environment = aug_environment
        
        self._structure = self._meta.structure
        self._lattice = self._structure.lattice.matrix               # Lattice row vectors in Angstroms
        self._reciprocal_lattice = np.linalg.inv(self._lattice).T   # Transpose of inverse cell matrix (B_mat without 2pi)
        self.scipy_workers = scipy_workers
        
        # Shift all eigenvalues relative to the Fermi Level at the start.
        # This forces the Fermi level to 0.0 eV for all downstream operations.
        self._meta.energies = self._meta.energies - self._meta.efermi
        self._meta.efermi = 0.0
        
        # valence counts
        if valence_counts is False:
            valence_counts = {}
        self._valence_counts = valence_counts
        
        # Determine minimum grid dims to enclose the full plane-wave cutoff sphere without aliasing errors.
        lattice_norm = np.linalg.norm(self._lattice, axis=1)
        CUTOFF = np.ceil(np.sqrt(self._meta.energy_cutoff / HSQDTM) / (2 * np.pi / lattice_norm))
        self._minimum_fft_size = np.array(2 * CUTOFF + 1, dtype=int)
        
        # High-performance analytical projector caching variables
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

        # Internal placeholders for properties lazily calculated on demand
        self._total_charge = None
        self._kpoint_multiplicities = None
        self._kpoint_weights = None
        self._grid_cache = {}  # Internal cache to avoid rebuilding meshgrids across calculation calls
        self._tetrahedra_indices = None
        
    @property
    def reference_environment(self):
        if getattr(self, "_reference_environment", None) is None:
            pdos = self.get_atom_projected_density_of_states()
            self._reference_environment = AtomicReferenceEnvironment(
                structure=self._structure,
                pdos_data=pdos,
                aug_environment=self._aug_environment,
                )
        return self._reference_environment
        
    def clear_projector_cache(self):
        """Flushes memory tied to static reciprocal projector matrices."""
        self._projector_cache.clear()

    @property
    def valence_counts(self) -> dict | None:
        """
        Returns
        -------
        dict | None
            A dictionary where each key is an atomic species in the system and each
            value is the number of valence electrons used in the pseudo potential.
        """
        return self._aug_environment.valence_counts

    @property
    def nspin(self):
        """Returns total active spin allocation dimension count (1 for restricted, 2 for collinear/LSDA)."""
        return self._meta.nspin
        
    @property
    def nkpoints(self):
        """Returns the total number of sampling coordinates located in the irreducible Brillouin zone wedge."""
        return self._meta.nkpts
        
    @property
    def nbands(self):
        """Returns total calculated band orbital channels loaded from the electronic output."""
        return self._meta.nbands
        
    @property
    def occupancies(self):
        """Returns the raw multidimensional occupancy numbers array of shape (nspin, nkpts, nbands)."""
        return self._meta.occupancies
        
    @property
    def energies(self):
        """Returns absolute state eigenvalues array in eV of shape (nspin, nkpts, nbands)."""
        return self._meta.energies
        
    @property
    def energy_cutoff(self):
        """Returns plane-wave basis kinetic energy cutoff threshold constraint in eV."""
        return self._meta.energy_cutoff
        
    @property
    def structure(self):
        """Returns crystal structure information metadata mapping cell geometry and sites."""
        return self._structure
        
    @property
    def lattice(self):
        """Returns real-space cell matrix row vectors matching crystal bounds."""
        return self._lattice
        
    @property
    def reciprocal_lattice(self):
        """Returns reciprocal-space cell matrix column vectors (A^-T layout)."""
        return self._reciprocal_lattice
        
    @property
    def efermi(self):
        """Returns baseline calculation Fermi level energy reference point in eV."""
        return self._meta.efermi
    
    @property
    def kpoints(self):
        """Returns fractional coordinates for the irreducible kpoint mesh."""
        return self._meta.kpoints
        
    @property
    def kpoints_cart(self):
        """Converts fractional sampling coordinates into Cartesian inverse Angstrom vectors (K = k * 2pi * B)."""
        return np.dot(self.kpoints, 2 * np.pi * self.reciprocal_lattice)
        
    @property
    def kpoint_multiplicities(self):
        """Maps star orbits across reciprocal point symmetries and time-reversal folding to find full-zone counts."""
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
        """Generates the full symmetric k-mesh grid and indexes its map back to the irreducible star coordinates."""
        recp_symm_ops = self.structure.lattice.get_recp_symmetry_operation()
        recip_rotations = []
        
        for op in recp_symm_ops:
            R_recip = np.round(op.rotation_matrix).astype(int)
            if not any(np.array_equal(R_recip, R) for R in recip_rotations):
                recip_rotations.append(R_recip)
        for R in [-R for R in recip_rotations]:
            if not any(np.array_equal(R, ex_R) for ex_R in recip_rotations):
                recip_rotations.append(R)

        kpts_full = []
        full_to_irr = []
        
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
        """Returns full Brillouin zone weights normalized to sum to 1.0."""
        if self._kpoint_weights is None:
            self._kpoint_weights = self.kpoint_multiplicities / np.sum(self.kpoint_multiplicities)
        return self._kpoint_weights
        
    @property
    def kpoints_full(self):
        """Returns fractional coordinates for the completely unfolded full Brillouin zone k-mesh."""
        if getattr(self, "_kpoints_full", None) is None:
            self._unfold_brillouin_zone()
        return self._kpoints_full

    @property
    def kpoints_cart_full(self):
        """Converts unfolded fractional sampling coordinates into Cartesian inverse Angstrom vectors."""
        return np.dot(self.kpoints_full, 2 * np.pi * self.reciprocal_lattice)
        
    @property
    def tetrahedra_indices(self):
        if self._tetrahedra_indices is None:
            self._tetrahedra_indices = self._get_tetrahedra()
        return self._tetrahedra_indices

    @property
    def full_to_irr_map(self):
        """An integer array mapping each full-zone k-point index back to its original irreducible k-point index."""
        if getattr(self, "_full_to_irr_map", None) is None:
            self._unfold_brillouin_zone()
        return self._full_to_irr_map

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
    
    def shift_energies(self, shift: float):
        """Shifts the eigenvalue energies and fermi energy by a constant value"""
        self._meta.energies += shift
        
    @property
    def total_charge(self):
        """Evaluates total integrated system valence electron content by summing occupied DOS profiles."""
        if self._total_charge is None:
            self._total_charge = self.get_electrons_in_energy_range(-np.inf, np.inf)
        return self._total_charge
        
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
        
    def calculate_charge_density(
        self, 
        grid_shape=None, 
        spin_channel=-1, 
        include_aug=True,
        energy_range=(-np.inf, np.inf), 
        use_partial_occ=True,
        return_density_matrices=False,
    ):
        """Constructs full real-space electronic charge density grid profiles (rho) with PAW augmentation."""
        if grid_shape is None: 
            grid_shape = self._minimum_fft_size * 2
        Nx, Ny, Nz = grid_shape
        
        normFac = np.sqrt((Nx * Ny * Nz) / self.structure.volume)
        kpoint_weights = self.kpoint_weights
        rho = np.zeros(grid_shape, dtype=float)
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
                    weight = rspin * kpoint_weights[ikpt] * (self.occupancies[ispin, ikpt, iband] if use_partial_occ else 1.0)
                    if weight > 0:
                        active_bands.append(iband)
                        weights.append(weight)
                if not active_bands: 
                    continue
                
                coeffs_list = self._wf_reader.read_coefficients_batch(ispin, ikpt, active_bands)
                gvectors, gvec_wrapped = self.get_plane_waves_basis_idx(ikpt, grid_shape, expected_npw=coeffs_list.shape[1])
                
                phi_k = np.zeros((len(active_bands), Nx, Ny, Nz), dtype=np.complex128)
                phi_k[:, gvec_wrapped[:, 0], gvec_wrapped[:, 1], gvec_wrapped[:, 2]] = coeffs_list
                
                with set_workers(self.scipy_workers):
                    phi_r = ifftn(phi_k * np.sqrt(Nx * Ny * Nz), axes=(1, 2, 3)) * normFac
                    
                rho += np.sum((phi_r.conj() * phi_r).real * np.array(weights)[:, np.newaxis, np.newaxis, np.newaxis], axis=0)
                
        if include_aug:
            density_matrices = self.calculate_onsite_occupancy(
                energy_range=energy_range, 
                spin_channel=spin_channel, 
                use_partial_occ=use_partial_occ
            )
            
            aug_rho, corr_rho = self._aug_environment.calculate_onsite_densities(
                grid_dims=grid_shape, 
                density_matrices=density_matrices,
                energy_range=energy_range
            )
            
            rho += aug_rho - corr_rho
        
        if return_density_matrices:
            return self._symmetrize_3d_grid(rho), density_matrices
        
        return self._symmetrize_3d_grid(rho)
    
    def calculate_nonbonding_charge_density(
        self, 
        grid_dims, 
        energy_range=None, 
        num_points=2000, 
        method="gaussian", 
        sigma=None,
    ) -> np.ndarray:
        """Converts electronic bounds into non-bonding environment metrics."""
        if energy_range is None:
            min_charge = 0.0
            max_charge = self.total_charge
        else:
            e_min, e_max = energy_range
            
            min_charge = self.get_electrons_in_energy_range(
                e_min=None, 
                e_max=e_min, 
                num_points=num_points, 
                method=method, 
                sigma=sigma
            )
            
            max_charge = self.get_electrons_in_energy_range(
                e_min=None, 
                e_max=e_max, 
                num_points=num_points, 
                method=method, 
                sigma=sigma
            )
            
        return self.reference_environment.generate_charge_density_grid(
            grid_dims=grid_dims,
            min_charge=min_charge,
            max_charge=max_charge,
        )
    
    def calculate_kinetic_energy_density(
            self, 
            grid_shape=None, 
            spin_channel=-1, 
            energy_range=(-np.inf, np.inf), 
            use_partial_occ=True, 
            include_aug=True,
            return_charge_density=False,
            return_density_matrices=False,
            ):
        """Computes full real-space non-negative electronic kinetic energy density profiles (tau) with PAW augmentation."""
        if grid_shape is None: 
            grid_shape = self._minimum_fft_size * 2
        grid_shape_tuple = tuple(grid_shape)
        Nx, Ny, Nz = grid_shape_tuple
        normFac = np.sqrt((Nx * Ny * Nz) / self.structure.volume)
        kpoint_weights = self.kpoint_weights
        
        tau = np.zeros(grid_shape_tuple, dtype=float)
        rho = np.zeros(grid_shape_tuple, dtype=float) if return_charge_density else None
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
                    weight = rspin * kpoint_weights[ikpt] * (self.occupancies[ispin, ikpt, iband] if use_partial_occ else 1.0)
                    if weight > 0:
                        active_bands.append(iband)
                        weights.append(weight)
                if not active_bands: 
                    continue
                
                coeffs_list = self._wf_reader.read_coefficients_batch(ispin, ikpt, active_bands)
                gvectors, gvec_wrapped = self.get_plane_waves_basis_idx(ikpt, grid_shape_tuple, expected_npw=coeffs_list.shape[1])
                rgvec = gvectors @ (2 * np.pi * self.reciprocal_lattice)
                
                k = self.kpoints_cart[ikpt]             
                gk2 = np.sum((rgvec + k[np.newaxis, :])**2, axis=1)             
                
                lap_coeffs = -gk2[np.newaxis, :] * coeffs_list
                
                phi_k = np.zeros((len(active_bands), Nx, Ny, Nz), dtype=np.complex128)
                phi_k[:, gvec_wrapped[:, 0], gvec_wrapped[:, 1], gvec_wrapped[:, 2]] = coeffs_list
                with set_workers(self.scipy_workers):
                    phi_r = ifftn(phi_k * np.sqrt(Nx * Ny * Nz), axes=(1, 2, 3)) * normFac
                
                lap_phi_k = np.zeros((len(active_bands), Nx, Ny, Nz), dtype=np.complex128)
                lap_phi_k[:, gvec_wrapped[:, 0], gvec_wrapped[:, 1], gvec_wrapped[:, 2]] = lap_coeffs
                with set_workers(self.scipy_workers):
                    lap_phi_r = ifftn(lap_phi_k * np.sqrt(Nx * Ny * Nz), axes=(1, 2, 3)) * normFac
                    
                w_arr = np.array(weights)[:, np.newaxis, np.newaxis, np.newaxis]
                tau += np.sum((-phi_r * lap_phi_r.conj()).real * w_arr, axis=0)
                if return_charge_density:
                    rho += np.sum((phi_r.conj() * phi_r).real * w_arr, axis=0)

        if include_aug:
            density_matrices = self.calculate_onsite_occupancy(
                energy_range=energy_range, 
                spin_channel=spin_channel, 
                use_partial_occ=use_partial_occ
            )
            aug_tau, corr_tau = self._aug_environment.calculate_ke_on_grid(
                grid_dims=grid_shape_tuple, 
                density_matrices=density_matrices,
                energy_range=energy_range
            )
            tau += aug_tau - corr_tau
        tau = self._symmetrize_3d_grid(tau)
        
        results = [tau]
        if return_charge_density: 
            aug_rho, corr_rho = self._aug_environment.calculate_on_grid(
                grid_dims=grid_shape_tuple, 
                density_matrices=density_matrices,
                energy_range=energy_range
            )
            rho += aug_rho - corr_rho
            results.append(self._symmetrize_3d_grid(rho))
            
        if return_density_matrices:
            results.append(density_matrices)
            
        return tuple(results) if len(results) > 1 else results[0]

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
        return_plot=False,
    ):
        """Calculates exact state-resolved kinetic and charge density metrics at a single point with PAW updates."""
        # Fallback to standard double unaliased grid dimensions if custom dimensions are omitted[cite: 2]
        grid_shape = grid_shape if grid_shape is not None else self._minimum_fft_size * 2
        nx, ny, nz = grid_shape
        
        # Map real continuous fractional coordinates into discrete periodic meshgrid indices
        ix = int(np.round(frac_coord[0] * nx)) % nx
        iy = int(np.round(frac_coord[1] * ny)) % ny
        iz = int(np.round(frac_coord[2] * nz)) % nz
    
        def point_callback(ispin, ikpt, coeffs_list, gvectors, kx_idx, ky_idx, kz_idx, weight, gshape, norm_factor):
            # Calculate structural coordinate phase shifting factors for the explicit point
            phases = np.exp(2j * np.pi * (kx_idx * ix / nx + ky_idx * iy / ny + kz_idx * iz / nz))
            
            # Construct Cartesian reciprocal momentum space vectors for the active block[cite: 2]
            rgvec = gvectors @ (2 * np.pi * self.reciprocal_lattice)
            k = self.kpoints_cart[ikpt]             
            K_cart = rgvec + k[np.newaxis, :]             
            gk2 = np.sum(K_cart**2, axis=1)             
            
            # Evaluate baseline pseudo-wavefunction and kinetic Laplacian contributions at the target coordinate
            phi_at_point = np.dot(coeffs_list, phases) * norm_factor
            lap_phi_at_point = np.dot(coeffs_list, -gk2 * phases) * norm_factor
            
            # Extract standard plane-wave tracking fields scaled by state k-weights and degeneracies
            rho_bands = (phi_at_point.conj() * phi_at_point).real * weight
            tau_bands = (-phi_at_point * lap_phi_at_point.conj()).real * weight
            
            # Isolate the index configurations of states matching the requested energy boundaries[cite: 2]
            active_bands = []
            for iband in range(self.nbands):
                rel_energy = self.energies[ispin, ikpt, iband]
                if energy_range is not None:
                    if not (energy_range[0] <= rel_energy <= energy_range[1]):
                        continue
                active_bands.append(iband)
            
            # Execute localized PAW sphere reconstructions if the matrix row dimensions line up correctly[cite: 2]
            if len(active_bands) == len(coeffs_list):
                point_cart = frac_coord @ self._aug_environment.lattice_matrix
                num_atoms = len(self.structure)
                
                # Gather raw projectivity overlaps for all structural sites[cite: 2]
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
                
                # Inject all-electron minus pseudo localized corrections inside the spheres[cite: 2]
                if include_aug:
                    aug_bands = np.zeros(len(active_bands), dtype=np.float64)
                    aug_ke_bands = np.zeros(len(active_bands), dtype=np.float64)
                    
                    # Compute individual 1D state corrections by establishing local density matrices
                    for n_idx in range(len(active_bands)):
                        band_density_matrices = []
                        for i_atom in range(num_atoms):
                            proj = projections_all_atoms[i_atom][:, n_idx]
                            dm = np.outer(proj, proj.conj()).real
                            band_density_matrices.append(dm)
                            
                        # Evaluate localized all-electron vs pseudo onsite charge adjustments (AE - PS)[cite: 2]
                        ae_rho, ps_rho = self._aug_environment.calculate_onsite_densities_at_point(
                            point_cart=point_cart, density_matrices=band_density_matrices
                        )
                        aug_bands[n_idx] = ae_rho - ps_rho
                        
                        # Evaluate localized all-electron vs pseudo onsite kinetic adjustments (AE - PS)[cite: 2]
                        ae_tau, ps_tau = self._aug_environment.calculate_onsite_ke_densities_at_point(
                            point_cart=point_cart, density_matrices=band_density_matrices
                        )
                        aug_ke_bands[n_idx] = ae_tau - ps_tau
                    
                    # Accumulate localized PAW terms onto the continuous background channels
                    rho_bands += aug_bands * weight
                    tau_bands += aug_ke_bands * weight
            
            metrics = [rho_bands, tau_bands]
            
            # Resolve complex spatial derivatives if gradient or laplacian flags are specified[cite: 2]
            if return_grad_rho_sq or return_lap_rho:
                grad_phi_at_point = np.zeros((len(coeffs_list), 3), dtype=complex)
                for idim in range(3):
                    grad_phi_at_point[:, idim] = np.dot(coeffs_list, 1j * K_cart[:, idim] * phases) * norm_factor
                
                if return_grad_rho_sq:
                    grad_rho_vec = 2.0 * (phi_at_point[:, np.newaxis].conj() * grad_phi_at_point).real
                    grad_rho_sq_bands = np.sum(grad_rho_vec**2, axis=1) * weight
                    metrics.append(grad_rho_sq_bands)
                else:
                    metrics.append(None)
                    
                if return_lap_rho:
                    grad_psi_sq = np.sum(np.abs(grad_phi_at_point)**2, axis=1)
                    lap_rho_bands = 2.0 * (grad_psi_sq + (phi_at_point.conj() * lap_phi_at_point.conj()).real) * weight
                    metrics.append(lap_rho_bands)
                else:
                    metrics.append(None)
                    
            return metrics
    
        # Allocate required collector arrays inside the spectral decomposition framework[cite: 2]
        num_metrics = 4 if (return_grad_rho_sq or return_lap_rho) else 2

        energy_grid, smeared = self._execute_spectral_engine(
            num_metrics=num_metrics, spin_channel=spin_channel, energy_range=energy_range, 
            num_points=num_points, method=method, sigma=sigma, eval_callback=point_callback
        )
        
        # Route directly to the graphing handler if return_plot flag is active
        if return_plot:
            plot_curves = {
                r"Charge Density $\rho$": smeared[0],
                r"Kinetic Density $\tau$": smeared[1]
            }
            if return_grad_rho_sq and smeared[2] is not None:
                plot_curves[r"Gradient $|\nabla\rho|^2$"] = smeared[2]
            if return_lap_rho and smeared[3] is not None:
                plot_curves[r"Laplacian $\nabla^2\rho$"] = smeared[3]
                
            return self._generate_property_plot(
                energy_grid=energy_grid,
                plot_curves=plot_curves,
                x_label="Differential Density Magnitude (per eV)",
                energy_range=energy_range
            )
            
        results = [energy_grid, smeared[0], smeared[1]]
        if return_grad_rho_sq: results.append(smeared[2])
        if return_lap_rho: results.append(smeared[3])
        return tuple(results)
    
    def get_integrated_rho_tau_vs_energy(self, frac_coord, return_plot=False, **kwargs) -> np.ndarray:
        """Calculates total cumulatively integrated metrics at a fraction coordinate point."""
        # Explicitly turn off derivative components to eliminate unnecessary matrix overhead calculations[cite: 2]
        kwargs["return_grad_rho_sq"] = False
        kwargs["return_lap_rho"] = False
        
        # Force return_plot=False internally to extract raw mathematical array trajectories[cite: 2]
        contributions = self.get_rho_tau_vs_energy(frac_coord, return_plot=False, **kwargs)
        energy_grid = contributions[0]
        smeared_rho = contributions[1]
        smeared_tau = contributions[2]
        
        # Pre-allocate integration accumulators matching the energy grid dimensions
        cum_charge = np.zeros(len(energy_grid), dtype=np.float64)
        cum_tau = np.zeros(len(energy_grid), dtype=np.float64)
        
        # Execute cumulative trapezoidal numerical integrations across the energy steps[cite: 2]
        if len(energy_grid) > 1:
            cum_charge[1:] = cumulative_trapezoid(smeared_rho, energy_grid)
            cum_tau[1:] = cumulative_trapezoid(smeared_tau, energy_grid)
            
        # If visual chart is requested, pass the integrated fields to the shared static engine
        if return_plot:
            plot_curves = {
                "Integrated Charge Density": cum_charge,
                "Integrated Kinetic Density": cum_tau
            }
            return self._generate_property_plot(
                energy_grid=energy_grid,
                plot_curves=plot_curves,
                x_label="Accumulated Integrated Value",
                energy_range=kwargs.get("energy_range", None)
            )
            
        return energy_grid, cum_charge, cum_tau
    
    def get_total_charge_vs_energy(
        self,
        spin_channel: int = -1,
        energy_range: list = None,
        num_points: int = 2000,
        method: str = "gaussian",
        sigma: float = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the total cell charge integrated as a function of energy 
        by evaluating the total cell Density of States (DOS).
        """
        # A callback that simply returns the state weights to compute total DOS
        def total_dos_callback(ispin, ikpt, coeffs_list, gvectors, kx_idx, ky_idx, kz_idx, weight, gshape, norm_factor):
            n_bands = coeffs_list.shape[0]
            return [np.full(n_bands, weight)]

        # Execute the engine to get the smeared total DOS
        energy_grid, smeared = self._execute_spectral_engine(
            num_metrics=1,
            spin_channel=spin_channel,
            energy_range=energy_range,
            num_points=num_points,
            method=method,
            sigma=sigma,
            eval_callback=total_dos_callback
        )
        
        total_dos = smeared[0]
        
        # Numerically integrate the total DOS to get cumulative cell charge Q(E)
        dx = np.diff(energy_grid)
        avg_dos = 0.5 * (total_dos[:-1] + total_dos[1:])
        total_charge = np.zeros_like(energy_grid)
        total_charge[1:] = np.cumsum(avg_dos * dx)
        
        return energy_grid, total_charge
    
    def get_nonbonding_rho_vs_energy(
        self, 
        frac_coord, 
        num_points = 2000,
        return_plot=False, 
        **kwargs,
    ) -> np.ndarray:
        """Computes the differential non-bonding reference charge density vs energy curve."""
        # 1. Retrieve the true global continuous charging profile of the cell
        energy_grid, total_charge = self.get_total_charge_vs_energy(
            num_points=num_points,
            **kwargs
        )
        
        # 2. Construct the array mapping column 0 to energies and column 1 to total cell charge Q(E)
        energy_charge_grid = np.column_stack((energy_grid, total_charge))
        
        # 3. Forward the mapped array into the target PDOS-driven reference environment
        reference_data = self.reference_environment.calculate_density_at_point_vs_energy(
            frac_coord=frac_coord,
            energy_charge_array=energy_charge_grid,
            num_interp_points=num_points
        )
        
        # Dispatch to plotting subsystem if the return_plot flag is active
        if return_plot:
            plot_curves = {
                "Non-Bonding Reference Density": reference_data[:, 1]
            }
            return self._generate_property_plot(
                energy_grid=energy_grid,
                plot_curves=plot_curves,
                x_label="Differential Density Magnitude (per eV)",
                energy_range=kwargs.get("energy_range", None)
            )
            
        return reference_data
    
    def calculate_localization_function(
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
            
        tau, rho = self.calculate_kinetic_energy_density(
            grid_shape=grid_shape, 
            include_aug=include_aug,
            energy_range=energy_range, 
            spin_channel=spin_channel, 
            use_partial_occ=use_partial_occ, 
            return_charge_density=True
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

        return energy_grid, dos
    
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

    def get_projected_density_of_states(
        self, 
        spin_channel=-1, 
        energy_range=None, 
        num_points=2000, 
        method="gaussian", 
        sigma=None, 
        orbital_types=None,
        return_plot=False,
    ):
        """
        Constructs the total electronic DOS along with individual projections 
        smeared across s, p, d, and f orbital character manifolds without external batching.
        """
        if orbital_types is None:
            orbital_types = ["s", "p", "d", "f"]
            
        orbital_map = {"s": 0, "p": 1, "d": 2, "f": 3}
        target_ls = [orbital_map[orb] for orb in orbital_types if orb in orbital_map]
        num_atoms = len(self.structure)

        def pdos_callback(ispin, ikpt, coeffs_list, gvectors, kx_idx, ky_idx, kz_idx, weight, gshape, norm_factor):
            num_bands = coeffs_list.shape[0]
            metrics = [np.full(num_bands, weight)]
            
            orbital_accumulators = {l: np.zeros(num_bands) for l in target_ls}
            total_projection_per_band = np.zeros(num_bands)
            
            rgvec = gvectors @ (2 * np.pi * self.reciprocal_lattice)
            k = self.kpoints_cart[ikpt]
            
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
                proj_sq = np.abs(proj_atom)**2
                total_projection_per_band += np.sum(proj_sq, axis=0)
                
                for p_idx, l in enumerate(dataset.angular_momenta):
                    if l in orbital_accumulators:
                        orbital_accumulators[l] += proj_sq[p_idx, :]
            
            zero_mask = total_projection_per_band < 1e-12
            safe_denominator = np.where(zero_mask, 1.0, total_projection_per_band)
            
            for l in target_ls:
                normalized_character = np.where(zero_mask, 1.0 / 4.0, orbital_accumulators[l] / safe_denominator)
                metrics.append(normalized_character * weight)
                
            return metrics

        num_metrics = 1 + len(target_ls)
        energy_grid, smeared = self._execute_spectral_engine(
            num_metrics=num_metrics, spin_channel=spin_channel, energy_range=energy_range, 
            num_points=num_points, method=method, sigma=sigma, eval_callback=pdos_callback
        )
        
        total_dos = smeared[0]
        projections_dict = {orb: smeared[1 + idx] for idx, orb in enumerate(orbital_types)}
            
        if return_plot:
            return self._generate_dos_plot(
                energy_grid=energy_grid, 
                total_dos=total_dos, 
                plot_curves=projections_dict, 
                energy_range=energy_range
            )
            
        projections_dict["energy_grid"] = energy_grid
        projections_dict["total_dos"] = total_dos
        return projections_dict

    def get_atom_projected_density_of_states(
        self, 
        atom_indices=None,
        spin_channel=-1, 
        energy_range=None, 
        num_points=2000, 
        method="gaussian", 
        sigma=None, 
        return_plot=False,
    ):
        """
        Constructs the total electronic DOS along with individual projections 
        onto specified individual atoms without external batching.
        """
        if atom_indices is None:
            atom_indices = list(range(len(self.structure)))
            
        def atom_dos_callback(ispin, ikpt, coeffs_list, gvectors, kx_idx, ky_idx, kz_idx, weight, gshape, norm_factor):
            num_bands = coeffs_list.shape[0]
            metrics = [np.full(num_bands, weight)]
            
            atom_accumulators = {i_atom: np.zeros(num_bands) for i_atom in atom_indices}
            total_projection_per_band = np.zeros(num_bands)
            
            rgvec = gvectors @ (2 * np.pi * self.reciprocal_lattice)
            k = self.kpoints_cart[ikpt]
            
            for i_all_atom in range(len(self.structure)):
                elem = self.structure[i_all_atom].species_string
                dataset = self._aug_environment.paw_datasets[elem]
                h = dataset.q_linear_grid[-1]
                
                P_G_matrix = compute_reciprocal_projectors(
                    k, rgvec, self.structure[i_all_atom].coords, h,
                    len(dataset.q_linear_grid), self.structure.volume, 
                    dataset.reciprocal_projectors, dataset.angular_momenta, dataset.magnetic_nums
                )
                
                proj_atom = np.dot(P_G_matrix, coeffs_list.T)
                proj_sq = np.abs(proj_atom)**2
                atom_sum = np.sum(proj_sq, axis=0)
                total_projection_per_band += atom_sum
                
                if i_all_atom in atom_accumulators:
                    atom_accumulators[i_all_atom] = atom_sum
            
            zero_mask = total_projection_per_band < 1e-12
            safe_denominator = np.where(zero_mask, 1.0, total_projection_per_band)
            
            for i_atom in atom_indices:
                normalized_character = np.where(zero_mask, 1.0 / len(self.structure), atom_accumulators[i_atom] / safe_denominator)
                metrics.append(normalized_character * weight)
                
            return metrics

        num_metrics = 1 + len(atom_indices)
        energy_grid, smeared = self._execute_spectral_engine(
            num_metrics=num_metrics, spin_channel=spin_channel, energy_range=energy_range, 
            num_points=num_points, method=method, sigma=sigma, eval_callback=atom_dos_callback
        )
        
        total_dos = smeared[0]
        projections_dict = {i_atom: smeared[1 + idx] for idx, i_atom in enumerate(atom_indices)}
            
        if return_plot:
            # Map structural data directly into readable label string annotations for the plot legend
            plot_curves = {
                f"Atom {i} ({self.structure[i].species_string})": projections_dict[i] 
                for i in atom_indices
            }
            return self._generate_dos_plot(
                energy_grid=energy_grid, 
                total_dos=total_dos, 
                plot_curves=plot_curves, 
                energy_range=energy_range
            )
            
        projections_dict["energy_grid"] = energy_grid
        projections_dict["total_dos"] = total_dos
        return projections_dict
        
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
        
    @classmethod
    def from_directory(
            cls, 
            directory: Path | str = Path("."), 
            fmt: str = "vasp", 
            scipy_workers: int = -1, 
            **kwargs,
            ):
        """Dynamic wf_reader factory routing file stream construction to selected code formats."""
        if fmt == "vasp": 
            from baderkit.post_wfc.wf_readers import VaspReader as wf_reader
        elif fmt == "qe": 
            from baderkit.post_wfc.wf_readers import QeReader as wf_reader
        else: 
            raise ValueError(f"Unknown format profile template string keyword: {fmt}")
            
        from baderkit.post_wfc.pseudopotentials.augmentation_environment import PAWAugmentationEnvironment
        aug_env = PAWAugmentationEnvironment.from_directory(
            directory=directory,
            fmt=fmt,
            )
        
        return cls(
            wf_reader(directory=Path(directory), **kwargs), 
            aug_environment=aug_env,
            valence_counts=kwargs.get("valence_counts", None),
            scipy_workers=scipy_workers, 
            **kwargs)