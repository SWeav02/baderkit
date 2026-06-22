# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
from scipy.fft import fftn, ifftn, set_workers
from scipy.integrate import trapezoid, cumulative_trapezoid

from baderkit.post_wfc.wf_parsers.base import HSQDTM

# TODO:
# Create method for plotting how much a state contributes or detracts from
# the localization at a point
# Further break that down into contributions from charge density vs. kinetic energy density

class PostWFC:
    """
    Standardized, code-agnostic post-processing driver executing 3D fast Fourier transforms 
    to map discrete reciprocal coefficients into dense localized real-space properties.
    """
    def __init__(
        self, 
        parser, 
        scipy_workers: int = -1
        ):
        """Initializes properties and calculates necessary minimum FFT grid boundaries."""
        self._parser = parser
        self._meta = parser.meta
        self._structure = self._meta.structure
        self._lattice = self._structure.lattice.matrix               # Lattice row vectors in Angstroms
        self._reciprocal_lattice = np.linalg.inv(self._lattice).T   # Transpose of inverse cell matrix (B_mat without 2pi)
        self.scipy_workers = scipy_workers
        
        # Determine minimum grid dims to enclose the full plane-wave cutoff sphere without aliasing errors.
        # Max integer Miller component index along direction i requires: |n_i| <= R * ||a_i||
        # Using scipy/numpy norm calls for clean vectorization
        lattice_norm = np.linalg.norm(self._lattice, axis=1)
        CUTOFF = np.ceil(np.sqrt(self._meta.energy_cutoff / HSQDTM) / (2 * np.pi / lattice_norm))
        self._minimum_fft_size = np.array(2 * CUTOFF + 1, dtype=int)
        
        # Internal placeholders for properties lazily calculated on demand
        self._energy_range = None
        self._total_electrons = None
        self._kpoint_multiplicities = None
        self._kpoint_weights = None
        self._grid_cache = {}  # Internal cache to avoid rebuilding meshgrids across calculation calls
        
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
            
            # 1. Use the operations directly since they are already in the reciprocal basis
            for op in recp_symm_ops:
                R_recip = np.round(op.rotation_matrix).astype(int)
                if not any(np.array_equal(R_recip, R) for R in recip_rotations):
                    recip_rotations.append(R_recip)
                    
            # Fold in time-reversal inversion symmetry (-R operations)
            for R in [-R for R in recip_rotations]:
                if not any(np.array_equal(R, ex_R) for ex_R in recip_rotations):
                    recip_rotations.append(R)
                    
            # 2. Trace equivalent points using the minimum image convention
            multiplicities = []
            for k in self.kpoints:
                star_kpts = []
                for R in recip_rotations:
                    k_wrapped = np.mod(R @ k, 1.0)
                    
                    # Robust distance checker handling 0.0 vs 1.0 boundary wrapping noise safely
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
    def full_to_irr_map(self):
        """An integer array mapping each full-zone k-point index back to its original irreducible k-point index."""
        if getattr(self, "_full_to_irr_map", None) is None:
            self._unfold_brillouin_zone()
        return self._full_to_irr_map
        
    @property
    def energy_range(self):
        """Returns relative boundary offsets scaled directly against the Fermi level (E - E_f)."""
        if self._energy_range is None:
            self._energy_range = np.min(self.energies) - self.efermi, np.max(self.energies) - self.efermi
        return self._energy_range
        
    @property
    def total_electrons(self):
        """Evaluates total integrated system valence electron content by summing occupied DOS profiles."""
        if self._total_electrons is None:
            self._total_electrons = self.get_electrons_in_energy_range(-np.inf, np.inf)
        return self._total_electrons
        
    def get_plane_waves_frac(self, grid_shape=None):
        """Generates fractional coordinate arrays for plane waves in standard FFT wrapped frequency order."""
        if grid_shape is None: 
            grid_shape = self._minimum_fft_size * 2
        grid_shape = tuple(grid_shape)
        if grid_shape in self._grid_cache: 
            return self._grid_cache[grid_shape]
            
        Nx, Ny, Nz = grid_shape
        # Wrap frequency array mapping indices matching standard 3D FFT layouts: [0, 1, 2, ..., -3, -2, -1]
        fx = [ii if ii < Nx // 2 + 1 else ii - Nx for ii in range(Nx)]
        fy = [jj if jj < Ny // 2 + 1 else jj - Ny for jj in range(Ny)]
        fz = [kk if kk < Nz // 2 + 1 else kk - Nz for kk in range(Nz)]
        
        # indexing='ij' ensures output configurations match dimensions shape (Nx, Ny, Nz) directly
        gx, gy, gz = np.meshgrid(fx, fy, fz, indexing='ij')
        self._grid_cache[grid_shape] = (gx, gy, gz)
        return gx, gy, gz

    def plane_waves_cart(self, grid_shape=None):
        """Transforms integer fractional meshgrid coordinate axes into explicit Cartesian grid coordinates (in A^-1)."""
        gx, gy, gz = self.get_plane_waves_frac(grid_shape)
        
        # Aligns reciprocal vectors (b1, b2, b3) row indices to match fractional grid components (x, y, z)
        cx, cy, cz = np.tensordot(
            self.reciprocal_lattice * np.pi * 2, [gx, gy, gz], axes=(0, 0))

        return cx, cy, cz
        
    def get_plane_waves_basis_idx(self, ikpt, grid_shape=None, expected_npw=None):
        """Gathers explicit active g-vectors and pre-wrapped grid coordinates from the current parser."""
        if grid_shape is None: 
            grid_shape = self._minimum_fft_size * 2
        gvectors = self._parser.read_gvectors(ikpt)
        # Wrap fractional components into modular array spaces matching the discrete FFT size index layout
        return gvectors, (gvectors % np.asarray(grid_shape)[np.newaxis, :]).astype(int)

    def get_plane_waves_basis_cart_from_idx(self, gvectors, grid_shape=None):
        """Projects integer Miller vector blocks straight into Cartesian inverse Angstrom coordinates."""
        return gvectors @ (2 * np.pi * self.reciprocal_lattice)
        
    def get_plane_wave_coefficients(self, ispin, ikpt, iband):
        """Single-band coefficient query acting as a backward-compatible parser bridge layer."""
        return self._parser.read_coefficients(ispin, ikpt, iband)
        
    def get_pseudo_wavefunction(self, ispin=0, ikpt=0, iband=0, grid_shape=None, kr_phase=False, coeffs=None):
        r'''
        Obtain the pseudo-wavefunction of the specified KS states in real space
        by performing FT transform on the reciprocal space planewave
        coefficients.  The 3D FT grid size is determined by grid_shape, which
        defaults to self._minimum_fft_size*2 if not given.  Gvectors of the KS states is used
        to put 1D planewave coefficients back to 3D grid.

        Inputs:
            ispin : spin index of the desired KS states, starting from 1
            ikpt  : k-point index of the desired KS states, starting from 1
            iband : band index of the desired KS states, starting from 1

        The return wavefunctions are normalized in a way that

                        \sum_{ijk} | \phi_{ijk} | ^ 2 = 1
        '''
        if grid_shape is None: 
            grid_shape = self._minimum_fft_size * 2
        grid_shape = tuple(grid_shape)
        Nx, Ny, Nz = grid_shape
        
        # By default, the WAVECAR only stores the periodic part of the Bloch
        # wavefunction. In order to get the full Bloch wavefunction, one need to
        # multiply the periodic part with the phase: exp(i k (r + r0). Below, the
        # k-point vector and the real-space grid are both in the direct coordinates.
        if kr_phase:
            phase = np.exp(1j * np.pi * 2 * np.sum(self.kpoints[ikpt] * (np.mgrid[0:Nx, 0:Ny, 0:Nz].reshape((3, Nx*Ny*Nz)).T / np.array(grid_shape, dtype=float)), axis=1)).reshape(grid_shape)
        else:
            phase = 1.0
            
        gvectors, gvec_wrapped = self.get_plane_waves_basis_idx(ikpt, grid_shape)
        phi_k = np.zeros(grid_shape, dtype=np.complex128)
        if coeffs is None: 
            coeffs = self.get_plane_wave_coefficients(ispin, ikpt, iband)
            
        # Reconstruct the 3D reciprocal space matrix by placing flat coefficients back onto wrapped grids
        phi_k[gvec_wrapped[:, 0], gvec_wrapped[:, 1], gvec_wrapped[:, 2]] = coeffs
        
        # Execute inverse FFT to map state profile directly into real coordinate space representation
        with set_workers(self.scipy_workers):
            pseudo_wfs = ifftn(phi_k * np.sqrt(Nx * Ny * Nz)) * phase
        return pseudo_wfs
        
    def calculate_charge_density(self, grid_shape=None, spin_channel=-1, energy_range=(-np.inf, np.inf), use_partial_occ=True):
        """Constructs full real-space electronic charge density grid profiles (rho)."""
        if grid_shape is None: 
            grid_shape = self._minimum_fft_size * 2
        Nx, Ny, Nz = grid_shape
        
        # Normalization adjustments ensuring that: \sum_{ijk} | \phi_{ijk} | ^ 2 * cell_volume / N_grid_points = 1
        normFac = np.sqrt((Nx * Ny * Nz) / self.structure.volume)
        kpoint_weights = self.kpoint_weights
        rho = np.zeros(grid_shape, dtype=float)
        spin_indices = [i for i in range(self.nspin)] if spin_channel == -1 else [spin_channel]
    
        # Loop over active spin allocations channels
        for ispin in spin_indices:
            # Loop over sampling irreducible k-points blocks
            for ikpt in range(self.nkpoints):
                active_bands = []
                weights = []
                # Filter bands matching user-specified energy window constraints
                for iband in range(self.nbands):
                    rel_energy = self.energies[ispin, ikpt, iband] - self.efermi
                    if not (energy_range[0] <= rel_energy <= energy_range[1]): 
                        continue
                        
                    # Scale weights based on calculation spin polarization channel limits
                    # Double occupancy factor (rspin=2) must be explicitly enforced if nspin == 1
                    rspin = 2.0 if self.nspin == 1 else 1.0
                    weight = rspin * kpoint_weights[ikpt] * (self.occupancies[ispin, ikpt, iband] if use_partial_occ else 1.0)
                    if weight > 0:
                        active_bands.append(iband)
                        weights.append(weight)
                if not active_bands: 
                    continue
                
                # High-performance batch read eliminates discrete single-band disk head seeking loops
                coeffs_list = self._parser.read_coefficients_batch(ispin, ikpt, active_bands)
                gvectors, gvec_wrapped = self.get_plane_waves_basis_idx(ikpt, grid_shape, expected_npw=coeffs_list.shape[1])
                
                # Multi-dimensional vectorized band allocation layout mapping
                phi_k = np.zeros((len(active_bands), Nx, Ny, Nz), dtype=np.complex128)
                phi_k[:, gvec_wrapped[:, 0], gvec_wrapped[:, 1], gvec_wrapped[:, 2]] = coeffs_list
                
                # Transform entire active energy band manifold collectively using a batched multi-dimensional parallel FFT
                with set_workers(self.scipy_workers):
                    phi_r = ifftn(phi_k * np.sqrt(Nx * Ny * Nz), axes=(1, 2, 3)) * normFac
                    
                # Accumulate real-space densities scaled by corresponding state weights: rho = sum( |psi|^2 * w )
                rho += np.sum((phi_r.conj() * phi_r).real * np.array(weights)[:, np.newaxis, np.newaxis, np.newaxis], axis=0)
                
        return self._symmetrize_3d_grid(rho)
    
    def calculate_localization_function(
            self, 
            grid_shape=None, 
            energy_range=(-np.inf, np.inf), 
            spin_channel=-1, 
            localization_function="elf", 
            savin_correction=True,
            use_partial_occ=True,
            ):
        """Calculates specific localized electron topological indicators (ELF, LOL, or ELI-D)."""
        if grid_shape is None: 
            grid_shape = self._minimum_fft_size * 2
            
        # Collect baseline real space fields required for evaluating topological descriptions
        tau, rho = self.calculate_kinetic_energy_density(grid_shape=grid_shape, energy_range=energy_range, spin_channel=spin_channel, use_partial_occ=use_partial_occ, return_charge_density=True)
        if localization_function == "lol":
            from baderkit.post_wfc.localization_functions import lol
            return lol(rho, tau, savin_correction, spin_channel != -1)
            
        # Both ELF and ELI-D evaluations require explicit field Laplacians and gradient norms
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
        G2 = Gx**2 + Gy**2 + Gz**2  # Compute squared norms magnitude values: G^2 = Gx^2 + Gy^2 + Gz^2
        
        if not is_reciprocal:
            with set_workers(self.scipy_workers): 
                recip_data = fftn(data, norm='ortho')
        else: 
            recip_data = data
            
        # Laplacian operator identity mapping in reciprocal space translates cleanly to: Del^2(rho) -> -G^2 * rho(G)
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
        # Gradient operator identity mapping in reciprocal space translates cleanly to: Grad(rho) -> i * G * rho(G)
        with set_workers(self.scipy_workers):
            grad_x = ifftn(1j * Gx * recip_data, norm='ortho')
            grad_y = ifftn(1j * Gy * recip_data, norm='ortho')
            grad_z = ifftn(1j * Gz * recip_data, norm='ortho')
        return grad_x.real, grad_y.real, grad_z.real
        
    def calculate_state_rho_tau_contributions(
            self, 
            frac_coord, 
            return_grad_rho_sq = True,
            return_lap_rho = True,
            spin_channel = -1, 
            energy_range=None, 
            num_points=2000, 
            method = "gaussian",  # "gaussian", "methfessel-paxton", "fermi-dirac", "tetrahedron"
            sigma=None,
            temperature=None,  # Added to allow physical Kelvin inputs for Fermi-Dirac
            ):
        """
        Computes the partial contributions to the charge density and kinetic
        energy density at a single requested fractional coordinate using direct 
        Fourier evaluation. Bypasses the expensive 3D FFT grid entirely.
        
        Supports Gaussian, Methfessel-Paxton, Fermi-Dirac, and Tetrahedron smearing methods.
        """
        method, sigma = self._get_default_sigma(method.lower(), sigma)

        kpoint_weights = self.kpoint_weights
        energies = self.energies
        
        tau = np.empty_like(energies, dtype=float)
        rho = np.empty_like(energies, dtype=float)
        
        if return_grad_rho_sq:
            grad_rho_sq = np.empty_like(energies, dtype=float)
        if return_lap_rho:
            lap_rho = np.empty_like(energies, dtype=float)
            
        spin_indices = [i for i in range(self.nspin)] if spin_channel == -1 else [spin_channel]
        
        grid_shape = self._minimum_fft_size * 2
        norm_factor = 1.0 / np.sqrt(self.structure.volume)

        # Core loop runs efficiently over the Irreducible Brillouin Zone (IBZ) only
        for ispin in spin_indices:
            for ikpt in range(self.nkpoints):
                active_bands = list(range(self.nbands))
                rspin = 2.0 if self.nspin == 1 else 1.0
                
                # Tetrahedron method handles weights intrinsically via grid cell volumes,
                # so kpoint_weight is omitted during this raw accumulation loop step.
                weight = rspin * kpoint_weights[ikpt] if method.lower() != "tetrahedron" else rspin
                
                coeffs_list = self._parser.read_coefficients_batch(ispin, ikpt, active_bands)
                gvectors, _ = self.get_plane_waves_basis_idx(ikpt, grid_shape, expected_npw=coeffs_list.shape[1])
                rgvec = self.get_plane_waves_basis_cart_from_idx(gvectors, grid_shape)
                
                k = self.kpoints_cart[ikpt]             
                gk2 = np.sum((rgvec + k[np.newaxis, :])**2, axis=1)             
                lap_coeffs = -gk2[np.newaxis, :] * coeffs_list
                
                phases = np.exp(2j * np.pi * np.dot(gvectors, frac_coord))
                
                phi_at_point = np.dot(coeffs_list, phases) * norm_factor
                lap_phi_at_point = np.dot(lap_coeffs, phases) * norm_factor
                
                tau[ispin, ikpt, :] = (-phi_at_point * lap_phi_at_point.conj()).real * weight
                rho[ispin, ikpt, :] = (phi_at_point.conj() * phi_at_point).real * weight
                
                if return_grad_rho_sq or return_lap_rho:
                    K_cart = rgvec + k[np.newaxis, :]
                    phase_grad = 1j * K_cart * phases[:, np.newaxis]
                    grad_phi_at_point = np.dot(coeffs_list, phase_grad) * norm_factor
                    
                    if return_grad_rho_sq:
                        grad_rho_vector = 2.0 * (phi_at_point[:, np.newaxis].conj() * grad_phi_at_point).real
                        grad_rho_sq[ispin, ikpt, :] = np.sum(grad_rho_vector**2, axis=1) * weight
                        
                    if return_lap_rho:
                        grad_psi_sq = np.sum(np.abs(grad_phi_at_point)**2, axis=1)
                        lap_rho[ispin, ikpt, :] = 2.0 * (grad_psi_sq + (phi_at_point.conj() * lap_phi_at_point).real) * weight
                    
        # Generate the energy grid
        e_min, e_max = energy_range if energy_range is not None else self.energy_range
        energy_grid = np.linspace(e_min, e_max, num_points)
        
        # --- SMEARING ENGINE SELECTION ---
        if method.lower() == "tetrahedron":
            smear_matrix = self._get_tetrahedron_weights(energy_grid, spin_indices)
        else:
            bands = (self.energies - self.efermi)[spin_indices].ravel()
            delta_E = energy_grid[:, None] - bands[None, :]
            smear_matrix = self._get_smear_matrix(delta_E / sigma, method, sigma)
        
        # Flatten target metrics to align with the generated smear matrix columns
        rho_vals = rho[spin_indices].ravel()
        tau_vals = tau[spin_indices].ravel()
        
        rho_smeared = np.dot(smear_matrix, rho_vals)
        tau_smeared = np.dot(smear_matrix, tau_vals)
        
        # --- GAUSSIAN POST-PROCESSING CONVOLUTION WINDOW ---
        if method.lower() == "tetrahedron" and sigma > 0.0:
            delta_e = energy_grid[1] - energy_grid[0]
            n_kernel = int(np.ceil(4.0 * sigma / delta_e))
            if n_kernel > 0:
                x_kernel = np.arange(-n_kernel, n_kernel + 1) * delta_e
                kernel = np.exp(-0.5 * (x_kernel / sigma)**2)
                kernel /= np.sum(kernel)
                
                rho_smeared = np.convolve(rho_smeared, kernel, mode='same')
                tau_smeared = np.convolve(tau_smeared, kernel, mode='same')
        
        results = [energy_grid, rho_smeared, tau_smeared]
        
        if return_grad_rho_sq:
            grad_rho_sq_vals = grad_rho_sq[spin_indices].ravel()
            grad_rho_sq_smeared = np.dot(smear_matrix, grad_rho_sq_vals)
            if method.lower() == "tetrahedron" and sigma > 0.0 and n_kernel > 0:
                grad_rho_sq_smeared = np.convolve(grad_rho_sq_smeared, kernel, mode='same')
            results.append(grad_rho_sq_smeared)
            
        if return_lap_rho:
            lap_rho_vals = lap_rho[spin_indices].ravel()
            lap_rho_smeared = np.dot(smear_matrix, lap_rho_vals)
            if method.lower() == "tetrahedron" and sigma > 0.0 and n_kernel > 0:
                lap_rho_smeared = np.convolve(lap_rho_smeared, kernel, mode='same')
            results.append(lap_rho_smeared)
            
        return tuple(results)
    
    def get_density_of_states(
            self, 
            spin_channel = -1, 
            energy_range=None, 
            num_points=2000, 
            method="gaussian", 
            sigma=None, 
            temperature=None, # Added to allow physical Kelvin inputs for Fermi-Dirac
            use_occupancies=False,
            ):
        """
        Constructs energy coordinate profiles outlining the Electronic Density of States (DOS).
        Supports 'gaussian', 'methfessel-paxton', 'fermi-dirac', and 'tetrahedron' smearing methods.
        For 'tetrahedron', an optional Gaussian post-smoothing can be applied via the 'sigma' parameter.
        """
        method, sigma = self._get_default_sigma(method.lower(), sigma)
        
        kpt_weights = self.kpoint_weights
        bands = self.energies - self.efermi
        e_min, e_max = energy_range if energy_range is not None else self.energy_range
        energy_grid = np.linspace(e_min, e_max, num_points)
        
        if spin_channel == 1 and self.nspin == 1:
            spin_channel = 0

        spin_all = spin_channel if spin_channel != -1 else list(range(self.nspin))
        
        if spin_channel == -1 and self.nspin == 1:
            factor = 2
        else:
            factor = 1
            
        if method.lower() == "tetrahedron":
            smear_matrix = self._get_tetrahedron_weights(energy_grid, spin_all)
            if use_occupancies:
                w_t = self.occupancies[spin_all].ravel() * factor
            else:
                w_t = np.ones(len(spin_all) * self.nkpoints * self.nbands) * factor
        else:
            bands_flat = bands[spin_all].ravel()
            if use_occupancies:
                w_t = (self.occupancies[spin_all] * kpt_weights[None, :, None]).ravel() * factor
            else:
                w_t = (np.ones_like(bands[spin_all]) * kpt_weights[None, :, None]).ravel() * factor
            smear_matrix = self._get_smear_matrix((energy_grid[:, None] - bands_flat[None, :]) / sigma, method, sigma)
        
        dos = np.dot(smear_matrix, w_t)
        
        if method.lower() == "tetrahedron" and sigma > 0.0:
            delta_e = energy_grid[1] - energy_grid[0]
            n_kernel = int(np.ceil(4.0 * sigma / delta_e))
            if n_kernel > 0:
                x_kernel = np.arange(-n_kernel, n_kernel + 1) * delta_e
                kernel = np.exp(-0.5 * (x_kernel / sigma)**2)
                kernel /= np.sum(kernel)
                dos = np.convolve(dos, kernel, mode='same')
                
        return energy_grid, dos

    def _get_default_sigma(self, method, sigma):
        shorthands = {
            "gaussian": "gaussian",
            "methfessel-paxton": "methfessel-paxton",
            "mp": "methfessel-paxton",
            "fermi-dirac": "fermi-dirac",
            "fm": "fermi-dirac",
            "tetrahedron": "tetrahedron",
            "tet": "tetrahedron",
            }
        formal_method = shorthands.get(method, None)
        if method is None:
            raise ValueError(f"Unknown smearing method: '{method}'")
        
        if sigma is None:
            
            default_sigma = {
                "gaussian": 0.1,
                "methfessel-paxton": 0.18,
                "fermi-dirac": 300,
                "tetrahedron": 0.04,
                }
            
            sigma = default_sigma[formal_method]
            
        if method == "fermi-dirac":
            sigma = 8.617333262e-5 * sigma
        
        return formal_method, sigma

    def _get_smear_matrix(self, x, method, sigma):
        """Helper matrix generator parsing customized analytical broadening distributions."""
        if method.lower() == "gaussian":
            return np.exp(-0.5 * x**2) / (sigma * np.sqrt(2 * np.pi))
        elif method.lower() in ["methfessel-paxton", "mp"]:
            term_0 = np.exp(-x**2) / np.sqrt(np.pi)
            return ((1.5 - x**2) * term_0) / sigma
        elif method.lower() in ["fermi-dirac", "fd"]:
            exp_term = np.exp(np.clip(x, -50, 50))
            return (exp_term / (exp_term + 1.0)**2) / sigma
        else:
            raise ValueError(f"Unknown smearing method: '{method}'")

    def _setup_tetrahedra(self):
        """
        Identifies the uniform k-point grid dimensions and splits each 
        micro-cell into 6 tetrahedra without any explicit loops or advanced indexing bugs.
        """
        if hasattr(self, '_tetrahedra'):
            return self._tetrahedra, self._ntetra
            
        _ = self.kpoints_full
        kpts = np.mod(np.round(self.kpoints_full, 6), 1.0)
        u1, u2, u3 = np.unique(kpts[:, 0]), np.unique(kpts[:, 1]), np.unique(kpts[:, 2])
        nk1, nk2, nk3 = len(u1), len(u2), len(u3)
        
        # Vectorized grid coordinate mapping
        i = np.argmin(np.abs(kpts[:, 0, None] - u1[None, :]), axis=1)
        j = np.argmin(np.abs(kpts[:, 1, None] - u2[None, :]), axis=1)
        k = np.argmin(np.abs(kpts[:, 2, None] - u3[None, :]), axis=1)
        
        grid_indices = np.zeros((nk1, nk2, nk3), dtype=int)
        grid_indices[i, j, k] = np.arange(len(kpts))
            
        # FIX: Use np.ogrid to create broadcastable 3D index vectors.
        # This prevents NumPy from collapsing dimensions during advanced indexing.
        I, J, K = np.ogrid[:nk1, :nk2, :nk3]
        Ip = (I + 1) % nk1
        Jp = (J + 1) % nk2
        Kp = (K + 1) % nk3
        
        # Grid slicing to construct the 8 corners of all micro-cubes simultaneously
        c000 = grid_indices[I,  J,  K ][:, :, :, None]
        c100 = grid_indices[Ip, J,  K ][:, :, :, None]
        c010 = grid_indices[I,  Jp, K ][:, :, :, None]
        c110 = grid_indices[Ip, Jp, K ][:, :, :, None]
        c001 = grid_indices[I,  J,  Kp][:, :, :, None]
        c101 = grid_indices[Ip, J,  Kp][:, :, :, None]
        c011 = grid_indices[I,  Jp, Kp][:, :, :, None]
        c111 = grid_indices[Ip, Jp, Kp][:, :, :, None]
        
        # Define the 6 space-filling tetrahedra templates
        t1 = np.concatenate([c000, c100, c110, c111], axis=-1).reshape(-1, 4)
        t2 = np.concatenate([c000, c100, c101, c111], axis=-1).reshape(-1, 4)
        t3 = np.concatenate([c000, c001, c101, c111], axis=-1).reshape(-1, 4)
        t4 = np.concatenate([c000, c010, c110, c111], axis=-1).reshape(-1, 4)
        t5 = np.concatenate([c000, c010, c011, c111], axis=-1).reshape(-1, 4)
        t6 = np.concatenate([c000, c001, c011, c111], axis=-1).reshape(-1, 4)
        
        self._tetrahedra = np.vstack([t1, t2, t3, t4, t5, t6])
        self._ntetra = len(self._tetrahedra)
        return self._tetrahedra, self._ntetra
    
    def _get_tetrahedron_weights(self, energy_grid, spin_all):
        """Calculates the tetrahedron smearing matrix using streamlined vector arrays."""
        tetrahedra, ntetra = self._setup_tetrahedra()
        num_points = len(energy_grid)
        smear_matrix_3d = np.zeros((num_points, len(spin_all), self.nkpoints, self.nbands))
        bands = self.energies - self.efermi
    
        for ispin, s in enumerate(spin_all):
            # Sort corner energies directly: shape (ntetra, 4, nbands)
            e = np.sort(bands[s][self.full_to_irr_map][tetrahedra], axis=1)
            e1, e2, e3, e4 = e[:, 0, :], e[:, 1, :], e[:, 2, :], e[:, 3, :]
            
            # Precompute differences safely
            e21, e31, e41, e32, e42, e43 = e2-e1, e3-e1, e4-e1, e3-e2, e4-e2, e4-e3
            
            c21 = np.where(e21 > 1e-12, 1.0 / np.maximum(e21 * e31 * e41, 1e-12), 0.0)
            c32_4 = np.where(e32 > 1e-12, -(e31 + e42) / np.maximum(e31 * e41 * e32 * e42, 1e-12), 0.0)
            c32_3 = np.where(e32 > 1e-12, 3.0 / np.maximum(e31 * e41, 1e-12), 0.0)
            c43 = np.where(e43 > 1e-12, -1.0 / np.maximum(e43 * e42 * e41, 1e-12), 0.0)
            
            band_indices = np.arange(self.nbands)[None, :] 
            
            # Energy grid evaluations
            for ie, E in enumerate(energy_grid):
                v_e1, v_e2, v_e3, v_e4 = E - e1, E - e2, E - e3, E - e4
                G = np.zeros_like(e1)
                
                # Case 1, 2, and 3 steps condensed cleanly
                G = np.where((v_e1 > 0) & (v_e2 <= 0), 3.0 * c21 * v_e1**2, G)
                G = np.where((v_e2 > 0) & (v_e3 <= 0), (c32_3 * e21) + v_e2 * (2.0 * c32_3 + 3.0 * v_e2 * c32_4), G)
                G = np.where((v_e3 > 0) & (v_e4 <= 0), -3.0 * c43 * v_e4**2, G)
                
                # Accumulate corner contributions natively 
                for c in range(4):
                    row_indices = self.full_to_irr_map[tetrahedra[:, c]][:, None]
                    np.add.at(smear_matrix_3d[ie, ispin], (row_indices, band_indices), G / (4.0 * ntetra))
    
        return smear_matrix_3d.reshape(num_points, -1)
    
    def get_electrons_in_energy_range(self, e_min, e_max, num_points=2000, method="gaussian", sigma=0.05):
        """Integrates occupied DOS profiles across designated energy limits windows relative to the Fermi Level."""
        bands = self.energies - self.efermi
        
        # Handle asymptotic bounds limitations by padding extreme values past band margins safely
        if e_min == -np.inf: e_min = np.min(bands) - (5 * sigma if method.lower() != "tetrahedron" else 1.0)
        if e_max == np.inf: e_max = np.max(bands) + (5 * sigma if method.lower() != "tetrahedron" else 1.0)
        
        # Leverages our updated unified DOS generator with occupancies turned on
        egrid, dens = self.get_density_of_states(
            spin_channel=-1, 
            energy_range=(e_min, e_max), 
            num_points=num_points, 
            method=method, 
            sigma=sigma, 
            use_occupancies=True
        )
        
        # Evaluate total accumulated valence electrons within target bounds using numerical integration
        return trapezoid(dens, egrid)
    
    def find_energy_for_electron_count(self, target_electrons, e_min=-np.inf, assume_full_occupancy=False, num_points=5000, method="gaussian", sigma=0.05):
        """Identifies relative energy cutoff limits enclosing specific targeted electron populations quantities."""
        e_min_calc, e_max_calc = self.energy_range
        pad = 5 * sigma if method.lower() != "tetrahedron" else 1.0
        e_min_calc -= pad
        e_max_calc += pad
        
        egrid, density = self.get_density_of_states(
            spin_channel=-1, 
            energy_range=(e_min_calc, e_max_calc), 
            num_points=num_points, 
            method=method, 
            sigma=sigma, 
            use_occupancies=not assume_full_occupancy
        )
        
        if assume_full_occupancy:
            density = density * np.max(self.occupancies)
            
        if e_min == -np.inf: e_min = egrid[0]
        mask = egrid >= e_min
        sub_g, sub_d = egrid[mask], density[mask]
        
        cum_charge = np.zeros(len(sub_g))
        cum_charge[1:] = cumulative_trapezoid(sub_d, sub_g)
        
        if target_electrons > cum_charge[-1]: 
            raise ValueError("Target electron allocation total exceeds evaluated capacity parameters grid envelope limits.")
        return np.interp(target_electrons, cum_charge, sub_g)
    
    def _symmetrize_3d_grid(self, field):
        """Averages real-space property grid profiles over space-group operations to enforce symmetry invariants."""
        symmetry = self.structure.symmetry_data
        Nx, Ny, Nz = field.shape
        sym_field = np.zeros_like(field)
        
        mx, my, mz = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing='ij')
        coords = np.stack([mx.ravel() / Nx, my.ravel() / Ny, mz.ravel() / Nz], axis=0)
        
        # Walk systematically through available crystal rotation and translation operations
        for R, t in zip(symmetry.rotations, symmetry.translations):
            R_inv = np.round(np.linalg.inv(R)).astype(int)
            # Map forward real space grid components onto inverted cell symmetry frames
            tc = ((R_inv @ coords - (R_inv @ t)[:, np.newaxis]) % 1.0)
            
            # Re-index coordinate components back onto periodic cell boundary limits using integer rounding masks
            sym_field += field[
                np.round(tc[0] * Nx).astype(int) % Nx, 
                np.round(tc[1] * Ny).astype(int) % Ny, 
                np.round(tc[2] * Nz).astype(int) % Nz
            ].reshape(Nx, Ny, Nz)
            
        return sym_field / len(symmetry.rotations)
    
    @classmethod
    def from_directory(cls, directory: Path | str = Path("."), fmt: str = "vasp", scipy_workers: int = -1, **kwargs):
        """Dynamic parser factory routing file stream construction to selected code formats templates handles."""
        
        # select proper parser
        if fmt == "vasp": 
            from baderkit.post_wfc.wf_parsers import VaspParser as Parser
        elif fmt == "qe": 
            from baderkit.post_wfc.wf_parsers import QeParser as Parser
        else: 
            raise ValueError(f"Unknown format profile template string keyword: {fmt}")
            
        # create instance
        return cls(Parser(directory=Path(directory), **kwargs), scipy_workers=scipy_workers)