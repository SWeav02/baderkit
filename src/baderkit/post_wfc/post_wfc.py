# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
from scipy.fft import fftn, ifftn, set_workers
from scipy.integrate import trapezoid, cumulative_trapezoid

from baderkit.post_wfc.wf_parsers.base import HSQDTM
from baderkit.post_wfc.all_electron_references.reference_environment import AtomicReferenceEnvironment
from baderkit.global_numba.file_parsers import load_pseudo

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
        valence_counts,
        scipy_workers: int = -1,
        **kwargs
        ):
        """Initializes properties and calculates necessary minimum FFT grid boundaries."""
        self._parser = parser
        self._meta = parser.meta
        self._structure = self._meta.structure
        self._lattice = self._structure.lattice.matrix               # Lattice row vectors in Angstroms
        self._reciprocal_lattice = np.linalg.inv(self._lattice).T   # Transpose of inverse cell matrix (B_mat without 2pi)
        self.scipy_workers = scipy_workers
        
        # valence counts
        if valence_counts is False:
            valence_counts = {}
            # The user requests a full electron calculation
            for site in self.structure:
                valence_counts[site.specie.symbol] = getattr(site.specie, "number", 0)
        self._valence_counts = valence_counts
        
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
        self._reference_environment = None # The equivalent non-bonding system
        
    @property
    def valence_counts(self) -> dict | None:
        """

        Returns
        -------
        dict | None
            A dictionary where each key is an atomic species in the system and each
            value is the number of valence electrons used in the pseudo potential.
            This is used for methods that calculate oxidation states.

        """
        return self._valence_counts

    @valence_counts.setter
    def valence_counts(self, value: dict):
        self._valence_counts = value
        
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
    
    @property
    def reference_environment(self):
        if self._reference_environment is None:
            self._reference_environment = AtomicReferenceEnvironment(
                self.structure, 
                self.valence_counts,
                )
        return self._reference_environment
        
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
    
    def calculate_kinetic_energy_density(self, grid_shape=None, spin_channel=-1, energy_range=(-np.inf, np.inf), use_partial_occ=True, return_charge_density=False):
        """Computes full real-space non-negative electronic kinetic energy density profiles (tau)."""
        # TODO: Add tag to determine if using canonical or shrodinger (-lap(rho)/2) form
        # Change LOL/ELF calls to assume canonical rather than shrodinger
        
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
                    rel_energy = self.energies[ispin, ikpt, iband] - self.efermi
                    if not (energy_range[0] <= rel_energy <= energy_range[1]): 
                        continue
                    rspin = 2.0 if self.nspin == 1 else 1.0
                    weight = rspin * kpoint_weights[ikpt] * (self.occupancies[ispin, ikpt, iband] if use_partial_occ else 1.0)
                    if weight > 0:
                        active_bands.append(iband)
                        weights.append(weight)
                if not active_bands: 
                    continue
                
                # Optimized batch reading extracts all active bands in a single file open transaction
                coeffs_list = self._parser.read_coefficients_batch(ispin, ikpt, active_bands)
                gvectors, gvec_wrapped = self.get_plane_waves_basis_idx(ikpt, grid_shape_tuple, expected_npw=coeffs_list.shape[1])
                rgvec = self.get_plane_waves_basis_cart_from_idx(gvectors, grid_shape_tuple)
                
                # Construct absolute momentum vector components coordinates: K = G + k
                k = self.kpoints_cart[ikpt]             
                gk2 = np.sum((rgvec + k[np.newaxis, :])**2, axis=1)             
                
                # Integration by Parts identity transforms kinetic matrix elements into a Laplacian representation:
                # tau(r) = sum( |Grad(psi)|^2 ) -> Re-mapped onto grid space fields via: -psi * Del^2(psi)*
                lap_coeffs = -gk2[np.newaxis, :] * coeffs_list
                
                # Map raw state coefficients arrays collectives onto dense 3D frequency grid structures
                phi_k = np.zeros((len(active_bands), Nx, Ny, Nz), dtype=np.complex128)
                phi_k[:, gvec_wrapped[:, 0], gvec_wrapped[:, 1], gvec_wrapped[:, 2]] = coeffs_list
                with set_workers(self.scipy_workers):
                    phi_r = ifftn(phi_k * np.sqrt(Nx * Ny * Nz), axes=(1, 2, 3)) * normFac
                
                # Map corresponding orbital Laplacian coefficients onto matching wrapped grid matrix structures
                lap_phi_k = np.zeros((len(active_bands), Nx, Ny, Nz), dtype=np.complex128)
                lap_phi_k[:, gvec_wrapped[:, 0], gvec_wrapped[:, 1], gvec_wrapped[:, 2]] = lap_coeffs
                with set_workers(self.scipy_workers):
                    lap_phi_r = ifftn(lap_phi_k * np.sqrt(Nx * Ny * Nz), axes=(1, 2, 3)) * normFac
                    
                w_arr = np.array(weights)[:, np.newaxis, np.newaxis, np.newaxis]
                # Accumulate kinetic density mapping terms (tau) in real space coordinates
                tau += np.sum((-phi_r * lap_phi_r.conj()).real * w_arr, axis=0)
                if return_charge_density:
                    rho += np.sum((phi_r.conj() * phi_r).real * w_arr, axis=0)

        tau = self._symmetrize_3d_grid(tau)
        if return_charge_density: 
            return tau, self._symmetrize_3d_grid(rho)
        return tau
    
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
                method = "gaussian", 
                sigma=None,
                ):
        
        grid_shape = self._minimum_fft_size * 2
        nx, ny, nz = grid_shape
        
        # Target voxel mapping
        ix = int(np.round(frac_coord[0] * nx)) % nx
        iy = int(np.round(frac_coord[1] * ny)) % ny
        iz = int(np.round(frac_coord[2] * nz)) % nz
    
        def point_callback(ispin, ikpt, coeffs_list, gvectors, kx_idx, ky_idx, kz_idx, weight, gshape, norm_factor):
            c_grid = np.zeros((self.nbands, nx, ny, nz), dtype=complex)
            for iband in range(self.nbands):
                c_grid[iband, kx_idx, ky_idx, kz_idx] = coeffs_list[iband, :]
            
            psi_r_all = np.fft.ifftn(c_grid, axes=(1, 2, 3)) * (nx * ny * nz) * norm_factor
            phi_at_point = psi_r_all[:, ix, iy, iz]
            
            rgvec = self.get_plane_waves_basis_cart_from_idx(gvectors, gshape)
            k = self.kpoints_cart[ikpt]             
            K_cart = rgvec + k[np.newaxis, :]             
            gk2 = np.sum(K_cart**2, axis=1)             
            
            c_grid_lap = np.zeros((self.nbands, nx, ny, nz), dtype=complex)
            for iband in range(self.nbands):
                c_grid_lap[iband, kx_idx, ky_idx, kz_idx] = -gk2 * coeffs_list[iband, :]
            
            lap_psi_r_all = np.fft.ifftn(c_grid_lap, axes=(1, 2, 3)) * (nx * ny * nz) * norm_factor
            lap_phi_at_point = lap_psi_r_all[:, ix, iy, iz]
            
            rho_bands = (phi_at_point.conj() * phi_at_point).real * weight
            tau_bands = (-phi_at_point * lap_phi_at_point.conj()).real * weight
            
            # Pack metrics dynamically to keep indexes clean
            metrics = [rho_bands, tau_bands]
            
            if return_grad_rho_sq or return_lap_rho:
                grad_phi_at_point = np.zeros((self.nbands, 3), dtype=complex)
                for idim in range(3):
                    c_grid_grad = np.zeros((self.nbands, nx, ny, nz), dtype=complex)
                    for iband in range(self.nbands):
                        c_grid_grad[iband, kx_idx, ky_idx, kz_idx] = 1j * K_cart[:, idim] * coeffs_list[iband, :]
                    grad_psi_r_all = np.fft.ifftn(c_grid_grad, axes=(1, 2, 3)) * (nx * ny * nz) * norm_factor
                    grad_phi_at_point[:, idim] = grad_psi_r_all[:, ix, iy, iz]
                
                if return_grad_rho_sq:
                    grad_rho_vec = 2.0 * (phi_at_point[:, np.newaxis].conj() * grad_phi_at_point).real
                    grad_rho_sq_bands = np.sum(grad_rho_vec**2, axis=1) * weight
                    metrics.append(grad_rho_sq_bands)
                else:
                    metrics.append(None)
                    
                if return_lap_rho:
                    grad_psi_sq = np.sum(np.abs(grad_phi_at_point)**2, axis=1)
                    lap_rho_bands = 2.0 * (grad_psi_sq + (phi_at_point.conj() * lap_phi_at_point).real) * weight
                    metrics.append(lap_rho_bands)
                else:
                    metrics.append(None)
                    
            return metrics
    
        # Determine total tracking dimensions dynamically 
        num_metrics = 4 if (return_grad_rho_sq or return_lap_rho) else 2

        energy_grid, smeared = self._execute_spectral_engine(
            num_metrics=num_metrics, spin_channel=spin_channel, energy_range=energy_range, 
            num_points=num_points, method=method, sigma=sigma, eval_callback=point_callback
        )
        
        results = [energy_grid, smeared[0], smeared[1]]
        if return_grad_rho_sq: results.append(smeared[2])
        if return_lap_rho: results.append(smeared[3])
        return tuple(results)
    
    def calculate_state_deformation_density(
                self, 
                frac_coord, 
                spin_channel=-1,
                energy_range=None, 
                num_points=2000, 
                method="gaussian", 
                sigma=None
                ):
        """
        Calculates the step-by-step delta deformation density spectrum matching the exact
        non-linear charge accumulation tracks of the true crystal states.
        
        Automatically masks out high-energy steps that exceed the physical capacity 
        of the reference atomic basis pool and emits a tracking warning.
        """
        reference_env = self.reference_environment

        # 1. Resolve formal smearing parameters using your native routing tool
        formal_method, formal_sigma = self._get_default_sigma(method, sigma)
        e_min, e_max = energy_range if energy_range is not None else self.energy_range
        
        # 2. Automatically calculate background charge accumulated up to e_min
        start_charge = self.get_electrons_in_energy_range(
            -np.inf, e_min, num_points=num_points, method=method, sigma=sigma
        )
        
        # 3. Compute natively smeared crystal profiles simultaneously
        egrid, dos = self.get_density_of_states(
            spin_channel=spin_channel, 
            energy_range=(e_min, e_max), 
            num_points=num_points, 
            method=method, 
            sigma=sigma,
            use_occupancies=False
        )
        
        _, rho_crys, _ = self.calculate_state_rho_tau_contributions(
            frac_coord=frac_coord, 
            return_grad_rho_sq=False, 
            return_lap_rho=False,
            spin_channel=spin_channel, 
            energy_range=(e_min, e_max), 
            num_points=num_points, 
            method=method,
            sigma=sigma
        )
        
        # 4. Convert the continuous spectral DOS into exact cell charge boundaries
        delta_e = egrid[1] - egrid[0]
        q_bounds = np.zeros(len(dos) + 1)
        q_bounds[0] = start_charge
        q_bounds[1:] = start_charge + np.cumsum(dos) * delta_e
        
        # 5. Extract the matching sequential reference steps from the environment
        delta_rho_ref_steps = reference_env.calculate_sequential_reference_deltas(
            frac_coord=frac_coord, q_bounds=q_bounds
        )
        
        # 6. Direct Point-by-Point Spectral Subtraction
        deformation_profile = rho_crys - (delta_rho_ref_steps / delta_e)
        
        # 7. Dynamic Capacity Checking & Saturation Masking
        max_capacity_charge = reference_env.get_maximum_cell_charge_capacity()
        
        # Check against the upper charge bound of each discrete energy interval
        invalid_mask = q_bounds[1:] > max_capacity_charge
        
        if np.any(invalid_mask):
            first_invalid_idx = np.where(invalid_mask)[0][0]
            meaningless_energy_threshold = egrid[first_invalid_idx]
            
            print(f"========================================================================\n"
                  f"WARNING: Reference atomic basis set capacity exceeded!\n"
                  f"Orbital saturation occurs at total cell charge: {max_capacity_charge:.4f} e-\n"
                  f"Energy values above {meaningless_energy_threshold:.4f} eV are unphysical.\n"
                  f"Masking subsequent deformation steps to 0.0.\n"
                  f"========================================================================")
            
            # Force the net delta to zero for all saturated intervals
            deformation_profile[invalid_mask] = 0.0
            
        return egrid, deformation_profile
    
    def get_density_of_states(
                self, 
                spin_channel = -1, 
                energy_range=None, 
                num_points=2000, 
                method="gaussian", 
                sigma=None, 
                use_occupancies=False,
                ):
            """
            Constructs energy coordinate profiles outlining the Electronic Density of States (DOS).
            Supports 'gaussian', 'methfessel-paxton', 'fermi-dirac', 'tetrahedron', and 'none' methods.
            For 'tetrahedron', an optional Gaussian post-smoothing can be applied via the 'sigma' parameter.
            """
            method, sigma = self._get_default_sigma(method, sigma)
            
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
                
            if method == "tetrahedron":
                smear_matrix = self._get_tetrahedron_weights(energy_grid, spin_all)
                if use_occupancies:
                    w_t = self.occupancies[spin_all].ravel() * factor
                else:
                    w_t = np.ones(len(spin_all) * self.nkpoints * self.nbands) * factor
            else:
                # Flatten bands and build weights identically across all continuum & delta methods
                bands_flat = bands[spin_all].ravel()
                if use_occupancies:
                    w_t = (self.occupancies[spin_all] * kpt_weights[None, :, None]).ravel() * factor
                else:
                    w_t = (np.ones_like(bands[spin_all]) * kpt_weights[None, :, None]).ravel() * factor
                
                # INTERCEPT HERE: Handle sharp delta-binning cleanly
                if method == "none":
                    delta_e = energy_grid[1] - energy_grid[0]
                    
                    # Fast vectorized nearest-neighbor grid index projection
                    closest_idx = np.round((bands_flat - e_min) / delta_e).astype(int)
                    valid_mask = (closest_idx >= 0) & (closest_idx < num_points)
                    
                    smear_matrix = np.zeros((num_points, len(bands_flat)))
                    if len(bands_flat) > 0:
                        # 1.0 / delta_e scale enforces perfect preservation of integrated total state counts
                        smear_matrix[closest_idx[valid_mask], np.where(valid_mask)[0]] = 1.0 / delta_e
                else:
                    # Continuous analytical broadening tracks
                    smear_matrix = self._get_smear_matrix((energy_grid[:, None] - bands_flat[None, :]) / sigma, method, sigma)
            
            dos = np.dot(smear_matrix, w_t)
            
            if method == "tetrahedron" and sigma > 0.0:
                delta_e = energy_grid[1] - energy_grid[0]
                n_kernel = int(np.ceil(4.0 * sigma / delta_e))
                if n_kernel > 0:
                    x_kernel = np.arange(-n_kernel, n_kernel + 1) * delta_e
                    kernel = np.exp(-0.5 * (x_kernel / sigma)**2)
                    kernel /= np.sum(kernel)
                    dos = np.convolve(dos, kernel, mode='same')
                    
            return energy_grid, dos
    
    def get_electrons_in_energy_range(self, e_min, e_max, num_points=2000, method="gaussian", sigma=None):
            """Integrates occupied DOS profiles across designated energy limits windows relative to the Fermi Level."""
            bands = self.energies - self.efermi
            
            # Resolve the formal method identifier and default sigma values
            method, sigma = self._get_default_sigma(method, sigma)
            
            # Handle asymptotic bounds limitations by padding extreme values past band margins safely
            if e_min == -np.inf: 
                pad = 0.1 if method == "none" else (1.0 if method == "tetrahedron" else 5 * sigma)
                e_min = np.min(bands) - pad
                
            if e_max == np.inf: 
                pad = 0.1 if method == "none" else (1.0 if method == "tetrahedron" else 5 * sigma)
                e_max = np.max(bands) + pad
            
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
        pad = 5 * sigma if method != "tetrahedron" else 1.0
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
    
    def _get_default_sigma(self, method, sigma):
        # Gracefully capture literal Python None types
        if method is None:
            method = "none"
            
        shorthands = {
            "none": "none",
            "gaussian": "gaussian",
            "methfessel-paxton": "methfessel-paxton",
            "mp": "methfessel-paxton",
            "fermi-dirac": "fermi-dirac",
            "fm": "fermi-dirac",
            "tetrahedron": "tetrahedron",
            "tet": "tetrahedron",
            }
        
        formal_method = shorthands.get(method if isinstance(method, str) else method, None)
        if formal_method is None:
            raise ValueError(f"Unknown smearing method: '{method}'")
        
        if sigma is None:
            default_sigma = {
                "none": 0.0,
                "gaussian": 0.1,
                "methfessel-paxton": 0.18,
                "fermi-dirac": 300,
                "tetrahedron": 0.04,
                }
            sigma = default_sigma[formal_method]
            
        if formal_method == "fermi-dirac":
            sigma = 8.617333262e-5 * sigma
        
        return formal_method, sigma
    
    def _get_smear_matrix(self, x, method, sigma):
        """Helper matrix generator parsing customized analytical broadening distributions."""
        if method == "none":
            raise ValueError("Smearing matrix for 'none' method must be constructed via discrete grid-binning, not continuous coordinate division.")
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
        """
        Unified high-performance orchestration engine for plane-wave spectral decompositions.
        Manages loops, weights, indexing, and the energy smearing/convolution backend.
        """
        method, sigma = self._get_default_sigma(method, sigma)
        kpoint_weights = self.kpoint_weights
        
        spin_indices = [i for i in range(self.nspin)] if spin_channel == -1 else [spin_channel]
        grid_shape = self._minimum_fft_size * 2
        nx, ny, nz = grid_shape
        norm_factor = 1.0 / np.sqrt(self.structure.volume)
        
        # Track metrics dynamically across an arbitrary number of collection vectors
        raw_data = np.zeros((num_metrics, self.nspin, self.nkpoints, self.nbands), dtype=float)
        
        # Execute loops over the Irreducible Brillouin Zone
        for ispin in spin_indices:
            for ikpt in range(self.nkpoints):
                active_bands = list(range(self.nbands))
                rspin = 2.0 if self.nspin == 1 else 1.0
                weight = rspin * kpoint_weights[ikpt] if method != "tetrahedron" else rspin
                
                coeffs_list = self._parser.read_coefficients_batch(ispin, ikpt, active_bands)
                gvectors, _ = self.get_plane_waves_basis_idx(ikpt, grid_shape, expected_npw=coeffs_list.shape[1])
                
                kx_idx = gvectors[:, 0] % nx
                ky_idx = gvectors[:, 1] % ny
                kz_idx = gvectors[:, 2] % nz
                
                # Defer custom numerical contractions to the provided function handle
                metrics_block = eval_callback(
                    ispin, ikpt, coeffs_list, gvectors, kx_idx, ky_idx, kz_idx, 
                    weight, grid_shape, norm_factor
                )
                
                for imetric, metric_bands in enumerate(metrics_block):
                    if metric_bands is not None:
                        raw_data[imetric, ispin, ikpt, :] = metric_bands
                        
        # Generate the energy spectrum mapping coordinate grid
        e_min, e_max = energy_range if energy_range is not None else self.energy_range
        energy_grid = np.linspace(e_min, e_max, num_points)
        delta_e = energy_grid[1] - energy_grid[0]
        
        if method == "tetrahedron":
            smear_matrix = self._get_tetrahedron_weights(energy_grid, spin_indices)
        elif method == "none":
            bands = (self.energies - self.efermi)[spin_indices].ravel()
            
            # Fast vectorized nearest-neighbor discrete binning
            closest_idx = np.round((bands - e_min) / delta_e).astype(int)
            valid_mask = (closest_idx >= 0) & (closest_idx < num_points)
            
            smear_matrix = np.zeros((num_points, len(bands)))
            if len(bands) > 0:
                # 1.0 / delta_e weight preserves exact density integrations under post-smearing
                smear_matrix[closest_idx[valid_mask], np.where(valid_mask)[0]] = 1.0 / delta_e
        else:
            bands = (self.energies - self.efermi)[spin_indices].ravel()
            delta_E = energy_grid[:, None] - bands[None, :]
            smear_matrix = self._get_smear_matrix(delta_E / sigma, method, sigma)
        
        # Build Gaussian post-processing convolution windows if necessary
        use_convolution = (method == "tetrahedron" and sigma > 0.0)
        if use_convolution:
            delta_e_kernel = energy_grid[1] - energy_grid[0]
            n_kernel = int(np.ceil(4.0 * sigma / delta_e_kernel))
            if n_kernel > 0:
                x_kernel = np.arange(-n_kernel, n_kernel + 1) * delta_e_kernel
                kernel = np.exp(-0.5 * (x_kernel / sigma)**2)
                kernel /= np.sum(kernel)
            else:
                use_convolution = False
                
        # Matrix multiply to apply spectral smearing over all channels simultaneously
        smeared_results = []
        for imetric in range(num_metrics):
            vals = raw_data[imetric, spin_indices].ravel()
            smeared = np.dot(smear_matrix, vals)
            if use_convolution:
                smeared = np.convolve(smeared, kernel, mode='same')
            smeared_results.append(smeared)
            
        return energy_grid, smeared_results
    
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
    def from_directory(
            cls, 
            directory: Path | str = Path("."), 
            fmt: str = "vasp", 
            scipy_workers: int = -1, 
            valence_counts: dict = None,
            pseudopotential_filename: Path | None | list | str | bool = None,
            **kwargs,
            ):
        """Dynamic parser factory routing file stream construction to selected code formats templates handles."""
        
        # select proper parser
        if fmt == "vasp": 
            from baderkit.post_wfc.wf_parsers import VaspParser as Parser
        elif fmt == "qe": 
            from baderkit.post_wfc.wf_parsers import QeParser as Parser
        else: 
            raise ValueError(f"Unknown format profile template string keyword: {fmt}")
            
        if valence_counts is None:
            valence_counts = load_pseudo(pseudopotential_filename)
            
        # create instance
        return cls(Parser(directory=Path(directory), **kwargs), scipy_workers=scipy_workers, valence_counts=valence_counts, **kwargs)