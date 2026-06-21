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
        self._lattice = self._structure.lattice.matrix
        self._reciprocal_lattice = np.linalg.inv(self._lattice).T
        self.scipy_workers = scipy_workers
        
        lattice_norm = np.linalg.norm(self._lattice, axis=1)
        CUTOFF = np.ceil(np.sqrt(self._meta.energy_cutoff / HSQDTM) / (2 * np.pi / lattice_norm))
        self._minimum_fft_size = np.array(2 * CUTOFF + 1, dtype=int)
        
        self._energy_range = None
        self._total_electrons = None
        self._kpoint_multiplicities = None
        self._kpoint_weights = None
        self._grid_cache = {}
        
    @property
    def nspin(self):
        """Returns total active spin allocation dimension count (1 or 2)."""
        return self._meta.nspin
        
    @property
    def nkpoints(self):
        """Returns the number of irreducible sampling coordinates."""
        return self._meta.nkpts
        
    @property
    def nbands(self):
        """Returns total calculated band orbital channels."""
        return self._meta.nbands
        
    @property
    def kpoints(self):
        """Returns fractional coordinates for the irreducible kpoint mesh."""
        return self._meta.kpoints
        
    @property
    def kpoints_cart(self):
        """Converts fractional sampling coordinates into Cartesian inverse Angstrom vectors."""
        return np.dot(self.kpoints, 2 * np.pi * self.reciprocal_lattice)
        
    @property
    def occupancies(self):
        """Returns the raw multidimensional occupancy numbers array."""
        return self._meta.occupancies
        
    @property
    def energies(self):
        """Returns absolute state eigenvalues in eV."""
        return self._meta.energies
        
    @property
    def energy_cutoff(self):
        """Returns kinetic energy basis set cutoff limitation in eV."""
        return self._meta.energy_cutoff
        
    @property
    def structure(self):
        """Returns crystal structure information metadata."""
        return self._structure
        
    @property
    def lattice(self):
        """Returns real-space cell matrix row vectors."""
        return self._lattice
        
    @property
    def reciprocal_lattice(self):
        """Returns reciprocal-space cell matrix column vectors."""
        return self._reciprocal_lattice
        
    @property
    def efermi(self):
        """Returns baseline calculation Fermi level energy in eV."""
        return self._meta.efermi

    @property
    def kpoint_multiplicities(self):
        """Maps star orbits across reciprocal rotations to determine full-zone multiplicities."""
        if self._kpoint_multiplicities is None:
            recp_symm_ops = self.structure.lattice.get_recp_symmetry_operation()
            recip_rotations = []
            for op in recp_symm_ops:
                R_recip = np.round(np.linalg.inv(op.rotation_matrix).T).astype(int)
                if not any(np.array_equal(R_recip, R) for R in recip_rotations):
                    recip_rotations.append(R_recip)
            for R in [-R for R in recip_rotations]:
                if not any(np.array_equal(R, ex_R) for ex_R in recip_rotations):
                    recip_rotations.append(R)
            multiplicities = []
            for k in self.kpoints:
                eq_kpts = []
                for R in recip_rotations:
                    k_wrapped = np.mod(np.round(R @ k, decimals=6), 1.0)
                    if not any(np.allclose(k_wrapped - eq, np.round(k_wrapped - eq), 1e-5) for eq in eq_kpts):
                        eq_kpts.append(k_wrapped)
                multiplicities.append(len(eq_kpts))
            self._kpoint_multiplicities = np.array(multiplicities)
        return self._kpoint_multiplicities
    
    @property
    def kpoint_weights(self):
        """Returns normalized full Brillouin zone weights configuration maps."""
        if self._kpoint_weights is None:
            self._kpoint_weights = self.kpoint_multiplicities / np.sum(self.kpoint_multiplicities)
        return self._kpoint_weights
    
    @property
    def energy_range(self):
        """Returns boundaries scaled relative to Fermi levels (E_min - E_f, E_max - E_f)."""
        if self._energy_range is None:
            self._energy_range = np.min(self.energies) - self.efermi, np.max(self.energies) - self.efermi
        return self._energy_range
    
    @property
    def total_electrons(self):
        """Evaluates total system electron allocation count by integrating DOS profiles."""
        if self._total_electrons is None:
            self._total_electrons = self.get_electrons_in_energy_range(-np.inf, np.inf)
        return self._total_electrons
    
    def get_plane_waves_frac(self, grid_shape=None):
        """Generates reciprocal fractional grid coordinates arrays."""
        if grid_shape is None: 
            grid_shape = self._minimum_fft_size * 2
        grid_shape_tuple = tuple(grid_shape)
        if grid_shape_tuple in self._grid_cache: 
            return self._grid_cache[grid_shape_tuple]
        Nx, Ny, Nz = grid_shape_tuple
        fx = [ii if ii < Nx // 2 + 1 else ii - Nx for ii in range(Nx)]
        fy = [jj if jj < Ny // 2 + 1 else jj - Ny for jj in range(Ny)]
        fz = [kk if kk < Nz // 2 + 1 else kk - Nz for kk in range(Nz)]
        gx, gy, gz = np.meshgrid(fx, fy, fz, indexing='ij')
        self._grid_cache[grid_shape_tuple] = (gx, gy, gz)
        return gx, gy, gz

    def plane_waves_cart(self, grid_shape=None):
        """Transforms reciprocal coordinate structures into Cartesian inverse Angstrom points."""
        gx, gy, gz = self.get_plane_waves_frac(grid_shape)
        return np.tensordot(self.reciprocal_lattice * np.pi * 2, [gz, gy, gx], axes=(0, 0))
    
    def get_plane_waves_basis_idx(self, ikpt, grid_shape=None, expected_npw=None):
        """Extracts sorted plane wave indices directly via the active parser strategy pattern."""
        if grid_shape is None: 
            grid_shape = self._minimum_fft_size * 2
        gvectors = self._parser.read_gvectors(ikpt)
        return gvectors, (gvectors % np.asarray(grid_shape)[np.newaxis, :]).astype(int)

    def get_plane_waves_basis_cart_from_idx(self, gvectors, grid_shape=None):
        """Maps unique indexed integer vectors straight into Cartesian inverse Angstrom structures."""
        return gvectors @ (2 * np.pi * self.reciprocal_lattice)
    
    def get_plane_wave_coefficients(self, ispin, ikpt, iband):
        """Single-band descriptor fallback utility query."""
        return self._parser.read_coefficients(ispin, ikpt, iband)
    
    def get_pseudo_wavefunction(self, ispin=0, ikpt=0, iband=0, grid_shape=None, kr_phase=False, coeffs=None):
        r'''
        Obtain the pseudo-wavefunction of the specified KS states in real space
        by performing FT transform on the reciprocal space planewave
        coefficients.  The 3D FT grid size is determined by grid_shape, which
        defaults to self._grid_shape if not given.  Gvectors of the KS states is used
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
        grid_shape_tuple = tuple(grid_shape)
        Nx, Ny, Nz = grid_shape_tuple
        
        # By default, the WAVECAR only stores the periodic part of the Bloch
        # wavefunction. In order to get the full Bloch wavefunction, one need to
        # multiply the periodic part with the phase: exp(i k (r + r0). Below, the
        # k-point vector and the real-space grid are both in the direct coordinates.
        if kr_phase:
            phase = np.exp(1j * np.pi * 2 * np.sum(self.kpoints[ikpt] * (np.mgrid[0:Nx, 0:Ny, 0:Nz].reshape((3, Nx*Ny*Nz)).T / np.array(grid_shape_tuple, dtype=float)), axis=1)).reshape(grid_shape_tuple)
        else:
            phase = 1.0
            
        gvectors, gvec_wrapped = self.get_plane_waves_basis_idx(ikpt, grid_shape_tuple)
        phi_k = np.zeros(grid_shape_tuple, dtype=np.complex128)
        if coeffs is None: 
            coeffs = self.get_plane_wave_coefficients(ispin, ikpt, iband)
        phi_k[gvec_wrapped[:, 0], gvec_wrapped[:, 1], gvec_wrapped[:, 2]] = coeffs
        with set_workers(self.scipy_workers):
            pseudo_wfs = ifftn(phi_k * np.sqrt(Nx * Ny * Nz)) * phase
        return pseudo_wfs
    
    def calculate_charge_density(self, grid_shape=None, spin_channel=-1, energy_range=(-np.inf, np.inf), use_partial_occ=True):
        """Constructs full real-space electronic charge density grid properties arrays."""
        if grid_shape is None: 
            grid_shape = self._minimum_fft_size * 2
        grid_shape_tuple = tuple(grid_shape)
        Nx, Ny, Nz = grid_shape_tuple
        normFac = np.sqrt((Nx * Ny * Nz) / self.structure.volume)
        kpoint_weights = self.kpoint_weights
        rho = np.zeros(grid_shape_tuple, dtype=float)
        spin_indices = [i for i in range(self.nspin)] if spin_channel == -1 else [spin_channel]
    
        # Loop over spins
        for ispin in spin_indices:
            # Loop over k-points
            for ikpt in range(self.nkpoints):
                active_bands = []
                weights = []
                # Loop over bands
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
                
                # Optimized batch loading extracts all coefficients in one disk read block
                coeffs_list = self._parser.read_coefficients_batch(ispin, ikpt, active_bands)
                gvectors, gvec_wrapped = self.get_plane_waves_basis_idx(ikpt, grid_shape_tuple, expected_npw=coeffs_list.shape[1])
                
                # Vectorized Band Layout mapping
                phi_k = np.zeros((len(active_bands), Nx, Ny, Nz), dtype=np.complex128)
                phi_k[:, gvec_wrapped[:, 0], gvec_wrapped[:, 1], gvec_wrapped[:, 2]] = coeffs_list
                with set_workers(self.scipy_workers):
                    phi_r = ifftn(phi_k * np.sqrt(Nx * Ny * Nz), axes=(1, 2, 3)) * normFac
                rho += np.sum((phi_r.conj() * phi_r).real * np.array(weights)[:, np.newaxis, np.newaxis, np.newaxis], axis=0)
        return self._symmetrize_3d_grid(rho)
    
    def calculate_laplacian(self, data, is_reciprocal=False):
        """Evaluates second-derivative field Laplacian shapes via fast Fourier transforms."""
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
        """Evaluates first-derivative vector fields using parallelized spatial FFT mapping loops."""
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
    
    def calculate_kinetic_energy_density(self, grid_shape=None, spin_channel=-1, energy_range=(-np.inf, np.inf), use_partial_occ=True, return_charge_density=False):
        """Computes full real-space non-negative electronic kinetic energy density profiles (tau)."""
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
                
                # Optimized batch loading extracts all coefficients in one disk read block
                coeffs_list = self._parser.read_coefficients_batch(ispin, ikpt, active_bands)
                gvectors, gvec_wrapped = self.get_plane_waves_basis_idx(ikpt, grid_shape_tuple, expected_npw=coeffs_list.shape[1])
                rgvec = self.get_plane_waves_basis_cart_from_idx(gvectors, grid_shape_tuple)
                
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

        tau = self._symmetrize_3d_grid(tau)
        if return_charge_density: 
            return tau, self._symmetrize_3d_grid(rho)
        return tau
    
    def calculate_localization_function(self, grid_shape=None, energy_range=(-np.inf, np.inf), spin_channel=-1, localization_function="elf", use_partial_occ=True):
        """Calculates specific localized electron topological indicator fields maps."""
        if grid_shape is None: 
            grid_shape = self._minimum_fft_size * 2
        tau, rho = self.calculate_kinetic_energy_density(grid_shape=grid_shape, energy_range=energy_range, spin_channel=spin_channel, use_partial_occ=use_partial_occ, return_charge_density=True)
        if localization_function == "lol":
            from baderkit.post_wfc.localization_functions import lol
            return lol(rho, tau, spin_channel != -1)
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
            return elf(rho, tau, lap_rho, grad_sq, spin_channel != -1)
    
    def get_density_of_states(self, spin=False, e_range=None, num_points=2000, sigma=0.05):
        """Constructs an integrated energy coordinate profile outlining Electronic Density of States."""
        kpt_weights = self.kpoint_weights / np.sum(self.kpoint_weights)
        bands = self.energies - self.efermi
        e_min, e_max = e_range if e_range is not None else self.energy_range
        energy_grid = np.linspace(e_min, e_max, num_points)
        if spin:
            b_alpha = bands[0].ravel() if bands.shape[0] > 1 else bands[0].ravel()
            b_beta = bands[1].ravel() if bands.shape[0] > 1 else bands[0].ravel()
            w_s = (np.ones((self.nkpoints, bands.shape[2])) * kpt_weights[:, None]).ravel()
            dos_a = np.dot(np.exp(-0.5 * ((energy_grid[:, None] - b_alpha[None, :]) / sigma)**2) / (sigma * np.sqrt(2 * np.pi)), w_s)
            dos_b = np.dot(np.exp(-0.5 * ((energy_grid[:, None] - b_beta[None, :]) / sigma)**2) / (sigma * np.sqrt(2 * np.pi)), w_s)
            return energy_grid, dos_a, dos_b
        else:
            w_t = (np.ones_like(bands) * kpt_weights[None, :, None]).ravel()
            dos = np.dot(np.exp(-0.5 * ((energy_grid[:, None] - bands.ravel()[None, :]) / sigma)**2) / (sigma * np.sqrt(2 * np.pi)), w_t)
            return energy_grid, dos
    
    def get_electrons_in_energy_range(self, e_min, e_max, num_points=2000, sigma=0.05):
        """Integrates occupied DOS profiles across designated energy limits windows."""
        kpt_weights = self.kpoint_weights / np.sum(self.kpoint_weights)
        bands = self.energies - self.efermi
        if e_min == -np.inf: e_min = np.min(bands) - (5 * sigma)
        if e_max == np.inf: e_max = np.max(bands) + (5 * sigma)
        w_occ = (self.occupancies * kpt_weights[None, :, None]).ravel()
        egrid = np.linspace(e_min, e_max, num_points)
        dens = np.dot(np.exp(-0.5 * ((egrid[:, None] - bands.ravel()[None, :]) / sigma)**2) / (sigma * np.sqrt(2 * np.pi)), w_occ)
        return trapezoid(dens, egrid) * (2 if self.nspin == 1 else 1)
    
    def find_energy_for_electron_count(self, target_electrons, e_min=-np.inf, assume_full_occupancy=False, num_points=5000, sigma=0.05):
        """Identifies relative cutoff boundaries enclosing specified target core electron quantities values."""
        egrid, t_dos = self.get_density_of_states(spin=False, num_points=num_points, sigma=sigma)
        if assume_full_occupancy:
            density = t_dos * np.max(self.occupancies)
        else:
            kpt_weights = self.kpoint_weights / np.sum(self.kpoint_weights)
            bands = self.energies - self.efermi
            w_occ = (self.occupancies * kpt_weights[None, :, None]).ravel()
            density = np.dot(np.exp(-0.5 * ((egrid[:, None] - bands.ravel()[None, :]) / sigma)**2) / (sigma * np.sqrt(2 * np.pi)), w_occ)
        if e_min == -np.inf: e_min = egrid[0]
        mask = egrid >= e_min
        sub_g, sub_d = egrid[mask], density[mask]
        cum_charge = np.zeros(len(sub_g))
        cum_charge[1:] = cumulative_trapezoid(sub_d, sub_g) * (2 if self.nspin == 1 else 1)
        if target_electrons > cum_charge[-1]: 
            raise ValueError("Target exceeds grid capacity.")
        return np.interp(target_electrons, cum_charge, sub_g)
    
    def _symmetrize_3d_grid(self, field):
        """Averages property grid profiles over space-group point symmetry operators."""
        symmetry = self.structure.symmetry_data
        Nx, Ny, Nz = field.shape
        sym_field = np.zeros_like(field)
        mx, my, mz = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing='ij')
        coords = np.stack([mx.ravel() / Nx, my.ravel() / Ny, mz.ravel() / Nz], axis=0)
        for R, t in zip(symmetry.rotations, symmetry.translations):
            R_inv = np.round(np.linalg.inv(R)).astype(int)
            tc = ((R_inv @ coords - (R_inv @ t)[:, np.newaxis]) % 1.0)
            sym_field += field[np.round(tc[0]*Nx).astype(int)%Nx, np.round(tc[1]*Ny).astype(int)%Ny, np.round(tc[2]*Nz).astype(int)%Nz].reshape(Nx, Ny, Nz)
        return sym_field / len(symmetry.rotations)
    
    @classmethod
    def from_directory(cls, directory: Path | str = Path("."), fmt: str = "vasp", scipy_workers: int = -1, **kwargs):
        """Constructs and binds specific execution parameters configurations profiles handles."""
        if fmt == "vasp": 
            from baderkit.post_wfc.wf_parsers import VaspParser as Parser
        elif fmt == "qe": 
            from baderkit.post_wfc.wf_parsers import QeParser as Parser
        else: 
            raise ValueError(f"Unknown format profile template string keyword: {fmt}")
        return cls(Parser(directory=Path(directory), **kwargs), scipy_workers=scipy_workers)