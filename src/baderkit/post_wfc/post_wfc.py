# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
from scipy.fft import fftn, ifftn
from scipy.fft import set_workers

from baderkit.post_wfc.wf_parsers.codes import DftMethod

BOHR_TO_ANG    = 0.529177249
# HSQDTM    =  hbar**2/(2*ELECTRON MASS)
HSQDTM = 3.8100198740807945

class PostWFC:
    def __init__(
        self,
        structure,
        kpoints,
        occupancies,
        energies,
        energy_cutoff,
        dft_code,
        dft_kwargs,
        efermi=None,
        scipy_workers=-1,
        ):
        
        self._nspin = energies.shape[0]
        self._nkpoints = energies.shape[1]
        self._nbands = energies.shape[2]
        
        self._kpoints = kpoints
        self._occupancies = occupancies
        self._energies = energies
        self._energy_cutoff = energy_cutoff
        
        self._structure = structure
        self._lattice = structure.lattice.matrix
        self._reciprocal_lattice = np.linalg.inv(structure.lattice.matrix).T
        self._dft_code = DftMethod(dft_code)
        self._dft_kwargs = dft_kwargs
        
        self.scipy_workers = scipy_workers
        
        # FIX: Corrected formula dependency grouping for the maximum FFT grid cutoff
        lattice_norm = np.linalg.norm(self._lattice, axis=1)
        CUTOFF = np.ceil(np.sqrt(energy_cutoff / HSQDTM) / (2 * np.pi / lattice_norm))
        self._minimum_fft_size = np.array(2 * CUTOFF + 1, dtype=int)
        
        self._efermi = efermi
        
        self._kpoint_multiplicities = None
        self._kpoint_weights = None
        
        self._charge_density = None
        self._kinetic_energy_density = None
        self._elf = None
        self._lol = None
        self._elid = None
        
    @property
    def nspin(self):
        return self._nspin
    
    @property
    def nkpoints(self):
        return self._nkpoints
    
    @property
    def nbands(self):
        return self._nbands
    
    @property
    def kpoints(self):
        return self._kpoints
    
    @property
    def kpoints_cart(self):
        return np.dot(self.kpoints, 2*np.pi*self.reciprocal_lattice)
    
    @property
    def occupancies(self):
        return self._occupancies
    
    @property
    def energies(self):
        return self._energies
    
    @property
    def energy_cutoff(self):
        return self._energy_cutoff
    
    @property
    def structure(self):
        return self._structure
    
    @property
    def lattice(self):
        return self._lattice
    
    @property
    def reciprocal_lattice(self):
        return self._reciprocal_lattice
    
    @property
    def kpoint_multiplicities(self):
        if self._kpoint_multiplicities is None:
            # Get reciprocal lattice rotation matrices
            recp_symm_ops = self.structure.lattice.get_recp_symmetry_operation()
            recip_rotations = []
            for op in recp_symm_ops:
                R_dir = op.rotation_matrix
                R_recip = np.round(np.linalg.inv(R_dir).T).astype(int)
                
                if not any(np.array_equal(R_recip, R) for R in recip_rotations):
                    recip_rotations.append(R_recip)
                    
            # Time-reversal symmetry
            tr_rotations = [-R for R in recip_rotations]
            for R in tr_rotations:
                if not any(np.array_equal(R, existing_R) for existing_R in recip_rotations):
                    recip_rotations.append(R)
                        
            # Calculate multiplicities
            multiplicities = []
            for k in self.kpoints:
                k = np.array(k)
                equivalent_kpts = []
                
                for R in recip_rotations:
                    # Rotate k-point
                    k_rot = R @ k
                    
                    # Wrap to [0, 1) interval, rounding slightly first to catch boundary floats
                    k_wrapped = np.mod(np.round(k_rot, decimals=6), 1.0)
                    
                    # Check if this periodic image is already recorded in the star
                    is_duplicate = False
                    for eq_k in equivalent_kpts:
                        diff = k_wrapped - eq_k
                        diff = diff - np.round(diff)  # Account for periodic boundary conditions
                        if np.allclose(diff, 0, 1e-5):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        equivalent_kpts.append(k_wrapped)
                        
                multiplicities.append(len(equivalent_kpts))
                
            multiplicities = np.array(multiplicities)
            self._kpoint_multiplicities = multiplicities
        return self._kpoint_multiplicities
    
    @property
    def kpoint_weights(self):
        if self._kpoint_weights is None:
            self._kpoint_weights = self.kpoint_multiplicities / np.sum(self.kpoint_multiplicities)
        return self._kpoint_weights
    
    @property
    def efermi(self):
        if self._efermi is None:
            # get energies and occupancies
            energies = self.energies
            occupances = self.occupancies
            
            # in case occupancies are doubled for non-spin-polarized calcs
            max_occ = occupances.max() / 2
            
            # partially occupied energies
            occupied_energies = energies[occupances >= max_occ]
            
            assert len(occupied_energies), "No occupied orbitals found in system"
            
            # get last occupied orbital
            self._efermi = occupied_energies.max()

        return self._efermi
    
    def get_plane_waves_frac(self, grid_shape=None):
        """
        The plane waves in reciprocal space fractional coordinates.
        """
        if grid_shape is None:
            grid_shape = self._minimum_fft_size * 2
        
        fx = [ii if ii < grid_shape[0] // 2 + 1 else ii - grid_shape[0]
              for ii in range(grid_shape[0])]
        fy = [jj if jj < grid_shape[1] // 2 + 1 else jj - grid_shape[1]
              for jj in range(grid_shape[1])]
        fz = [kk if kk < grid_shape[2] // 2 + 1 else kk - grid_shape[2]
              for kk in range(grid_shape[2])]

        # plane-waves: Reciprocal coordinate
        # indexing = 'ij' so that outputs are of shape (ngrid[0], ngrid[1], ngrid[2])
        gz, gy, gx = np.meshgrid(fx, fy, fz, indexing='ij')

        return gx, gy, gz
    

    def plane_waves_cart(self, grid_shape=None):
        """
        The plane waves in reciprocal space cartesian coordinates.
        """
        gx, gy, gz = self.get_plane_waves_frac(grid_shape)
        
        reciprocal_lattice = self.reciprocal_lattice
        cx, cy, cz = np.tensordot(
            reciprocal_lattice * np.pi * 2, [gx, gy, gz], axes=(0, 0))

        return cx, cy, cz
    
    def get_plane_waves_basis(self, ikpt, grid_shape=None):
        """
        The plane waves that are within the energy cutoff at each kpoint
        """
        plane_waves_frac = np.array(self.get_plane_waves_frac(grid_shape)).reshape((3,-1)).T
        
        # calculate kinetic energy at this kpoint
        kvec = self.kpoints[ikpt]
        # Kinetic_Energy = (G + k)**2 / 2
        # HSQDTM    =  hbar**2/(2*ELECTRON MASS)
        KENERGY = HSQDTM * np.linalg.norm(
            np.dot(plane_waves_frac + kvec[np.newaxis, :], 2*np.pi*self.reciprocal_lattice), axis=1
        )**2
        # find Gvectors where (G + k)**2 / 2 < ENCUT
        return plane_waves_frac[np.where(KENERGY < self.energy_cutoff)[0]]
    
    def get_plane_waves_basis_cart(self, ikpt, grid_shape=None):
        """
        The plane waves that are within the energy cutoff at each kpoint.
        Returns array of shape (N_basis, 3).
        """
        plane_waves_frac = np.array(self.get_plane_waves_frac(grid_shape)).reshape((3,-1)).T
        
        # calculate kinetic energy at this kpoint
        kvec = self.kpoints[ikpt]
        KENERGY = HSQDTM * np.linalg.norm(
            np.dot(plane_waves_frac + kvec[np.newaxis, :], 2*np.pi*self.reciprocal_lattice), axis=1
        )**2
        
        # FIX: Directly compute dot product to naturally return a (N_basis, 3) matrix shape
        # instead of a (3, N_basis) shape, preventing broadcast ValueError crashes.
        gvec_frac = plane_waves_frac[np.where(KENERGY < self.energy_cutoff)[0]]
        return np.dot(gvec_frac, 2 * np.pi * self.reciprocal_lattice)
    
    def get_plane_wave_coefficients(self, ispin, ikpt, iband):
        return self._dft_code.coeff_reader(
            ispin=ispin,
            ikpt=ikpt,
            iband=iband,
            **self._dft_kwargs
            )
    
    def get_pseudo_wavefunction(
            self, 
            ispin=0, 
            ikpt=0, 
            iband=0,
            grid_shape=None,
            kr_phase=False,
            coeffs=None,
            ):
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
        
        # By default, the WAVECAR only stores the periodic part of the Bloch
        # wavefunction. In order to get the full Bloch wavefunction, one need to
        # multiply the periodic part with the phase: exp(i k (r + r0). Below, the
        # k-point vector and the real-space grid are both in the direct
        # coordinates.
        if kr_phase:
            phase = np.exp(1j * np.pi * 2 *
                           np.sum(
                               self._kpoints[ikpt] * # FIX: Adjusted from missing `_kvecs` to `_kpoints`
                               (
                                   # r
                                   np.mgrid[
                                       0:grid_shape[0], 0:grid_shape[1], 0:grid_shape[2]
                                   ].reshape((3, np.prod(grid_shape))).T /
                                   grid_shape.astype(float) #+
                                   # r0
                                   # np.array(r0, dtype=float)
                               ),
                               axis=1
                           )).reshape(grid_shape)
        else:
            phase = 1.0

        # default normalization factor so that
        # \sum_{ijk} | \phi_{ijk} | ^ 2 = 1
        normFac = np.sqrt(np.prod(grid_shape))

        gvec = self.get_plane_waves_basis(ikpt, grid_shape)

        phi_k = np.zeros(grid_shape, dtype=np.complex128)
        grid_shape = np.asarray(grid_shape)
        gvec %= grid_shape[np.newaxis, :]

        if coeffs is None:
            coeffs = self.get_plane_wave_coefficients(ispin, ikpt, iband)
            
        phi_k[gvec[:, 0], gvec[:, 1], gvec[:, 2]
              ] = coeffs

        # perform complex2complex FFT
        with set_workers(self.scipy_workers):
            pseudo_wfs = ifftn(phi_k * normFac) * phase
        return pseudo_wfs
    
    def calculate_charge_density(
            self, 
            grid_shape = None,
            spin_channel = -1, # -1 is total, otherwise index
            energy_range=(-np.inf, np.inf),
            use_partial_occ = True,
            ):
        # get grid shape
        if grid_shape is None:
            grid_shape = self._minimum_fft_size * 2
        
        # normalization factor so that
        # \sum_{ijk} | \phi_{ijk} | ^ 2 * volume / Ngrid = 1
        normFac = np.sqrt(np.prod(grid_shape) / self.structure.volume)

        kpoint_weights = self.kpoint_weights

        # Charge density grid initialization
        rho = np.zeros(grid_shape, dtype=complex)
        
        if spin_channel == -1:
            spin_indices = [i for i in range(self.nspin)]
        else:
            spin_indices = [spin_channel]

        # Loop over spins
        for ispin in spin_indices:
            # FIX: Total array accumulation initialization (rho[:] = 0.0) removed from here 
            # to prevent subsequent spin channels from overwriting earlier ones.

            # Loop over k-points
            for ikpt in range(self.nkpoints):

                # Loop over bands
                for iband in range(self.nbands):
                    # omit the bands outside our range
                    abs_energy = self.energies[ispin, ikpt, iband]
                    rel_energy = abs_energy - self.efermi
                    in_window = (energy_range[0] <= rel_energy <= energy_range[1])
                    
                    if not in_window:
                        continue
                    
                    # FIX: Always force rspin = 2.0 when nspin == 1. This accurately accounts
                    # for VASP WAVECAR storing occupancies capped at 1.0 for non-spin-polarized cells.
                    rspin = 2.0 if self.nspin == 1 else 1.0
                    if use_partial_occ:
                        weight = rspin * kpoint_weights[ikpt] * self.occupancies[ispin, ikpt, iband]
                    else:
                        weight = rspin * kpoint_weights[ikpt]

                    # wavefunction in reciprocal space
                    phi_q = self.get_plane_wave_coefficients(ispin, ikpt, iband)

                    # wavefunction in real space
                    
                    phi_r = self.get_pseudo_wavefunction(
                        grid_shape=grid_shape,
                        ispin=ispin,
                        ikpt=ikpt,
                        iband=iband,
                        coeffs=phi_q
                        ) * normFac

                    # charge density in real space
                    rho += phi_r.conj() * phi_r * weight
                    
        rho = rho.real
        rho = self._symmetrize_3d_grid(rho)
        return rho
    
    def calculate_laplacian(
        self,
        data,
        is_reciprocal=False,
            ):
        # FIX: Changed from uncallable method attribute `.T` to direct function call and tuple unpacking
        Gx, Gy, Gz = self.plane_waves_cart()
        # the norm squared of the G-vectors
        G2 = Gx**2 + Gy**2 + Gz**2
        
        if not is_reciprocal:
            # Convert data to reciprocal space
            with set_workers(self.scipy_workers):
                recip_data = fftn(data, norm='ortho')
        else:
            recip_data = data
        # Calculate laplacian in reciprocal space
        recip_lap = -G2 * recip_data
        # Convert back to real space
        with set_workers(self.scipy_workers):
            real_lap = ifftn(recip_lap, norm='ortho')
        
        return real_lap.real
    
    def calculate_gradient(
        self,
        data,
        is_reciprocal=False,
            ):
        
        if not is_reciprocal:
            # Convert data to reciprocal space
            with set_workers(self.scipy_workers):
                recip_data = fftn(data, norm='ortho')
        else:
            recip_data = data

        # charge density gradient: grad rho
        ########################################
        # correct method for gradient using FFT
        ########################################
        # FIX: Changed from uncallable method attribute `.T` to direct function call and tuple unpacking
        Gx, Gy, Gz = self.plane_waves_cart()
        with set_workers(self.scipy_workers):
            grad_rho_x = ifftn(1j * Gx * recip_data, norm='ortho')
            grad_rho_y = ifftn(1j * Gy * recip_data, norm='ortho')
            grad_rho_z = ifftn(1j * Gz * recip_data, norm='ortho')
        return grad_rho_x.real, grad_rho_y.real, grad_rho_z.real
    
    def calculate_kinetic_energy_density(
            self, 
            grid_shape = None, 
            spin_channel = -1,
            energy_range=(-np.inf, np.inf),
            use_partial_occ = True,
            return_charge_density = False,
            ):
        
        # get grid shape
        if grid_shape is None:
            grid_shape = self._minimum_fft_size * 2

        # normalization factor so that
        # \sum_{ijk} | \phi_{ijk} | ^ 2 * volume / Ngrid = 1
        normFac = np.sqrt(np.prod(grid_shape) / self.structure.volume)

        kpoint_weights = self.kpoint_weights
        
        # Density array initializations
        tau = np.zeros(grid_shape, dtype=complex)
        if return_charge_density:
            rho = np.zeros(grid_shape, dtype=complex)
            
        if spin_channel == -1:
            spin_indices = [i for i in range(self.nspin)]
        else:
            spin_indices = [spin_channel]

        # Loop over spin channels
        for ispin in spin_indices:
            # FIX: Corrected loop structure to properly iterate over ALL available k-points 
            # instead of mistakenly looping over spin_indices twice.
            for ikpt in range(self.nkpoints):
                
                # plane-wave G-vectors (now correctly shaped as N_basis x 3)
                rgvec = self.get_plane_waves_basis_cart(ikpt, grid_shape)
                
                k = self.kpoints_cart[ikpt]             # k
                gk = rgvec + k[np.newaxis, :]           # G + k
                gk2 = np.linalg.norm(gk, axis=1)**2     # | G + k |^2

                # Loop over bands
                for iband in range(self.nbands):
                    # omit the bands outside our range
                    abs_energy = self.energies[ispin, ikpt, iband]
                    rel_energy = abs_energy - self.efermi
                    in_window = (energy_range[0] <= rel_energy <= energy_range[1])
                    
                    if not in_window:
                        continue

                    # FIX: Multiplied by rspin=2.0 under nspin=1 calculations to handle the 
                    # binary WAVECAR fractional weight cap, guaranteeing proper electron count integrations.
                    rspin = 2.0 if self.nspin == 1 else 1.0
                    if use_partial_occ:
                        weight = rspin * kpoint_weights[ikpt] * self.occupancies[ispin, ikpt, iband]
                    else:
                        weight = rspin * kpoint_weights[ikpt]
                        
                    if weight == 0:
                        continue

                    # wavefunction in reciprocal space
                    phi_q = self.get_plane_wave_coefficients(ispin, ikpt, iband)

                    # wavefunction in real space
                    phi_r = self.get_pseudo_wavefunction(
                        grid_shape=grid_shape,
                        ispin=ispin,
                        ikpt=ikpt,
                        iband=iband,
                        coeffs=phi_q
                        ) * normFac

                    # grad^2 \phi in reciprocal space
                    lap_phi_q = -gk2 * phi_q
                    # grad^2 \phi in real space
                    lap_phi_r = self.get_pseudo_wavefunction(
                        grid_shape=grid_shape,
                        ispin=ispin,
                        ikpt=ikpt,
                        iband=iband,
                        coeffs=lap_phi_q
                        ) * normFac
                    
                    tau += (-phi_r * lap_phi_r.conj()) * weight

                    # charge density in real space
                    if return_charge_density:
                        rho += phi_r.conj() * phi_r * weight

        # symmetry averaging
        tau = tau.real
        tau = self._symmetrize_3d_grid(tau)
        if return_charge_density:
            rho = rho.real
            rho = self._symmetrize_3d_grid(rho)
            return tau, rho
            
        return tau
    
    def calculate_localization_function(
            self,
            grid_shape = None, 
            energy_range=(-np.inf, np.inf),
            spin_channel=-1,
            localization_function="elf",
            use_partial_occ = True,
            ):
        # get grid shape
        if grid_shape is None:
            grid_shape = self._minimum_fft_size * 2
            
        # get total charge density and tau
        tau, rho = self.calculate_kinetic_energy_density(
            grid_shape=grid_shape,
            energy_range=(-np.inf, np.inf),
            spin_channel=-1,
            use_partial_occ = use_partial_occ,
            return_charge_density = True,
            )
        
        # check for partial density request
        is_total = (energy_range[0] == -np.inf) and (energy_range[1] == np.inf)
        
        if not is_total:
            partial_rho = self.calculate_charge_density(
                grid_shape=grid_shape,
                energy_range=energy_range,
                spin_channel=spin_channel,
                use_partial_occ = use_partial_occ,
                )
            
        # get rho in reciprocal space
        with set_workers(self.scipy_workers):
            rho_q = fftn(rho, norm='ortho')
        # get gradient and laplacian (.real applied to discard vanishing complex numeric noise)
        lap_rho = self.calculate_laplacian(rho_q, is_reciprocal=True).real
        grad_rho_x, grad_rho_y, grad_rho_z = self.calculate_gradient(rho_q, is_reciprocal=True)
        
        # get grad2 (using absolute values to keep the array strictly real-valued)
        grad_rho_sq = np.abs(grad_rho_x)**2 + np.abs(grad_rho_y)**2 + np.abs(grad_rho_z)**2
        
        # calculate localization function baseline factors
        prefactor = 3./5 * (3.0 * np.pi**2)**(2./3)
        D0 = np.where(rho > 0.0, prefactor * rho**(5./3), 0.0)
        eps = 1E-8
        D0[D0 < eps] = eps
        
        # Normalize function string format to accept case-insensitive 'elid' or 'eli-d'
        loc_func = localization_function.lower().replace("-", "")
        
        if loc_func == "elf":
            # Calculate chi: D0 = T + TCORR - TBOS
            rho_safe = np.where(rho > 1e-14, rho, 1e-14)
            D = tau + 0.5 * lap_rho - 0.25 * grad_rho_sq / rho_safe
            # calculate ELF kernel, including Savin's shifting constant
            X = (D + 2.871e-5) / D0
        elif loc_func == "lol":
            X = tau / D0
        elif loc_func == "elid":
            # ELI-D formula: D * rho^(-8/3)
            rho_safe = np.where(rho > 1e-14, rho, 1e-14)
            D = tau + 0.5 * lap_rho - 0.25 * grad_rho_sq / rho_safe
            X = D * (rho_safe ** (-8. / 3.))
        else:
            raise ValueError(f"Localization Function {localization_function} is not implemented.")
        
        # calculate partial charge ratio adjustments
        if not is_total:
            
            partial_sum = partial_rho.sum()
            if partial_sum < 1e-14:
                raise ValueError("No populated states found in energy range")
            
            nonzero_mask = rho > eps
            x = np.zeros_like(rho)
            x[nonzero_mask] = partial_rho[nonzero_mask] / rho[nonzero_mask]
            x = np.clip(x, 0.0, 1.0)
            
            factor = x * (rho.sum() / partial_sum)
            if loc_func == "elid":
                # Because ELI-D scales proportionally with localized states,
                # partial ELI-D scales linearly with the orbital fraction window
                kerx = X * factor
            else:
                # ELF and LOL kernels are inversely proportional to localizability
                kerx = np.where(x > 0.0, X / factor, np.inf)
        else:
            kerx = X
            
        # Map to the final descriptor values
        if loc_func == "elf":
            result = 1 / (1 + kerx**2)
        elif loc_func == "lol":
            result = 1 / (1 + kerx)
        elif loc_func == "elid":
            # ELI-D maps directly to the raw kernel value
            result = kerx
            
        return result
        
    
    def _symmetrize_3d_grid(self, field):
        """Vectorized real-space grid symmetrization via inverse coordinate mapping."""
        # get rotations/translations
        symmetry = self.structure.symmetry_data
        rotations = symmetry.rotations
        translations = symmetry.translations
        
        Nx, Ny, Nz = field.shape
        sym_field = np.zeros_like(field)
        
        x_grid = np.arange(Nx)
        y_grid = np.arange(Ny)
        z_grid = np.arange(Nz)
        
        mx, my, mz = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
        coords_frac = np.stack([mx.ravel() / Nx, my.ravel() / Ny, mz.ravel() / Nz], axis=0)
        
        num_ops = len(rotations)
        
        for R, t in zip(rotations, translations):
            R_inv = np.round(np.linalg.inv(R)).astype(int)
            t_inv = R_inv @ t
            
            trans_coords = R_inv @ coords_frac - t_inv[:, np.newaxis]
            trans_coords = trans_coords % 1.0
            
            idx_x = np.round(trans_coords[0] * Nx).astype(int) % Nx
            idx_y = np.round(trans_coords[1] * Ny).astype(int) % Ny
            idx_z = np.round(trans_coords[2] * Nz).astype(int) % Nz
            
            sym_field += field[idx_x, idx_y, idx_z].reshape(Nx, Ny, Nz)
            
        return sym_field / num_ops
    
    @classmethod
    def from_vasp(
        cls,
        poscar_filename: Path | str = "POSCAR",
        wavecar_filename: Path | str = "WAVECAR",
        scipy_workers: int = -1,
        use_vasp4: bool = False,
            ):
        
        poscar_filename = Path(poscar_filename)
        wavecar_filename = Path(wavecar_filename)
        
        vasp = DftMethod("vasp")
        
        structure, kpoints, occs, bands, encut, efermi, vasp_dict = vasp.wf_reader(
            poscar_filename=poscar_filename,
            wavecar_filename=wavecar_filename,
            use_vasp4=use_vasp4
            )
        return cls(
            structure=structure,
            kpoints=kpoints,
            occupancies=occs,
            energies=bands,
            energy_cutoff=encut,
            dft_code=vasp,
            dft_kwargs=vasp_dict,
            efermi=efermi,
            scipy_workers=scipy_workers,
            )