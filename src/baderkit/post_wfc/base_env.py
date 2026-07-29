# -*- coding: utf-8 -*-

from pathlib import Path
import logging
import psutil
import h5py
from rich.progress import track
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid
from scipy.fft import fftn, ifftn, set_workers

from baderkit.post_wfc.wf_readers.base import HSQDTM

from baderkit.post_wfc.wfc_numba import (
    integrate_tetrahedra_spectral_density, 
    integrate_tetrahedra_analytic_charge,
    )

@dataclass
class PAWPatch:
    """Stores precomputed geometric masks and local partial wave matrices for an atomic sphere."""

    atom_idx: int
    mask: np.ndarray  # Boolean 1D mask array over flat real-space FFT grid
    start_ch: int
    end_ch: int
    ae_mat: np.ndarray  # Shape: (N_local, n_proj_a)
    ps_mat: np.ndarray  # Shape: (N_local, n_proj_a)
    grad_ae_mat: np.ndarray | None  # Shape: (3, N_local, n_proj_a)
    grad_ps_mat: np.ndarray | None  # Shape: (3, N_local, n_proj_a)

class PostWFC:
    """
    Abstract baseline context managing shared structural parameters, k-point meshes,
    and cell symmetries. Supports master-companion delegation to share identical
    memory spaces between active representations.
    """
    def __init__(
        self, 
        directory=Path("."),
        fmt="vasp",
        smearing="tet",
        sigma=None,
        **kwargs
        ):
        """Initializes state or points straight to a companion reference environment."""
            
        self.directory = Path(directory)
        # Make sure post_wfc.h5 exists for this calculation
        self._postwfc_file = self.directory / "post_wfc.h5"
        wf_reader = self._get_reader(
            directory=directory,
            fmt=fmt,
            postwfc_file=self._postwfc_file,
            **kwargs
            )
        
        self._spectral_resolution = 100
        
        self._meta = wf_reader.metadata
        self._paw_datasets = wf_reader.paw_datasets
        
        self._smearing, self._sigma = self._get_default_sigma(smearing, sigma)
        
        self._lattice = self._meta.structure.lattice.matrix
        self._reciprocal_lattice = 2 * np.pi * np.linalg.inv(self._lattice).T   
        
        self._meta.energies = self._meta.energies - self._meta.efermi
        self._meta.efermi = 0.0
        
        lattice_norm = np.linalg.norm(self._lattice, axis=1)
        CUTOFF = np.ceil(np.sqrt(self._meta.energy_cutoff / HSQDTM) / (2 * np.pi / lattice_norm))
        self._minimum_fft_shape = np.array(2 * CUTOFF + 1, dtype=int)
        
    ###########################################################################
    # Base Metadata about System
    ###########################################################################
    
    @property
    def paw_datasets(self):
        return self._paw_datasets
    
    @property
    def structure(self):
        return self._meta.structure
        
    @property
    def lattice(self):
        return self._lattice
        
    @property
    def reciprocal_lattice(self):
        return self._reciprocal_lattice

    @property
    def valence_counts(self):
        """The number of valence electrons assigned to each species"""
        valence_counts = {}
        for element, dataset in self.paw_datasets.items():
            valence_counts[element] = dataset.Z
        return valence_counts

    @property
    def nspin(self) -> int:
        return self._meta.nspin
    
    @property
    def nkpoints(self) -> int:
        return self._meta.nkpts
        
    @property
    def nbands(self) -> int:
        return self._meta.nbands
        
    @property
    def occupancies(self):
        return self._meta.occupancies
        
    @property
    def energies(self):
        return self._meta.energies
    
    @property
    def energy_cutoff(self):
        return self._meta.energy_cutoff
    
    @property
    def efermi(self) -> float:
        return self._meta.efermi
    
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
            dE = (e_max - e_min) / self._spectral_resolution
            
            if self._smearing == "none":
                pad = 0.5 * dE
            elif self._smearing in ["gaussian", "methfessel-paxton", "mp"]:
                pad = 6.5 * self._sigma
            elif self._smearing in ["fermi-dirac", "fd"]:
                pad = 14.0 * self._sigma
            elif self._smearing == "tetrahedron":
                # FIXED: Reduced to exactly 14.0 * self._sigma to perfectly align the boundaries
                # with the Fermi-Dirac kernel decay envelope, removing the large unnecessary zero buffers.
                pad = 14.0 * self._sigma if self._sigma > 0.0 else 0.0
            else:
                pad = 5.0 * self._sigma
                
            self._energy_range = e_min - pad, e_max + pad
        return self._energy_range
    
    @property
    def _num_spectral_points(self):
        return len(self.energy_grid)
    
    @property
    def energy_grid(self):
        if getattr(self, "_energy_grid", None) is None:
            emin, emax = self.energy_range
            diff = emax-emin
            num_points = int(diff * self._spectral_resolution) # 100 pts per ev
            self._energy_grid = np.linspace(emin, emax, num_points)
        return self._energy_grid
            
    @property
    def unsmeared_energy_range(self) -> tuple[float, float]:
        if getattr(self, "_unsmeared_energy_range", None) is None:
            self._unsmeared_energy_range = np.min(self.energies), np.max(self.energies)
        return self._unsmeared_energy_range
    
    @property
    def smear_matrix(self) -> NDArray:
        if getattr(self, "_smear_matrix", None) is None:
            self._smear_matrix = self._get_smear_matrix()
        return self._smear_matrix
    
    ###########################################################################
    # KPOINTS
    ###########################################################################
    @property
    def kpoints(self):
        return self._meta.kpoints
    
    @property
    def kpoints_cart(self):
        return np.dot(self.kpoints, 2 * np.pi * self.reciprocal_lattice)
    
    @property
    def kpoint_multiplicities(self):
        if getattr(self, "_kpoint_multiplicities", None) is None:
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
        if getattr(self, "_kpoint_weights", None) is None:
            self._kpoint_weights = self.kpoint_multiplicities / np.sum(self.kpoint_multiplicities)
        return self._kpoint_weights
        
    @property
    def kpoints_full(self):
        if getattr(self, "_kpoints_full", None) is None:
            self._unfold_brillouin_zone()
        return self._kpoints_full

    @property
    def kpoints_cart_full(self):
        if getattr(self, "_kpoints_cart_full", None) is None:
            self._kpoints_cart_full = np.dot(self.kpoints_full, 2 * np.pi * self.reciprocal_lattice)
        return self._kpoints_cart_full
        
    @property
    def full_to_irr_map(self):
        if getattr(self, "_full_to_irr_map", None) is None:
            self._unfold_brillouin_zone()
        return self._full_to_irr_map
    
    @property
    def tetrahedra_indices(self):
        if getattr(self, "_tetrahedra_indices", None) is None:
            self._tetrahedra_indices = self._get_tetrahedra()
        return self._tetrahedra_indices
    
    ###########################################################################
    # FFT Methods
    ###########################################################################
    def get_fft_grid(self, grid_shape=None):
        if grid_shape is None: 
            grid_shape = self._minimum_fft_shape
        grid_shape = tuple(grid_shape)
            
        Nx, Ny, Nz = grid_shape
        fx = [ii if ii < Nx // 2 + 1 else ii - Nx for ii in range(Nx)]
        fy = [jj if jj < Ny // 2 + 1 else jj - Ny for jj in range(Ny)]
        fz = [kk if kk < Nz // 2 + 1 else kk - Nz for kk in range(Nz)]
        
        gx, gy, gz = np.meshgrid(fx, fy, fz, indexing='ij')
        return (gx, gy, gz)
        
    def get_fft_grid_cart(self, grid_shape=None):
        """Transforms integer fractional meshgrid coordinate axes into explicit Cartesian grid coordinates (in A^-1)."""
        gx, gy, gz = self.get_fft_grid(grid_shape)
        cx, cy, cz = np.tensordot(
            self.reciprocal_lattice * np.pi * 2, [gx, gy, gz], axes=(0, 0))
        return cx, cy, cz
        
    ###########################################################################
    # Fetch Methods
    ###########################################################################
    
    def fetch_psi(
        self,
        ikpt: int | list[int] | np.ndarray,
        ispin: int | list[int] | np.ndarray = 0,
        iband: int | list[int] | np.ndarray = 0,
        order: int = 0,
    ) -> np.ndarray | list[np.ndarray]:
        """Retrieves pseudo wavefunctions or derivatives from the saved HDF5 file.
    
        Parameters
        ----------
        ikpt : int, list, or np.ndarray
            k-point index or array of indices.
        ispin : int, list, or np.ndarray, optional
            Spin index or array of indices. Defaults to 0.
        iband : int, list, or np.ndarray, optional
            Band index or array of indices. Defaults to 0.
        order : int, optional
            Derivative order: 0 for psi, 1 for gradient, 2 for laplacian. Defaults to 0.
        pseudo : bool, optional
            Must be True. Reciprocal all-electron quantities are not stored to avoid
            Fourier-space truncation errors. Defaults to True.
    
        Returns
        -------
        np.ndarray or list of np.ndarray
            If ikpt is a scalar, returns a single NumPy array for that k-point.
            If ikpt is a list/array, returns a list of NumPy arrays (one per k-point).
        """
        if not self._postwfc_file.exists():
            raise FileNotFoundError(f"HDF5 cache file not found: {self._postwfc_file}")
    
        # Validate order parameter
        if order not in (0, 1, 2):
            raise ValueError(
                f"Invalid order {order}. Must be 0 (psi), 1 (gradient), or 2 (laplacian)."
            )
    
        order_map = {0: "psi", 1: "grad", 2: "lap"}
        subgroup = order_map[order]
        prefix = "ps"
    
        # Helper function for explicit bounds checking
        def _validate_indices(idx, max_val, name):
            arr = np.asarray(idx)
            if np.any(arr < 0) or np.any(arr >= max_val):
                raise IndexError(
                    f"{name} index out of bounds. Requested: {idx}, valid range: [0, {max_val - 1}]"
                )
    
        # Validate inputs against system dimensions
        _validate_indices(ikpt, self.nkpoints, "k-point")
        _validate_indices(ispin, self.nspin, "Spin")
        _validate_indices(iband, self.nbands, "Band")
    
        # Flag whether ikpt was originally passed as a scalar
        is_scalar_kpt = np.ndim(ikpt) == 0
    
        # Standardize inputs to 1D arrays or slices for indexing
        k_indices = np.atleast_1d(ikpt)
    
        results = []
    
        with h5py.File(self._postwfc_file, "r") as file:
            for k in k_indices:
                dataset_path = f"{prefix}/{subgroup}/{k}"
    
                if dataset_path not in file:
                    raise KeyError(f"Dataset '{dataset_path}' not found in HDF5 file.")
    
                dset = file[dataset_path]
                if order == 1:
                    data = dset[ispin, :, iband]
                else:
                    data = dset[ispin, iband]
                results.append(data)
    
        return results[0] if is_scalar_kpt else results
    
    
    def fetch_projector_overlaps(
        self,
        ikpt: int | list[int] | np.ndarray,
        ispin: int | list[int] | np.ndarray = 0,
        iband: int | list[int] | np.ndarray = 0,
    ) -> np.ndarray | list[np.ndarray]:
        """Retrieves precomputed PAW projector overlap scalar coefficients P_a = <p_a | psi_ps>
    
        from the saved HDF5 file.
    
        Parameters
        ----------
        ikpt : int, list, or np.ndarray
            k-point index or array of indices.
        ispin : int, list, or np.ndarray, optional
            Spin index or array of indices. Defaults to 0.
        iband : int, list, or np.ndarray, optional
            Band index or array of indices. Defaults to 0.
    
        Returns
        -------
        np.ndarray or list of np.ndarray
            If ikpt is a scalar, returns array of shape (..., total_n_proj).
            If ikpt is a list/array, returns a list of NumPy arrays (one per k-point).
        """
        if not self._postwfc_file.exists():
            raise FileNotFoundError(f"HDF5 cache file not found: {self._postwfc_file}")
    
        def _validate_indices(idx, max_val, name):
            arr = np.asarray(idx)
            if np.any(arr < 0) or np.any(arr >= max_val):
                raise IndexError(
                    f"{name} index out of bounds. Requested: {idx}, valid range: [0, {max_val - 1}]"
                )
    
        _validate_indices(ikpt, self.nkpoints, "k-point")
        _validate_indices(ispin, self.nspin, "Spin")
        _validate_indices(iband, self.nbands, "Band")
    
        is_scalar_kpt = np.ndim(ikpt) == 0
        k_indices = np.atleast_1d(ikpt)
    
        results = []
    
        with h5py.File(self._postwfc_file, "r") as file:
            for k in k_indices:
                dataset_path = f"projector_overlaps/{k}"
    
                if dataset_path not in file:
                    raise KeyError(f"Dataset '{dataset_path}' not found in HDF5 file.")
    
                dset = file[dataset_path]
                data = dset[ispin, iband]
                results.append(data)
    
        return results[0] if is_scalar_kpt else results
    
    
    def fetch_paw_coefficients(
        self,
        ikpt: int | list[int] | np.ndarray,
        ispin: int | list[int] | np.ndarray = 0,
        iband: int | list[int] | np.ndarray = 0,
    ) -> np.ndarray | list[np.ndarray]:
        """Retrieves raw plane-wave expansion coefficients from the saved HDF5 file.
    
        Parameters
        ----------
        ikpt : int, list, or np.ndarray
            k-point index or array of indices.
        ispin : int, list, or np.ndarray, optional
            Spin index or array of indices. Defaults to 0.
        iband : int, list, or np.ndarray, optional
            Band index or array of indices. Defaults to 0.
    
        Returns
        -------
        np.ndarray or list of np.ndarray
            If ikpt is a scalar, returns a single NumPy array for that k-point.
            If ikpt is a list/array, returns a list of NumPy arrays (one per k-point).
        """
        if not self._postwfc_file.exists():
            raise FileNotFoundError(f"HDF5 cache file not found: {self._postwfc_file}")
    
        def _validate_indices(idx, max_val, name):
            arr = np.asarray(idx)
            if np.any(arr < 0) or np.any(arr >= max_val):
                raise IndexError(
                    f"{name} index out of bounds. Requested: {idx}, valid range: [0, {max_val - 1}]"
                )
    
        _validate_indices(ikpt, self.nkpoints, "k-point")
        _validate_indices(ispin, self.nspin, "Spin")
        _validate_indices(iband, self.nbands, "Band")
    
        is_scalar_kpt = np.ndim(ikpt) == 0
        k_indices = np.atleast_1d(ikpt)
    
        results = []
    
        with h5py.File(self._postwfc_file, "r") as file:
            for k in k_indices:
                dataset_path = f"coefficients/{k}"
    
                if dataset_path not in file:
                    raise KeyError(f"Dataset '{dataset_path}' not found in HDF5 file.")
    
                dset = file[dataset_path]
    
                # Slice raw coefficients dataset of shape (nspin, nbands, ngvecs)
                data = dset[ispin, iband]
                results.append(data)
    
        return results[0] if is_scalar_kpt else results
    
    def fetch_gvectors(
        self,
        ikpt: int | list[int] | np.ndarray,
        return_cart: bool = False,
        add_k: bool = False,
    ) -> np.ndarray | list[np.ndarray]:
        """Retrieves reciprocal-space G-vectors from the saved HDF5 file.
    
        Parameters
        ----------
        ikpt : int, list, or np.ndarray
            k-point index or array of indices.
        return_cart : bool, optional
            If True, transforms Miller indices to Cartesian reciprocal coordinates (in Å⁻¹).
            If False, returns integer Miller indices (h, k, l). Defaults to False.
        add_k : bool, optional
            If True, adds the k-point wavevector to yield K = k + G. Defaults to False.
    
        Returns
        -------
        np.ndarray or list of np.ndarray
            If ikpt is a scalar, returns a single array of shape (ngvecs, 3).
            If ikpt is a list/array, returns a list of arrays (one per k-point).
            Data type is int32 if return_cart=False and add_k=False, otherwise float64.
        """
        if not self._postwfc_file.exists():
            raise FileNotFoundError(f"HDF5 cache file not found: {self._postwfc_file}")
    
        def _validate_indices(idx, max_val, name):
            arr = np.asarray(idx)
            if np.any(arr < 0) or np.any(arr >= max_val):
                raise IndexError(
                    f"{name} index out of bounds. Requested: {idx}, valid range: [0, {max_val - 1}]"
                )
    
        _validate_indices(ikpt, self.nkpoints, "k-point")
    
        is_scalar_kpt = np.ndim(ikpt) == 0
        k_indices = np.atleast_1d(ikpt)
    
        recip_matrix = self.reciprocal_lattice
        results = []
    
        with h5py.File(self._postwfc_file, "r") as file:
            for k in k_indices:
                dataset_path = f"gvectors/{k}"
    
                if dataset_path not in file:
                    raise KeyError(f"Dataset '{dataset_path}' not found in HDF5 file.")
    
                dset = file[dataset_path]
                g_int = dset[:]  # Integer Miller indices (ngvecs, 3)
    
                # 1. Coordinate transformation: integer Miller -> Cartesian
                if return_cart:
                    data = g_int @ recip_matrix  # Shape: (ngvecs, 3) in Å⁻¹
                else:
                    data = g_int.astype(np.float64) if add_k else g_int
    
                # 2. Add k-point wavevector K = k + G
                if add_k:
                    if return_cart:
                        k_vec = self.kpoints_cart[k]
                    else:
                        k_vec = self.kpoints[k]
    
                    data = data + k_vec
    
                results.append(data)
    
        return results[0] if is_scalar_kpt else results
    
    def fetch_density_matrix(
        self,
        spin_channel: int = -1,
        energy_range: tuple[float, float] = (-np.inf, np.inf),
        use_partial_occ: bool = True,
    ) -> np.ndarray:
        """Fetches or calculates the atomic density matrix D_{a, ij}.
    
        Returns shape: (num_atoms, max_n_proj, max_n_proj)
        """
        has_energy_filter = energy_range != (-np.inf, np.inf)
    
        # Use precalculated cached density matrix if evaluating full ground state
        if not has_energy_filter and self._postwfc_file.exists():
            with h5py.File(self._postwfc_file, "r") as file:
                if "density_matrices" in file:
                    D_cached = file["density_matrices"][:]
                    if spin_channel == -1:
                        return np.sum(D_cached, axis=0)
                    return D_cached[spin_channel]
    
        # Otherwise, compute D_{a, ij} dynamically for the requested subset
        spins = list(range(self.nspin)) if spin_channel == -1 else [spin_channel]
        spin_weight = 2 if (self.nspin == 1 or spin_channel != -1) else 1
        atom_offsets = self._get_atom_channel_offsets()
        num_atoms = len(self.structure)
        max_n_proj = max(end - start for start, end in atom_offsets)
    
        D_out = np.zeros((num_atoms, max_n_proj, max_n_proj), dtype=np.float64)
    
        e_min, e_max = energy_range
    
        for ikpt in range(self.nkpoints):
            k_weight = self.kpoint_weights[ikpt]
            for ispin in spins:
                energies = self.energies[ispin, ikpt, :]
                occupancies = self.occupancies[ispin, ikpt, :]
    
                mask = (energies >= e_min) & (energies <= e_max)
                if use_partial_occ:
                    mask = mask & (occupancies >= 1e-8)
    
                if not np.any(mask):
                    continue
    
                occ_vals = occupancies[mask] if use_partial_occ else np.ones(np.sum(mask))
                weights = k_weight * occ_vals * spin_weight
    
                P_all = self.fetch_projector_overlaps(ikpt=ikpt, ispin=ispin, iband=mask)
    
                for atom_idx in range(num_atoms):
                    start_ch, end_ch = atom_offsets[atom_idx]
                    n_proj_a = end_ch - start_ch
                    P_a = P_all[:, start_ch:end_ch]
    
                    D_out[atom_idx, :n_proj_a, :n_proj_a] += np.real(
                        (P_a.conj().T * weights) @ P_a
                    )
    
        return D_out
    
    ###########################################################################
    # Property Calculations
    ###########################################################################
    
    def calculate_densities_along_line(
        self,
        start_frac,
        end_frac,
        num_points: int = 100,
        spin_channel: int = -1,
        energy_range: tuple = (-np.inf, np.inf),
        use_partial_occ: bool = True,
        pseudo: bool = False,
        include_aug: bool = None,
        return_grad_rho_sq: bool = False,
        return_lap_rho: bool = False,
        use_shrod_tau: bool = False,
    ) -> tuple:
        """Calculates density fields (rho, tau, etc.) along a linear path between
    
        two fractional coordinates. Ideal for drawing 1D bonding profiles.
        """
        if include_aug is not None:
            pseudo = not include_aug
    
        start_frac = np.asarray(start_frac, dtype=np.float64)
        end_frac = np.asarray(end_frac, dtype=np.float64)
    
        # Linearly interpolate fractional coordinates between start and end
        t = np.linspace(0.0, 1.0, num_points)[:, np.newaxis]
        frac_coords = start_frac + t * (end_frac - start_frac)
    
        # Delegate the calculation to point-wise calculation engine
        return self.calculate_densities_at_points(
            points=frac_coords,
            coords_are_cartesian=False,
            spin_channel=spin_channel,
            energy_range=energy_range,
            use_partial_occ=use_partial_occ,
            pseudo=pseudo,
            return_grad_rho_sq=return_grad_rho_sq,
            return_lap_rho=return_lap_rho,
            use_shrod_tau=use_shrod_tau,
        )
    
    def calculate_densities_on_grid(
        self,
        grid_size: tuple[int, int, int] = None,
        spin_channel: int = -1,
        energy_range: tuple[float, float] = (-np.inf, np.inf),
        use_partial_occ: bool = True,
        pseudo: bool = False,
        return_grad_rho_sq: bool = False,
        return_lap_rho: bool = False,
        use_shrod_tau: bool = False,
    ):
        """Calculates charge density n(r) and kinetic energy density tau(r) on a 3D real-space grid via Inverse 3D FFT.

        Symmetrizes the smooth pseudo fields first to eliminate FFT aliasing, then applies
        exact atomic density matrix (D_ij) real-space PAW sphere augmentations if pseudo=False.
        """
        if grid_size is None:
            grid_size = self._minimum_fft_shape

        Nx, Ny, Nz = grid_size
        N_grid = Nx * Ny * Nz

        # Determine spin channels
        if spin_channel == -1:
            spins = list(range(self.nspin))
        else:
            spins = [spin_channel]
        spin_weight = 2 if (self.nspin == 1 or spin_channel != -1) else 1

        # Scaling factor relating Fourier coefficients to real-space grid values
        fft_scale = Nx * Ny * Nz

        # Initialize pseudo real-space grids
        density = np.zeros(grid_size, dtype=np.float64)
        tau = np.zeros(grid_size, dtype=np.float64)

        e_min, e_max = energy_range

        for ikpt in track(
            range(self.nkpoints), description="[bold green]Building Densities..."
        ):
            k_weight = self.kpoint_weights[ikpt]

            # 1. Map integer G vectors onto positive FFT grid indices
            G_int = self.fetch_gvectors(ikpt=ikpt)
            idx_x = G_int[:, 0] % Nx
            idx_y = G_int[:, 1] % Ny
            idx_z = G_int[:, 2] % Nz

            K_vecs = self.fetch_gvectors(ikpt=ikpt, return_cart=True, add_k=True)

            # 2. Collect coefficients & weights for active states in energy window
            active_psi_list = []
            active_weights_list = []

            for ispin in spins:
                energies = self.energies[ispin, ikpt, :]
                occupancies = self.occupancies[ispin, ikpt, :]

                mask = (energies >= e_min) & (energies <= e_max)
                if use_partial_occ:
                    mask = mask & (occupancies >= 1e-8)

                if not np.any(mask):
                    continue

                occ_vals = (
                    occupancies[mask] if use_partial_occ else np.ones(np.sum(mask))
                )

                psi_g = self.fetch_psi(
                    ikpt=ikpt, ispin=ispin, iband=mask, order=0
                )
                active_psi_list.append(psi_g)

                w = (k_weight * occ_vals * spin_weight)[
                    :, np.newaxis, np.newaxis, np.newaxis
                ]
                active_weights_list.append(w)

            if not active_psi_list:
                continue

            psi_g_all = np.concatenate(active_psi_list, axis=0)  # (N_active, ngvecs)
            weights_all = np.concatenate(
                active_weights_list, axis=0
            )  # (N_active, 1, 1, 1)
            N_active = psi_g_all.shape[0]

            # 3. Smooth Pseudo Wavefunctions via Inverse 3D FFT
            grid_g = np.zeros((N_active, Nx, Ny, Nz), dtype=np.complex128)
            grid_g[:, idx_x, idx_y, idx_z] = psi_g_all
            psi_r = (
                ifftn(grid_g, axes=(-3, -2, -1)) * fft_scale
            )  # (N_active, Nx, Ny, Nz)

            # 4. Smooth Pseudo Gradients via Inverse 3D FFT
            grid_grad_g = np.zeros((N_active, 3, Nx, Ny, Nz), dtype=np.complex128)
            for dim in range(3):
                grad_g_dim = 1j * K_vecs[:, dim][np.newaxis, :] * psi_g_all
                grid_grad_g[:, dim, idx_x, idx_y, idx_z] = grad_g_dim

            grad_r = (
                ifftn(grid_grad_g, axes=(-3, -2, -1)) * fft_scale
            )  # (N_active, 3, Nx, Ny, Nz)

            # 5. Accumulate Smooth Pseudo Charge Density n_ps(r)
            density += np.sum(weights_all * (np.abs(psi_r) ** 2), axis=0)

            # 6. Accumulate Smooth Pseudo Kinetic Energy Density tau_ps(r)
            grad_sq_r = np.sum(
                np.abs(grad_r) ** 2, axis=1
            )  # (N_active, Nx, Ny, Nz)
            tau += 0.5 * np.sum(weights_all * grad_sq_r, axis=0)

        # 7. Symmetrize smooth pseudo fields to remove FFT grid aliasing
        rho = self._symmetrize_3d_grid(density)
        tau = self._symmetrize_3d_grid(tau)

        # 8. Apply atomic density matrix (D_ij) PAW real-space augmentations post-symmetrization
        if not pseudo:
            fx = np.linspace(0, 1, Nx, endpoint=False)
            fy = np.linspace(0, 1, Ny, endpoint=False)
            fz = np.linspace(0, 1, Nz, endpoint=False)
            FFx, FFy, FFz = np.meshgrid(fx, fy, fz, indexing="ij")
            grid_frac = np.stack([FFx, FFy, FFz], axis=-1)
            grid_cart_flat = (grid_frac @ self.lattice).reshape(N_grid, 3)

            patches = self._precompute_paw_patches(
                pts_cart=grid_cart_flat, compute_gradients=True
            )
            D_matrix = self.fetch_density_matrix(
                spin_channel=spin_channel,
                energy_range=energy_range,
                use_partial_occ=use_partial_occ,
            )

            self._apply_paw_augmentation(
                density_matrix=D_matrix,
                density=rho,
                patches=patches,
                tau=tau,
            )

        # 9. Calculate Laplacian and optional fields on total density
        if return_lap_rho or use_shrod_tau:
            lap_rho = self.calculate_laplacian(rho)
            if use_shrod_tau:
                tau -= lap_rho / 4.0

        # Collect requested results
        results = [rho, tau]

        if return_grad_rho_sq:
            gradx, grady, gradz = self.calculate_gradient(rho)
            results.append(gradx**2 + grady**2 + gradz**2)

        if return_lap_rho:
            results.append(lap_rho)

        return tuple(results)

    def calculate_densities_at_points(
        self,
        points: np.ndarray,
        coords_are_cartesian: bool = True,
        spin_channel: int = -1,
        energy_range: tuple[float, float] = (-np.inf, np.inf),
        use_partial_occ: bool = True,
        pseudo: bool = False,
        return_grad_rho_sq: bool = False,
        return_lap_rho: bool = False,
        use_shrod_tau: bool = False,
        mem_safety_fraction: float = 0.10,
    ):
        """Calculates all-electron or pseudo charge density n(r) and kinetic energy density

        tau(r) at an arbitrary set of real-space coordinates using safe memory chunking.
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("`points` must be an array of shape (N_points, 3).")

        # Standardize points to Cartesian coordinates (Å)
        pts_cart = pts @ self.lattice if not coords_are_cartesian else pts

        N_pts = len(pts_cart)
        volume = self.structure.volume
        sqrt_vol = np.sqrt(volume)

        # Determine spin channels
        if spin_channel == -1:
            spins = list(range(self.nspin))
        else:
            spins = [spin_channel]
        spin_weight = 2 if (self.nspin == 1 or spin_channel != -1) else 1

        # Initialize density accumulators across all points
        density = np.zeros(N_pts, dtype=np.float64)
        tau = np.zeros(N_pts, dtype=np.float64)

        grad_rho = (
            np.zeros((3, N_pts), dtype=np.float64) if return_grad_rho_sq else None
        )
        lap_rho = (
            np.zeros(N_pts, dtype=np.float64)
            if (return_lap_rho or use_shrod_tau)
            else None
        )

        e_min, e_max = energy_range

        for ikpt in track(
            range(self.nkpoints),
            description="[bold green]Building Point-Wise Densities...",
        ):
            k_weight = self.kpoint_weights[ikpt]

            # Get Cartesian wavevectors K = k + G in Å⁻¹
            K_vecs = self.fetch_gvectors(ikpt=ikpt, return_cart=True, add_k=True)
            ngvecs = len(K_vecs)

            active_psi_list = []
            active_grad_list = []
            active_lap_list = []
            active_weights_list = []

            for ispin in spins:
                energies = self.energies[ispin, ikpt, :]
                occupancies = self.occupancies[ispin, ikpt, :]

                mask = (energies >= e_min) & (energies <= e_max)
                if use_partial_occ:
                    mask = mask & (occupancies >= 1e-8)

                if not np.any(mask):
                    continue

                occ_vals = (
                    occupancies[mask] if use_partial_occ else np.ones(np.sum(mask))
                )

                # Fetch smooth pseudo wavefunctions and gradients
                psi_g = self.fetch_psi(
                    ikpt=ikpt, ispin=ispin, iband=mask, order=0
                )
                active_psi_list.append(psi_g)

                grad_g = self.fetch_psi(
                    ikpt=ikpt, ispin=ispin, iband=mask, order=1
                ).transpose(1, 0, 2)
                active_grad_list.append(grad_g)

                if return_lap_rho or use_shrod_tau:
                    lap_g = self.fetch_psi(
                        ikpt=ikpt, ispin=ispin, iband=mask, order=2
                    )
                    active_lap_list.append(lap_g)

                w = (k_weight * occ_vals * spin_weight)[:, np.newaxis]
                active_weights_list.append(w)

            if not active_psi_list:
                continue

            psi_g_all = np.concatenate(active_psi_list, axis=0)  # (N_active, ngvecs)
            grad_g_all = np.concatenate(
                active_grad_list, axis=0
            )  # (N_active, 3, ngvecs)
            weights_all = np.concatenate(
                active_weights_list, axis=0
            )  # (N_active, 1)
            N_active = psi_g_all.shape[0]

            if return_lap_rho or use_shrod_tau:
                lap_g_all = np.concatenate(
                    active_lap_list, axis=0
                )  # (N_active, ngvecs)

            # Calculate safe chunk size for current RAM and k-point dimensions
            chunk_size = self._get_safe_chunk_size(
                n_gvectors=ngvecs,
                n_bands=N_active,
                mem_safety_fraction=mem_safety_fraction,
            )

            # Evaluate points in memory-safe chunks
            for p_start in range(0, N_pts, chunk_size):
                p_end = min(p_start + chunk_size, N_pts)
                pts_chunk = pts_cart[p_start:p_end]

                # Phase matrix for chunk: shape (N_pts_chunk, ngvecs)
                phase_chunk = np.exp(1j * (pts_chunk @ K_vecs.T))

                # Smooth Pseudo Wavefunctions and Gradients
                psi_r = (psi_g_all @ phase_chunk.T) / sqrt_vol
                grad_r = (
                    np.matmul(grad_g_all, phase_chunk.T) / sqrt_vol
                )

                if return_lap_rho or use_shrod_tau:
                    lap_r = (lap_g_all @ phase_chunk.T) / sqrt_vol

                # 1. Accumulate smooth charge density n_ps(r)
                density[p_start:p_end] += np.sum(
                    weights_all * (np.abs(psi_r) ** 2), axis=0
                )

                # 2. Accumulate smooth kinetic energy density tau_ps(r)
                grad_sq_r = np.sum(np.abs(grad_r) ** 2, axis=1)
                tau[p_start:p_end] += 0.5 * np.sum(weights_all * grad_sq_r, axis=0)

                # 3. Accumulate smooth density gradient \nabla n_ps(r)
                if return_grad_rho_sq:
                    psi_conj = np.conj(psi_r)[:, np.newaxis, :]
                    grad_rho[:, p_start:p_end] += 2.0 * np.real(
                        np.sum(
                            weights_all[:, :, np.newaxis] * psi_conj * grad_r,
                            axis=0,
                        )
                    )

                # 4. Accumulate smooth density Laplacian \nabla^2 n_ps(r)
                if return_lap_rho or use_shrod_tau:
                    term1 = np.conj(psi_r) * lap_r
                    term2 = np.sum(np.abs(grad_r) ** 2, axis=1)
                    lap_rho[p_start:p_end] += 2.0 * np.real(
                        np.sum(weights_all * (term1 + term2), axis=0)
                    )

        # Apply atomic density matrix (D_ij) PAW real-space augmentations
        if not pseudo:
            patches = self._precompute_paw_patches(
                pts_cart=pts_cart, compute_gradients=True
            )
            D_matrix = self.fetch_density_matrix(
                spin_channel=spin_channel,
                energy_range=energy_range,
                use_partial_occ=use_partial_occ,
            )

            self._apply_paw_augmentation(
                density_matrix=D_matrix,
                density=density,
                patches=patches,
                tau=tau,
                grad_rho=grad_rho,
            )

        if use_shrod_tau and lap_rho is not None:
            tau -= lap_rho / 4.0

        results = [density, tau]

        if return_grad_rho_sq:
            grad_rho_sq = np.sum(grad_rho**2, axis=0)
            results.append(grad_rho_sq)

        if return_lap_rho:
            results.append(lap_rho)

        return tuple(results) if len(results) > 2 else (density, tau)
    
    def _get_atom_channel_offsets(self) -> list[tuple[int, int]]:
        """Returns list of (start_idx, end_idx) channel slices for each atom in structure."""
        offsets = []
        curr = 0
        for site in self.structure:
            n_proj = len(self.paw_datasets[site.specie.symbol].angular_momenta)
            offsets.append((curr, curr + n_proj))
            curr += n_proj
        return offsets
    
    def _precompute_paw_patches(
        self,
        pts_cart: np.ndarray,
        compute_gradients: bool = True,
    ) -> list[PAWPatch]:
        """Precomputes 3D real-space partial wave matrices for all atoms on a target point set.
    
        Called ONCE before the k-point loop.
        """
        recip_lattice_T_div_twopi = self.reciprocal_lattice.T / (2.0 * np.pi)
        lattice = self.structure.lattice.matrix
        atom_offsets = self._get_atom_channel_offsets()
    
        patches = []
    
        for atom_idx, site in enumerate(self.structure):
            start_ch, end_ch = atom_offsets[atom_idx]
            paw_sp = self.paw_datasets[site.specie.symbol]
            pos = self.structure.cart_coords[atom_idx]
            max_rc = np.max(paw_sp.paw_cutoffs)
    
            # 1. Minimum image displacement
            dr = pts_cart - pos
            dr_frac = dr @ recip_lattice_T_div_twopi
            shift_frac = np.round(dr_frac)
            dr_frac_mic = dr_frac - shift_frac
            vecs = dr_frac_mic @ lattice
    
            r_sq = np.sum(vecs**2, axis=1)
            mask = r_sq < (max_rc**2)
    
            if not np.any(mask):
                continue
    
            local_vecs = vecs[mask]  # (N_local, 3)
    
            # 2. Evaluate AE and PS partial wave matrices
            ae_mat, ps_mat = paw_sp.evaluate_partial_waves(local_vecs)
    
            # 3. Evaluate AE and PS gradient matrices if requested
            grad_ae_mat, grad_ps_mat = None, None
            if compute_gradients:
                g_ae, g_ps = paw_sp.evaluate_partial_wave_gradients(local_vecs)
                # Transpose to shape (3, N_local, n_proj_a)
                grad_ae_mat = np.swapaxes(g_ae, 0, 1)
                grad_ps_mat = np.swapaxes(g_ps, 0, 1)
    
            patches.append(
                PAWPatch(
                    atom_idx=atom_idx,
                    mask=mask,
                    start_ch=start_ch,
                    end_ch=end_ch,
                    ae_mat=ae_mat,
                    ps_mat=ps_mat,
                    grad_ae_mat=grad_ae_mat,
                    grad_ps_mat=grad_ps_mat,
                )
            )
    
        return patches
    
    
    def _apply_paw_augmentation(
        self,
        density_matrix: np.ndarray,
        density: np.ndarray,
        patches: list[PAWPatch],
        tau: np.ndarray = None,
        grad_rho: np.ndarray = None,
    ):
        """Applies atomic density matrix (D_ij) valence augmentations in-place.

        Parameters
        ----------
        density_matrix : np.ndarray, shape (num_atoms, n_proj_max, n_proj_max)
            Accumulated atomic density matrix D_{a, ij} = sum_{n, k} w_nk * Re(P_nki* P_nkj).
        density : np.ndarray, shape (Nx, Ny, Nz) or (N_pts,)
            Real-space electron charge density array. Modified in-place.
        patches : list[PAWPatch]
            Precomputed geometric and partial wave patches.
        tau : np.ndarray, optional, shape (Nx, Ny, Nz) or (N_pts,)
            Real-space kinetic energy density array. Modified in-place if provided.
        grad_rho : np.ndarray, optional, shape (3, N_pts)
            Real-space charge density gradient array. Modified in-place if provided.
        """
        if density_matrix is None or not patches:
            return

        density_flat = density.reshape(-1)
        tau_flat = tau.reshape(-1) if tau is not None else None

        for patch in patches:
            atom_idx = patch.atom_idx
            mask = patch.mask
            n_proj_a = patch.ae_mat.shape[1]

            # Extract real atomic density matrix D_{a, ij} for this atom
            D_a = density_matrix[atom_idx, :n_proj_a, :n_proj_a]

            # ------------------------------------------------------------------
            # 1. Valence Charge Density Augmentation:
            # \Delta n_a = \sum_{ij} D_{ij} [\Phi_i^AE \Phi_j^AE - \Phi_i^PS \Phi_j^PS]
            # ------------------------------------------------------------------
            ae_term = np.sum((patch.ae_mat @ D_a) * patch.ae_mat, axis=1)
            ps_term = np.sum((patch.ps_mat @ D_a) * patch.ps_mat, axis=1)

            density_flat[mask] += ae_term - ps_term

            # ------------------------------------------------------------------
            # 2. Valence Kinetic Energy Density Augmentation:
            # \Delta \tau_a = 1/2 \sum_{dim=1}^3 \sum_{ij} D_{ij} [\nabla_dim \Phi_i^AE \nabla_dim \Phi_j^AE - ...]
            # ------------------------------------------------------------------
            if tau_flat is not None and patch.grad_ae_mat is not None:
                d_tau = np.zeros(np.count_nonzero(mask), dtype=np.float64)
                for dim in range(3):
                    g_ae_dim = patch.grad_ae_mat[dim]  # (N_local, n_proj_a)
                    g_ps_dim = patch.grad_ps_mat[dim]  # (N_local, n_proj_a)

                    g_ae_term = np.sum((g_ae_dim @ D_a) * g_ae_dim, axis=1)
                    g_ps_term = np.sum((g_ps_dim @ D_a) * g_ps_dim, axis=1)

                    d_tau += 0.5 * (g_ae_term - g_ps_term)

                tau_flat[mask] += d_tau

            # ------------------------------------------------------------------
            # 3. Valence Density Gradient Augmentation:
            # \nabla \Delta n_a = 2 \sum_{ij} D_{ij} [(\nabla \Phi_i^AE) \Phi_j^AE - (\nabla \Phi_i^PS) \Phi_j^PS]
            # ------------------------------------------------------------------
            if grad_rho is not None and patch.grad_ae_mat is not None:
                for dim in range(3):
                    g_ae_dim = patch.grad_ae_mat[dim]
                    g_ps_dim = patch.grad_ps_mat[dim]

                    d_grad_ae = np.sum((g_ae_dim @ D_a) * patch.ae_mat, axis=1)
                    d_grad_ps = np.sum((g_ps_dim @ D_a) * patch.ps_mat, axis=1)

                    grad_rho[dim, mask] += 2.0 * (d_grad_ae - d_grad_ps)

    def calculate_densities_vs_energy(
        self,
        frac_coord: list | np.ndarray,
        return_grad_rho_sq: bool = False,
        return_lap_rho: bool = False,
        spin_channel: int = -1,
        pseudo: bool = False,
        cumulative: bool = False,
        return_plot: bool = False,
        plot_range: tuple[float, float] = None,
        use_shrod_tau: bool = False,
    ):
        """Calculates energy-resolved local electron densities n(r, E), kinetic energy densities

        tau(r, E), and density derivatives at target real-space coordinates.
        """
        pts_frac = np.atleast_2d(np.asarray(frac_coord, dtype=np.float64))
        pts_cart = pts_frac @ self.lattice
        sqrt_vol = np.sqrt(self.structure.volume)

        if not pseudo:
            patches = self._precompute_paw_patches(
                pts_cart=pts_cart, compute_gradients=True
            )
        else:
            patches = None

        def point_callback(
            ispin,
            ikpt,
            weight,
            **kwargs,
        ):
            K_vecs = self.fetch_gvectors(ikpt=ikpt, return_cart=True, add_k=True)
            phase = np.exp(1j * (pts_cart @ K_vecs.T))

            # 1. Fetch smooth pseudo wavefunctions and gradients
            psi_g = self.fetch_psi(
                ikpt=ikpt,
                ispin=ispin,
                iband=np.arange(self.nbands),
                order=0,
            )
            psi_r = (psi_g @ phase.T) / sqrt_vol  # (nbands, N_pts)

            grad_g = self.fetch_psi(
                ikpt=ikpt,
                ispin=ispin,
                iband=np.arange(self.nbands),
                order=1,
            ).transpose(1, 0, 2)
            grad_r = np.matmul(grad_g, phase.T) / sqrt_vol  # (nbands, 3, N_pts)

            # 2. Smooth state-resolved quantities
            rho_n = weight * np.mean(np.abs(psi_r) ** 2, axis=1)
            grad_sq_r = np.sum(np.abs(grad_r) ** 2, axis=1)
            tau_n = 0.5 * weight * np.mean(grad_sq_r, axis=1)

            # 3. Add single-state density matrix augmentations post-eval if not pseudo
            if not pseudo and patches:
                P_all = self.fetch_projector_overlaps(
                    ikpt=ikpt, ispin=ispin, iband=np.arange(self.nbands)
                )  # (nbands, total_n_proj)

                for patch in patches:
                    P_a = P_all[:, patch.start_ch : patch.end_ch]  # (nbands, n_proj_a)

                    # Vectorized single-state partial wave evaluations across all bands
                    phi_ae = P_a @ patch.ae_mat.T  # (nbands, N_pts)
                    phi_ps = P_a @ patch.ps_mat.T  # (nbands, N_pts)

                    # Delta charge density per band
                    d_rho_n = np.mean(np.abs(phi_ae) ** 2 - np.abs(phi_ps) ** 2, axis=1)
                    rho_n += weight * d_rho_n

                    # Delta kinetic energy density per band
                    if patch.grad_ae_mat is not None:
                        d_tau_n = np.zeros(self.nbands, dtype=np.float64)
                        for dim in range(3):
                            g_ae = P_a @ patch.grad_ae_mat[dim].T  # (nbands, N_pts)
                            g_ps = P_a @ patch.grad_ps_mat[dim].T  # (nbands, N_pts)
                            d_tau_n += 0.5 * np.mean(
                                np.abs(g_ae) ** 2 - np.abs(g_ps) ** 2, axis=1
                            )
                        tau_n += weight * d_tau_n

            metrics = [rho_n, tau_n]

            # 4. Density gradient magnitude |grad n_i(r)|^2
            if return_grad_rho_sq:
                grad_rho_n = 2.0 * np.real(
                    psi_r[:, np.newaxis, :] * np.conj(grad_r)
                )
                grad_rho_sq_n = weight * np.mean(
                    np.sum(grad_rho_n**2, axis=1), axis=1
                )
                metrics.append(grad_rho_sq_n)

            return metrics

        num_metrics = 2 + (1 if return_grad_rho_sq else 0)

        # Execute spectral broadening engine across energy grid
        smeared = self._execute_spectral_engine(
            num_metrics=num_metrics,
            spin_channel=spin_channel,
            eval_callback=point_callback,
        )

        if cumulative:
            smeared = [
                cumulative_trapezoid(curve, self.energy_grid, initial=0)
                for curve in smeared
            ]

        if return_plot:
            prefix = "Integrated " if cumulative else ""
            x_label = (
                "Accumulated Integrated Value"
                if cumulative
                else "Differential Density Magnitude (per eV)"
            )

            plot_curves = {
                f"{prefix}Charge Density $\\rho$": smeared[0],
                f"{prefix}Kinetic Density $\\tau$": smeared[1],
            }

            if return_grad_rho_sq:
                plot_curves[f"{prefix}Gradient $|\\nabla\\rho|^2$"] = smeared[2]

            return self._generate_property_plot(
                plot_curves=plot_curves, x_label=x_label, plot_range=plot_range
            )

        return smeared
    
    def get_density_of_states(
        self,
        spin_channel: int = -1,
        cumulative: bool = False,
        return_plot: bool = False,
        plot_range: tuple[float, float] = None,
    ):
        """Calculates the Total or Spin-Resolved Density of States (DOS) across the energy grid.
    
        Parameters
        ----------
        spin_channel : int, optional
            Spin channel index (0 or 1), or -1 for total spin sum.
        cumulative : bool, optional
            If True, integrates the DOS over energy to yield N(E) (integrated electron count).
        return_plot : bool, optional
            If True, generates and returns a Matplotlib property plot.
        plot_range : tuple of (float, float), optional
            Energy window range (E_min, E_max) relative to Fermi level for plot output.
    
        Returns
        -------
        dos : np.ndarray
            Density of states array evaluated across `self.energy_grid`.
        """
    
        def dos_callback(
            ispin,
            ikpt,
            weight,
            **kwargs,
        ):
            # Every electronic state contributes a uniform weight of 1.0 * BZ_weight
            return [np.full(self.nbands, weight)]
    
        # Execute spectral engine for 1 metric (DOS)
        smeared = self._execute_spectral_engine(
            num_metrics=1,
            spin_channel=spin_channel,
            eval_callback=dos_callback,
        )
        dos = smeared[0]
    
        # Optionally compute cumulative DOS N(E)
        if cumulative:
            dos = cumulative_trapezoid(dos, self.energy_grid, initial=0)
    
        if return_plot:
            prefix = "Integrated " if cumulative else ""
            y_label = "States" if cumulative else "States / eV"
            plot_curves = {f"{prefix}Density of States": dos}
    
            return self._generate_property_plot(
                plot_curves=plot_curves, y_label=y_label, plot_range=plot_range
            )
    
        return dos

    ###########################################################################
    # LOCALIZATION METHODS
    ###########################################################################
    def get_elf_at_points(
        self,
        frac_coord,
        spin_channel: int = -1,
        energy_range: tuple = (-np.inf, np.inf),
        use_partial_occ: bool = True,
        pseudo: bool = False,
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
        contributions = self.calculate_densities_at_points(
            frac_coord=frac_coord,
            return_grad_rho_sq=need_derivatives,
            return_lap_rho=False,
            spin_channel=spin_channel,
            energy_range=energy_range,
            use_partial_occ=use_partial_occ,
            pseudo=pseudo,
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

    def get_elf_on_grid(
            self, 
            pseudo=False,
            energy_range=(-np.inf, np.inf), 
            spin_channel=-1, 
            localization_function="elf", 
            savin_correction=True,
            use_partial_occ=True,
            ):
        """Calculates specific localized electron topological indicators (ELF, LOL, or ELI-D)."""
            
        loc_fn_lower = localization_function.lower()
        if loc_fn_lower not in ["elf", "lol", "elid"]:
            raise ValueError(f"Unknown localization function identifier profile: {localization_function}")

        need_derivatives = loc_fn_lower in ["elf", "elid"]
        
        contributions = self.calculate_densities_on_grid(
            return_grad_rho_sq=need_derivatives,
            return_lap_rho=False,
            spin_channel=spin_channel,
            energy_range=energy_range,
            use_partial_occ=use_partial_occ,
            pseudo=pseudo,
            use_shrod_tau=False,
            )
        
        rho = contributions[0]
        tau = contributions[1]
        grad_sq = contributions[2] if need_derivatives else None
        
        if localization_function == "lol":
            from baderkit.post_wfc.localization_functions import lol
            return lol(rho, tau, savin_correction, spin_channel != -1)
            
        if localization_function == "elid":
            from baderkit.post_wfc.localization_functions import elid
            return elid(rho, tau, grad_sq)
        
        elif localization_function == "elf":
            from baderkit.post_wfc.localization_functions import elf
            return elf(rho, tau, grad_sq, savin_correction, spin_channel != -1)
        
    def get_elf_vs_energy(
        self, 
        frac_coord, 
        spin_channel=-1, 
        pseudo=False,
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
        contributions = self.calculate_densities_vs_energy(
            frac_coord=frac_coord,
            return_grad_rho_sq=need_derivatives,
            return_lap_rho=False,
            spin_channel=spin_channel,
            pseudo=pseudo,
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
    
    ###########################################################################
    # Public Helper Functions
    ###########################################################################
    def calculate_laplacian(self, data, is_reciprocal=False):
        """Evaluates second-derivative field Laplacian grid profiles via algebraic Fourier space multiplication."""
        
        if not is_reciprocal:
            with set_workers(-1): 
                recip_data = fftn(data, norm='ortho')
        else: 
            recip_data = data
            
        Gx, Gy, Gz = self.get_fft_grid_cart(recip_data.shape)
        G2 = Gx**2 + Gy**2 + Gz**2  
            
        recip_lap = -G2 * recip_data
        with set_workers(-1): 
            real_lap = ifftn(recip_lap, norm='ortho')
        return real_lap.real
        
    def calculate_gradient(self, data, is_reciprocal=False):
        """Evaluates first-derivative components partial vectors fields using spatial Fourier transforms."""
        if not is_reciprocal:
            with set_workers(-1): 
                recip_data = fftn(data, norm='ortho')
        else: 
            recip_data = data
            
        Gx, Gy, Gz = self.get_fft_grid_cart(recip_data.shape)
        with set_workers(-1):
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
    def _execute_spectral_engine(
        self, num_metrics: int, spin_channel: int, eval_callback
    ) -> list[np.ndarray]:
        """Unified orchestration engine for plane-wave spectral decompositions.
    
        Dispatches state collection paths for PAW and localized atomic environments,
        then evaluates spectral broadenings or analytical tetrahedron integrations.
        """
        # 1. Setup loop invariants
        spins = range(self.nspin) if spin_channel == -1 else [spin_channel]
        rspin = 2.0 if self.nspin == 1 else 1.0
        nx, ny, nz = self._minimum_fft_shape
    
        # Pre-allocate continuous state cache: (num_metrics, nspin, nkpoints, nbands)
        raw_data = np.zeros(
            (num_metrics, self.nspin, self.nkpoints, self.nbands), dtype=np.float64
        )
    
        # 2. Main execution loop across spin channels and k-points
        for ispin in spins:
            for ikpt in range(self.nkpoints):
                weight = (
                    rspin
                    if self._smearing == "tetrahedron"
                    else rspin * self.kpoint_weights[ikpt]
                )
    
                # Fire callback to evaluate spatial/spectral properties
                metrics_block = eval_callback(
                    ispin,
                    ikpt,
                    weight,
                )
    
                # Store metrics into state cache
                for imetric, metric_bands in enumerate(metrics_block):
                    if metric_bands is not None:
                        raw_data[imetric, ispin, ikpt] = metric_bands
    
        # 3. Route 1: Analytical Tetrahedron Brillouin Zone Integration
        if self._smearing == "tetrahedron":
            full_map = self.full_to_irr_map
            eigenvalues = self.energies[:, full_map, :]
            cached_metrics = np.ascontiguousarray(
                np.transpose(raw_data, (1, 2, 3, 0))[:, full_map, :, :]
            )
    
            tetra_indices = self.tetrahedra_indices
            tetra_weight = 1.0 / len(tetra_indices)
    
            smeared_output = integrate_tetrahedra_spectral_density(
                self.energy_grid,
                tetra_indices,
                eigenvalues,
                cached_metrics,
                tetra_weight,
            )
    
            results = [np.sum(smeared_output[i], axis=0) for i in range(num_metrics)]
    
            # Symmetrically smooth step features if convolution smoothing is active
            if self._sigma > 0.0:
                delta_e = self.energy_grid[1] - self.energy_grid[0]
                n_kernel = int(np.ceil(14.0 * self._sigma / delta_e))
                if n_kernel > 0:
                    x = np.arange(-n_kernel, n_kernel + 1) * (delta_e / self._sigma)
                    exp_term = np.exp(np.clip(x, -50, 50))
                    kernel = exp_term / (exp_term + 1.0) ** 2
                    kernel /= np.sum(kernel)
                    results = [
                        np.convolve(res, kernel, mode="same") for res in results
                    ]
    
            return results
    
        # 4. Route 2: Continuous Smearing Matrix Fallback
        smear_matrix = self.smear_matrix
        return [
            np.dot(smear_matrix, raw_data[i, spins].ravel())
            for i in range(num_metrics)
        ]
    
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
        
        smearing = shorthands.get(method if isinstance(method, str) else method, None)
        if smearing is None:
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
            sigma = default_sigma[smearing]
            invalid_sigma=False
        
        if invalid_sigma:
            logging.warning(f"Invalid sigma: {old_sigma}. Using {self._smearing} default: {sigma}.")
            
        if smearing == "fermi-dirac" or smearing == "tetrahedron":
            sigma = 8.617333262e-5 * sigma
        return smearing, sigma
        
    def _get_smear_matrix(self):
        """Helper matrix generator parsing customized analytical broadening distributions."""
        
        # Manual construction for None
        if self._smearing == "none":
            e_min, e_max = self.energy_range
            delta_e = self.energy_grid[1] - self.energy_grid[0]
            smear_matrix = np.zeros((self._num_spectral_points, len(self.energies)))
            closest_idx = np.round((self.energies - e_min) / delta_e).astype(int)
            valid_mask = (closest_idx >= 0) & (closest_idx < self._num_spectral_points)
            smear_matrix[closest_idx[valid_mask], np.where(valid_mask)[0]] = 1.0 / delta_e
            return smear_matrix
            
        delta_E = self.energy_grid[:, None] - self.energies[None, :]
        x = delta_E / self._sigma
        
        if self._smearing == "gaussian":
            return np.exp(-0.5 * x**2) / (self._sigma * np.sqrt(2 * np.pi))
        elif self._smearing in ["methfessel-paxton", "mp"]:
            term_0 = np.exp(-x**2) / np.sqrt(np.pi)
            return ((1.5 - x**2) * term_0) / self._sigma
        elif self._smearing in ["fermi-dirac", "fd"]:
            exp_term = np.exp(np.clip(x, -50, 50))
            return (exp_term / (exp_term + 1.0)**2) / self._sigma
        else:
            raise ValueError(f"Unknown smearing self._smearing: '{self._smearing}'")
            
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
        if self._smearing == "tetrahedron" and (self._sigma is None or self._sigma == 0.0):
            tetra_indices = self.tetrahedra_indices
            tetra_weight = 1.0 / len(tetra_indices)
            rspin = 2.0 if self.nspin == 1 else 1.0
            
            
            full_map = self.full_to_irr_map
            eigenvalues_full = self.energies[spin_indices][:, full_map, :]
            total_charge = integrate_tetrahedra_analytic_charge(
                energy_grid=energy_grid,
                tetra_indices=tetra_indices,
                eigenvalues=eigenvalues_full,
                tetra_weight=tetra_weight,
                rspin=rspin
            )
        
        # Route 2: Convolved Tetrahedron Method (Direct Cumulative Charge Convolution)
        elif self._smearing == "tetrahedron" and self._sigma > 0.0:
            tetra_indices = self.tetrahedra_indices
            tetra_weight = 1.0 / len(tetra_indices)
            rspin = 2.0 if self.nspin == 1 else 1.0
            
            full_map = self.full_to_irr_map
            eigenvalues_full = self.energies[spin_indices][:, full_map, :]
            
            # 1. Compute the exact analytical un-smeared cumulative charge profile
            total_charge_unsmeared = integrate_tetrahedra_analytic_charge(
                energy_grid=energy_grid,
                tetra_indices=tetra_indices,
                eigenvalues=eigenvalues_full,
                tetra_weight=tetra_weight,
                rspin=rspin
            )
            
            # 2. Build the exact Fermi-Dirac derivative kernel matching your post-processing environment
            n_kernel = int(np.ceil(14.0 * self._sigma / delta_e))
            if n_kernel > 0:
                x_kernel = np.arange(-n_kernel, n_kernel + 1) * delta_e
                scaled_x = x_kernel / self._sigma
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
    
    def _get_safe_chunk_size(
        self, n_gvectors: int, n_bands: int, mem_safety_fraction: float = 0.10
    ) -> int:
        """Dynamically calculates a safe coordinate chunk size based on available system memory.
    
        Budgeting defaults to using only 10% of currently available RAM to remain fully
        safe alongside existing array allocations and overhead.
        """
        available_mem = psutil.virtual_memory().available
    
        # Safe fraction of available RAM for this chunk's workspace
        mem_budget = available_mem * mem_safety_fraction
    
        # np.complex128 elements take up 16 bytes
        c128_bytes = 16
    
        # Bytes required per single point in evaluation workspace:
        # 1. phase matrix: n_gvectors * 16 bytes
        # 2. state matrices: psi (1) + lap (1) + grad (3) = 5 * n_bands * 16 bytes
        bytes_per_point = (n_gvectors * c128_bytes) + (5 * n_bands * c128_bytes)
    
        chunk_size = int(mem_budget // bytes_per_point)
    
        # Guarantee at least a chunk size of 1 so execution never halts
        return max(1, chunk_size)
    
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
    def _get_reader(
            self, 
            directory: Path | str = Path("."), 
            fmt: str = "vasp", 
            nbands: int = None, # starts from band 0
            bands: list = None, # only reads selected bands
            **kwargs,
            ):
        """Dynamic factory mapping disk files to code-specific parsing instances."""
        if fmt == "vasp": 
            from baderkit.post_wfc.wf_readers import VaspReader as wf_reader
            
        elif fmt == "qe": 
            from baderkit.post_wfc.wf_readers import QeReader as wf_reader
        else: 
            raise ValueError(f"Unknown reader profile format string: {fmt}")
            
        
        # read wave functions
        wf_reader=wf_reader(directory=Path(directory), nbands=nbands, **kwargs)
        
        return wf_reader