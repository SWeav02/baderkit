# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path
import h5py
import numpy as np
from rich.progress import track

from baderkit.post_wfc.paw.paw_dataset import PAWSpecies
from baderkit.toolkit import Structure

# bohr to angstrom
BOHR_TO_ANG = 0.529177249
# rydberg to eV
RY_TO_EV = 13.605826
# hartree to eV
HARTREE_TO_EV = 2.0 * RY_TO_EV
# 2*pi
TWOPI = 2 * np.pi
# Kinetic energy factor
HSQDTM = 3.8100198740807945


@dataclass
class WfcMetadata:
    """Standardized immutable data container for keeping system metadata.

    All physical length values are in Angstroms, and energies are in eV.
    """

    structure: Structure  # Crystal lattice vectors and atomic positions
    kpoints: np.ndarray  # Fractional coordinates array of shape (nkpts, 3)
    occupancies: (
        np.ndarray
    )  # State occupation numbers of shape (nspin, nkpts, nbands)
    energies: (
        np.ndarray
    )  # Absolute band eigenvalues of shape (nspin, nkpts, nbands)
    energy_cutoff: float  # Kinetic energy basis set cutoff threshold
    efermi: float  # Electronic Fermi Level energy baseline (always 0 by construction)
    original_efermi: float  # The efermi before shift
    nspin: int  # Total number of spin dimensions (1 or 2)
    nkpts: int  # Number of k-points in the irreducible wedge
    nbands: int  # Number of selected bands
    max_nbands: int  # Maximum number of physical bands in WAVECAR file
    bands: np.ndarray  # 1D array of selected 0-indexed band coordinates
    cplx_dtype: complex  # Precision requirement datatype (complex64 or complex128)


class BaseWfcReader(ABC):
    """Abstract Base Class outlining the mandatory interface contract required for

    implementing code-specific periodic electronic structure Readers.
    """

    def __init__(
        self,
        directory: Path | str = Path("."),
        nbands: int = None,
        bands: list[int] = None,
        postwfc_file: Path | str = "post_wfc.h5",
        force_recompute: bool = False,
        **kwargs,
    ):
        """Initializes the Reader base class, loads metadata, and executes wave-

        function precomputation if cached results are absent or invalid.
        """
        self.directory = Path(directory)
        self.postwfc_file = Path(postwfc_file)
        self._gvec_cache = (
            {}
        )  # Internal lifecycle cache to prevent re-building plane-wave spheres

        # Load metadata directly from cache if valid, otherwise parse calculation folder
        metadata_exists = not force_recompute and self._is_cache_valid(
            nbands=nbands, bands=bands
        )
        if metadata_exists:
            self.metadata = self._read_metadata_from_cache()
            self.paw_datasets = self._read_paw_dataset_from_cache()
        else:
            self.metadata = self.read_metadata(nbands=nbands, bands=bands)
            self.paw_datasets = self.read_paw_dataset()

        # Expose convenient attributes
        self.structure = self.metadata.structure
        self.lattice = self.structure.lattice.matrix
        self.reciprocal_lattice = (
            2 * np.pi * np.linalg.inv(self.structure.lattice.matrix).T
        )
        self.nspin = self.metadata.nspin
        self.nkpoints = self.metadata.nkpts
        self.nbands = self.metadata.nbands
        self.occupations = self.metadata.occupancies
        self.kpoints_cart = self.metadata.kpoints @ self.reciprocal_lattice

        # save if metadata didn't already exist
        if not metadata_exists:
            self._save_psi()

    @abstractmethod
    def read_metadata(
        self,
        nbands=None,
        bands=None,
    ) -> WfcMetadata:
        """Parses code-native output headers to extract system geometry and electronic details.

        Must return a populated WfcMetadata instance normalized to Angstrom and
        eV. nbands can optionally be set to ignore bands above a given range.
        bands can optionally be set to specify exact band indices to parse.
        """
        pass

    @abstractmethod
    def read_paw_dataset(
        self,
    ) -> dict[str, PAWSpecies]:
        """Parses pseudo potential files.

        Must return a dictionary mapping species to PAWSpecies datasets.
        """
        pass

    @abstractmethod
    def read_coefficients(
        self, ispin: int, ikpt: int, iband: int
    ) -> np.ndarray:
        """Extracts a flat 1D array of complex plane-wave expansion coefficients

        for a targeted single electronic state.
        """
        pass

    @abstractmethod
    def read_coefficients_batch(
        self, ispin: int, ikpt: int, bands: list
    ) -> np.ndarray:
        """Streams complex plane-wave coefficients for multiple bands simultaneously

        during a single file handle session to eliminate disk I/O thrashing.

        shape: nbands x nplanewaves
        """
        pass

    @abstractmethod
    def read_gvectors(self, ikpt: int) -> np.ndarray:
        """Returns a 2D integer array of shape (npw, 3) containing the explicit

        reciprocal space Miller indices (h, k, l) for a target k-point.
        """
        pass

    def _write_metadata(self, file: h5py.File):
        """Dynamically writes WfcMetadata parameters, structure, and arrays to an HDF5 group."""
        meta_grp = file.create_group("metadata")

        # Save structure datasets for full independent reloading
        meta_grp.create_dataset(
            "cart_coords", data=self.structure.cart_coords
        )
        meta_grp.create_dataset(
            "lattice_matrix", data=self.structure.lattice.matrix
        )
        species_symbols = [site.specie.symbol for site in self.structure]
        meta_grp.create_dataset(
            "species", data=np.array(species_symbols, dtype=h5py.string_dtype())
        )

        # Dynamically write all fields of WfcMetadata
        for f in fields(self.metadata):
            if f.name == "structure":
                continue

            val = getattr(self.metadata, f.name)
            if val is None:
                continue
            elif f.name == "cplx_dtype":
                meta_grp.attrs[f.name] = str(np.dtype(val))
            elif isinstance(val, np.ndarray):
                meta_grp.create_dataset(f.name, data=val)
            elif isinstance(val, (int, float, bool, str, np.number)):
                meta_grp.attrs[f.name] = val

    def _write_paw_datasets(self, file: h5py.File):
        """Dynamically writes PAWSpecies dataset parameters and arrays to an HDF5 group."""
        paw_grp = file.create_group("paw_datasets")

        # Skip derived runtime attributes (splines and diffs recomputed in __post_init__)
        EXCLUDED_FIELDS = {
            "q_projector_splines",
            "partial_wave_diffs",
            "partial_wave_diff_splines",
        }

        for symbol, paw_sp in self.paw_datasets.items():
            sp_grp = paw_grp.create_group(symbol)

            for f in fields(paw_sp):
                if f.name in EXCLUDED_FIELDS:
                    continue

                val = getattr(paw_sp, f.name)
                if val is None:
                    sp_grp.attrs[f.name] = ""
                elif isinstance(val, np.ndarray):
                    sp_grp.create_dataset(f.name, data=val)
                elif isinstance(val, (int, float, bool, str, np.number)):
                    sp_grp.attrs[f.name] = val

    def _read_metadata_from_cache(self) -> WfcMetadata:
        """Reconstructs a WfcMetadata object dynamically from the saved HDF5 cache file."""
        with h5py.File(self.postwfc_file, "r") as file:
            meta_grp = file["metadata"]

            # Reconstruct Structure
            lattice_matrix = meta_grp["lattice_matrix"][:]
            species = [
                s.decode("utf-8") if isinstance(s, bytes) else str(s)
                for s in meta_grp["species"][:]
            ]
            cart_coords = meta_grp["cart_coords"][:]
            structure = Structure(
                lattice_matrix, species, cart_coords, coords_are_cartesian=True
            )

            kwargs = {"structure": structure}

            # Dynamically read dataclass fields
            for f in fields(WfcMetadata):
                if f.name == "structure":
                    continue

                if f.name in meta_grp.attrs:
                    val = meta_grp.attrs[f.name]
                    if f.name == "cplx_dtype":
                        val = np.dtype(val).type
                    kwargs[f.name] = val
                elif f.name in meta_grp:
                    kwargs[f.name] = meta_grp[f.name][:]

            return WfcMetadata(**kwargs)

    def _read_paw_dataset_from_cache(self) -> dict[str, PAWSpecies]:
        """Reconstructs PAWSpecies dataset objects dynamically from the saved HDF5 cache file."""
        paw_datasets = {}
        with h5py.File(self.postwfc_file, "r") as file:
            if "paw_datasets" not in file:
                raise KeyError(
                    "Dataset group 'paw_datasets' not found in HDF5 file."
                )

            paw_grp = file["paw_datasets"]
            for symbol in paw_grp.keys():
                sp_grp = paw_grp[symbol]
                kwargs = {}

                for f in fields(PAWSpecies):
                    if f.name in sp_grp.attrs:
                        val = sp_grp.attrs[f.name]
                        if f.name == "source" and val == "":
                            val = None
                        kwargs[f.name] = val
                    elif f.name in sp_grp:
                        kwargs[f.name] = sp_grp[f.name][:]

                # Instantiating PAWSpecies automatically invokes __post_init__ to rebuild splines
                paw_datasets[symbol] = PAWSpecies(**kwargs)

        return paw_datasets

    def _is_cache_valid(
        self, nbands: int = None, bands: list[int] = None
    ) -> bool:
        """Checks if a valid HDF5 postwfc file exists with required metadata and datasets."""
        if not self.postwfc_file.exists():
            return False

        try:
            with h5py.File(self.postwfc_file, "r") as file:
                if "metadata" not in file or "paw_datasets" not in file:
                    return False

                meta_grp = file["metadata"]

                # Validate band user constraints against cached metadata
                if nbands is not None and meta_grp.attrs.get("nbands") != nbands:
                    return False

                if bands is not None:
                    cached_bands = meta_grp["bands"][:]
                    if not np.array_equal(cached_bands, np.asarray(bands)):
                        return False

                nkpts = meta_grp.attrs.get("nkpts")
                if nkpts is None:
                    return False

                # Verify presence of reciprocal datasets and PAW overlaps for all k-points
                for ikpt in range(nkpts):
                    required_keys = [
                        f"ps/psi/{ikpt}",
                        f"ps/grad/{ikpt}",
                        f"ps/lap/{ikpt}",
                        f"coefficients/{ikpt}",
                        f"gvectors/{ikpt}",
                        f"projector_overlaps/{ikpt}",
                    ]
                    if any(key not in file for key in required_keys):
                        return False

                return True
        except Exception:
            return False

    def _save_psi(self):
        """Calculates and saves pseudo wavefunctions, derivatives, projector overlap scalars,
    
        raw plane-wave coefficients, and precomputed total atomic density matrices (D_ij).
        """
        structure = self.structure
        atom_positions = structure.cart_coords
        nspin = self.nspin
        nkpoints = self.nkpoints
        nbands = self.nbands
        sqrt_vol = np.sqrt(structure.volume)
    
        atom_offsets = self._get_atom_channel_offsets()
        num_atoms = len(structure)
    
        # Determine max number of channels across all species for array allocation
        max_n_proj = max(
            end - start for start, end in atom_offsets
        )
    
        # Initialize full ground-state density matrix accumulator
        # Shape: (nspin, num_atoms, max_n_proj, max_n_proj)
        D_total = np.zeros(
            (nspin, num_atoms, max_n_proj, max_n_proj), dtype=np.float64
        )
    
        spin_weight = 2 if self.nspin == 1 else 1
    
        with h5py.File(self.postwfc_file, "w") as file:
            self._write_metadata(file)
            self._write_paw_datasets(file)
    
            for ikpt in track(
                range(nkpoints),
                description="[bold blue]Caching Pseudo Psi & PAW Overlaps...",
                total=nkpoints,
            ):
                k_weight = self.kpoint_weights[ikpt]
    
                # Get G vectors
                g_int = self.read_gvectors(ikpt)
                G_basis_cart = g_int @ self.reciprocal_lattice
                ngvecs = len(G_basis_cart)
                k_cart = self.kpoints_cart[ikpt]
    
                K_vecs = G_basis_cart + k_cart[np.newaxis, :]
                K_sq = np.sum(K_vecs**2, axis=1)
    
                file.create_dataset(
                    f"gvectors/{ikpt}",
                    data=g_int,
                    dtype=np.int32,
                    compression="lzf",
                )
    
                raw_coeffs = file.create_dataset(
                    f"coefficients/{ikpt}",
                    shape=(nspin, nbands, ngvecs),
                    dtype=np.complex128,
                    compression="lzf",
                )
                ps_psi = file.create_dataset(
                    f"ps/psi/{ikpt}",
                    shape=(nspin, nbands, ngvecs),
                    dtype=np.complex128,
                    compression="lzf",
                )
                ps_grad = file.create_dataset(
                    f"ps/grad/{ikpt}",
                    shape=(nspin, 3, nbands, ngvecs),
                    dtype=np.complex128,
                    compression="lzf",
                )
                ps_lap = file.create_dataset(
                    f"ps/lap/{ikpt}",
                    shape=(nspin, nbands, ngvecs),
                    dtype=np.complex128,
                    compression="lzf",
                )
    
                # Evaluate PAW reciprocal projectors for all atomic centers
                paw_projector_matrices = []
                for atom_idx, site in enumerate(structure):
                    paw_basis = self.paw_datasets[site.specie.symbol]
                    pos = atom_positions[atom_idx]
                    spatial_phase = np.exp(-1j * np.dot(K_vecs, pos))
    
                    proj = paw_basis.evaluate_q_projectors(
                        K_vecs, pos, spatial_phase=spatial_phase
                    )
                    paw_projector_matrices.append(proj)
    
                paw_projector_matrix = np.vstack(paw_projector_matrices)
                total_n_proj = paw_projector_matrix.shape[0]
    
                proj_overlaps = file.create_dataset(
                    f"projector_overlaps/{ikpt}",
                    shape=(nspin, nbands, total_n_proj),
                    dtype=np.complex128,
                    compression="lzf",
                )
    
                for ispin in range(nspin):
                    coeffs = self.read_coefficients_batch(
                        ispin, ikpt, np.arange(nbands)
                    ).T  # Shape: (ngvecs, nbands)
    
                    raw_coeffs[ispin] = coeffs.T
    
                    # 1. Pseudo Wavefunction
                    psi_ps_val = coeffs / sqrt_vol
                    ps_psi[ispin] = psi_ps_val.T
    
                    # 2. Projector Overlaps P_a = <p_a | psi_ps>
                    p_psi_ps = (
                        paw_projector_matrix.conj() @ coeffs
                    ) / sqrt_vol
                    P_all = p_psi_ps.T  # Shape: (nbands, total_n_proj)
                    proj_overlaps[ispin] = P_all
    
                    # 3. Accumulate Atomic Density Matrix D_{a, ij}
                    occ = self.metadata.occupancies[ispin, ikpt, :]
                    weights = k_weight * occ * spin_weight  # Shape: (nbands,)
    
                    for atom_idx in range(num_atoms):
                        start_ch, end_ch = atom_offsets[atom_idx]
                        n_proj_a = end_ch - start_ch
                        P_a = P_all[:, start_ch:end_ch]  # Shape: (nbands, n_proj_a)
    
                        # D_a = Re( P_a^H @ diag(weights) @ P_a )
                        D_a = np.real((P_a.conj().T * weights) @ P_a)
                        D_total[ispin, atom_idx, :n_proj_a, :n_proj_a] += D_a
    
                    # 4. Pseudo Laplacian & Gradient
                    ps_lap[ispin] = (-K_sq[:, np.newaxis] * psi_ps_val).T
                    grad_ps = (
                        1j
                        * K_vecs[:, :, np.newaxis]
                        * psi_ps_val[:, np.newaxis, :]
                    )
                    ps_grad[ispin] = np.moveaxis(
                        grad_ps, [0, 1, 2], [2, 0, 1]
                    )
    
            # Save precomputed total density matrices to HDF5
            file.create_dataset("density_matrices", data=D_total, compression="lzf")