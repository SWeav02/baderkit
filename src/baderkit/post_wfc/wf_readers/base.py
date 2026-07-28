# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from pathlib import Path
from baderkit.toolkit import Structure

# bohr to angstrom
BOHR_TO_ANG    = 0.529177249
# rydberg to eV
RY_TO_EV   = 13.605826
# hartree to eV
HARTREE_TO_EV  = 2.0 * RY_TO_EV
# 2*pi
TWOPI = 2*np.pi
# Kinetic energy factor
HSQDTM = 3.8100198740807945

@dataclass
class WfcMetadata:
    """
    Standardized immutable data container for keeping system metadata.
    All physical length values are in Angstroms, and energies are in eV.
    """
    structure: Structure       # Crystal lattice vectors and atomic positions
    kpoints: np.ndarray        # Fractional coordinates array of shape (nkpts, 3)
    occupancies: np.ndarray    # State occupation numbers of shape (nspin, nkpts, nbands)
    energies: np.ndarray       # Absolute band eigenvalues of shape (nspin, nkpts, nbands)
    energy_cutoff: float       # Kinetic energy basis set cutoff threshold
    efermi: float              # Electronic Fermi Level energy baseline
    nspin: int                 # Total number of spin dimensions (1 or 2)
    nkpts: int                 # Number of k-points in the irreducible wedge
    nbands: int                # Number of selected bands
    max_nbands: int            # Maximum number of physical bands in WAVECAR file
    bands: np.ndarray          # 1D array of selected 0-indexed band coordinates
    cplx_dtype: complex        # Precision requirement datatype (complex64 or complex128)

class BaseWfcReader(ABC):
    """
    Abstract Base Class outlining the mandatory interface contract required
    for implementing code-specific periodic electronic structure Readers.
    """
    def __init__(
        self,
        directory: Path | str = Path("."),
        nbands: int = None,
        bands: list[int] = None,
        **kwargs
    ):
        """Initializes the Reader base class and prepares the internal g-vector lookups."""
        self.directory = Path(directory)
        self._gvec_cache = {}  # Internal lifecycle cache to prevent re-building plane-wave spheres
        self.read_metadata(
            nbands=nbands,
            bands=bands,
        )
        
    @abstractmethod
    def read_metadata(
        self,
        nbands = None,
        bands = None,
    ) -> WfcMetadata:
        """
        Parses code-native output headers to extract system geometry and electronic details.
        Must return a populated WfcMetadata instance normalized to Angstrom and eV.
        nbands can optionally be set to ignore bands above a given range.
        bands can optionally be set to specify exact band indices to parse.
        """
        pass

    @abstractmethod
    def read_coefficients(self, ispin: int, ikpt: int, iband: int) -> np.ndarray:
        """
        Extracts a flat 1D array of complex plane-wave expansion coefficients 
        for a targeted single electronic state.
        """
        pass

    @abstractmethod
    def read_coefficients_batch(self, ispin: int, ikpt: int, bands: list) -> np.ndarray:
        """
        Streams complex plane-wave coefficients for multiple bands simultaneously 
        during a single file handle session to eliminate disk I/O thrashing.
        """
        pass
    
    @abstractmethod
    def read_gvectors(self, ikpt: int) -> np.ndarray:
        """
        Returns a 2D integer array of shape (npw, 3) containing the explicit 
        reciprocal space Miller indices (h, k, l) for a target k-point.
        """
        pass