# -*- coding: utf-8 -*-

from abc import ABC

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

@dataclass
class BaseSpecies(ABC):
    """
    Standardized, code-agnostic data representation of atomic core reconstruction 
    parameters. Overcomplete basis channels are canonically orthogonalized and sorted 
    by energy on initialization to yield clear, independent radial charge densities.
    All calculations and quantities are handled natively in Angstrom and eV units.
    """
    
    # BASIC INFORMATION
    name: str
    """Name of this all electron reference"""
    
    element: str
    """Chemical element symbol (e.g., 'Ca')."""
    
    # GRID INFORMATION
    radial_grid: NDArray
    """1D array containing the radial coordinate mesh grid points, r (in Angstroms)."""
    
    q_radial_grid: NDArray
    """1D array containing the radial coordinate mesh grid points in reciprocal space (in 1/Angstroms)"""
    
    real_is_log: bool
    """Whether or not the real grid is on a logarithmic scale. Typically it is."""
    
    q_is_log: bool
    """Whether or not the reciprocal grid is on a logarithmic scale."""
    
    paw_cutoffs: NDArray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    """1D array of the core truncation radii, r_c, matching each specific channel."""
    
    q_paw_cutoffs: NDArray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    """1D array of the reciprocal truncation radii matching each specific channel."""
    
    max_paw_cutoff: float = 0.0
    """The maximum boundary radius defining the global core reconstruction sphere."""

    # QUANTUM NUMBERS
    principal_quantum_numbers: NDArray = field(default_factory=lambda: np.empty(0, dtype=np.int_))
    """1D array of principle quantum numbers quantum numbers for each active channel."""
    
    angular_momenta: NDArray = field(default_factory=lambda: np.empty(0, dtype=np.int_))
    """1D array of orbital angular momentum quantum numbers, l, for each active channel."""
    
    magnetic_quantum_numbers: NDArray = field(default_factory=lambda: np.empty(0, dtype=np.int_))
    """1D array of magnetic quantum numbers, m, for each active channel."""

    # ENERGIES AND OCCUPATIONS
    eigenvalues: NDArray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    """1D array of atomic reference state energy eigenvalues for each channel (in eV)."""
    
    reference_occupations: NDArray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    """1D array of reference atomic occupations for each channel."""

    def _create_1d_splines(self, grid, values):
        """Generates scipy CubicSpline objects for a list of 1d functions."""
        splines = []
        
        # pad grid
        ref_idx = 1 if grid[0] == 0 else 0
        padded_grid = np.insert(grid, 0, -grid[ref_idx])

        # generate splines        
        for idx in range(len(values)):
            # pad projector
            padded = np.insert(values[idx], 0, values[idx][ref_idx])
            # ensure grid and values are the same length (important in PAW for example)
            max_len = min(len(padded_grid), len(padded))
            
            splines.append(CubicSpline(padded_grid[:max_len], padded[:max_len], extrapolate=False))
            
        return splines
