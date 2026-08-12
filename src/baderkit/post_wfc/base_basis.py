# -*- coding: utf-8 -*-

from abc import ABC

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

# Angular momentum letter mapping
L_SYMBOLS = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g", 5: "h", 6: "i"}

# Real spherical harmonic orbital sub-labels by (l, m)
M_REAL_HARMONICS = {
    1: {-1: "y", 0: "z", 1: "x"},
    2: {-2: "xy", -1: "yz", 0: "z2", 1: "xz", 2: "x2-y2"},
    3: {
        -3: "y(3x2-y2)",
        -2: "xyz",
        -1: "yz2",
        0: "z3",
        1: "xz2",
        2: "z(x2-y2)",
        3: "x(x2-3y2)",
    },
}

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
    
    orbital_labels: list = field(default_factory=list)
    """Orbital label strings, e.g. 1s, 2px, etc."""

    # ENERGIES AND OCCUPATIONS
    eigenvalues: NDArray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    """1D array of atomic reference state energy eigenvalues for each channel (in eV)."""
    
    reference_occupations: NDArray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    """1D array of reference atomic occupations for each channel."""
    
    def __post_init__(self):
        """
        Generates the radial rho, kinetic energy density, and NAO radial splines on initialization
        only if they are not already provided (bypasses recalculation for valence subsets).
        """
        self._get_channel_labels()

    
    def _create_1d_splines(self, grid, values, pad_points=3):
        """Generates scipy CubicSpline objects for a list of 1d functions
    
        with anti-symmetric/symmetric padded boundary near 0.
        """
        splines = []
    
        # Filter out 0 if present at start to build proper negative grid reflection
        start_idx = 1 if grid[0] == 0 else 0
    
        # Ensure pad_points doesn't exceed available grid data
        n_pad = min(pad_points, len(grid) - start_idx)
    
        # Reflect initial grid points into the negative region (e.g., [x1, x2, x3] -> [-x3, -x2, -x1])
        pad_grid = -grid[start_idx : start_idx + n_pad][::-1]
        padded_grid = np.concatenate([pad_grid, grid])
    
        for val in values:
            # Reflect values corresponding to the negative grid slice
            # (For even symmetry, keep original order flipped; for odd symmetry, adjust sign)
            pad_vals = val[start_idx : start_idx + n_pad][::-1]
            padded = np.concatenate([pad_vals, val])
    
            # Align lengths in case grid and values differ
            max_len = min(len(padded_grid), len(padded))
    
            splines.append(
                CubicSpline(
                    padded_grid[:max_len], padded[:max_len], extrapolate=False
                )
            )
    
        return splines
    
    def _get_channel_labels(
        self,
        use_real_harmonics: bool = True,
        disambiguate_duplicates: bool = True,
    ) -> list[str]:
        """Generates string labels for each channel in a BaseSpecies object.
    
        Parameters
        ----------
        use_real_harmonics : bool, optional
            If True, converts magnetic quantum numbers (m) into real orbital
            labels (e.g., 'px', 'dz2'). If False, appends '_m=...' or omits m.
        disambiguate_duplicates : bool, optional
            If duplicate channel labels exist (e.g. multiple projectors for the
            same quantum state), appends an index suffix (e.g. '3s_1', '3s_2').
    
        Returns
        -------
        List[str]
            A list of channel labels matching the length of the channel arrays.
        """
        n_arr = self.principal_quantum_numbers
        l_arr = self.angular_momenta
        m_arr = self.magnetic_quantum_numbers
    
        # Determine total channels from available arrays
        num_channels = max(len(n_arr), len(l_arr), len(m_arr), len(self.eigenvalues))
        labels = []
    
        for idx in range(num_channels):
            n = n_arr[idx] if idx < len(n_arr) else ""
            l = l_arr[idx] if idx < len(l_arr) else None
            m = m_arr[idx] if idx < len(m_arr) else None
    
            # Format n and l
            n_str = str(int(n)) if n != "" else ""
            l_str = L_SYMBOLS.get(int(l), f"l={l}") if l is not None else ""
    
            # Format m
            m_str = ""
            if m is not None and l is not None and l > 0:
                m_val = int(m)
                if use_real_harmonics and l in M_REAL_HARMONICS:
                    m_str = M_REAL_HARMONICS[l].get(m_val, f"m={m_val}")
                else:
                    m_str = f"_m{m_val:+d}"
    
            label = f"{n_str}{l_str}{m_str}"
            labels.append(label if label else f"channel_{idx}")
    
        # Optionally disambiguate duplicate labels
        if disambiguate_duplicates:
            counts = {}
            for lab in labels:
                counts[lab] = counts.get(lab, 0) + 1
    
            seen = {}
            disambiguated = []
            for lab in labels:
                if counts[lab] > 1:
                    seen[lab] = seen.get(lab, 0) + 1
                    disambiguated.append(f"{lab}_{seen[lab]}")
                else:
                    disambiguated.append(lab)
            labels = disambiguated
    
        self.orbital_labels = labels
        return labels
            
        breakpoint()
