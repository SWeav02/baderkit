# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from pathlib import Path

from baderkit.post_wfc.wfc_numba import evaluate_real_harmonics_multi
from baderkit.post_wfc.base_basis import BaseSpecies

@dataclass
class PAWSpecies(BaseSpecies):
    """
    Standardized, code-agnostic data representation of atomic core reconstruction 
    parameters. All channel-specific properties are stored in synchronized, 
    ordered NumPy arrays for direct, high-performance iteration.
    """
    # BASIC INFORMATION
    
    source: str | None = field(default_factory=None)
    """The DFT code this PAW was used for"""
    
    # PARTIAL WAVES
    all_electron_partial_waves: NDArray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    """2D array of shape (num_channels, num_pts) containing core-orthogonalized atomic partial waves, phi(r)."""
    
    pseudo_partial_waves: NDArray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    """2D array of shape (num_channels, num_pts) containing smooth, nodeless pseudo partial waves, tilde_phi(r)."""
    
    # PROJECTORS
    q_projectors: NDArray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    """2D array of shape (num_channels, ndata) containing raw continuous momentum-space projector fields, tilde_p(q)."""

    q_projector_splines: list = field(default_factory=list)
    """List of scipy.interpolate.CubicSpline objects representing 1D reciprocal-space projector profiles."""
    
    partial_wave_diffs: list = field(default_factory=list)
    """List of 1D arrays representing the difference between the all electorn and pseudo partial waves"""
    
    partial_wave_diff_splines: list = field(default_factory=list)
    """List of scipy.interpolate.CubicSpline objects representing 1D partial plane differences."""
    
    # OVERLAP MATRIX
    # paw_overlap_matrix: NDArray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    # """2D array of shape (num_channels, num_channels) representing the overlap matrix"""

    
    def __post_init__(self):
        # Automatically determine the outermost core boundary from the channel limits
        if self.paw_cutoffs.size > 0 and self.max_paw_cutoff == 0.0:
            self.max_paw_cutoff = float(np.max(self.paw_cutoffs))

        self.q_projector_splines = self._create_1d_splines(self.q_radial_grid, self.q_projectors)
        self._precompute_radial_diffs()
        
            
    def _precompute_radial_diffs(self):
        # get grid. Matches PAW grid but extends beyond it.
        grid = self.radial_grid
        
        # Get AW partial waves
        paw_ae_waves = self.all_electron_partial_waves
        paw_ps_waves = self.pseudo_partial_waves
        
        # Get the cutoff radii for each paw channel
        r_cs = self.paw_cutoffs
        
        # Initialize containers (orbital_index, projector_index)
        self.partial_wave_diffs = []
        
        # Loop over each basis function
        for r_c, ae, ps in zip(r_cs, paw_ae_waves, paw_ps_waves):
            
            # Get cutoff radius and mask
            mask = np.where(grid < r_c)[0]
            
            phi_ps = ae[mask]
            phi_ae = ps[mask]
            diff = phi_ae-phi_ps
            self.partial_wave_diffs.append(diff)
        
        self.partial_wave_diff_splines = self._create_1d_splines(self.radial_grid, self.partial_wave_diffs)
        
    def evaluate_q_projectors(
        self, 
        K_vecs,
        coord,
        spatial_phase = None,
    ) -> NDArray:
        """
        Maps POTCAR reciprocal projectors onto a discrete G+k plane-wave grid using Numba acceleration.
        
        Returns:
            NDArray: Complex matrix array of shape (num_channels, N_plane_waves)
        """
        n_qvecs = len(K_vecs)
        n_proj = len(self.angular_momenta)
        # get phase shift due to atom position
        if spatial_phase is None:
            spatial_phase = np.exp(-1j * np.dot(K_vecs, coord))
                
        # loop over projectors        
        atom_projector = np.zeros((n_proj,n_qvecs), np.complex128)
        for proj_idx in range(n_proj):
            spline = self.q_projector_splines[proj_idx]
            l = self.angular_momenta[proj_idx]
            m = self.magnetic_quantum_numbers[proj_idx]
            
            # evaluate angular part
            p_a, K_mags = evaluate_real_harmonics_multi(l, m, K_vecs)
            
            # evaluate radial part
            p_r = spline(K_mags)
            
            # add this projectors contributions
            atom_projector[proj_idx] = (spatial_phase * p_r * p_a)
        
        return atom_projector
    
    def evaluate_partial_diffs(
        self, 
        K_vecs,
        coord,
        spatial_phase = None,
    ) -> NDArray:
        """
        Maps POTCAR reciprocal projectors onto a discrete G+k plane-wave grid using Numba acceleration.
        
        Returns:
            NDArray: Complex matrix array of shape (num_channels, N_plane_waves)
        """
        n_qvecs = len(K_vecs)
        n_proj = len(self.angular_momenta)
        # get phase shift due to atom position
        if spatial_phase is None:
            spatial_phase = np.exp(-1j * np.dot(K_vecs, coord))
                
        # loop over projectors        
        partial_diff = np.zeros((n_proj,n_qvecs), np.complex128)
        for proj_idx in range(n_proj):
            spline = self.partial_wave_diff_splines[proj_idx]
            l = self.angular_momenta[proj_idx]
            m = self.magnetic_quantum_numbers[proj_idx]
            
            # evaluate angular part
            p_a, K_mags = evaluate_real_harmonics_multi(l, m, K_vecs)
            
            # evaluate radial part
            p_r = spline(K_mags)
            
            # add this projectors contributions
            partial_diff[proj_idx] = (spatial_phase * p_r * p_a)
        
        return partial_diff
    
    @classmethod
    def from_filename(
            cls, 
            filename: Path | str = Path("."), 
            fmt: str = "vasp", 
            **kwargs,
            ):
        filename = Path(filename)
        """Dynamic wf_reader factory routing file stream construction to selected code formats."""
        if fmt == "vasp": 
            from baderkit.post_wfc.pseudopotentials.vasp import parse_vasp_potcar
            dataset = parse_vasp_potcar(filename)
        else: 
            raise ValueError(f"Unknown format profile template string keyword: {fmt}")
        
        if len(dataset) == 1:
            dataset = list(dataset.values())[0]
        return dataset