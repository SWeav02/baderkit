# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from scipy.integrate import simpson

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
    
    partial_radial_diffs: list = field(default_factory=list)
    """List of 1D arrays representing the difference between the all electorn and pseudo partial waves"""
    
    partial_radial_diff_splines: list = field(default_factory=list)
    """List of scipy.interpolate.CubicSpline objects representing 1D partial plane differences."""
    
    # OVERLAP MATRIX
    paw_overlap_matrix: NDArray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    """2D array of shape (num_channels, num_channels) representing the overlap matrix"""

    
    def __post_init__(self):
        # Automatically determine the outermost core boundary from the channel limits
        if self.paw_cutoffs.size > 0 and self.max_paw_cutoff == 0.0:
            self.max_paw_cutoff = float(np.max(self.paw_cutoffs))

        self.q_projector_splines = self._create_1d_splines(self.q_radial_grid, self.q_projectors)
        self._precompute_radial_diffs()
        self._precompute_overlap_matrices()
        
            
    def _precompute_radial_diffs(self):
        # get grid. Matches PAW grid but extends beyond it.
        grid = self.radial_grid
        
        # Get AW partial waves
        paw_ae_waves = self.all_electron_partial_waves
        paw_ps_waves = self.pseudo_partial_waves
        
        # Get the cutoff radii for each paw channel
        r_cs = self.paw_cutoffs
        
        # Initialize containers (orbital_index, projector_index)
        self.partial_radial_diffs = []
        
        # Loop over each basis function
        for r_c, ae, ps in zip(r_cs, paw_ae_waves, paw_ps_waves):
            
            # Get cutoff radius and mask
            mask = np.where(grid < r_c)[0]
            
            phi_ps = ae[mask]
            phi_ae = ps[mask]
            diff = phi_ae-phi_ps
            self.partial_radial_diffs.append(diff)
        
        self.partial_radial_diff_splines = self._create_1d_splines(self.radial_grid, self.partial_radial_diffs)
        
    def build_g_space_conj_projectors(
        self, 
        q_vecs: NDArray, 
        spatial_phase: float,
    ) -> NDArray:
        """
        Maps POTCAR reciprocal projectors onto a discrete G+k plane-wave grid using Numba acceleration.
        
        Returns:
            NDArray: Complex matrix array of shape (num_channels, N_plane_waves)
        """
        # get q vectors (k+G) and their norms
        q_norms = np.linalg.norm(q_vecs, axis=1)
        n_qvecs = len(q_norms)
        
        # loop over projections
        n_proj = len(self.angular_momenta)
        atom_projector = np.zeros((n_proj,n_qvecs), np.complex128)
        for proj_idx in range(n_proj):
            spline = self.q_projector_splines[proj_idx]
            l = self.angular_momenta[proj_idx]
            m = self.magnetic_quantum_numbers[proj_idx]
            
            # evaluate radial part
            p_r = spline(q_norms)
            
            # evaluate angular part
            p_a = evaluate_real_harmonics_multi(l, m, q_vecs/q_norms[:, np.newaxis])
            
            # add this projectors contributions
            atom_projector[proj_idx] = (spatial_phase * p_r * p_a)
        
        return atom_projector.conj()
    
    def _precompute_overlap_matrices(self):
        """
        Precomputes the PAW augmentation overlap matrix q_ij for each atom species.
        q_ij = <phi_i_AE | phi_j_AE>_rc - <phi_i_PS | phi_j_PS>_rc
        """
        grid = self.radial_grid
        grid_sq = grid ** 2
        grid_cu = grid ** 3
        is_logarithmic = self.real_is_log
        dx = np.diff(np.log(grid))[0] if is_logarithmic else np.diff(grid)[0]
        
        self.q_matrices = []
        
        paw_ae_waves = self.all_electron_partial_waves
        paw_ps_waves = self.pseudo_partial_waves
        r_cs = self.paw_cutoffs
        num_projectors = len(paw_ae_waves)
        
        q_mat = np.zeros((num_projectors, num_projectors), dtype=np.float64)
        
        for i in range(num_projectors):
            # get quantum nums and radial cutoff
            l_i = self.angular_momenta[i]
            m_i = self.magnetic_quantum_numbers[i]
            r_c_i = r_cs[i]
            
            for j in range(num_projectors):
                # get quantum nums and radial cutoff
                l_j = self.angular_momenta[j]
                m_j = self.magnetic_quantum_numbers[j]
                r_c_j = r_cs[j]
                
                # Kronecker delta
                if l_i != l_j or m_i != m_j:
                    continue
                    
                r_c = min(r_c_i, r_c_j)
                mask = np.where(grid < r_c)[0]
                
                phi_ae_i = paw_ae_waves[i][mask]
                phi_ae_j = paw_ae_waves[j][mask]
                phi_ps_i = paw_ps_waves[i][mask]
                phi_ps_j = paw_ps_waves[j][mask]
                
                grid_cut = grid_cu[mask] if is_logarithmic else grid_sq[mask]
                integrand = grid_cut * (phi_ae_i * phi_ae_j - phi_ps_i * phi_ps_j)
                
                q_mat[i, j] = simpson(y=integrand, dx=dx)
        self.paw_overlap_matrix = q_mat
                    
    
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