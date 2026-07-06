# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

from .augmentation_numba import precompute_cubic_spline_derivs, build_g_space_projectors

@dataclass
class PAWSpecies:
    """
    Standardized, code-agnostic data representation of atomic core reconstruction 
    parameters. All channel-specific properties are stored in synchronized, 
    ordered NumPy arrays for direct, high-performance iteration.
    """
    # --- Fields WITHOUT default values first ---
    name: str
    """Name of this pseudopotential"""
    
    element: str
    """Chemical element symbol (e.g., 'Ca')."""
    
    Z: float
    """Total electrons in this pseudopotential"""
    
    radial_grid: NDArray
    """1D array containing the radial coordinate mesh grid points, r."""
    
    # --- Fields WITH default values second ---
    source: str | None = None
    """The DFT code this PAW was used for"""
    
    cutoff_radii: NDArray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    """1D array of the core truncation radii, r_c, matching each specific channel."""
    
    max_cutoff_radius: float = 0.0
    """The maximum boundary radius defining the global core reconstruction sphere."""
    
    angular_momenta: NDArray = field(default_factory=lambda: np.empty(0, dtype=np.int_))
    """1D array of orbital angular momentum quantum numbers, l, for each active channel."""
    
    magnetic_nums: NDArray = field(default_factory=lambda: np.empty(0, dtype=np.int_))
    """1D array of orbital magnetic momentum quantum numbers, m, for each active channel."""
    
    all_electron_partial_waves: NDArray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    """2D array of shape (num_channels, num_pts) containing core-orthogonalized atomic partial waves, phi(r)."""
    
    pseudo_partial_waves: NDArray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    """2D array of shape (num_channels, num_pts) containing smooth, nodeless pseudo partial waves, tilde_phi(r)."""
    
    q_linear_grid: NDArray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    """1D array containing the uniform reciprocal space wavevector grid, q, parsed from the POTCAR step metrics."""
    
    reciprocal_projectors: NDArray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    """2D array of shape (num_channels, ndata) containing raw continuous momentum-space projector fields, tilde_p(q)."""

    reciprocal_projectors_padded: NDArray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    """2D array of shape (num_channels, ndata + 1) containing parity-padded reciprocal projectors including q=-h for spline matching."""

    dion_matrix: NDArray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    """2D array of shape (num_channels, num_channels) storing the strength matrix elements coupling channels."""
    
    # Add these fields inside PAWSpecies (e.g., right below reciprocal_projectors_padded)
    eigenvalues: NDArray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    """1D array of atomic reference state energy eigenvalues for each expanded channel."""
    
    reference_occupations: NDArray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    """1D array of reference atomic occupations for each expanded channel."""
    
    # --- Derived fields computed post-initialization ---
    radial_all_electron_pairs_matrix: NDArray = field(default_factory=lambda: np.empty(0), init=False)
    radial_pseudo_pairs_matrix: NDArray = field(default_factory=lambda: np.empty(0), init=False)
    radial_all_electron_ke_pairs_matrix: NDArray = field(default_factory=lambda: np.empty(0), init=False)
    radial_pseudo_ke_pairs_matrix: NDArray = field(default_factory=lambda: np.empty(0), init=False)
    
    # --- Spline derivative cache matrices (added) ---
    radial_all_electron_pairs_derivs: NDArray = field(default_factory=lambda: np.empty(0), init=False)
    radial_pseudo_pairs_derivs: NDArray = field(default_factory=lambda: np.empty(0), init=False)
    radial_all_electron_ke_pairs_derivs: NDArray = field(default_factory=lambda: np.empty(0), init=False)
    radial_pseudo_ke_pairs_derivs: NDArray = field(default_factory=lambda: np.empty(0), init=False)
    
    reciprocal_projectors_derivs: NDArray = field(default_factory=lambda: np.empty(0), init=False)
    
    def __post_init__(self):
        # Automatically determine the outermost core boundary from the channel limits
        if self.cutoff_radii.size > 0 and self.max_cutoff_radius == 0.0:
            self.max_cutoff_radius = float(np.max(self.cutoff_radii))
        # compute radial cross terms
        self._precompute_radial_cross_terms()
        self._precompute_kinetic_cross_terms() 
        self._precompute_reciprocal_projector_terms()
            
    def _precompute_reciprocal_projector_terms(self):
        """Generates second derivatives for raw continuous momentum-space projectors."""
        if self.q_linear_grid.size == 0 or self.reciprocal_projectors.size == 0:
            return
            
        num_channels = self.reciprocal_projectors.shape[0]
        derivs = np.zeros_like(self.reciprocal_projectors)
        
        for idx in range(num_channels):
            derivs[idx] = precompute_cubic_spline_derivs(self.q_linear_grid, self.reciprocal_projectors[idx])
            
        self.reciprocal_projectors_derivs = derivs
        
    def _precompute_radial_cross_terms(self):
        """
        Generates the 2D radial cross-product arrays and computes their
        cubic spline second derivatives row-by-row on the radial grid.
        """
        num_channels = len(self.angular_momenta)
        num_pts = len(self.radial_grid)
        r = self.radial_grid
        
        # Safely handle the 1/r^2 division singularity at r=0
        with np.errstate(divide='ignore', invalid='ignore'):
            r2_inv = np.where(r > 1e-10, 1.0 / r**2, 0.0)
        
        pairs_mat = np.zeros((num_channels * num_channels, num_pts), dtype=np.float64)
        pseudo_pairs_mat = np.zeros((num_channels * num_channels, num_pts), dtype=np.float64)
        
        pairs_derivs = np.zeros_like(pairs_mat)
        pseudo_pairs_derivs = np.zeros_like(pseudo_pairs_mat)
        
        pair_idx = 0
        for i in range(num_channels):
            for j in range(num_channels):
                u_ae_i = self.all_electron_partial_waves[i]
                u_ae_j = self.all_electron_partial_waves[j]
                u_ps_i = self.pseudo_partial_waves[i]
                u_ps_j = self.pseudo_partial_waves[j]
                
                # Compute raw division for all coordinates > 0
                with np.errstate(divide='ignore', invalid='ignore'):
                    ae_vals = (u_ae_i * u_ae_j) * r2_inv
                    ps_vals = (u_ps_i * u_ps_j) * r2_inv
                
                # FIX: Catch the r=0 limit dynamically if grid begins at the origin
                if r[0] < 1e-10:
                    ae_vals[0] = ae_vals[1]
                    ps_vals[0] = ps_vals[1]
                
                pairs_mat[pair_idx] = ae_vals - ps_vals
                pseudo_pairs_mat[pair_idx] = ps_vals
                
                # Precompute cubic spline derivatives for the current row
                pairs_derivs[pair_idx] = precompute_cubic_spline_derivs(r, pairs_mat[pair_idx])
                pseudo_pairs_derivs[pair_idx] = precompute_cubic_spline_derivs(r, pseudo_pairs_mat[pair_idx])
                
                pair_idx += 1
                
        self.radial_all_electron_pairs_matrix = pairs_mat
        self.radial_pseudo_pairs_matrix = pseudo_pairs_mat
        self.radial_all_electron_pairs_derivs = pairs_derivs
        self.radial_pseudo_pairs_derivs = pseudo_pairs_derivs
        
    def _precompute_kinetic_cross_terms(self):
        """
        Generates 2D radial cross-product arrays for kinetic energy density components
        and computes their cubic spline second derivatives row-by-row on the radial grid.
        """
        num_channels = len(self.angular_momenta)
        num_pts = len(self.radial_grid)
        r = self.radial_grid
        
        ke_pairs_mat = np.zeros((num_channels * num_channels, num_pts), dtype=np.float64)
        ke_pseudo_pairs_mat = np.zeros((num_channels * num_channels, num_pts), dtype=np.float64)
        
        ke_pairs_derivs = np.zeros_like(ke_pairs_mat)
        ke_pseudo_pairs_derivs = np.zeros_like(ke_pseudo_pairs_mat)
        
        # Take direct numerical derivatives dphi/dr 
        d_phi_ae = [np.gradient(phi, r) for phi in self.all_electron_partial_waves]
        d_phi_ps = [np.gradient(t_phi, r) for t_phi in self.pseudo_partial_waves]
        
        pair_idx = 0
        for i in range(num_channels):
            l_i = self.angular_momenta[i]
            for j in range(num_channels):
                l_j = self.angular_momenta[j]
                
                # Centrifugal component: l*(l+1)/r^2 * phi_i * phi_j
                with np.errstate(divide='ignore', invalid='ignore'):
                    centrifugal_filter = np.where(r > 1e-10, l_i * (l_i + 1) / r**2, 0.0)
                    
                ang_ae = centrifugal_filter * self.all_electron_partial_waves[i] * self.all_electron_partial_waves[j] if l_i == l_j else 0.0
                ang_ps = centrifugal_filter * self.pseudo_partial_waves[i] * self.pseudo_partial_waves[j] if l_i == l_j else 0.0
                
                # Positive-definite KED: 0.5 * (nabla_i . nabla_j)
                ke_ae = 0.5 * (d_phi_ae[i] * d_phi_ae[j] + ang_ae)
                ke_ps = 0.5 * (d_phi_ps[i] * d_phi_ps[j] + ang_ps)
                
                ke_pairs_mat[pair_idx] = ke_ae - ke_ps
                ke_pseudo_pairs_mat[pair_idx] = ke_ps
                
                # Precompute cubic spline derivatives for the current row
                ke_pairs_derivs[pair_idx] = precompute_cubic_spline_derivs(r, ke_pairs_mat[pair_idx])
                ke_pseudo_pairs_derivs[pair_idx] = precompute_cubic_spline_derivs(r, ke_pseudo_pairs_mat[pair_idx])
                
                pair_idx += 1
                
        self.radial_all_electron_ke_pairs_matrix = ke_pairs_mat
        self.radial_pseudo_ke_pairs_matrix = ke_pseudo_pairs_mat
        self.radial_all_electron_ke_pairs_derivs = ke_pairs_derivs
        self.radial_pseudo_ke_pairs_derivs = ke_pseudo_pairs_derivs
        
    def build_g_space_projectors(
        self, 
        k_cart: NDArray, 
        g_vectors_cart: NDArray, 
        atom_cart_pos: NDArray, 
        cell_volume: float
    ) -> NDArray:
        """
        Maps POTCAR reciprocal projectors onto a discrete G+k plane-wave grid using Numba acceleration.
        
        Returns:
            NDArray: Complex matrix array of shape (num_channels, N_plane_waves)
        """
        
        return build_g_space_projectors(
            self.q_linear_grid,
            self.reciprocal_projectors,
            self.reciprocal_projectors_derivs,
            self.angular_momenta,
            self.magnetic_nums,
            k_cart,
            g_vectors_cart,
            atom_cart_pos,
            float(cell_volume)
        )