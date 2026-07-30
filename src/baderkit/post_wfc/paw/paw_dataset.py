# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

from baderkit.post_wfc.base_basis import BaseSpecies
from baderkit.post_wfc.wfc_numba import evaluate_real_harmonics_multi

@dataclass
class PAWBasisFields:
    """Holds evaluated partial wave basis fields on a set of spatial coordinates."""
    phi_ae: np.ndarray          # Shape: (N_pts, n_proj)
    phi_ps: np.ndarray          # Shape: (N_pts, n_proj)
    grad_phi_ae: np.ndarray | None = None  # Shape: (N_pts, n_proj, 3)
    grad_phi_ps: np.ndarray | None = None  # Shape: (N_pts, n_proj, 3)
    lap_phi_ae: np.ndarray | None = None   # Shape: (N_pts, n_proj)
    lap_phi_ps: np.ndarray | None = None   # Shape: (N_pts, n_proj)

@dataclass
class PAWSpecies(BaseSpecies):
    """Standardized, code-agnostic data representation of atomic core reconstruction

    parameters configured for valence-only density matrix (D_ij) expansions.
    """

    # BASIC INFORMATION
    Z: float = field(default_factory=float)
    
    source: str | None = field(default_factory=None)

    # PARTIAL WAVES
    all_electron_partial_waves: NDArray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )
    
    pseudo_partial_waves: NDArray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )

    # PROJECTORS & SPLINES
    q_projectors: NDArray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )
    q_projector_splines: list = field(default_factory=list)

    ae_partial_wave_splines: list = field(default_factory=list)
    ps_partial_wave_splines: list = field(default_factory=list)
    square_partial_wave_diff_splines: list = field(default_factory=list)

    def __post_init__(self):
        if self.paw_cutoffs.size > 0 and self.max_paw_cutoff == 0.0:
            self.max_paw_cutoff = float(np.max(self.paw_cutoffs))

        self.q_projector_splines = self._create_1d_splines(
            self.q_radial_grid, self.q_projectors
        )
        self._precompute_wave_splines()

    def _precompute_wave_splines(self):
        """Builds 1D radial splines for partial waves R(r), dR(r)/dr, and d2R(r)/dr2."""
        grid = self.radial_grid
    
        self.ae_partial_wave_splines = self._create_1d_splines(grid, self.all_electron_partial_waves)
        self.ps_partial_wave_splines = self._create_1d_splines(grid, self.pseudo_partial_waves)
    
        # 1st derivatives dR/dr
        self.ae_d_partial_wave_splines = [spl.derivative(1) for spl in self.ae_partial_wave_splines]
        self.ps_d_partial_wave_splines = [spl.derivative(1) for spl in self.ps_partial_wave_splines]
    
        # 2nd derivatives d2R/dr2
        self.ae_d2_partial_wave_splines = [spl.derivative(2) for spl in self.ae_partial_wave_splines]
        self.ps_d2_partial_wave_splines = [spl.derivative(2) for spl in self.ps_partial_wave_splines]

    def evaluate_basis_fields(
        self,
        vecs: np.ndarray,
        compute_gradients: bool = True,
        compute_laplacian: bool = False,
    ) -> PAWBasisFields:
        """Evaluates 1D radial splines & real spherical harmonics for all partial wave channels."""
        n_voxels = len(vecs)
        n_proj = len(self.angular_momenta)

        if n_voxels == 0:
            empty = np.zeros((0, n_proj), dtype=np.float64)
            empty_grad = np.zeros((0, n_proj, 3), dtype=np.float64) if compute_gradients else None
            empty_lap = np.zeros((0, n_proj), dtype=np.float64) if compute_laplacian else None
            return PAWBasisFields(
                phi_ae=empty,
                phi_ps=empty,
                grad_phi_ae=empty_grad,
                grad_phi_ps=empty_grad,
                lap_phi_ae=empty_lap,
                lap_phi_ps=empty_lap,
            )

        r_mags = np.linalg.norm(vecs, axis=1)
        r_safe = np.maximum(r_mags, 1e-12)
        r_hat = vecs / r_safe[:, np.newaxis]

        phi_ae = np.zeros((n_voxels, n_proj), dtype=np.float64)
        phi_ps = np.zeros((n_voxels, n_proj), dtype=np.float64)

        grad_phi_ae = np.zeros((n_voxels, n_proj, 3), dtype=np.float64) if compute_gradients else None
        grad_phi_ps = np.zeros((n_voxels, n_proj, 3), dtype=np.float64) if compute_gradients else None

        lap_phi_ae = np.zeros((n_voxels, n_proj), dtype=np.float64) if compute_laplacian else None
        lap_phi_ps = np.zeros((n_voxels, n_proj), dtype=np.float64) if compute_laplacian else None

        for a in range(n_proj):
            l = self.angular_momenta[a]
            m = self.magnetic_quantum_numbers[a]

            y_lm, grad_y_lm = evaluate_real_harmonics_multi(
                l, m, vecs, compute_gradients=compute_gradients
            )

            r_ae = np.nan_to_num(self.ae_partial_wave_splines[a](r_mags), nan=0.0)
            r_ps = np.nan_to_num(self.ps_partial_wave_splines[a](r_mags), nan=0.0)

            phi_ae[:, a] = r_ae * y_lm
            phi_ps[:, a] = r_ps * y_lm

            # Radial 1st derivatives needed for either gradient or laplacian evaluation
            if compute_gradients or compute_laplacian:
                dr_ae = np.nan_to_num(self.ae_d_partial_wave_splines[a](r_mags), nan=0.0)
                dr_ps = np.nan_to_num(self.ps_d_partial_wave_splines[a](r_mags), nan=0.0)

            if compute_gradients:
                grad_phi_ae[:, a, :] = (
                    dr_ae[:, np.newaxis] * y_lm[:, np.newaxis] * r_hat
                    + r_ae[:, np.newaxis] * grad_y_lm
                )
                grad_phi_ps[:, a, :] = (
                    dr_ps[:, np.newaxis] * y_lm[:, np.newaxis] * r_hat
                    + r_ps[:, np.newaxis] * grad_y_lm
                )

            if compute_laplacian:
                d2r_ae = np.nan_to_num(self.ae_d2_partial_wave_splines[a](r_mags), nan=0.0)
                d2r_ps = np.nan_to_num(self.ps_d2_partial_wave_splines[a](r_mags), nan=0.0)

                angular_factor = (l * (l + 1)) / (r_safe**2)
                radial_lap_ae = d2r_ae + (2.0 / r_safe) * dr_ae - angular_factor * r_ae
                radial_lap_ps = d2r_ps + (2.0 / r_safe) * dr_ps - angular_factor * r_ps

                lap_phi_ae[:, a] = np.nan_to_num(radial_lap_ae * y_lm, nan=0.0)
                lap_phi_ps[:, a] = np.nan_to_num(radial_lap_ps * y_lm, nan=0.0)

        return PAWBasisFields(
            phi_ae=phi_ae,
            phi_ps=phi_ps,
            grad_phi_ae=grad_phi_ae,
            grad_phi_ps=grad_phi_ps,
            lap_phi_ae=lap_phi_ae,
            lap_phi_ps=lap_phi_ps,
        )

    def contract_density_matrix(
        self,
        basis: PAWBasisFields,
        D_atom: np.ndarray,
        compute_tau: bool = True,
        compute_grad_rho_sq: bool = False,
        compute_lap_rho: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Contracts precomputed basis fields with a single accumulated density matrix D_ij."""
        n_voxels = len(basis.phi_ae)
        if n_voxels == 0:
            z = np.zeros(0, dtype=np.float64)
            return z, z, z, z

        n_proj = basis.phi_ae.shape[1]
        D_sub = D_atom[:n_proj, :n_proj]

        # Charge density delta_rho
        n_ae = np.sum((basis.phi_ae @ D_sub) * basis.phi_ae, axis=-1)
        n_ps = np.sum((basis.phi_ps @ D_sub) * basis.phi_ps, axis=-1)
        delta_rho = n_ae - n_ps

        # Kinetic energy density delta_tau (positive definite)
        delta_tau = np.zeros(n_voxels, dtype=np.float64)
        if compute_tau and basis.grad_phi_ae is not None:
            tau_ae = 0.5 * np.einsum("ab, vak, vbk -> v", D_sub, basis.grad_phi_ae, basis.grad_phi_ae, optimize=True)
            tau_ps = 0.5 * np.einsum("ab, vak, vbk -> v", D_sub, basis.grad_phi_ps, basis.grad_phi_ps, optimize=True)
            delta_tau = tau_ae - tau_ps

        # Gradient squared delta_|grad(rho)|^2
        delta_grad_sq = np.zeros(n_voxels, dtype=np.float64)
        if compute_grad_rho_sq and basis.grad_phi_ae is not None:
            g_rho_ae = 2.0 * np.einsum("ab, va, vbk -> vk", D_sub, basis.phi_ae, basis.grad_phi_ae, optimize=True)
            g_rho_ps = 2.0 * np.einsum("ab, va, vbk -> vk", D_sub, basis.phi_ps, basis.grad_phi_ps, optimize=True)
            delta_grad_sq = np.sum(g_rho_ae**2, axis=-1) - np.sum(g_rho_ps**2, axis=-1)

        # Density Laplacian delta_lap(rho)
        delta_lap = np.zeros(n_voxels, dtype=np.float64)
        if compute_lap_rho and basis.lap_phi_ae is not None and basis.grad_phi_ae is not None:
            g_dot_ae = np.einsum("ab, vak, vbk -> v", D_sub, basis.grad_phi_ae, basis.grad_phi_ae, optimize=True)
            g_dot_ps = np.einsum("ab, vak, vbk -> v", D_sub, basis.grad_phi_ps, basis.grad_phi_ps, optimize=True)

            lap_term_ae = np.sum((basis.lap_phi_ae @ D_sub) * basis.phi_ae, axis=-1)
            lap_term_ps = np.sum((basis.lap_phi_ps @ D_sub) * basis.phi_ps, axis=-1)

            delta_lap = 2.0 * ((lap_term_ae + g_dot_ae) - (lap_term_ps + g_dot_ps))

        return delta_rho, delta_tau, delta_grad_sq, delta_lap

    def contract_state_overlaps(
        self,
        basis: PAWBasisFields,
        P_a: np.ndarray,
        total_n_pts: int,
        compute_tau: bool = True,
        compute_grad_rho_sq: bool = False,
        compute_lap_rho: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Contracts precomputed basis fields with per-band projector overlaps P_a (shape: nbands, n_proj)."""
        n_bands = len(P_a)
        if len(basis.phi_ae) == 0:
            z = np.zeros(n_bands, dtype=np.float64)
            return z, z, z, z

        n_proj = basis.phi_ae.shape[1]
        P_sub = P_a[:, :n_proj]

        # Band-resolved wavefunctions: (nbands, N_local)
        phi_ae = P_sub @ basis.phi_ae.T
        phi_ps = P_sub @ basis.phi_ps.T

        # Delta rho per band
        d_rho_n = np.sum(np.abs(phi_ae)**2 - np.abs(phi_ps)**2, axis=1) / total_n_pts

        # Delta tau per band
        d_tau_n = np.zeros(n_bands, dtype=np.float64)
        g_sq_ae = g_sq_ps = None
        if (compute_tau or compute_lap_rho or compute_grad_rho_sq) and basis.grad_phi_ae is not None:
            g_ae = np.einsum("bp, vpk -> bvk", P_sub, basis.grad_phi_ae)  # (nbands, N_local, 3)
            g_ps = np.einsum("bp, vpk -> bvk", P_sub, basis.grad_phi_ps)

            g_sq_ae = np.sum(np.abs(g_ae)**2, axis=-1)  # (nbands, N_local)
            g_sq_ps = np.sum(np.abs(g_ps)**2, axis=-1)

            if compute_tau:
                d_tau_n = 0.5 * np.sum(g_sq_ae - g_sq_ps, axis=1) / total_n_pts

        # Delta |grad rho|^2 per band
        d_grad_sq_n = np.zeros(n_bands, dtype=np.float64)
        if compute_grad_rho_sq and basis.grad_phi_ae is not None:
            grad_ae = 2.0 * np.real(phi_ae[:, :, np.newaxis] * np.conj(g_ae))
            grad_ps = 2.0 * np.real(phi_ps[:, :, np.newaxis] * np.conj(g_ps))
            d_grad_sq_n = np.sum(np.sum(grad_ae**2, axis=-1) - np.sum(grad_ps**2, axis=-1), axis=1) / total_n_pts

        # Delta lap(rho) per band
        d_lap_n = np.zeros(n_bands, dtype=np.float64)
        if compute_lap_rho and basis.lap_phi_ae is not None and basis.grad_phi_ae is not None:
            lap_ae_wave = P_sub @ basis.lap_phi_ae.T
            lap_ps_wave = P_sub @ basis.lap_phi_ps.T

            lap_ae = 2.0 * (np.real(np.conj(phi_ae) * lap_ae_wave) + g_sq_ae)
            lap_ps = 2.0 * (np.real(np.conj(phi_ps) * lap_ps_wave) + g_sq_ps)
            d_lap_n = np.sum(lap_ae - lap_ps, axis=1) / total_n_pts

        return d_rho_n, d_tau_n, d_grad_sq_n, d_lap_n

    def evaluate_paw_sphere_augmentations(
        self,
        vecs: np.ndarray,
        D_atom: np.ndarray,
        compute_tau: bool = True,
        compute_grad_rho_sq: bool = False,
        compute_lap_rho: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convenience wrapper for single-density-matrix grid/point evaluations."""
        needs_gradients = compute_tau or compute_grad_rho_sq or compute_lap_rho
        basis = self.evaluate_basis_fields(
            vecs, compute_gradients=needs_gradients, compute_laplacian=compute_lap_rho
        )
        return self.contract_density_matrix(
            basis, D_atom, compute_tau, compute_grad_rho_sq, compute_lap_rho
        )
        
    def evaluate_q_projectors(
        self,
        K_vecs: np.ndarray,
        coord: np.ndarray,
        spatial_phase: np.ndarray = None,
    ) -> np.ndarray:
        n_qvecs = len(K_vecs)
        n_proj = len(self.angular_momenta)

        if spatial_phase is None:
            spatial_phase = np.exp(-1j * np.dot(K_vecs, coord))

        atom_projector = np.zeros((n_proj, n_qvecs), dtype=np.complex128)
        for proj_idx in range(n_proj):
            spline = self.q_projector_splines[proj_idx]
            l = self.angular_momenta[proj_idx]
            m = self.magnetic_quantum_numbers[proj_idx]

            p_a, K_mags = evaluate_real_harmonics_multi(l, m, K_vecs)
            p_r = np.nan_to_num(spline(K_mags), nan=0.0)

            phase_l = (1j) ** l
            atom_projector[proj_idx] = phase_l * spatial_phase * p_r * p_a

        return atom_projector

    @classmethod
    def from_filename(
        cls,
        filename: Path | str = Path("."),
        fmt: str = "vasp",
        **kwargs,
    ):
        filename = Path(filename)
        if fmt == "vasp":
            from baderkit.post_wfc.pseudopotentials.vasp import (
                parse_vasp_potcar,
            )

            dataset = parse_vasp_potcar(filename)
        else:
            raise ValueError(
                f"Unknown format profile template string keyword: {fmt}"
            )

        if len(dataset) == 1:
            dataset = list(dataset.values())[0]
        return dataset