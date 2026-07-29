# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

from baderkit.post_wfc.base_basis import BaseSpecies
from baderkit.post_wfc.wfc_numba import evaluate_real_harmonics_multi


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

    def __post_init__(self):
        if self.paw_cutoffs.size > 0 and self.max_paw_cutoff == 0.0:
            self.max_paw_cutoff = float(np.max(self.paw_cutoffs))

        self.q_projector_splines = self._create_1d_splines(
            self.q_radial_grid, self.q_projectors
        )
        self._precompute_wave_splines()

    def _precompute_wave_splines(self):
        """Builds 1D radial splines for individual AE and PS partial waves phi(r)."""
        grid = self.radial_grid

        self.ae_partial_wave_splines = self._create_1d_splines(
            grid, self.all_electron_partial_waves
        )
        self.ps_partial_wave_splines = self._create_1d_splines(
            grid, self.pseudo_partial_waves
        )

    def evaluate_partial_waves(
        self, vecs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluates 3D real-space AE and PS partial wave matrices [phi(r) * Y_lm(r_hat)].

        Returns
        -------
        ae_mat : np.ndarray, shape (N_voxels, n_proj)
            All-electron partial wave evaluation matrix.
        ps_mat : np.ndarray, shape (N_voxels, n_proj)
            Pseudo partial wave evaluation matrix.
        """
        n_voxels = len(vecs)
        n_proj = len(self.angular_momenta)

        ae_mat = np.zeros((n_voxels, n_proj), dtype=np.float64)
        ps_mat = np.zeros((n_voxels, n_proj), dtype=np.float64)

        if n_voxels == 0:
            return ae_mat, ps_mat

        for proj_idx in range(n_proj):
            ae_spline = self.ae_partial_wave_splines[proj_idx]
            ps_spline = self.ps_partial_wave_splines[proj_idx]
            l = self.angular_momenta[proj_idx]
            m = self.magnetic_quantum_numbers[proj_idx]
            r_c = self.paw_cutoffs[proj_idx]

            y_lm, r_mags = evaluate_real_harmonics_multi(l, m, vecs)

            phi_ae_r = np.nan_to_num(ae_spline(r_mags), nan=0.0)
            phi_ps_r = np.nan_to_num(ps_spline(r_mags), nan=0.0)

            mask_cut = r_mags >= r_c
            phi_ae_r[mask_cut] = 0.0
            phi_ps_r[mask_cut] = 0.0

            ae_mat[:, proj_idx] = phi_ae_r * y_lm
            ps_mat[:, proj_idx] = phi_ps_r * y_lm

        return ae_mat, ps_mat

    def evaluate_partial_wave_gradients(
        self, vecs: np.ndarray, eps: float = 1e-5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluates Cartesian gradients of 3D real-space AE and PS partial waves via finite differences.

        Returns
        -------
        grad_ae : np.ndarray, shape (N_voxels, 3, n_proj)
        grad_ps : np.ndarray, shape (N_voxels, 3, n_proj)
        """
        n_points = len(vecs)
        n_proj = len(self.angular_momenta)

        grad_ae = np.zeros((n_points, 3, n_proj), dtype=np.float64)
        grad_ps = np.zeros((n_points, 3, n_proj), dtype=np.float64)

        for dim in range(3):
            shift = np.zeros((1, 3))
            shift[0, dim] = eps

            ae_plus, ps_plus = self.evaluate_partial_waves(vecs + shift)
            ae_minus, ps_minus = self.evaluate_partial_waves(vecs - shift)

            grad_ae[:, dim, :] = (ae_plus - ae_minus) / (2.0 * eps)
            grad_ps[:, dim, :] = (ps_plus - ps_minus) / (2.0 * eps)

        return grad_ae, grad_ps

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