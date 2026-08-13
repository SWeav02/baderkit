#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PeriodicFrameRelaxer: A basis-driven, periodic-graph relaxation engine for 
constructing rotationally covariant reference orbital frames prior to IAO projection.
"""

from typing import Any, NamedTuple
import warnings

import numpy as np
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.spatial.transform import Rotation

from baderkit.post_wfc.projection.all_electron_dataset import AESpecies


class SymmetryOrbit(NamedTuple):
    """Container for the space-group symmetry equivalences of a ground-state frame assignment."""

    canonical_rotations: list[np.ndarray]  # (N_atoms, 3, 3) primary rotations
    space_group_symbol: str  # e.g., "Fm-3m" or "P4/mmm"
    symmetry_operations: list[
        np.ndarray
    ]  # List of (3, 3) SO(3) matrices for equivalent states
    orbit_rotations: list[
        list[np.ndarray]
    ]  # All degenerate valid frame assignments across the orbit


class PeriodicFrameRelaxer:
    """
    Drives pre-IAO atomic reference frame alignment using continuous radial overlap 
    weights, point-group symmetry candidate sets, topological multipole proxies, 
    and periodic graph relaxation.
    """

    def __init__(
        self,
        structure: Structure,
        atom_bases: list[AESpecies] | dict[str, AESpecies],
        rcut_overlap_tol: float = 1e-6,
        topo_hierarchy_weights: dict[str, float] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        structure : Structure
            Pymatgen Structure object representing the periodic unit cell.
        atom_bases : list[AESpecies] | dict[str, AESpecies]
            List or dictionary mapping element symbols to AESpecies objects.
        rcut_overlap_tol : float, optional
            Numerical threshold below which radial basis overlaps are truncated (default 1e-6).
        topo_hierarchy_weights : dict[str, float] | None, optional
            Weight hierarchy for topologically non-equivalent subshells 
            (e.g., {'e_g': 1.0, 't_2g': 0.8, 'p_z': 1.0, 'p_xy': 0.8}).
        """
        self.structure = structure

        if isinstance(atom_bases, list):
            self.atom_bases = {b.element: b for b in atom_bases}
        else:
            self.atom_bases = atom_bases

        self.rcut_overlap_tol = rcut_overlap_tol
        self.topo_weights = topo_hierarchy_weights or {"e_g": 1.0, "t_2g": 0.8}

        self.natoms = len(structure)

        # Internal cache attributes populated during relaxation
        self._periodic_graph: list[list[dict[str, Any]]] = []
        self._candidate_sets: list[list[np.ndarray]] = []
        self._multipole_proxies: list[dict[str, Any]] = []

    # =========================================================================
    # PUBLIC API ENTRY POINT
    # =========================================================================

    def relax_frames(
        self,
        max_sweeps: int = 10,
        energy_tol: float = 1e-8,
    ) -> SymmetryOrbit:
        """
        Executes the full multi-stage relaxation pipeline to produce 
        rotationally covariant, canonical atomic reference frames.

        Returns
        -------
        SymmetryOrbit
            NamedTuple containing the canonical frame assignment and the full 
            space-group symmetry orbit generating all degenerate valid states.
        """
        # Stage 1: Build physical weights and periodic graph
        self._build_periodic_graph()

        # Stage 2: Detect local point groups & generate SO(3) quotient candidates
        self._generate_all_candidate_sets()

        # Stage 3: Construct topological subshell multipole proxies
        self._construct_multipole_proxies()

        # Stage 4: Relax candidate selections across periodic lattice graph
        relaxed_rotations = self._relax_lattice_graph(
            max_sweeps=max_sweeps, energy_tol=energy_tol
        )

        # Stage 5: Fix global crystal gauge and enumerate full symmetry orbit
        return self._finalize_symmetry_orbit(relaxed_rotations)

    # =========================================================================
    # STAGE 1: BASIS-DERIVED RADIAL OVERLAPS & PERIODIC GRAPH
    # =========================================================================

    def _compute_radial_overlap_tail(
        self, elem_a: str, elem_b: str, r_distance: float
    ) -> float:
        """
        Computes the radial wavefunction overlap S_ab(r) between basis tails at distance r_distance.
        """
        basis_a = self.atom_bases[elem_a]
        basis_b = self.atom_bases[elem_b]

        rcut_a = basis_a.effective_cutoff
        rcut_b = basis_b.effective_cutoff
        rcut_sum = rcut_a + rcut_b

        if r_distance >= rcut_sum:
            return 0.0

        # Sample outermost valence radial splines along the inter-atomic line
        r_a = r_distance * (rcut_a / rcut_sum)
        r_b = r_distance * (rcut_b / rcut_sum)

        val_a = float(np.nan_to_num(basis_a.radial_splines[-1](r_a), 0.0))
        val_b = float(np.nan_to_num(basis_b.radial_splines[-1](r_b), 0.0))

        t = r_distance / rcut_sum
        envelope = (1.0 - t**2) ** 2

        S_ab = np.abs(val_a * val_b) * envelope
        return float(np.clip(S_ab, 0.0, 1.0))

    def _build_periodic_graph(self) -> None:
        """
        Constructs the periodic neighbor interaction graph using CrystalNN 
        to isolate the first coordination shell for every site in the crystal.
        """
        structure = self.structure
        species_list = [site.specie.symbol for site in structure]

        # Suppress pymatgen's CrystalNN/Voronoi UserWarnings during local environment search
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            cnn = CrystalNN()
            all_nn_info = cnn.get_all_nn_info(structure)

        self._periodic_graph = [[] for _ in range(self.natoms)]

        for a_idx, nn_list in enumerate(all_nn_info):
            sp_a = species_list[a_idx]
            pos_a = structure[a_idx].coords

            for neigh in nn_list:
                b_idx = int(neigh["site_index"])
                sp_b = species_list[b_idx]
                site_b = neigh["site"]
                img_off = tuple(int(x) for x in neigh["image"])

                # Displacement vector to the periodic neighbor image
                dr_vec = site_b.coords - pos_a
                r_dist = float(np.linalg.norm(dr_vec))

                if r_dist < 1e-5:
                    continue

                r_hat = dr_vec / r_dist
                S_ab = self._compute_radial_overlap_tail(sp_a, sp_b, r_dist)

                # CrystalNN weight represents normalized Voronoi solid-angle fraction (0 to 1)
                cnn_weight = float(neigh.get("weight", 1.0))

                # Edge weight combines solid-angle coordination weight with orbital overlap
                w_ab = (S_ab**2) * cnn_weight

                self._periodic_graph[a_idx].append({
                    "b_idx": b_idx,
                    "dist": r_dist,
                    "r_hat": r_hat,
                    "S_ab": S_ab,
                    "weight": w_ab,
                    "cnn_weight": cnn_weight,
                    "image_offset": img_off,
                })

    # =========================================================================
    # STAGE 2: SITE-SYMMETRY DETECTION & CANDIDATE GENERATION
    # =========================================================================

    def _detect_site_point_group(self, atom_idx: int) -> str:
        """
        Analyzes the local neighbor environment of site `atom_idx` to identify 
        its local point group symmetry G_a (e.g., C1, Cs, C3v, Oh).

        Parameters
        ----------
        atom_idx : int
            Index of the target atom in `self.structure`.

        Returns
        -------
        symbol : str
            Hermann-Mauguin site symmetry symbol (e.g., "m-3m", "4/mmm", "3m", "1").
        """
        # 1. Attempt exact space-group site symmetry detection via SpacegroupAnalyzer
        dataset = self.structure.get_symmetry_dataset()
        if dataset is not None and "site_symmetry_symbols" in dataset:
            return str(dataset["site_symmetry_symbols"][atom_idx])

    def _generate_candidate_sets_for_site(self, atom_idx: int) -> list[np.ndarray]:
        """
        Generates the discrete set of SO(3)/G_a candidate rotation seeds 
        for site `atom_idx` based on its local point group symmetry.

        Parameters
        ----------
        atom_idx : int
            Index of the target atom in `self.structure`.

        Returns
        -------
        candidate_rotations : list[np.ndarray]
            List of 3x3 proper rotation matrices R in SO(3) representing 
            symmetry-equivalent candidate orientations for the site.
        """
        symbol = self._detect_site_point_group(atom_idx)

        # 1. High Polyhedral / Cubic Symmetry (e.g., "m-3m", "-43m", "432", "23", "m-3")
        cubic_symbols = {"m-3m", "-43m", "432", "23", "m-3", "m3m", "m3"}
        if symbol in cubic_symbols or ("3" in symbol and "m" in symbol and len(symbol) > 3):
            cubic_seeds = []
            for x_rot in [0, 90, 180, 270]:
                for y_rot in [0, 90, 180, 270]:
                    for z_rot in [0, 90, 180, 270]:
                        R = Rotation.from_euler("xyz", [x_rot, y_rot, z_rot], degrees=True).as_matrix()
                        if not any(np.allclose(R, s, atol=1e-3) for s in cubic_seeds):
                            cubic_seeds.append(R)
            return cubic_seeds

        # Build local density quadrupole tensor Q_a
        Q_a = np.zeros((3, 3), dtype=np.float64)
        for edge in self._periodic_graph[atom_idx]:
            r_hat = edge["r_hat"]
            w = edge["weight"]
            Q_a += w * np.outer(r_hat, r_hat)

        eigvals, eigvecs = np.linalg.eigh(Q_a)

        # Ensure right-handed orthonormal basis det(V) = +1
        V = eigvecs.copy()
        if np.linalg.det(V) < 0:
            V[:, 2] *= -1.0

        # 2. Axially Symmetric Sites (e.g., "4/mmm", "6/mmm", "3m", "4mm")
        # One unique non-degenerate eigenvector v_3; transverse plane is isotropic
        d12 = abs(eigvals[1] - eigvals[0]) / (np.trace(Q_a) + 1e-12)
        d23 = abs(eigvals[2] - eigvals[1]) / (np.trace(Q_a) + 1e-12)

        if d12 < 1e-3 or d23 < 1e-3:
            axial_candidates = []
            # Sample 8 discrete azimuthal rotations around the principal axis v_3
            angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
            for theta in angles:
                R_z = Rotation.from_rotvec(theta * V[:, 2]).as_matrix()
                R_cand = R_z @ V

                # Direct alignment
                axial_candidates.append(R_cand)

                # 180-degree flip of principal axis
                R_flip = Rotation.from_rotvec(np.pi * V[:, 0]).as_matrix() @ R_cand
                axial_candidates.append(R_flip)

            # Deduplicate candidates
            unique_axial = []
            for R in axial_candidates:
                if not any(np.allclose(R, u, atol=1e-3) for u in unique_axial):
                    unique_axial.append(R)
            return unique_axial

        # 3. Low-Symmetry / Anisotropic Sites (e.g., "1", "m", "2/m", "2", "222")
        # All 3 eigenvalues distinct: Frame uniquely determined up to 4 sign permutations
        sign_permutations = [
            np.diag([1.0, 1.0, 1.0]),
            np.diag([-1.0, -1.0, 1.0]),
            np.diag([-1.0, 1.0, -1.0]),
            np.diag([1.0, -1.0, -1.0]),
        ]

        candidates = [V @ P for P in sign_permutations]
        return candidates

    def _generate_all_candidate_sets(self) -> None:
        """Populates `self._candidate_sets` for all atoms in the unit cell."""
        self._candidate_sets = []
        for atom_idx in range(self.natoms):
            cands = self._generate_candidate_sets_for_site(atom_idx)
            self._candidate_sets.append(cands)

    # =========================================================================
    # STAGE 3: TOPOLOGICAL MULTIPOLE PROXIES & PAIRWISE ENERGIES
    # =========================================================================

    def _construct_multipole_proxies(self) -> None:
        """
        Builds subshell multipole tensor proxies M_l for each atom, applying 
        topological preference weights (e.g., separating l=2 into e_g and t_2g).
        """
        self._multipole_proxies = []

        for atom_idx in range(self.natoms):
            site = self.structure[atom_idx]
            elem = site.specie.symbol
            basis_obj = self.atom_bases[elem]

            ang_mom = getattr(basis_obj, "angular_momenta", np.array([], dtype=int))

            # Count subshell channels present
            count_p = int(np.sum(ang_mom == 1))
            count_d = int(np.sum(ang_mom == 2))
            count_f = int(np.sum(ang_mom == 3))

            proxy = {
                "elem": elem,
                "has_p": count_p > 0,
                "has_d": count_d > 0,
                "has_f": count_f > 0,
                "count_p": count_p,
                "count_d": count_d,
                "count_f": count_f,
                # Topological hierarchy weights
                "w_p_z": float(self.topo_weights.get("p_z", 1.0)),
                "w_p_xy": float(self.topo_weights.get("p_xy", 0.8)),
                "w_eg": float(self.topo_weights.get("e_g", 1.0)),
                "w_t2g": float(self.topo_weights.get("t_2g", 0.8)),
            }

            self._multipole_proxies.append(proxy)

    def _eval_site_proxy_score(self, proxy: dict, r_loc_hat: np.ndarray) -> float:
        """
        Evaluates the directional multipole projection score along a local 
        unit vector r_loc_hat = (x, y, z) in the atom's local coordinate frame.
        """
        x, y, z = r_loc_hat
        x2, y2, z2 = x**2, y**2, z**2

        score = 1.0  # Isotropic monopole baseline (s-orbitals)

        # p-orbital directional projection (p_z primary axis vs p_x, p_y)
        if proxy["has_p"]:
            p_proj = proxy["w_p_z"] * z2 + proxy["w_p_xy"] * (x2 + y2)
            score += proxy["count_p"] * p_proj

        # d-orbital topological projection (e_g axial vs t_2g inter-axial)
        if proxy["has_d"]:
            p_eg = 0.25 * ((3.0 * z2 - 1.0) ** 2) + 0.75 * ((x2 - y2) ** 2)
            p_t2g = 3.0 * (x2 * y2 + y2 * z2 + z2 * x2)

            d_proj = proxy["w_eg"] * p_eg + proxy["w_t2g"] * p_t2g
            score += proxy["count_d"] * d_proj

        return score

    def _compute_pairwise_proxy_energy(
        self,
        atom_a: int,
        atom_b: int,
        R_a: np.ndarray,
        R_b: np.ndarray,
        r_hat: np.ndarray,
        edge_weight: float,
    ) -> float:
        """
        Evaluates the directional proxy coupling energy between subshell multipoles 
        at site `atom_a` (rotated by R_a) and site `atom_b` (rotated by R_b) 
        along bond vector `r_hat`.

        Parameters
        ----------
        atom_a : int
            Index of origin atom A.
        atom_b : int
            Index of neighbor atom B.
        R_a : np.ndarray
            3x3 rotation matrix for atom A.
        R_b : np.ndarray
            3x3 rotation matrix for atom B.
        r_hat : np.ndarray
            Unit vector pointing from atom A to atom B in global Cartesian space.
        edge_weight : float
            Physical C^infinity smooth weight for the (A, B) bond edge.

        Returns
        -------
        energy : float
            Pairwise coupling energy scalar.
        """
        proxy_a = self._multipole_proxies[atom_a]
        proxy_b = self._multipole_proxies[atom_b]

        # Transform global bond direction into local Cartesian frame of atom A
        r_loc_a = R_a.T @ r_hat

        # Transform reverse bond direction (-r_hat) into local Cartesian frame of atom B
        r_loc_b = R_b.T @ (-r_hat)

        # Evaluate directional multipole projection scores
        score_a = self._eval_site_proxy_score(proxy_a, r_loc_a)
        score_b = self._eval_site_proxy_score(proxy_b, r_loc_b)

        # Bilinear coupling energy weighted by radial edge weight
        return float(edge_weight * score_a * score_b)

    # =========================================================================
    # STAGE 4: PERIODIC LATTICE GRAPH RELAXATION
    # =========================================================================

    def _relax_lattice_graph(
        self, max_sweeps: int, energy_tol: float
    ) -> list[np.ndarray]:
        """
        Performs iterative conditional modes (ICM) / mean-field relaxation across 
        the periodic interaction graph to select optimal candidate rotations.

        Parameters
        ----------
        max_sweeps : int
            Maximum number of mean-field relaxation sweeps across the lattice.
        energy_tol : float
            Convergence threshold for total graph energy change between sweeps.

        Returns
        -------
        current_rotations : list[np.ndarray]
            List of optimal 3x3 rotation matrices R_a for each atom in the cell.
        """
        # 1. Initialize all atomic frames with candidate 0
        current_rotations = [cands[0] for cands in self._candidate_sets]

        prev_total_energy = -float("inf")

        # 2. Mean-field Iterated Conditional Modes (ICM) relaxation loops
        for sweep in range(max_sweeps):
            total_sweep_energy = 0.0
            num_updated = 0

            for a_idx in range(self.natoms):
                cands = self._candidate_sets[a_idx]
                edges = self._periodic_graph[a_idx]

                best_cand = current_rotations[a_idx]
                best_site_energy = -float("inf")

                # Evaluate local interaction energy for every candidate orientation
                for R_cand in cands:
                    site_energy = 0.0
                    for edge in edges:
                        b_idx = edge["b_idx"]
                        r_hat = edge["r_hat"]
                        w_ab = edge["weight"]
                        R_b = current_rotations[b_idx]

                        # Directional multipole coupling score with neighbor b
                        site_energy += self._compute_pairwise_proxy_energy(
                            a_idx, b_idx, R_cand, R_b, r_hat, w_ab
                        )

                    if site_energy > best_site_energy:
                        best_site_energy = site_energy
                        best_cand = R_cand

                # Update site orientation if a higher-energy candidate is found
                if not np.allclose(best_cand, current_rotations[a_idx], atol=1e-5):
                    num_updated += 1
                    current_rotations[a_idx] = best_cand

                total_sweep_energy += best_site_energy

            # Each undirected bond interaction is summed twice (once from a, once from b)
            total_energy = total_sweep_energy / 2.0
            energy_diff = abs(total_energy - prev_total_energy)

            # 3. Check for graph convergence
            if num_updated == 0 or energy_diff < energy_tol:
                break

            prev_total_energy = total_energy

        return current_rotations

    # =========================================================================
    # STAGE 5: GAUGE CANONICALIZATION & SYMMETRY ORBIT ENUMERATION
    # =========================================================================

    def _get_crystal_proper_rotations(self) -> list[np.ndarray]:
        """Extracts proper SO(3) point-group rotation matrices of the crystal structure."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sga = SpacegroupAnalyzer(self.structure, symprec=1e-3)
            symm_ops = sga.get_symmetry_operations(cartesian=True)
            proper_rots = []
            for op in symm_ops:
                R_op = op.rotation_matrix
                if abs(np.linalg.det(R_op) - 1.0) < 1e-3:
                    if not any(
                        np.allclose(R_op, existing, atol=1e-3)
                        for existing in proper_rots
                    ):
                        proper_rots.append(R_op)
            if proper_rots:
                return proper_rots
        return [np.eye(3, dtype=np.float64)]

    def _canonicalize_global_gauge(
        self, rotations: list[np.ndarray]
    ) -> list[np.ndarray]:
        """
        Anchors the whole-system ground state to global Cartesian axes by selecting 
        the crystal point-group operation Q in P_crystal that maximizes alignment.

        Parameters
        ----------
        rotations : list[np.ndarray]
            List of 3x3 rotation matrices R_a resulting from lattice relaxation.

        Returns
        -------
        canonical_rotations : list[np.ndarray]
            List of gauge-fixed 3x3 rotation matrices R_a aligned with global Cartesian space.
        """
        proper_rots = self._get_crystal_proper_rotations()

        best_Q = proper_rots[0]
        best_score = -float("inf")

        # Find the global crystal point-group rotation Q that maximizes alignment
        # of all site frames with the global Cartesian identity matrix I_3x3
        for Q_cand in proper_rots:
            score = sum(np.trace(Q_cand @ R_a) for R_a in rotations)
            if score > best_score:
                best_score = score
                best_Q = Q_cand

        canonical_rotations = [best_Q @ R_a for R_a in rotations]
        return canonical_rotations

    def _finalize_symmetry_orbit(
        self, canonical_rotations: list[np.ndarray]
    ) -> SymmetryOrbit:
        """
        Identifies space-group symmetry operations of the crystal structure and 
        generates the full set of degenerate valid frame assignments across the orbit.

        Parameters
        ----------
        canonical_rotations : list[np.ndarray]
            List of 3x3 canonical rotation matrices R_a resulting from relaxation.

        Returns
        -------
        SymmetryOrbit
            NamedTuple containing the canonical frame assignment, space group symbol, 
            SO(3) symmetry matrices, and the full list of degenerate orbit frame states.
        """
        # 1. Fix global crystal gauge deterministically using crystal point-group operations
        canon_rot = self._canonicalize_global_gauge(canonical_rotations)

        # 2. Extract space group and proper symmetry operations
        proper_rots = self._get_crystal_proper_rotations()

        sg_symbol = self.structure.get_space_group_info()[0]

        orbit_rotations: list[list[np.ndarray]] = []

        # 3. Propagate canonical state across proper space-group rotations
        for R_op in proper_rots:
            orbit_state = [R_op @ R_a for R_a in canon_rot]

            # Deduplicate orbit states
            if not any(
                all(
                    np.allclose(s_a, o_a, atol=1e-3)
                    for s_a, o_a in zip(orbit_state, existing_state)
                )
                for existing_state in orbit_rotations
            ):
                orbit_rotations.append(orbit_state)

        if not orbit_rotations:
            orbit_rotations = [canon_rot]
            proper_rots = [np.eye(3, dtype=np.float64)]

        return SymmetryOrbit(
            canonical_rotations=canon_rot,
            space_group_symbol=sg_symbol,
            symmetry_operations=proper_rots,
            orbit_rotations=orbit_rotations,
        )