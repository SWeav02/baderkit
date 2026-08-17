
from pathlib import Path
from typing import Literal
import json
import re

import h5py
import numpy as np
from rich import print as rprint
from rich.progress import track
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import eigh, block_diag
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.structure import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from baderkit.post_wfc.base_env import PostWFC
from baderkit.post_wfc.projection.all_electron_dataset import AESpecies
from baderkit.post_wfc.wfc_numba import get_pbc_displacements
from baderkit import Structure
import finufft

def write_cube(
    filename: str | Path,
    structure,
    data: np.ndarray,
    origin_cart: np.ndarray,
    voxel_vectors: np.ndarray,
    comment: str = "IAO Orbital Grid",
    atom_mode: Literal["center_atom", "full_cell", "in_box"] = "in_box",
    center_atom_idx: int = 0,
) -> None:
    """Writes 3D volumetric grid data and atomic structure to a Gaussian Cube file in
    Cartesian space.

    Parameters
    ----------
    filename : str | Path
        Path to the output .cube file.
    structure : Structure
        Pymatgen structure object containing sites and elements.
    data : np.ndarray
        3D numpy array of shape (nx, ny, nz) with real orbital scalar values.
    origin_cart : np.ndarray
        Cartesian coordinates of the grid origin (0, 0, 0) in Angstroms.
    voxel_vectors : np.ndarray
        Matrix of shape (3, 3) where rows are voxel step vectors along x, y, z in Angstroms.
    comment : str, optional
        Header comment string for the cube file, by default "IAO Orbital Grid".
    atom_mode : Literal["center_atom", "full_cell", "in_box"], optional
        Mode for writing atomic sites:
        - "center_atom" (1): Writes only the single centered atom.
        - "full_cell"   (2): Writes all atoms in the unit cell structure.
        - "in_box"      (3): Writes all periodic atom images falling inside the grid box.
    center_atom_idx : int, optional
        Index of the target centered atom in `structure` for "center_atom" mode, by default 0.
    """
    ANG_TO_BOHR = 1.8897261245650618
    nx, ny, nz = data.shape

    # Convert origin and step vectors from Angstroms to Bohr directly
    origin_bohr = origin_cart * ANG_TO_BOHR
    voxels_bohr = voxel_vectors * ANG_TO_BOHR

    mode = str(atom_mode).lower().strip()
    atoms_to_write: list[tuple[int, np.ndarray]] = []

    if mode in ("1", "center_atom", "center", "single"):
        site = structure[center_atom_idx]
        z_num = getattr(site.specie, "Z", getattr(site.specie, "number", 1))
        atoms_to_write.append((z_num, site.coords * ANG_TO_BOHR))

    elif mode in ("2", "full_cell", "cell", "full"):
        for site in structure:
            z_num = getattr(site.specie, "Z", getattr(site.specie, "number", 1))
            atoms_to_write.append((z_num, site.coords * ANG_TO_BOHR))

    elif mode in ("3", "in_box", "box", "grid_box"):
        grid_box = np.array([
            nx * voxel_vectors[0],
            ny * voxel_vectors[1],
            nz * voxel_vectors[2],
        ])
        inv_grid_box = np.linalg.inv(grid_box)

        corners_unit = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
        ])
        corners_cart = origin_cart + corners_unit @ grid_box

        inv_lattice = np.linalg.inv(structure.lattice.matrix)
        corners_frac = corners_cart @ inv_lattice

        min_frac = np.floor(corners_frac.min(axis=0)).astype(int) - 1
        max_frac = np.ceil(corners_frac.max(axis=0)).astype(int) + 1

        for site in structure:
            z_num = getattr(site.specie, "Z", getattr(site.specie, "number", 1))
            for i in range(min_frac[0], max_frac[0] + 1):
                for j in range(min_frac[1], max_frac[1] + 1):
                    for k in range(min_frac[2], max_frac[2] + 1):
                        shift = (
                            i * structure.lattice.matrix[0]
                            + j * structure.lattice.matrix[1]
                            + k * structure.lattice.matrix[2]
                        )
                        pos_cart = site.coords + shift

                        grid_frac = (pos_cart - origin_cart) @ inv_grid_box
                        if np.all(grid_frac >= -0.1) and np.all(grid_frac <= 1.1):
                            atoms_to_write.append((z_num, pos_cart * ANG_TO_BOHR))
    else:
        raise ValueError(
            f"Invalid atom_mode '{atom_mode}'. Choose from 'center_atom', 'full_cell', or 'in_box'."
        )

    n_atoms = len(atoms_to_write)

    with open(filename, "w") as f:
        f.write(f"{comment}\n")
        f.write("Generated for VESTA visualization\n")

        f.write(
            f"{n_atoms:5d} {origin_bohr[0]:12.6f} {origin_bohr[1]:12.6f} {origin_bohr[2]:12.6f}\n"
        )

        f.write(
            f"{nx:5d} {voxels_bohr[0, 0]:12.6f} {voxels_bohr[0, 1]:12.6f} {voxels_bohr[0, 2]:12.6f}\n"
        )
        f.write(
            f"{ny:5d} {voxels_bohr[1, 0]:12.6f} {voxels_bohr[1, 1]:12.6f} {voxels_bohr[1, 2]:12.6f}\n"
        )
        f.write(
            f"{nz:5d} {voxels_bohr[2, 0]:12.6f} {voxels_bohr[2, 1]:12.6f} {voxels_bohr[2, 2]:12.6f}\n"
        )

        for z_num, pos_bohr in atoms_to_write:
            f.write(
                f"{z_num:5d} {0.0:12.6f} {pos_bohr[0]:12.6f} {pos_bohr[1]:12.6f} {pos_bohr[2]:12.6f}\n"
            )

        flat_data = data.ravel()
        for i in range(0, len(flat_data), 6):
            chunk = flat_data[i : i + 6]
            f.write("".join(f"{val:13.5e}" for val in chunk) + "\n")


class AtomicProjectionEnvironment:
    """
    Manages the non-bonding atomic reference states by parsing compressed analytical 
    basis binaries with pre-applied primitive normalization constants.
    """
    BOHR_TO_ANGSTROM = 0.5291772109

    def __init__(
        self, 
        post_wfc=None,
        basis_dir=None,
        align_aos=True,
        spillage_cutoff: float = 0.10,
        **kwargs
    ):
        # Register post_wfc as the reference state context link
        if post_wfc is None:
            post_wfc = PostWFC(**kwargs)

        self.post_wfc = post_wfc
        self._meta = post_wfc._meta
        self.directory = post_wfc.directory
        self._spillage_cutoff = spillage_cutoff
        
        self.basis_dir = Path(basis_dir) if basis_dir is not None else (Path(__file__).parent / "bases" / "dyall")
        
        # Pulls structure and valence_counts directly from master post_wfc context smoothly
        self.total_charge = sum(self.valence_counts[site.specie.symbol] for site in self.structure)
        
        self._cache_voxel_footprints = {}
        
        # Initialize basis structures and run heavy orbital projections
        self._load_bases()
        self._build_iaos(align_aos=align_aos)

    ###########################################################################
    # Copied Properties
    ###########################################################################
    
    @property
    def structure(self):
        return self._meta.structure
        
    @property
    def lattice(self):
        return self.post_wfc._lattice
        
    @property
    def reciprocal_lattice(self):
        return self.post_wfc._reciprocal_lattice

    @property
    def valence_counts(self):
        """The number of valence electrons assigned to each species"""
        return self.post_wfc.valence_counts

    @property
    def nspin(self) -> int:
        return self._meta.nspin
    
    @property
    def nkpoints(self) -> int:
        return self._meta.nkpts
        
    @property
    def nbands(self) -> int:
        return self._meta.nbands
        
    @property
    def occupancies(self):
        return self._meta.occupancies
        
    @property
    def energies(self):
        return self._meta.energies
    
    @property
    def energy_cutoff(self):
        return self._meta.energy_cutoff
    
    @property
    def efermi(self) -> float:
        return self._meta.efermi

    ###########################################################################
    # Convenient Properties
    ###########################################################################
    
    @property
    def basis_map(self):
        if getattr(self, "_basis_map", None) is None:
            basis_map = []
            for i_atom, site in enumerate(self.structure):
                elem = site.species_string
                basis_map.append(self.atom_bases[elem])
            self._basis_map = basis_map
        return self._basis_map
    
    @property
    def nbasis(self):
        if getattr(self, "_nbasis", None) is None:
            self._nbasis = sum(len(i.angular_momenta) for i in self.basis_map)
        return self._nbasis

    @property
    def all_bases(self):
        if getattr(self, "_all_bases", None) is None:
            all_bases = []
            atom_to_basis_indices = {i: [] for i in range(len(self.structure))}
            basis_to_atom_indices = []
            subshell_map = {}
            global_basis_idx = 0

            for atom_idx, basis in enumerate(self.basis_map):
                subshell_map[atom_idx] = {}
                atom_g_indices = []
                n_orbitals = len(basis.angular_momenta)
                has_n = hasattr(basis, "principal_quantum_numbers") and (
                    len(getattr(basis, "principal_quantum_numbers", [])) == n_orbitals
                )

                # Pass 1: Build basis list and record global indices
                for basis_idx in range(n_orbitals):
                    all_bases.append({
                        "atom_idx": atom_idx,
                        "l": basis.angular_momenta[basis_idx],
                        "m": getattr(basis, "magnetic_quantum_numbers", [0] * n_orbitals)[basis_idx],
                        "n": getattr(basis, "principal_quantum_numbers", [0] * n_orbitals)[basis_idx],
                        "q_radial_spline": getattr(basis, "q_radial_splines", [None] * n_orbitals)[basis_idx],
                    })
                    atom_g_indices.append(global_basis_idx)
                    basis_to_atom_indices.append(atom_idx)
                    global_basis_idx += 1

                atom_to_basis_indices[atom_idx] = atom_g_indices

                # Pass 2: Precalculate subshell groupings safely without index overflow
                for local_idx in range(n_orbitals):
                    l = basis.angular_momenta[local_idx]
                    n_m = 2 * l + 1

                    if l > 0:
                        # Scan backward to find the subshell starting index
                        start = local_idx
                        while start > 0:
                            prev = start - 1
                            if basis.angular_momenta[prev] != l:
                                break
                            if has_n and basis.principal_quantum_numbers[prev] != basis.principal_quantum_numbers[local_idx]:
                                break
                            if (local_idx - prev) >= n_m:
                                break
                            start = prev

                        # Clamp start index to ensure range stays within bounds
                        start = min(start, max(0, n_orbitals - n_m))
                        shell_mask = list(range(start, start + n_m))
                    else:
                        shell_mask = [local_idx]

                    target_in_shell = shell_mask.index(local_idx) if local_idx in shell_mask else 0

                    subshell_map[atom_idx][local_idx] = {
                        "global_idx": atom_g_indices[local_idx],
                        "l": l,
                        "shell_global_indices": [atom_g_indices[i] for i in shell_mask],
                        "target_in_shell_idx": target_in_shell,
                        "shell_index": target_in_shell,  # Alias for backward compatibility
                    }

            self._all_bases = all_bases
            self._atom_to_basis_indices = atom_to_basis_indices
            self._basis_to_atom_indices = basis_to_atom_indices
            self._subshell_map = subshell_map

        return self._all_bases

    @property
    def subshell_map(self):
        if getattr(self, "_subshell_map", None) is None:
            self.all_bases  # noqa: B018
        return self._subshell_map

    @property
    def atom_to_basis_indices(self):
        if getattr(self, "_atom_to_basis_indices", None) is None:
            self.all_bases  # noqa: B018
        return self._atom_to_basis_indices

    @property
    def basis_to_atom_indices(self):
        if getattr(self, "_basis_to_atom_indices", None) is None:
            self.all_bases  # noqa: B018
        return self._basis_to_atom_indices

    @property
    def basis_aug_overlaps(self):
        if getattr(self, "_basis_aug_overlaps", None) is None:
            self._basis_aug_overlaps = [i.basis_aug_overlaps for i in self.basis_map]
        return self._basis_aug_overlaps

    @property
    def voxels_near_atoms(self):
        if getattr(self, "_voxels_near_atoms", None) is None:
            self._voxels_near_atoms = self._get_voxel_footprints()
        return self._voxels_near_atoms
    
    @property
    def spillage_cutoff(self):
        return self._spillage_cutoff
    
    @property
    def spillage(self):
        return self._spillage
    
    @property
    def _iao_file(self):
        return self.directory / "iao.h5"
    
    @property
    def standardized_rotation(self):
        if getattr(self, "_standardized_rotation", None) is None:
            self._standardized_rotation = self._compute_standardization_rotation()
        return self._standardized_rotation

    def _compute_standardization_rotation(self) -> np.ndarray:
        """Computes the 3x3 rigid rotation matrix standardized_rotation that maps the input lattice
    
        into pymatgen's standardized primitive crystallographic frame.
        """
        # initialize spacegroup analyzer
        sga = SpacegroupAnalyzer(self.structure, symprec=1e-3)
        
        # get standard structure
        std_struct = sga.get_primitive_standard_structure()
    
        L_in = self.structure.lattice.matrix   # Rows are a1, a2, a3
        L_std = std_struct.lattice.matrix
    
        # Kabsch / SVD polar decomposition: L_std = L_in @ standardized_rotation.T
        H = L_in.T @ L_std
        U, S, Vt = np.linalg.svd(H)
        
        standardized_rotation = Vt.T @ U.T
        # Ensure proper rotation (det(standardized_rotation) = +1)
        if np.linalg.det(standardized_rotation) < 0.0:
            Vt_adj = Vt.copy()
            Vt_adj[-1, :] *= -1.0
            standardized_rotation = Vt_adj.T @ U.T
    
        return standardized_rotation
    
    ###########################################################################
    # Helper Orbital Functions
    ###########################################################################

    def _get_basis_labels(self) -> list[str]:
        """Generates descriptive labels for each basis function in self.all_bases including n."""
        l_map = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g"}
        m_maps = {
            0: {0: "s"},
            1: {-1: "py", 0: "pz", 1: "px"},
            2: {-2: "dxy", -1: "dyz", 0: "dz2", 1: "dxz", 2: "dx2-y2"},
            3: {
                -3: "fy(3x2-y2)", -2: "fxyz", -1: "fyz2",
                0: "fz3", 1: "fxz2", 2: "fz(x2-y2)", 3: "fx(x2-3y2)",
            },
        }

        labels = []
        atom_lm_counts = {}

        for b_info in self.all_bases:
            atom_idx = b_info["atom_idx"]
            l = b_info["l"]
            m = b_info["m"]
            n = b_info.get("n", None)

            n_str = str(n) if n is not None else ""
            l_char = l_map.get(l, f"l={l}")
            comp_str = m_maps.get(l, {}).get(m, f"m={m}")

            if comp_str:
                orb_str = f"{n_str}{comp_str}"
            else:
                orb_str = f"{n_str}{l_char}"

            key = (atom_idx, l, m, n)
            count = atom_lm_counts.get(key, 0)
            atom_lm_counts[key] = count + 1

            suffix = f"_{count}" if count > 0 else ""
            labels.append(f"{orb_str}{suffix}")

        return labels

    @staticmethod
    def _is_orbital_match(b_info: dict, label: str, global_idx: int, spec: int | str) -> bool:
        """Checks whether a basis function matches a target orbital specifier (e.g. '4s', '3d', 'px')."""
        if isinstance(spec, int):
            # Matches by angular momentum l (0, 1, 2, 3) OR global basis index
            return (spec == b_info["l"]) or (spec == global_idx)

        if isinstance(spec, str):
            s = spec.strip().lower()
            l_map = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g"}
            l = b_info["l"]
            l_str = l_map.get(l, "").lower()
            n = b_info.get("n", None)

            clean_lbl = label.lower()
            base_lbl = clean_lbl.split("_")[0]

            # Direct match against generated label
            if s == clean_lbl or s == base_lbl:
                return True

            # Pure angular momentum character match (e.g. 's', 'p', 'd', 'f')
            if s == l_str:
                return True

            # Parse optional principal quantum number prefix, e.g. '4s', '3dxy', '4px'
            import re
            match = re.match(r"^(\d+)?([a-z0-9\-]+)$", s)
            if match:
                spec_n_str, spec_orb = match.groups()
                if spec_n_str is not None:
                    spec_n = int(spec_n_str)
                    if n is not None and n != spec_n:
                        return False

                # Match remaining orbital specifier
                if spec_orb == l_str:
                    return True

                n_str = str(n) if n is not None else ""
                lbl_no_n = base_lbl[len(n_str):] if n_str and base_lbl.startswith(n_str) else base_lbl

                if spec_orb == lbl_no_n or spec_orb == base_lbl:
                    return True

        return False

    ###########################################################################
    # Primary PDOS & COOP/COHP Methods
    ###########################################################################

    def get_projected_density_of_states(
        self,
        atoms: int | str | list[int | str] | None = None,
        orbitals: str | int | list[str | int] | None = None,
        by_orbital: bool = False,
        spin_channel: int = -1,
        cumulative: bool = False,
        return_plot: bool = False,
        plot_range: tuple[float, float] | None = None,
    ) -> dict[int | str | tuple[int, str], np.ndarray] | np.ndarray:
        """
        Calculates the Atom- and/or Orbital-Projected Density of States (PDOS) 
        using precalculated orthogonalized IAOs.

        Parameters
        ----------
        atoms : int | str | list[int | str] | None, optional
            Atom index (e.g. `0`), element symbol (e.g. `"Ca"`), or a list of these.
            If None, defaults to all atoms in the structure.
        orbitals : str | int | list[str | int] | None, optional
            Orbital character or compound selection specs:
            - General orbital character: `'s'`, `'p'`, `'d'`, `'4s'`, `'3d'`
            - Element-level grouping: `'Ca-s'`, `'Ca-4s'`, `'Ca-3d'`, `'Ca-px'`
            - Site-level selection: `'0-4s'`, `'0-px'`, `'1-dz2'`
            - Specific orbital component: `'px'`, `'py'`, `'pz'`, `'dxy'`, etc.
            - Integer angular momentum $l$ (0, 1, 2, 3) or global basis index.
        by_orbital : bool, default=False
            If False, sums matched orbitals per target group.
            If True, decomposes projections down to individual atomic orbitals.
        spin_channel : int, default=-1
            Spin channel index (-1 for total/both, 0 for spin up, 1 for spin down).
        cumulative : bool, default=False
            If True, computes integrated PDOS.
        return_plot : bool, default=False
            If True, returns a plot object.
        plot_range : tuple[float, float], optional
            (E_min, E_max) energy range for plotting.

        Returns
        -------
        dict | np.ndarray
            - If `atoms` is a single `int` and `orbitals` is None (and `by_orbital=False`): np.ndarray
            - Otherwise: dictionary mapping selection keys (e.g. `0`, `'Ca-4s'`, `'3d'`, `(0, '4px')`) -> np.ndarray
        """
        def _parse_atoms(a_spec):
            if a_spec is None:
                return list(range(len(self.structure)))
            if isinstance(a_spec, int):
                return [a_spec]
            if isinstance(a_spec, str):
                if a_spec.isdigit():
                    return [int(a_spec)]
                return [
                    i for i, site in enumerate(self.structure)
                    if site.species_string.lower() == a_spec.lower() or site.specie.symbol.lower() == a_spec.lower()
                ]
            res = []
            for item in a_spec:
                res.extend(_parse_atoms(item))
            return list(dict.fromkeys(res))

        base_atoms = _parse_atoms(atoms)
        basis_labels = self._get_basis_labels()
        target_groups = []

        if orbitals is None:
            if not by_orbital:
                for atom_idx in base_atoms:
                    indices = self.atom_to_basis_indices[atom_idx]
                    target_groups.append((atom_idx, indices))
            else:
                for atom_idx in base_atoms:
                    for g_idx in self.atom_to_basis_indices[atom_idx]:
                        lbl = basis_labels[g_idx]
                        target_groups.append(((atom_idx, lbl), [g_idx]))
        else:
            orb_specs = [orbitals] if isinstance(orbitals, (int, str)) else list(orbitals)

            for spec in orb_specs:
                if isinstance(spec, str) and "-" in spec and not spec.startswith("-"):
                    parts = spec.split("-", 1)
                    spec_atoms = _parse_atoms(parts[0])
                    orb_part = parts[1]
                    group_key_override = spec
                else:
                    spec_atoms = base_atoms
                    orb_part = spec
                    group_key_override = None

                matched_indices = []
                for a_idx in spec_atoms:
                    for g_idx in self.atom_to_basis_indices[a_idx]:
                        b_info = self.all_bases[g_idx]
                        lbl = basis_labels[g_idx]
                        if self._is_orbital_match(b_info, lbl, g_idx, orb_part):
                            matched_indices.append(g_idx)

                if by_orbital:
                    for g_idx in matched_indices:
                        lbl = basis_labels[g_idx]
                        a_idx = self.all_bases[g_idx]["atom_idx"]
                        target_groups.append(((a_idx, lbl), [g_idx]))
                else:
                    key = group_key_override if group_key_override is not None else str(spec)
                    target_groups.append((key, matched_indices))

        def pdos_callback(ispin, ikpt, weight, **kwargs):
            # Fetch IAO coefficients using FBZ indexing
            C_orth_k = self.fetch_iao_coeffs(ispin, ikpt, ikpt_is_fbz=True)
            weights = np.squeeze(np.abs(C_orth_k) ** 2)
        
            band_weights = []
            for _, indices in target_groups:
                if len(indices) == 0:
                    band_weights.append(np.zeros(self.nbands, dtype=np.float64))
                    continue
        
                if weights.ndim == 1:
                    w = weights
                elif weights.shape[0] == self.nbands:
                    w = np.sum(weights[:, indices], axis=1)
                else:
                    w = np.sum(weights[indices, :], axis=0)
        
                # Include weight factor (contains rspin and BZ integration weights)
                band_weights.append(np.asarray(w * weight, dtype=np.float64).ravel())
        
            return band_weights
        
        smeared = self.post_wfc._execute_spectral_engine(
            num_metrics=len(target_groups),
            spin_channel=spin_channel,
            eval_callback=pdos_callback,
            ikpt_is_fbz=True,
        )

        pdos_dict = {}
        for i, (group_key, _) in enumerate(target_groups):
            pdos_val = smeared[i]
            if cumulative:
                pdos_val = cumulative_trapezoid(pdos_val, self.post_wfc.energy_grid, initial=0)
            pdos_dict[group_key] = pdos_val

        if return_plot:
            prefix = "Integrated " if cumulative else ""
            x_label = "States" if cumulative else "States / eV"

            tdos = self.post_wfc.get_density_of_states(
                spin_channel=spin_channel,
                cumulative=cumulative,
                return_plot=False,
            )

            plot_curves = {"total": tdos}
            for group_key, pdos_val in pdos_dict.items():
                if isinstance(group_key, tuple):
                    atom_idx, orb_lbl = group_key
                    symbol = self.structure[atom_idx].species_string
                    label = f"{prefix}PDOS ({symbol} #{atom_idx} - {orb_lbl})"
                elif isinstance(group_key, int):
                    atom_idx = group_key
                    symbol = self.structure[atom_idx].species_string
                    label = f"{prefix}PDOS ({symbol} #{atom_idx})"
                else:
                    label = f"{prefix}PDOS ({group_key})"

                plot_curves[label] = pdos_val

            return self.post_wfc._generate_property_plot(
                plot_curves=plot_curves, x_label=x_label, plot_range=plot_range, subplots=False
            )

        if isinstance(atoms, int) and orbitals is None and not by_orbital:
            return pdos_dict[atoms]

        return pdos_dict
    
    
    
    ###########################################################################
    # Spillage visualization
    ###########################################################################
    def get_iao_dos(
        self,
        spin_channel: int = -1,
        cumulative: bool = False,
        return_plot: bool = False,
        plot_range: tuple[float, float] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Calculates the total Density of States (DOS) projected onto the full IAO basis,
        the exact Total DOS from the electronic structure calculation, and the difference
        (unprojected spillage density) between them.
        """
        def iao_dos_callback(ispin, ikpt, weight, **kwargs):
            # Fetch IAO coefficients using FBZ indexing
            C_orth_k = self.fetch_iao_coeffs(ispin, ikpt, ikpt_is_fbz=True)
            # Sum norm over all IAO basis functions for each band and include integration weight
            band_iao_weight = np.sum(np.abs(C_orth_k) ** 2, axis=1) * weight
            return [band_iao_weight]
    
        smeared = self.post_wfc._execute_spectral_engine(
            num_metrics=1,
            spin_channel=spin_channel,
            eval_callback=iao_dos_callback,
            ikpt_is_fbz=True,
        )
    
        iao_dos = smeared[0]
        if cumulative:
            iao_dos = cumulative_trapezoid(iao_dos, self.post_wfc.energy_grid, initial=0)
    
        total_dos = self.post_wfc.get_density_of_states(
            spin_channel=spin_channel,
            cumulative=cumulative,
            return_plot=False,
        )
    
        diff_dos = total_dos - iao_dos
    
        dos_results = {
            "total_dos": total_dos,
            "iao_dos": iao_dos,
            "difference": diff_dos,
        }
    
        if return_plot:
            prefix = "Integrated " if cumulative else ""
            x_label = "States" if cumulative else "States / eV"
    
            plot_curves = {
                f"{prefix}Total DOS (Exact)": total_dos,
                f"{prefix}IAO Projected DOS": iao_dos,
                f"{prefix}Difference (Spillage)": diff_dos,
            }
    
            return self.post_wfc._generate_property_plot(
                plot_curves=plot_curves,
                x_label=x_label,
                plot_range=plot_range,
                subplots=False,
            )
    
        return dos_results
    
    
    def get_spillage_vs_energy(
        self,
        spin_channel: int = -1,
        as_percent: bool = True,
        return_plot: bool = False,
        plot_range: tuple[float, float] | None = None,
        tol: float = 1e-8,
    ):
        """
        Calculates spillage of the Density of States (DOS) vs energy as either an
        absolute state density difference or a percentage.
        """
        def iao_dos_callback(ispin, ikpt, weight, **kwargs):
            # Fetch IAO coefficients using FBZ indexing
            C_orth_k = self.fetch_iao_coeffs(ispin, ikpt, ikpt_is_fbz=True)
            # Sum norm over all IAO basis functions for each band and include integration weight
            band_iao_weight = np.sum(np.abs(C_orth_k) ** 2, axis=1) * weight
            return [band_iao_weight]
    
        smeared = self.post_wfc._execute_spectral_engine(
            num_metrics=1,
            spin_channel=spin_channel,
            eval_callback=iao_dos_callback,
            ikpt_is_fbz=True,
        )
    
        iao_dos = smeared[0]
    
        total_dos = self.post_wfc.get_density_of_states(
            spin_channel=spin_channel,
            cumulative=False,
            return_plot=False,
        )
    
        diff = np.maximum(0.0, total_dos - iao_dos)
    
        if as_percent:
            with np.errstate(divide="ignore", invalid="ignore"):
                spillage = np.where(total_dos > tol, (diff / total_dos) * 100.0, 0.0)
                spillage = np.clip(spillage, 0.0, 100.0)
            units_label = "Spillage (%)"
        else:
            spillage = diff
            units_label = "Spillage (States / eV)"
    
        if return_plot:
            prefix = ""
            x_label = f"{prefix}{units_label}"
    
            plot_curves = {
                f"{prefix}IAO {units_label}": spillage,
            }
    
            fig = self.post_wfc._generate_property_plot(
                plot_curves=plot_curves,
                x_label=x_label,
                plot_range=plot_range,
                subplots=False,
            )
    
            if as_percent:
                for ax in fig.get_axes():
                    ax.set_xlim(0.0, 100.0)
    
            return fig
    
        return spillage

    ###########################################################################
    # Helper Fetch Functions
    ###########################################################################
    
    def _get_mapped_atom(self, atom_idx: int, ikpt_fbz: int) -> int:
        """Finds the preimage atom index i such that rotation R maps i -> atom_idx."""
        R_recip = self.post_wfc.kpoint_rotations[ikpt_fbz]
        # Real-space transformation matrix on fractional coordinates
        R_real = np.round(np.linalg.inv(R_recip).T).astype(int)
        
        site_coords_frac = self.structure[atom_idx].frac_coords
        target_frac = R_real.T @ site_coords_frac
        
        for i, site in enumerate(self.structure):
            diff = np.mod(site.frac_coords - target_frac + 0.5, 1.0) - 0.5
            if np.all(np.abs(diff) < 1e-4):
                return i
        return atom_idx
    
    def fetch_iao_coeffs(
        self,
        ispin: int = 0,
        ikpt: int = 0,
        atom_idx: int | None = None,
        orbital_identifier: str | int | None = None,
        ikpt_is_fbz: bool = False,
        depolarized: bool = False,
    ) -> np.ndarray:
        """Retrieves IAO or Depolarized AO expansion coefficients A(k) directly from the FBZ dataset."""
        if ikpt_is_fbz:
            ikpt_idx = ikpt
        else:
            if hasattr(self.post_wfc, "irr_to_full_map"):
                ikpt_idx = self.post_wfc.irr_to_full_map[ikpt]
            else:
                ikpt_idx = np.where(self.post_wfc.full_to_irr_map == ikpt)[0][0]

        dset_name = "A_depol_coeffs" if depolarized else "A_coeffs"

        with h5py.File(self._iao_file, "r") as file:
            C_k = file[dset_name][ispin, ikpt_idx]

        if atom_idx is None:
            return C_k

        if orbital_identifier is None:
            cols = self.atom_to_basis_indices[atom_idx]
            return C_k[:, cols]

        local_idx = self.basis_map[atom_idx].get_basis_idx(orbital_identifier)
        info = self.subshell_map[atom_idx][local_idx]
        return C_k[:, info["global_idx"]]
    
    
    def fetch_iao_hamiltonian(
        self,
        ispin: int = 0,
        ikpt: int = 0,
        ikpt_is_fbz: bool = False,
    ) -> np.ndarray:
        """Retrieves the IAO Hamiltonian matrix H(k) directly from the FBZ dataset."""
        if ikpt_is_fbz:
            ikpt_idx = ikpt
        else:
            if hasattr(self.post_wfc, "irr_to_full_map"):
                ikpt_idx = self.post_wfc.irr_to_full_map[ikpt]
            else:
                ikpt_idx = np.where(self.post_wfc.full_to_irr_map == ikpt)[0][0]

        with h5py.File(self._iao_file, "r") as file:
            return file["H"][ispin, ikpt_idx]
        
    def fetch_atomic_rotations(self, atom_idx: int | None = None) -> np.ndarray:
        """
        Retrieves the optimal 3x3 SO(3) spatial rotation matrix R_a (or all matrices)
        used to frame-align the reference atomic orbitals.
    
        Parameters
        ----------
        atom_idx : int | None, optional
            If provided, returns the 3x3 rotation matrix for the specified atom index.
            If None, returns an array of shape (natoms, 3, 3) for all atoms.
    
        Returns
        -------
        rotations : np.ndarray
            3x3 rotation matrix for a single atom or (natoms, 3, 3) array for all atoms.
        """
        with h5py.File(self._iao_file, "r") as file:
            rotations = file["atomic_rotations"][:]
    
        if atom_idx is not None:
            return rotations[atom_idx]
        
        return rotations

    ###########################################################################
    # Primary Projection Functions
    ###########################################################################
    def _load_bases(self):
        """Parses NPZ binaries, filters out target elements matching cell contents,
        and enforces site-wise L2 normalization on all radial atomic orbital bases."""
        unique_elements = {site.specie.symbol for site in self.structure}
        atom_bases = {}

        for element in unique_elements:
            file_path = self.basis_dir / f"{element}.npz"
            if not file_path.exists():
                raise FileNotFoundError(f"Missing analytical basis binary for element: {file_path}")
            
            basis = AESpecies.from_file(
                file_path, 
                paw_species=self.post_wfc.paw_datasets[element],
                cutoff_radius=15,
                g_cutoff_radius=15,
                energy_range=self.post_wfc.unsmeared_energy_range,
                energy_tol=0.1
            )

            # Enforce strict L2 normalization per site orbital basis
            if hasattr(basis, "normalize"):
                basis.normalize()
            elif hasattr(basis, "orbitals"):
                for orb in basis.orbitals:
                    if hasattr(orb, "normalize"):
                        orb.normalize()
                    elif hasattr(orb, "data") and hasattr(orb, "grid"):
                        # Fallback explicit radial integration: norm = sqrt(int |R(r)|^2 r^2 dr)
                        norm_sq = np.trapz((np.abs(orb.data) ** 2) * (orb.grid ** 2), orb.grid)
                        if norm_sq > 1e-12:
                            orb.data /= np.sqrt(norm_sq)

            atom_bases[element] = basis

        self.atom_bases = atom_bases
        
    def _evaluate_spillage(self) -> tuple[float, str]:
        """
        Evaluates the maximum safe energy threshold where the continuous DOS spillage ratio
        remains below the specified cutoff, using fast 1D energy histogramming and 1D smearing.

        Returns
        -------
        tuple[float, str]
            - `max_safe_energy`: Lower bound energy limit (in eV) below which IAO projections are safe.
            - `safe_range_str`: Formatted string for UI/diagnostic logging.
        """
        spillage_cutoff = getattr(self, "spillage_cutoff", 0.20)
        sigma = getattr(self.post_wfc, "_sigma", 0.05)

        # Map IBZ band energies to FBZ k-points to match FBZ spillage dataset shape
        full_map = self.post_wfc.full_to_irr_map
        energies_fbz = self.post_wfc.energies[:, full_map, :]

        energies = energies_fbz.ravel()
        spillages = self._spillage.ravel()

        spins, spin_weight = self.post_wfc._get_spin_channels_weights(-1)

        # Assign uniform BZ weights across all FBZ k-points
        n_kpts_full = len(full_map)
        w_k = spin_weight / n_kpts_full
        weights = np.full_like(energies, w_k)

        # Filter to unoccupied states at or above Fermi level (E >= 0.0 eV)
        mask = energies >= 0.0
        if not np.any(mask):
            max_safe = float(np.max(energies))
            return max_safe, f"> {max_safe:.4f} eV (All states within cutoff)"

        e_unocc = energies[mask]
        s_unocc = spillages[mask]
        w_unocc = weights[mask]

        e_min, e_max = float(e_unocc.min()), float(e_unocc.max())
        if e_max - e_min < 1e-5:
            return e_max, f"> {e_max:.4f} eV"

        # Construct fine 1D energy grid (300 energy bins)
        num_bins = 300
        grid, delta_e = np.linspace(e_min, e_max, num_bins, retstep=True)

        # Compute k-weighted 1D histograms for total DOS and spillage DOS
        total_dos_hist, _ = np.histogram(e_unocc, bins=num_bins, range=(e_min, e_max), weights=w_unocc)
        spill_dos_hist, _ = np.histogram(e_unocc, bins=num_bins, range=(e_min, e_max), weights=w_unocc * s_unocc)

        # Apply 1D Gaussian convolution matching smearing width sigma
        if sigma > 0.0 and delta_e > 0.0:
            n_kernel = int(np.ceil(4.0 * sigma / delta_e))
            if n_kernel > 0:
                x = np.arange(-n_kernel, n_kernel + 1) * delta_e
                kernel = np.exp(-0.5 * (x / sigma) ** 2)
                kernel /= kernel.sum()

                total_dos_hist = np.convolve(total_dos_hist, kernel, mode="same")
                spill_dos_hist = np.convolve(spill_dos_hist, kernel, mode="same")

        # Compute continuous spillage ratio
        tol = 1e-8
        with np.errstate(divide="ignore", invalid="ignore"):
            dos_spillage_ratio = np.where(total_dos_hist > tol, spill_dos_hist / total_dos_hist, 0.0)

        # Locate first energy bin where continuous DOS spillage exceeds cutoff
        exceeded_indices = np.where(dos_spillage_ratio > spillage_cutoff)[0]

        if len(exceeded_indices) > 0:
            max_safe_energy = float(grid[exceeded_indices[0]])
            return max_safe_energy, f"Up to {max_safe_energy:.4f} eV"
        else:
            max_safe_energy = e_max
            return max_safe_energy, f"> {max_safe_energy:.4f} eV (All states within cutoff)"
        
    def _build_iaos(
        self,
        temperature: float = 300.0,
        align_aos: bool = True,
    ) -> None:
        """Executes the complete IAO construction pipeline.

        Parameters
        ----------
        temperature : float, optional
            Smearing temperature in Kelvin (default is 300.0 K).
        align_aos : bool, optional
            If True, executes Pass 1 & Pass 2 to calculate canonical local SO(3)
            coordinate rotations R_a for each atom before constructing frame-aligned
            IAOs in Pass 3. If False, bypasses orientation optimization and constructs
            IAOs using identity rotations (R_a = I) in a single pass (default is True).
        """
        rprint("\n" + "=" * 80)
        rprint("[bold green]          INITIATING CANONICAL IAO GENERATION PIPELINE          [/bold green]")
        rprint("=" * 80)

        if align_aos:
            # Pass 1: Unaligned projection to compute M_a tensors
            M_k_list = self._project_system(
                temperature=temperature, 
                atomic_rotations=None
            )

            if M_k_list is None:
                raise RuntimeError("Pass 1 failed to return the overlap tensor list M_k_list.")

            # Pass 2: Calculate canonical local rotations R_a per atom
            atomic_rotations = self._optimize_atomic_orientations(
                M_k_list=M_k_list, 
                deg_tol=1e-4
            )
        else:
            rprint("[bold yellow]INFO: Local frame alignment bypassed (align_aos=False). Using identity rotations (R_a = I).[/bold yellow]")
            atomic_rotations = [np.eye(3, dtype=np.float64) for _ in range(len(self.basis_map))]

        # Construct IAOs using rotations (optimized or identity) & save to HDF5
        self._project_system(
            temperature=temperature, 
            atomic_rotations=atomic_rotations
        )

        rprint("[bold green]SUCCESS: Canonical IAO construction completed successfully![/bold green]\n")


    @staticmethod
    def _lowdin_ortho(M: np.ndarray, max_iter: int = 100, tol: float = 1e-12) -> np.ndarray:
        """
        Computes M @ (M^dagger @ M)^(-1/2) using Newton-Schulz iteration
        with residual tracking on the overlap matrix X^\dagger X.
        """
        norm = np.linalg.norm(M, ord=2)
        if norm < 1e-14:
            return M

        X = M / (norm * 1.01)
        I = np.eye(M.shape[1], dtype=M.dtype)

        for _ in range(max_iter):
            X_next = 0.5 * X @ (3.0 * I - X.conj().T @ X)

            # Check orthogonality error ||X^\dagger X - I||_\infty directly
            ortho_err = np.max(np.abs(X_next.conj().T @ X_next - I))
            if ortho_err < tol:
                return X_next
            X = X_next

        return X


    def _get_atom_subshells(self, local_basis) -> list[int]:
        """Extracts true subshell quantum numbers l from local_basis.angular_momenta."""
        subshells = []
        cursor = 0
        l_list = list(local_basis.angular_momenta)
        while cursor < len(l_list):
            l = int(l_list[cursor])
            subshells.append(l)
            cursor += 2 * l + 1
        return subshells


    def _get_atom_wigner_d(self, local_basis, R_a: np.ndarray) -> np.ndarray:
        """Builds the block-diagonal Wigner D-matrix for an atom using existing helper."""
        blocks = []
        subshells = self._get_atom_subshells(local_basis)
        for l in subshells:
            D_l = self.post_wfc._get_real_sph_rotation_matrix(l, R_a)
            blocks.append(D_l)

        return block_diag(*blocks)
    
    def _get_site_symmetry_rotations(self, site_index: int) -> list[np.ndarray]:
        """
        Helper function to get the valid symmetry rotation matrices for a given site.
        
        Args:
            site_index: The index of the site in the structure.
            
        Returns:
            A list of 3x3 numpy arrays representing the symmetry rotations.
        """
        # Initialize CrystalNN to find local environment
        cnn = CrystalNN()
        
        # Get the local environment
        neighbor_sites = cnn.get_nn(self.structure, site_index)
        neighbor_sites.append(self.structure[site_index])
        site_coords = self.structure[site_index].coords
        
        # Create a Molecule object from the neighbor sites to analyze point group
        coords = [s.coords - site_coords for s in neighbor_sites]
        species = [s.species for s in neighbor_sites]
        molecule = Molecule(species, coords)
        # Analyze point group symmetry
        pga = PointGroupAnalyzer(molecule)

        # Return the valid symmetry rotation operations
        # pga.get_symmetry_operations() returns SymmOp objects
        return [op.rotation_matrix for op in pga.get_symmetry_operations()]

    @staticmethod
    def _generate_uniform_so3_rotvecs(n_samples: int = 4000, seed: int = 42) -> np.ndarray:
        """Generates uniformly distributed 3D rotation vectors over SO(3) using 
        Shoemake's quaternion sampling.
        """
        rng = np.random.default_rng(seed)
        u = rng.random((n_samples, 3))
        
        q = np.empty((n_samples, 4))
        q[:, 0] = np.sqrt(1.0 - u[:, 0]) * np.sin(2.0 * np.pi * u[:, 1])
        q[:, 1] = np.sqrt(1.0 - u[:, 0]) * np.cos(2.0 * np.pi * u[:, 1])
        q[:, 2] = np.sqrt(u[:, 0]) * np.sin(2.0 * np.pi * u[:, 2])
        q[:, 3] = np.sqrt(u[:, 0]) * np.cos(2.0 * np.pi * u[:, 2])
        
        return Rotation.from_quat(q).as_rotvec()
    
    
    def _optimize_atomic_orientations(
        self, M_k_list: list[np.ndarray], deg_tol: float = 1e-6
    ) -> list[np.ndarray]:
        """Computes optimal local coordinate rotations R_a for each atom using uniform SO(3)
        sampling and symmetry-breaking seed jitter.
        """
        atomic_rotations = []
        
        # 1. Generate isotropic uniform SO(3) rotation seeds
        base_rotvecs = self._generate_uniform_so3_rotvecs(n_samples=5000, seed=42)
    
        for atom_idx, local_basis in enumerate(self.basis_map):
            subshells = self._get_atom_subshells(local_basis)
    
            if all(l == 0 for l in subshells):
                atomic_rotations.append(np.eye(3, dtype=np.float64))
                continue
    
            non_s_indices = []
            cursor = 0
            p_subblock_idx = None
    
            for l in subshells:
                dim = 2 * l + 1
                if l > 0:
                    if l == 1 and p_subblock_idx is None:
                        p_subblock_idx = (cursor, cursor + dim)
                    non_s_indices.extend(range(cursor, cursor + dim))
                cursor += dim
    
            M_a_real = M_k_list[atom_idx].real
    
            def objective(rotvec: np.ndarray) -> float:
                R = Rotation.from_rotvec(rotvec).as_matrix()
                D = self._get_atom_wigner_d(local_basis, R)
                M_rot = np.matmul(np.matmul(D.T, M_a_real), D)
                diag_vals = np.diagonal(M_rot, axis1=-2, axis2=-1)[..., non_s_indices]
                return -float(np.sum(diag_vals**2))
    
            # Evaluate objective across uniform SO(3) grid
            grid_vals = np.array([objective(rv) for rv in base_rotvecs])
    
            # Pick top 20 candidate seeds
            top_k_indices = np.argsort(grid_vals)[:20]
            seed_rotvecs = [base_rotvecs[i] for i in top_k_indices]
    
            # Add p-orbital eigenvector seed
            if p_subblock_idx is not None:
                s_idx, e_idx = p_subblock_idx
                M_p = np.sum(M_a_real, axis=(0, 1))[s_idx:e_idx, s_idx:e_idx]
                _, V = np.linalg.eigh(M_p)
                if np.linalg.det(V) < 0:
                    V[:, -1] *= -1.0
                seed_rotvecs.append(Rotation.from_matrix(V).as_rotvec())
    
            # MULTI-START WITH SYMMETRY-BREAKING JITTER
            best_fun = np.inf
            best_R = np.eye(3)
            rng = np.random.default_rng(1234)
    
            for seed_rv in seed_rotvecs:
                # Apply 2-degree random rotvec jitter to break exact Cartesian saddle points
                jitter = rng.normal(scale=np.radians(2.0), size=3)
                jittered_rv = Rotation.from_rotvec(seed_rv) * Rotation.from_rotvec(jitter)
    
                res = minimize(
                    objective, jittered_rv.as_rotvec(), method="L-BFGS-B",
                    options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 300}
                )
                if res.fun < best_fun:
                    best_fun = res.fun
                    best_R = Rotation.from_rotvec(res.x).as_matrix()
    
            atomic_rotations.append(best_R)
    
        # 2. Collect site symmetry operations
        degen_transforms = [self._get_site_symmetry_rotations(i) for i in range(len(self.structure))]
    
        # 3. Canonicalize rotation using globally invariant R_rel = R_std @ R_cand
        R_std = self.standardized_rotation
    
        for atom_idx, local_basis in enumerate(self.basis_map):
            rotation = atomic_rotations[atom_idx]
            deg_trans = degen_transforms[atom_idx]
    
            best_trans = rotation
            best_metric = (-np.inf, -np.inf, -np.inf, -np.inf)
    
            for trans in deg_trans:
                # LEFT-MULTIPLICATION: Cartesian symmetry operation trans acts on left of rotation
                R_cand = trans @ rotation
                R_rel = R_std @ R_cand
    
                trace_val = float(np.trace(R_rel))
                metric = (trace_val, float(R_rel[0, 0]), float(R_rel[1, 1]), float(R_rel[2, 2]))
    
                if metric > best_metric:
                    best_metric = metric
                    best_trans = R_cand
    
            atomic_rotations[atom_idx] = best_trans
    
        return atomic_rotations
    
    def _project_system(
        self, 
        temperature: float = 300.0, 
        atomic_rotations: list[np.ndarray] | None = None
    ) -> list[np.ndarray] | None:
        """Constructs IAOs directly on the Full Brillouin Zone (FBZ) mesh using Fermi-Dirac 
    
        continuous occupations and Lowdin symmetric orthogonalization. Saves full standalone 
        data to HDF5 during Pass 3.
    
        Parameters
        ----------
        temperature : float, optional
            Smearing temperature in Kelvin (default is 300.0 K).
        atomic_rotations : list[np.ndarray] | None, optional
            List of 3x3 SO(3) rotation matrices R_a for each atom.
            If None, computes initial unrotated IAOs and returns the k-resolved 
            overlap tensors M_k_list across all k-points and spin channels.
            If provided, applies R_a to reference basis functions and saves rotated IAOs to disk.
    
        Returns
        -------
        M_k_list : list[np.ndarray] | None
            If atomic_rotations is None (Pass 1), returns a list of length natoms where each 
            element is an array of shape (nspin, nkpoints_full, nbasis_atom, nbasis_atom) 
            containing the k-resolved overlap matrices M_a(s, k) = S21_a(s, k) @ A_a(s, k).
            Otherwise returns None (Pass 3).
        """
        structure = self.structure
        atom_positions = structure.cart_coords
        nspin = self.nspin
        kpts_cart_full = self.post_wfc.kpoints_cart_full
        nkpoints_full = len(kpts_cart_full)
        full_to_irr = self.post_wfc.full_to_irr_map
        nbands = self.nbands
        volume = structure.volume
        nbasis = self.nbasis
    
        # PAW channel offsets (for PAW augmentation projectors)
        paw_offsets = self.post_wfc._get_atom_channel_offsets()
        total_paw_channels = paw_offsets[-1][1] if paw_offsets else 0
    
        # Reference basis channel offsets (for S21 slicing)
        basis_offsets = [
            (self.atom_to_basis_indices[i][0], self.atom_to_basis_indices[i][-1] + 1)
            for i in range(len(self.basis_map))
        ]
    
        # Calculate smearing width sigma = k_B * T in eV
        k_B = 8.617333262145e-5
        sigma = k_B * temperature
    
        # Initialize FBZ spillage array and band energy accumulator
        self._spillage = np.zeros((nspin, nkpoints_full, nbands), dtype=np.float64)
        energies_fbz = np.zeros((nspin, nkpoints_full, nbands), dtype=np.float64)
    
        # Pre-allocate 4D k-resolved overlap tensors for each atom in Pass 1:
        # Shape per atom: (nspin, nkpoints_full, nbasis_atom, nbasis_atom)
        M_k_list = (
            [
                np.zeros((nspin, nkpoints_full, end_b - start_b, end_b - start_b), dtype=np.complex128)
                for start_b, end_b in basis_offsets
            ]
            if atomic_rotations is None
            else None
        )
    
        # Precompute block-diagonal Wigner D-matrices for each atom if rotations are provided
        wigner_D_blocks = []
        if atomic_rotations is not None:
            for atom_idx, local_basis in enumerate(self.basis_map):
                R_a = atomic_rotations[atom_idx]
                D_a = self._get_atom_wigner_d(local_basis, R_a)
                wigner_D_blocks.append(D_a)
    
        rprint("\n" + "=" * 80)
        pass_str = "PASS 1: UNALIGNED" if atomic_rotations is None else "PASS 3: FRAME-ALIGNED"
        rprint(f"[bold green]          STARTING FBZ PROJECTION ({pass_str})          [/bold green]")
        rprint("=" * 80)
        rprint(f"[bold white]System Dimensions:[/bold white] Spin={nspin}, FBZ k-points={nkpoints_full}, Bands={nbands}")
        rprint(f"[bold white]Basis Dimensions :[/bold white] Total={nbasis}")
        rprint(f"[bold white]Fermi Level (E_F):[/bold white] {self.efermi:.4f} eV")
        rprint(f"[bold white]Smearing Temp (T):[/bold white] {temperature:.1f} K (sigma = {sigma:.4f} eV)")
        rprint(f"[bold white]Unit Cell Volume :[/bold white] {volume:.6f} Å^3")
    
        rprint("\n" + "=" * 80)
        rprint("[bold blue]INFO: Executing Reciprocal Projections & PAW Augmentation across FBZ[/bold blue]")
        rprint("=" * 80)
    
        # Diagnostic trackers
        max_energy_err = 0.0
        max_charge_err = 0.0
        max_trace_err = 0.0
        max_cond_num_raw = 0.0
        max_cond_num_final = 0.0
    
        # -------------------------------------------------------------------------
        # PASS 3: Open HDF5 File and Initialize Datasets / Metadata
        # -------------------------------------------------------------------------
        h5_file = None
        if atomic_rotations is not None:
            h5_file = h5py.File(self._iao_file, "w")
    
            # Save Root Metadata Attributes
            h5_file.attrs["structure_json"] = json.dumps(structure.as_dict())
            h5_file.attrs["efermi"] = float(self.efermi)
            h5_file.attrs["temperature"] = float(temperature)
            h5_file.attrs["sigma"] = float(sigma)
            h5_file.attrs["spillage_cutoff"] = float(self.spillage_cutoff)
            h5_file.attrs["nspin"] = int(nspin)
            h5_file.attrs["nkpoints_full"] = int(nkpoints_full)
            h5_file.attrs["nbands"] = int(nbands)
            h5_file.attrs["nbasis"] = int(nbasis)
            h5_file.attrs["atom_to_basis_indices_json"] = json.dumps(
                {int(k): [int(x) for x in v] for k, v in self.atom_to_basis_indices.items()}
            )
            h5_file.attrs["paw_offsets_json"] = json.dumps([(int(s), int(e)) for s, e in paw_offsets])
            h5_file.attrs["basis_offsets_json"] = json.dumps([(int(s), int(e)) for s, e in basis_offsets])
    
            # Create Core Datasets
            dset_H = h5_file.create_dataset(
                "H",
                shape=(nspin, nkpoints_full, nbasis, nbasis),
                dtype=np.complex128,
                chunks=(nspin, 1, nbasis, nbasis),
                compression="lzf",
            )
    
            dset_A = h5_file.create_dataset(
                "A_coeffs",
                shape=(nspin, nkpoints_full, nbands, nbasis),
                dtype=np.complex128,
                chunks=(nspin, 1, nbands, nbasis),
                compression="lzf",
            )
    
            dset_A_depol = h5_file.create_dataset(
                "A_depol_coeffs",
                shape=(nspin, nkpoints_full, nbands, nbasis),
                dtype=np.complex128,
                chunks=(nspin, 1, nbands, nbasis),
                compression="lzf",
            )
    
            h5_file.create_dataset("atomic_rotations", data=np.array(atomic_rotations))
            grid_group = h5_file.create_group("grid_data")
    
        try:
            # Loop over Full Brillouin Zone k points
            for ikpt_full in track(range(nkpoints_full), description="[bold blue]Constructing IAOs (FBZ)...", total=nkpoints_full):
                ikpt_irr = full_to_irr[ikpt_full]
    
                K_vecs = self.post_wfc.fetch_gvectors(
                    ikpt=ikpt_full, return_cart=True, add_k=True, ikpt_is_fbz=True
                )
                n_gvecs = K_vecs.shape[0]
    
                local_basis_matrices = []
                local_partial_overlaps = []
    
                # Collect <chi|phi_aug-phi_ps> and <chi|chi>
                for atom_idx, local_basis in enumerate(self.basis_map):
                    spatial_phase = np.exp(-1j * np.dot(K_vecs, atom_positions[atom_idx]))
    
                    local_basis_matrices.append(
                        local_basis.evaluate_q_functions(
                            K_vecs,
                            atom_positions[atom_idx],
                            spatial_phase=spatial_phase,
                        )
                    )
                    local_partial_overlaps.append(local_basis.basis_aug_overlaps)
    
                local_basis_matrix = np.vstack(local_basis_matrices)
    
                # Prepare per-kpoint grid dataset placeholders in HDF5 (Pass 3)
                dset_C_pw, dset_P_paw = None, None
                dset_C_ao_pw, dset_P_ao_paw = None, None
                if h5_file is not None:
                    k_group = grid_group.create_group(f"k_{ikpt_full}")
                    k_group.create_dataset("K_vecs", data=K_vecs, compression="lzf")
    
                    dset_C_pw = k_group.create_dataset(
                        "C_iao_pw", shape=(nspin, nbasis, n_gvecs), dtype=np.complex128, compression="lzf"
                    )
                    dset_P_paw = k_group.create_dataset(
                        "P_iao_paw", shape=(nspin, nbasis, total_paw_channels), dtype=np.complex128, compression="lzf"
                    )
                    dset_C_ao_pw = k_group.create_dataset(
                        "C_ao_pw", shape=(nspin, nbasis, n_gvecs), dtype=np.complex128, compression="lzf"
                    )
                    dset_P_ao_paw = k_group.create_dataset(
                        "P_ao_paw", shape=(nspin, nbasis, total_paw_channels), dtype=np.complex128, compression="lzf"
                    )
    
                for ispin in range(nspin):
    
                    ###################################################################
                    # S_12 and S_21 Construction
                    ###################################################################
                    # Get pseudo part of Psi on FBZ (shape: n_gvecs, nbands)
                    psi_ps = self.post_wfc.fetch_psi(
                        ikpt=ikpt_full, ispin=ispin, iband=np.arange(nbands), ikpt_is_fbz=True
                    ).T
    
                    # Calculate <chi | psi_ps> (pseudo part of projection)
                    chi_psi_ps = local_basis_matrix.conj() @ psi_ps
    
                    # Get projector overlaps <p | psi_ps> on FBZ (shape: nbands, total_paw_channels)
                    paw_psi_ps_all = self.post_wfc.fetch_projector_overlaps(
                        ikpt=ikpt_full, ispin=ispin, iband=np.arange(nbands), ikpt_is_fbz=True
                    )
    
                    # Get augmentation part
                    chi_psi_aug = []
                    for atom_idx, paw_overlap in enumerate(local_partial_overlaps):
                        start_ch, end_ch = paw_offsets[atom_idx]
                        paw_psi_ps = paw_psi_ps_all[:, start_ch:end_ch]
                        chi_psi_aug.append(np.dot(paw_overlap, paw_psi_ps.T))
                    chi_psi_aug = np.vstack(chi_psi_aug)
    
                    # Construct unrotated S21
                    S21 = chi_psi_ps + chi_psi_aug
    
                    ###################################################################
                    # Apply SO(3) Atomic Rotations to S21
                    ###################################################################
                    if atomic_rotations is not None:
                        for atom_idx in range(len(self.basis_map)):
                            start_b, end_b = basis_offsets[atom_idx]
                            D_a = wigner_D_blocks[atom_idx]
                            # Use D_a.T (or D_a.conj().T) to project into local atomic frame
                            S21[start_b:end_b, :] = D_a.T @ S21[start_b:end_b, :]
    
                    S12 = S21.conj().T
    
                    ###################################################################
                    # P_12 and P_21
                    ###################################################################
                    P12 = S12
                    P21 = S21
    
                    # Orthogonalized Depolarized AO coefficients (P12)
                    A_depol = self._lowdin_ortho(P12)
    
                    ###################################################################
                    # Get Occupations with Fermi-Dirac Smearing
                    ###################################################################
                    E = self.energies[ispin, ikpt_irr]
                    energies_fbz[ispin, ikpt_full] = E
    
                    if sigma > 0.0:
                        x = np.clip((E - self.efermi) / sigma, -100.0, 100.0)
                        f = 1.0 / (1.0 + np.exp(x))
                    else:
                        f = (E <= self.efermi).astype(np.float64)
    
                    n_occ_eff = float(np.sum(f))
                    O = np.diag(f)
    
                    ###################################################################
                    # C_tilde: depolarized space with continuous occupations
                    ###################################################################
                    f_threshold = 1e-10
                    active_mask = f > f_threshold
                    n_active = int(np.sum(active_mask))
    
                    if n_active > 0:
                        sqrt_f_active = np.sqrt(f[active_mask])
                        P21_occ = P21[:, active_mask] * sqrt_f_active[None, :]
                        C_tilde_raw = P12 @ P21_occ
    
                        C_tilde = self._lowdin_ortho(C_tilde_raw)
                        O_tilde = C_tilde @ C_tilde.conj().T
                    else:
                        O_tilde = np.zeros((nbands, nbands), dtype=np.complex128)
    
                    ###################################################################
                    # IAO Coefficients A
                    ###################################################################
                    I_bands = np.eye(nbands)
                    A_raw = (I_bands + 2 * (O @ O_tilde) - O_tilde - O) @ P12
    
                    # Measure RAW condition number before orthogonalization
                    S_raw = A_raw.conj().T @ A_raw
                    cond_num_raw = np.linalg.cond(S_raw)
    
                    # Lowdin symmetric orthogonalization
                    A = self._lowdin_ortho(A_raw)
                    A_T = A.conj().T
    
                    # Measure FINAL condition number after orthogonalization
                    S_final = A_T @ A
                    cond_num_final = np.linalg.cond(S_final)
    
                    # Track max values
                    max_cond_num_raw = max(max_cond_num_raw, cond_num_raw)
                    max_cond_num_final = max(max_cond_num_final, cond_num_final)
    
                    ###################################################################
                    # Store k-Resolved Overlap Tensor M_a(s, k) (Pass 1)
                    ###################################################################
                    if M_k_list is not None:
                        for atom_idx in range(len(self.basis_map)):
                            start_b, end_b = basis_offsets[atom_idx]
                            S21_a = S21[start_b:end_b, :]  # Shape: (nbasis_atom, nbands)
                            A_a = A[:, start_b:end_b]      # Shape: (nbands, nbasis_atom)
                            M_k_list[atom_idx][ispin, ikpt_full] = S21_a @ A_a
    
                    ###################################################################
                    # Spillage Calculation
                    ###################################################################
                    spillage_k = 1.0 - np.sum(np.abs(A) ** 2, axis=1)
                    self._spillage[ispin, ikpt_full] = np.maximum(0.0, spillage_k)
    
                    ###################################################################
                    # H_IAO Construction & Disk Persistence (Pass 3)
                    ###################################################################
                    H_IAO = A_T @ (E[:, None] * A)
    
                    if h5_file is not None:
                        dset_A[ispin, ikpt_full] = A
                        dset_A_depol[ispin, ikpt_full] = A_depol
                        dset_H[ispin, ikpt_full] = H_IAO
    
                        # Contract IAO plane waves & PAW overlaps
                        dset_C_pw[ispin] = A.T @ psi_ps.T
                        dset_P_paw[ispin] = A.T @ paw_psi_ps_all
    
                        # Contract Depolarized AO plane waves & PAW overlaps
                        dset_C_ao_pw[ispin] = A_depol.T @ psi_ps.T
                        dset_P_ao_paw[ispin] = A_depol.T @ paw_psi_ps_all
    
                    ###################################################################
                    # Verify Results
                    ###################################################################
                    IAO_density = A.conj().T @ (f[:, None] * A)
                    q_iao = np.trace(IAO_density).real
                    q_err = np.abs(q_iao - n_occ_eff)
                    max_charge_err = max(max_charge_err, q_err)
    
                    t_err = np.abs(np.trace(O_tilde).real - n_active)
                    max_trace_err = max(max_trace_err, t_err)
    
                    if n_occ_eff > nbasis:
                        rprint(
                            f"[bold red]WARNING (ikpt={ikpt_full}, ispin={ispin}):[/bold red] Effective occupied bands ({n_occ_eff:.2f}) > nbasis ({nbasis})! "
                            "Atomic basis is too small to span the occupied space."
                        )
                        max_energy_err = float("nan")
                    else:
                        eigvals_gen = eigh(H_IAO, np.eye(nbasis), eigvals_only=True)
                        check_mask = f > 0.5
                        n_check = int(np.sum(check_mask))
                        if n_check > 0:
                            E_check = np.sort(E[check_mask])
                            iao_check = np.sort(eigvals_gen[:n_check])
                            e_err = np.max(np.abs(iao_check - E_check))
                            max_energy_err = max(max_energy_err, e_err)
    
            # Evaluate energy range cutoff using helper method
            self.max_safe_energy, safe_range_str = self._evaluate_spillage()
    
            # Finalize Pass 3 HDF5 persistence
            if h5_file is not None:
                h5_file.create_dataset("energies", data=energies_fbz)
                h5_file.create_dataset("kpoints_cart", data=kpts_cart_full)
                h5_file.create_dataset("spillage", data=self._spillage)
                h5_file.attrs["max_safe_energy"] = float(self.max_safe_energy)
    
        finally:
            if h5_file is not None:
                h5_file.close()
    
        # Print final summary
        status_energy = "[bold green]PASSED[/bold green]" if max_energy_err < 1e-6 else "[bold red]FAILED[/bold red]"
        status_charge = "[bold green]PASSED[/bold green]" if max_charge_err < 1e-6 else "[bold red]FAILED[/bold red]"
        status_trace = "[bold green]PASSED[/bold green]" if max_trace_err < 1e-6 else "[bold red]FAILED[/bold red]"
        status_cond_raw = "[bold green]PASSED[/bold green]" if max_cond_num_raw < 1e8 else "[bold yellow]MODERATE[/bold yellow]"
        status_cond_final = "[bold green]PASSED[/bold green]" if max_cond_num_final < 1.01 else "[bold red]FAILED[/bold red]"
    
        rprint("\n" + "=" * 80)
        rprint(f"[bold green]          IAO PROJECTION DIAGNOSTIC SUMMARY ({pass_str})          [/bold green]")
        rprint("=" * 80)
        rprint(f" • [bold white]Occupied Energy Recovery Max Err :[/bold white] {max_energy_err:.2e} eV  [{status_energy}]")
        rprint(f" • [bold white]Integrated Charge Max Err (Q_IAO):[/bold white] {max_charge_err:.2e}     [{status_charge}]")
        rprint(f" • [bold white]Projector Rank Trace Max Err     :[/bold white] {max_trace_err:.2e}     [{status_trace}]")
        rprint(f" • [bold white]Raw Basis Max Condition Num κ(S_raw)   :[/bold white] {max_cond_num_raw:.2e}     [{status_cond_raw}]")
        rprint(f" • [bold white]Final Basis Max Condition Num κ(S_final):[/bold white] {max_cond_num_final:.6f}     [{status_cond_final}]")
        rprint(
            f" • [bold white]Safe Energy Range (spillage < {self.spillage_cutoff:.0%}):[/bold white] [bold cyan]{safe_range_str}[/bold cyan]"
        )
        rprint("=" * 80 + "\n")
    
        return M_k_list
    
    # =============================================================================
    # Real space evaluation methods
    # =============================================================================
    def _evaluate_pw_sum(
        self,
        box_size_angstrom: tuple[float, float, float],
        K_all_local: np.ndarray,
        V_all: np.ndarray,
        grid_size: tuple[int, int, int],
        eps: float = 1e-6,
        sigma_factor: float = 0.85,
        enable_apodization: bool = True,
    ) -> np.ndarray:
        """Evaluates plane-wave Fourier sums onto an unrotated 3D local grid using FINUFFT Type-1
    
        with strictly isotropic spherical K-space truncation and super-Gaussian apodization.
        """
        # 1. Clean scalar grid steps in local space
        steps = np.array(box_size_angstrom) / np.array(grid_size)
        targets_raw = K_all_local * steps  # Targets in [-pi, pi]
    
        # 2. Isotropic Spherical Nyquist Filter
        k_cart_norm = np.linalg.norm(K_all_local, axis=1)
        k_nyquist_iso = np.pi / np.max(steps)
        nyquist_mask = k_cart_norm <= k_nyquist_iso
    
        V_valid = V_all[nyquist_mask]
        targets_valid = targets_raw[nyquist_mask].T
        k_cart_norm_valid = k_cart_norm[nyquist_mask]
    
        # 3. Isotropic Super-Gaussian Apodization Filter
        if enable_apodization:
            k_ratio = k_cart_norm_valid / k_nyquist_iso
            window = np.exp(-((k_ratio / sigma_factor) ** 8))
        else:
            window = 1.0
    
        # 4. Apply windowing (phase offset already handled in _evaluate_grid_field)
        c_coeffs = np.ascontiguousarray(V_valid * window, dtype=np.complex128)
    
        # 5. Execute FINUFFT Type-1 on unrotated local grid
        phi_3d = finufft.nufft3d1(
            np.ascontiguousarray(targets_valid[0], dtype=np.float64),
            np.ascontiguousarray(targets_valid[1], dtype=np.float64),
            np.ascontiguousarray(targets_valid[2], dtype=np.float64),
            c_coeffs,
            n_modes=grid_size,
            isign=1,
            modeord=0,
            eps=eps,
        )
    
        return phi_3d.reshape(-1)
    
    def _resolve_target_orbitals(
        self,
        atom_idx: int | None = None,
        orbital_identifier: str | int | tuple[int, int] | None = None,
        sites: list[int] | None = None,
    ) -> list[tuple[int, int, int]]:
        """Resolves atom and orbital arguments into a list of tuples: (atom_idx, local_ch, global_ch)."""
        structure = self.structure
        target_items = []
    
        if atom_idx is not None and orbital_identifier is not None:
            atom_basis = self.atom_bases[structure[atom_idx].specie.symbol]
            local_ch = atom_basis.get_basis_idx(orbital_identifier)
            global_ch = self.atom_to_basis_indices[atom_idx][local_ch]
            target_items.append((atom_idx, local_ch, global_ch))
        elif atom_idx is not None:
            for local_ch, global_ch in enumerate(self.atom_to_basis_indices[atom_idx]):
                target_items.append((atom_idx, local_ch, global_ch))
        else:
            if sites is None:
                sites = list(range(len(structure)))
            for a_idx in sites:
                for local_ch, global_ch in enumerate(self.atom_to_basis_indices[a_idx]):
                    target_items.append((a_idx, local_ch, global_ch))
    
        return target_items
    
    
    def _setup_paw_augmentation(
        self,
        grid_cart_flat: np.ndarray,
        center_coords: np.ndarray,
        box_size_angstrom: tuple[float, float, float],
        atomic_rotations: np.ndarray | None,
        paw_offsets: list[tuple[int, int]],
        structure: Structure | None = None,
    ) -> list[dict]:
        """Calculates active PAW sphere intersections and radial delta_phi fields on a 3D grid."""
        if structure is None:
            structure = self.structure
    
        inv_lattice = structure.lattice.inv_matrix
        lattice_matrix = structure.lattice.matrix
        box_radius = max(box_size_angstrom) / 2.0
    
        paw_aug_data = []
    
        for a_idx, site in enumerate(structure):
            atom_basis = self.atom_bases[site.specie.symbol]
            paw_sp = atom_basis.paw_species
            rcut = atom_basis.max_paw_cutoff
    
            if np.linalg.norm(site.coords - center_coords) > (rcut + box_radius + 0.5):
                continue
    
            diff_cart = grid_cart_flat - site.coords
            cand_indices = np.where(np.all(np.abs(diff_cart) <= rcut + 0.5, axis=1))[0]
            if cand_indices.size == 0:
                continue
    
            dr_cart_min, inside_mask = get_pbc_displacements(
                diff_cart[cand_indices], inv_lattice, lattice_matrix, rcut**2
            )
    
            if dr_cart_min.shape[0] > 0:
                R_a = atomic_rotations[a_idx] if atomic_rotations is not None else np.eye(3)
                dr_local = dr_cart_min @ R_a
    
                fields = paw_sp.evaluate_basis_fields(dr_local, compute_gradients=False)
                start_ch, end_ch = paw_offsets[a_idx]
    
                paw_aug_data.append({
                    "active_indices": cand_indices[inside_mask],
                    "delta_phi": fields.phi_ae - fields.phi_ps,
                    "start_ch": start_ch,
                    "end_ch": end_ch,
                })
    
        return paw_aug_data
    
    
    def _evaluate_bare_orbitals(
        self,
        r_cart: np.ndarray,
        target_items: list[tuple[int, int, int]],
        atomic_rotations: np.ndarray | None,
        spins: list[int],
    ) -> dict[int, np.ndarray]:
        """Evaluates bare unperturbed analytical reference atomic orbitals directly from AESpecies."""
        structure = self.structure
        inv_lattice = structure.lattice.inv_matrix
        lattice_matrix = structure.lattice.matrix
        n_eval = len(target_items)
        n_pts = r_cart.shape[0]
    
        phi_real_dict = {s: np.zeros((n_eval, n_pts), dtype=np.float64) for s in spins}
    
        # Group target channels by atom site to reuse coordinate transformations
        atom_targets: dict[int, list[tuple[int, int]]] = {}
        for slot, (a_idx, local_ch, _) in enumerate(target_items):
            atom_targets.setdefault(a_idx, []).append((slot, local_ch))
    
        for a_idx, slot_info in atom_targets.items():
            site = structure[a_idx]
            atom_basis = self.atom_bases[site.specie.symbol]
    
            diff_cart = r_cart - site.coords
            dr_cart_min, _ = get_pbc_displacements(
                diff_cart, inv_lattice, lattice_matrix, rcut_sq=1e10
            )
    
            R_a = atomic_rotations[a_idx] if atomic_rotations is not None else np.eye(3)
            dr_local = dr_cart_min @ R_a
    
            phi_all = atom_basis.evaluate_r_functions(dr_local)
    
            for slot, local_ch in slot_info:
                val_arr = phi_all[local_ch]
                for s in spins:
                    phi_real_dict[s][slot, :] = val_arr
    
        return phi_real_dict
    
    def _evaluate_grid_field(
        self,
        file: h5py.File,
        spin_channel: int,
        global_basis_idx: int,
        center_coords: np.ndarray,
        box_size_angstrom: tuple[float, float, float],
        grid_size: tuple[int, int, int],
        R_a: np.ndarray,
        paw_aug_data: list[dict],
        mode_str: str,
    ) -> np.ndarray:
        """Performs FBZ plane-wave FINUFFT summation and radial PAW augmentation for one basis function."""
        grid_group = file["grid_data"]
        n_kpts_full = len(file["kpoints_cart"])
        w_fbz = 1.0 / n_kpts_full
    
        pw_key = "C_ao_pw" if mode_str == "depolarized" else "C_iao_pw"
        paw_key = "P_ao_paw" if mode_str == "depolarized" else "P_iao_paw"
    
        all_K_vecs_local, all_V_coeffs = [], []
        c_P_a_total = {
            item["start_ch"]: np.zeros(item["end_ch"] - item["start_ch"], dtype=np.complex128)
            for item in paw_aug_data
        }
    
        for ikpt_full in range(n_kpts_full):
            k_group = grid_group[f"k_{ikpt_full}"]
            K_cart = k_group["K_vecs"][:]
            
            # Rotate plane-wave vectors into local atomic frame
            K_local = K_cart @ R_a
            C_pw = k_group[pw_key][spin_channel, global_basis_idx, :]
    
            phase_K = np.exp(1j * (K_cart @ center_coords))
            all_K_vecs_local.append(K_local)
            all_V_coeffs.append(w_fbz * C_pw * phase_K)
    
            if paw_aug_data:
                P_paw = k_group[paw_key][spin_channel, global_basis_idx, :]
                for item in paw_aug_data:
                    P_a = P_paw[item["start_ch"] : item["end_ch"]]
                    c_P_a_total[item["start_ch"]] += w_fbz * P_a
    
        K_all_local = np.vstack(all_K_vecs_local)
        V_all = np.concatenate(all_V_coeffs)
    
        phi_grid_flat = self._evaluate_pw_sum(
            box_size_angstrom, K_all_local, V_all, grid_size=grid_size
        )
    
        for item in paw_aug_data:
            c_P_a = c_P_a_total[item["start_ch"]]
            phi_grid_flat[item["active_indices"]] += item["delta_phi"] @ c_P_a
    
        return np.real(phi_grid_flat).reshape(grid_size)
    
    
    # =============================================================================
    # PUBLIC EVALUATION METHODS
    # =============================================================================
    
    def evaluate_iao_on_grid(
        self,
        atom_idx: int,
        orbital_identifier: str | int | tuple[int, int],
        box_size_angstrom: tuple[float, float, float] = (8.0, 8.0, 8.0),
        grid_size: tuple[int, int, int] = (61, 61, 61),
        spin_channel: int = 0,
        include_paw_aug: bool = True,
        mode: Literal["iao", "depolarized", "bare"] = "iao",
        filename: str | Path | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluates an IAO on a 3D real-space grid in the target atom's native local coordinate frame."""
        mode_str = str(mode).lower().strip()
        structure = self.structure
        site_coords = structure[atom_idx].coords
        atomic_rotations = self.fetch_atomic_rotations()
        R_a_target = atomic_rotations[atom_idx] if atomic_rotations is not None else np.eye(3, dtype=np.float64)
    
        # 1. Native local grid coordinates
        steps = np.array(box_size_angstrom) / np.array(grid_size)
        axes_local = [(np.arange(n) - (n - 1) / 2.0) * s for n, s in zip(grid_size, steps)]
        grid_rel_3d_local = np.stack(np.meshgrid(*axes_local, indexing="ij"), axis=-1)
    
        # 2. Simulation Cartesian sampling points (for PW & PAW evaluation)
        grid_cart_3d = (grid_rel_3d_local @ R_a_target.T) + site_coords
        grid_cart_flat = grid_cart_3d.reshape(-1, 3)
    
        if mode_str == "bare":
            target_items = self._resolve_target_orbitals(atom_idx, orbital_identifier)
            local_ch = target_items[0][1]
            atom_basis = self.atom_bases[structure[atom_idx].specie.symbol]
    
            dr_local = grid_rel_3d_local.reshape(-1, 3)
            data = atom_basis.evaluate_r_functions(dr_local)[local_ch].reshape(grid_size)
        else:
            is_depol = (mode_str == "depolarized")
            c_sample = self.fetch_iao_coeffs(
                ispin=spin_channel, ikpt=0, atom_idx=atom_idx,
                orbital_identifier=orbital_identifier, ikpt_is_fbz=True, depolarized=is_depol
            )
            A_sample = self.fetch_iao_coeffs(ispin=spin_channel, ikpt=0, ikpt_is_fbz=True, depolarized=is_depol)
            iao_idx = int(np.argmin(np.linalg.norm(A_sample - c_sample[:, None], axis=0)))
    
            with h5py.File(self._iao_file, "r") as file:
                paw_offsets = json.loads(file.attrs.get("paw_offsets_json", "[]")) or self.post_wfc._get_atom_channel_offsets()
                paw_aug_data = (
                    self._setup_paw_augmentation(
                        grid_cart_flat, site_coords, box_size_angstrom, atomic_rotations, paw_offsets, structure
                    )
                    if include_paw_aug else []
                )
                data = self._evaluate_grid_field(
                    file, spin_channel, iao_idx, site_coords, box_size_angstrom, grid_size, R_a_target, paw_aug_data, mode_str
                )
    
        origin_native = np.array([axes_local[0][0], axes_local[1][0], axes_local[2][0]])
        grid_native_3d = grid_rel_3d_local
    
        if filename is not None:
            symbol = structure[atom_idx].specie.symbol
            comment = f"{mode_str.upper()} Atom {atom_idx} ({symbol}) - {orbital_identifier}"
            voxel_vectors = np.diag(steps)
    
            # Rotate real crystal lattice and coordinates into target atom's native frame
            native_lattice = structure.lattice.matrix @ R_a_target
            native_coords = (structure.cart_coords - site_coords) @ R_a_target
            native_structure = Structure(
                native_lattice,
                structure.species,
                native_coords,
                coords_are_cartesian=True
            )
    
            write_cube(filename, native_structure, data, origin_native, voxel_vectors, comment, center_atom_idx=atom_idx)
    
        return data, grid_native_3d
    
    
    def evaluate_iao_real_space(
        self,
        r_point: tuple[float, float, float] | np.ndarray,
        atom_idx: int | None = None,
        orbital_identifier: str | int | tuple[int, int] | None = None,
        sites: list | None = None,
        spin_channel: int = -1,
        coords_are_cartesian: bool = False,
        mode: Literal["iao", "depolarized", "bare"] = "iao",
    ) -> dict[int, np.ndarray]:
        """Evaluates IAO, Depolarized AO, or Bare Reference AO fields in real space at arbitrary coordinates."""
        mode_str = str(mode).lower().strip()
        if mode_str not in ("iao", "depolarized", "bare"):
            raise ValueError(f"Invalid mode '{mode}'. Choose from 'iao', 'depolarized', or 'bare'.")
    
        structure = self.structure
        r_cart = np.asarray(r_point, dtype=np.float64)
        if not coords_are_cartesian:
            r_cart = r_cart @ structure.lattice.matrix
    
        is_single_point = (r_cart.ndim == 1)
        if is_single_point:
            r_cart = r_cart[None, :]
    
        atomic_rotations = self.fetch_atomic_rotations()
        target_items = self._resolve_target_orbitals(atom_idx, orbital_identifier, sites)
    
        with h5py.File(self._iao_file, "r") as file:
            nspin = int(file.attrs.get("nspin", self.nspin))
        spins = list(range(nspin)) if spin_channel == -1 else [spin_channel]
    
        # Bare path: evaluated analytically via AESpecies
        if mode_str == "bare":
            res_dict = self._evaluate_bare_orbitals(r_cart, target_items, atomic_rotations, spins)
            return {s: arr[:, 0] for s, arr in res_dict.items()} if is_single_point else res_dict
    
        # Band/IAO path: evaluated via FINUFFT grid summation + 3D interpolation
        r_min, r_max = np.min(r_cart, axis=0), np.max(r_cart, axis=0)
        center_coords = (r_min + r_max) / 2.0
        span = r_max - r_min
    
        max_rcut = max(self.atom_bases[s.specie.symbol].max_paw_cutoff for s in structure)
        box_size_angstrom = tuple(np.maximum(span + 2 * (max_rcut + 2.0), 8.0))
    
        spacing = 0.13
        grid_size = tuple(int(np.ceil(s / spacing)) for s in box_size_angstrom)
        steps = np.array(box_size_angstrom) / np.array(grid_size)
    
        axes_rel = [(np.arange(n) - (n - 1) / 2.0) * s for n, s in zip(grid_size, steps)]
        grid_rel_3d = np.stack(np.meshgrid(*axes_rel, indexing="ij"), axis=-1)
        grid_cart_flat = (grid_rel_3d + center_coords).reshape(-1, 3)
        grid_axes_cart = [axes_rel[i] + center_coords[i] for i in range(3)]
    
        phi_real_dict = {s: np.zeros((len(target_items), r_cart.shape[0]), dtype=np.float64) for s in spins}
    
        with h5py.File(self._iao_file, "r") as file:
            paw_offsets = json.loads(file.attrs.get("paw_offsets_json", "[]")) or self.post_wfc._get_atom_channel_offsets()
            paw_aug_data = self._setup_paw_augmentation(grid_cart_flat, center_coords, box_size_angstrom, atomic_rotations, paw_offsets)
    
            for ispin in spins:
                for b_i, (_, _, global_ch) in enumerate(target_items):
                    grid_data_real = self._evaluate_grid_field(
                        file, ispin, global_ch, center_coords, grid_rel_3d, grid_size, paw_aug_data, mode_str
                    )
    
                    interpolator = RegularGridInterpolator(
                        grid_axes_cart, grid_data_real, method="cubic", bounds_error=False, fill_value=0.0
                    )
                    phi_real_dict[ispin][b_i, :] = interpolator(r_cart)
    
        return {s: arr[:, 0] for s, arr in phi_real_dict.items()} if is_single_point else phi_real_dict

    def get_atom_pair_cohp(
        self,
        site_idx_A: int,
        site_idx_B: int,
        cell_translation: tuple[int, int, int] = (0, 0, 0),
        spin_channel: int = -1,
        negative_cohp: bool = True,
        cumulative: bool = False,
        return_plot: bool = False,
        plot_range: tuple[float, float] | None = None,
        **kwargs,
    ) -> dict[str, np.ndarray] | np.ndarray:
        """
        Calculates the atom-pair Crystal Orbital Hamilton Population (COHP) between
        site_idx_A and site_idx_B (shifted by cell_translation), matching LOBSTER.
    
        Parameters
        ----------
        site_idx_A : int
            Index of the target site A.
        site_idx_B : int
            Index of the target site B.
        cell_translation : tuple[int, int, int], default=(0, 0, 0)
            Lattice unit cell translation vector [R_x, R_y, R_z] for site B.
        spin_channel : int, default=-1
            Spin channel index (-1 for total/both, 0 for spin up, 1 for spin down).
        negative_cohp : bool, default=True
            If True, returns -COHP (bonding states > 0, antibonding states < 0).
        cumulative : bool, default=False
            If True, computes integrated COHP (iCOHP).
        return_plot : bool, default=False
            If True, returns a Matplotlib Figure object.
        plot_range : tuple[float, float], optional
            (E_min, E_max) energy range for plotting.
    
        Returns
        -------
        np.ndarray | matplotlib.figure.Figure
            1D numpy array containing COHP values or a Matplotlib Figure if `return_plot=True`.
        """
        structure = self.structure
        spins, _ = self.post_wfc._get_spin_channels_weights(spin_channel)
    
        basis_A = self.atom_to_basis_indices[site_idx_A]
        basis_B = self.atom_to_basis_indices[site_idx_B]
    
        R_cell = np.asarray(cell_translation, dtype=np.float64)
        R_lattice = R_cell @ structure.lattice.matrix
    
        kpts_cart_full = self.post_wfc.kpoints_cart_full
        n_kpts_full = len(kpts_cart_full)
    
        # Precompute FBZ phase factors for real-space Fourier transforms
        fbz_phases_minus = np.exp(-1j * (kpts_cart_full @ R_lattice))  # e^{-i k . R} for H(R)
        fbz_phases_plus = np.exp(1j * (kpts_cart_full @ R_lattice))    # e^{+i k . R} for P(k)
    
        # 1. Compute constant real-space Hamiltonian matrix H_{A,B}(R) via FBZ Fourier transform
        H_AB_R = np.zeros(
            (self.post_wfc.nspin, len(basis_A), len(basis_B)), dtype=np.complex128
        )
    
        for ispin in spins:
            H_sum = np.zeros((len(basis_A), len(basis_B)), dtype=np.complex128)
            for ikpt_full in range(n_kpts_full):
                H_k = self.fetch_iao_hamiltonian(
                    ispin=ispin, ikpt=ikpt_full, ikpt_is_fbz=True
                )
                H_sub = H_k[np.ix_(basis_A, basis_B)]
                H_sum += H_sub * fbz_phases_minus[ikpt_full]
            H_AB_R[ispin] = H_sum / n_kpts_full
    
        # 2. Define population callback using constant H_{A,B}(R) and FBZ density matrix
        def population_callback(ispin, ikpt, weight, **kwargs):
            C_k = self.fetch_iao_coeffs(ispin, ikpt, ikpt_is_fbz=True)
            if C_k is None or len(C_k) == 0:
                return [None]
    
            C_A = C_k[:, basis_A]  # Shape: (nbands, n_A)
            C_B = C_k[:, basis_B]  # Shape: (nbands, n_B)
    
            H_R = H_AB_R[ispin]    # Shape: (n_A, n_B)
            phase = fbz_phases_plus[ikpt]
    
            # Matrix contraction: Re[ Sum_{mu, nu} C_{j, mu}^* H_{mu, nu}(R) C_{j, nu} e^{i k . R} ]
            band_pop = np.real(np.sum((C_A.conj() @ H_R) * C_B, axis=1) * phase)
    
            band_pop *= weight
            if negative_cohp:
                band_pop = -band_pop
    
            return [band_pop]
    
        # 3. Execute spectral engine across FBZ k-points
        smeared = self.post_wfc._execute_spectral_engine(
            num_metrics=1,
            spin_channel=spin_channel,
            eval_callback=population_callback,
            ikpt_is_fbz=True,
        )[0]
    
        if cumulative:
            smeared = cumulative_trapezoid(
                smeared, self.post_wfc.energy_grid, initial=0
            )
    
        if return_plot:
            site_A_name = f"{structure[site_idx_A].specie.symbol}({site_idx_A})"
            site_B_name = f"{structure[site_idx_B].specie.symbol}({site_idx_B})"
            prefix = "Integrated " if cumulative else ""
            cohp_label = "-COHP" if negative_cohp else "COHP"
            label = f"{prefix}{cohp_label} ({site_A_name}-{site_B_name})"
            if np.any(cell_translation):
                label += f" R={cell_translation}"
    
            pdos_dict = {label: smeared}
            return self.post_wfc._generate_property_plot(
                plot_curves=pdos_dict,
                x_label=f"{prefix}{cohp_label}",
                plot_range=plot_range,
                subplots=False,
            )
    
        return smeared
    
    def get_orbital_pair_cohp(
        self,
        orbital_pairs: list[tuple[str, str]] | tuple[str, str],
        spin_channel: int = -1,
        negative_cohp: bool = True,
        cumulative: bool = False,
        return_plot: bool = False,
        plot_range: tuple[float, float] | None = None,
        **kwargs,
    ) -> dict[str, np.ndarray] | np.ndarray:
        """
        Calculates Crystal Orbital Hamilton Population (COHP) for specific orbital pairs
        across lattice images.
    
        Parameters
        ----------
        orbital_pairs : list[tuple[str, str]] | tuple[str, str]
            List of orbital pair tuples specified as strings:
            e.g. [("0 3s 000", "0 3s 100"), ("0 3px 000", "0 3px -1-10")]
        spin_channel : int, default=-1
            Spin channel index (-1 for total/both, 0 for spin up, 1 for spin down).
        negative_cohp : bool, default=True
            If True, returns -COHP (bonding > 0, antibonding < 0).
        cumulative : bool, default=False
            If True, computes integrated COHP (iCOHP).
        return_plot : bool, default=False
            If True, returns a Matplotlib Figure object.
        plot_range : tuple[float, float], optional
            (E_min, E_max) energy range for plotting.
    
        Returns
        -------
        np.ndarray | matplotlib.figure.Figure
            1D numpy array containing COHP values or a Matplotlib Figure if return_plot=True.
        """
        if isinstance(orbital_pairs, tuple) and len(orbital_pairs) == 2 and isinstance(orbital_pairs[0], str):
            pairs_list = [orbital_pairs]
        else:
            pairs_list = list(orbital_pairs)
    
        basis_labels = self._get_basis_labels()
    
        def _parse_token(token: str) -> tuple[int, int, np.ndarray]:
            parts = token.strip().split()
            if len(parts) < 3:
                raise ValueError(
                    f"Invalid orbital specification token: '{token}'. "
                    "Expected format 'site_idx orbital_spec translation' (e.g. '0 3s 100' or '0 3s 1 0 0')."
                )
    
            site_idx = int(parts[0])
            orb_spec = parts[1]
    
            # Space-separated translation tokens (e.g., '0', '3s', '1', '0', '0')
            if len(parts) >= 5:
                R_cell = np.array([int(p) for p in parts[2:5]], dtype=np.float64)
            else:
                trans_str = parts[2]
                # Match optional minus sign followed by a single digit (handles '000', '100', '-1-10')
                trans_matches = re.findall(r"-?\d", trans_str)
                if len(trans_matches) != 3:
                    # Fallback to multi-digit matching if separated by punctuation/delimiters
                    trans_matches = re.findall(r"-?\d+", trans_str)
    
                if len(trans_matches) != 3:
                    raise ValueError(
                        f"Could not parse 3 translation integers from '{trans_str}' in token '{token}'."
                    )
    
                R_cell = np.array([int(x) for x in trans_matches], dtype=np.float64)
    
            # Match global basis index on the target atom
            matched_g_idx = None
            for g_idx in self.atom_to_basis_indices[site_idx]:
                b_info = self.all_bases[g_idx]
                lbl = basis_labels[g_idx]
                if self._is_orbital_match(b_info, lbl, g_idx, orb_spec):
                    matched_g_idx = g_idx
                    break
    
            if matched_g_idx is None:
                raise ValueError(f"Orbital specification '{orb_spec}' not found on atom {site_idx}.")
    
            return site_idx, matched_g_idx, R_cell
    
        # Parse requested orbital pairs and calculate relative translations R_rel = R_B - R_A
        parsed_pairs = []
        for pair in pairs_list:
            site_A, g_A, R_A = _parse_token(pair[0])
            site_B, g_B, R_B = _parse_token(pair[1])
            R_rel = R_B - R_A
            parsed_pairs.append((site_A, g_A, site_B, g_B, R_rel))
    
        structure = self.structure
        spins, _ = self.post_wfc._get_spin_channels_weights(spin_channel)
        kpts_cart_full = self.post_wfc.kpoints_cart_full
        n_kpts_full = len(kpts_cart_full)
    
        # Precompute real-space Hamiltonian elements H_{g_A, g_B}(R_rel) per pair
        H_pair_R = np.zeros((self.post_wfc.nspin, len(parsed_pairs)), dtype=np.complex128)
        fbz_phases_plus = []
    
        for i_pair, (_, g_A, _, g_B, R_rel) in enumerate(parsed_pairs):
            R_lattice = R_rel @ structure.lattice.matrix
            fbz_phase_minus = np.exp(-1j * (kpts_cart_full @ R_lattice))
            fbz_phase_plus = np.exp(1j * (kpts_cart_full @ R_lattice))
            fbz_phases_plus.append(fbz_phase_plus)
    
            for ispin in spins:
                H_sum = 0.0j
                for ikpt_full in range(n_kpts_full):
                    H_k = self.fetch_iao_hamiltonian(ispin=ispin, ikpt=ikpt_full, ikpt_is_fbz=True)
                    H_sum += H_k[g_A, g_B] * fbz_phase_minus[ikpt_full]
                H_pair_R[ispin, i_pair] = H_sum / n_kpts_full
    
        fbz_phases_plus = np.array(fbz_phases_plus)  # Shape: (num_pairs, n_kpts_full)
    
        # Callback mapping orbital-pair density matrices and real-space Hamiltonians across FBZ
        def population_callback(ispin, ikpt, weight, **kwargs):
            C_k = self.fetch_iao_coeffs(ispin, ikpt, ikpt_is_fbz=True)
            if C_k is None or len(C_k) == 0:
                return [None]
    
            total_band_pop = np.zeros(self.nbands, dtype=np.float64)
    
            for i_pair, (_, g_A, _, g_B, _) in enumerate(parsed_pairs):
                C_A = C_k[:, g_A]  # Shape: (nbands,)
                C_B = C_k[:, g_B]  # Shape: (nbands,)
                H_R = H_pair_R[ispin, i_pair]
                phase = fbz_phases_plus[i_pair, ikpt]
    
                # Re[ C_{j, mu}^* H_{mu, nu}(R) C_{j, nu} e^{i k . R} ]
                band_pop = np.real(C_A.conj() * H_R * C_B * phase)
                total_band_pop += band_pop
    
            total_band_pop *= weight
            if negative_cohp:
                total_band_pop = -total_band_pop
    
            return [total_band_pop]
    
        smeared = self.post_wfc._execute_spectral_engine(
            num_metrics=1,
            spin_channel=spin_channel,
            eval_callback=population_callback,
            ikpt_is_fbz=True,
        )[0]
    
        if cumulative:
            smeared = cumulative_trapezoid(smeared, self.post_wfc.energy_grid, initial=0)
    
        if return_plot:
            prefix = "Integrated " if cumulative else ""
            cohp_label = "-COHP" if negative_cohp else "COHP"
            label = f"{prefix}{cohp_label} (Orbital Pairs)"
    
            pdos_dict = {label: smeared}
            return self.post_wfc._generate_property_plot(
                plot_curves=pdos_dict,
                x_label=f"{prefix}{cohp_label}",
                plot_range=plot_range,
                subplots=False,
            )
    
        return smeared

    def get_rCOHP(
        self,
        r_point: tuple[float, float, float] | np.ndarray,
        r_cut: float,
        cumulative: bool = False,
        spin_channel: int = -1,
        return_plot: bool = False,
        plot_range: tuple[float, float] | None = None,
        negative_cohp: bool = True,
        coords_are_cartesian: bool = False,
        max_h_energy: float | None = None,
        **kwargs,
    ) -> dict[str, np.ndarray] | np.ndarray:
        """
        Calculates real-space COHP (rCOHP) using precalculated orthogonal IAOs 
        evaluated at r_point with PAW sphere augmentations.
        Uses star phase unfolding across the Full Brillouin Zone to preserve exact spatial symmetry.
        """
        if max_h_energy is None:
            max_h_energy = getattr(self, "max_safe_energy", np.inf)
    
        structure = self.structure
        if coords_are_cartesian:
            r_cart = np.asarray(r_point, dtype=np.float64)
        else:
            r_cart = np.asarray(r_point, dtype=np.float64) @ structure.lattice.matrix
    
        spins, spin_weight = self.post_wfc._get_spin_channels_weights(spin_channel)
    
        # ---------------------------------------------------------------------
        # 1. EVALUATE IAO DENSITY MATRICES AND MAP METADATA PER SPIN
        # ---------------------------------------------------------------------
        sites = structure.get_sites_in_sphere(r_cart, r_cut)
    
        phi_r_dict = self.evaluate_iao_real_space(
            r_cart, sites=sites, spin_channel=spin_channel, coords_are_cartesian=True
        )
    
        basis_coords = []
        basis_map = []
        site_map = []
    
        for i_site, site in enumerate(sites):
            unit_indices = self.atom_to_basis_indices[site.index]
            basis_map.extend(unit_indices)
            basis_coords.append(np.tile(site.coords, (len(unit_indices), 1)))
            site_map.extend([i_site] * len(unit_indices))
    
        basis_coords = np.vstack(basis_coords) if basis_coords else np.empty((0, 3))
        basis_map = np.array(basis_map, dtype=np.intp)
        site_map = np.array(site_map, dtype=np.intp)
    
        same_site_mask = site_map[:, None] == site_map[None, :]
    
        # ---------------------------------------------------------------------
        # 2. CALCULATE H & D_H PER SPIN CHANNEL (FILTERING HIGH VIRTUAL NOISE)
        # ---------------------------------------------------------------------
        kpts_cart_full = self.post_wfc.kpoints_cart_full
        full_to_irr = self.post_wfc.full_to_irr_map
        k_weights = self.post_wfc.kpoint_weights  # Shape: (nkpoints,)
        nkpts_irr = self.post_wfc.nkpoints
    
        D_H_real_dict = {}
        home_basis_pairs_dict = {}
    
        for s in spins:
            phi_r_s = phi_r_dict[s]
            D_s = np.outer(phi_r_s, phi_r_s)
    
            # Zero out intraatomic pairs (\mu, \nu on same atom)
            D_s[same_site_mask] = 0.0
    
            all_basis_pairs_s = np.argwhere(np.abs(D_s) > 1e-12)
            if len(all_basis_pairs_s) == 0:
                home_basis_pairs_dict[s] = np.empty((0, 2), dtype=np.intp)
                D_H_real_dict[s] = np.empty(0, dtype=np.float64)
                continue
    
            D_s_vec = D_s[all_basis_pairs_s[:, 0], all_basis_pairs_s[:, 1]]
    
            home_basis_pairs_s = basis_map[all_basis_pairs_s]
            home_basis_pairs_dict[s] = home_basis_pairs_s
    
            basis_vecs_s = basis_coords[all_basis_pairs_s[:, 1]] - basis_coords[all_basis_pairs_s[:, 0]]
    
            # Construct H_real filtered to bands below max_h_energy
            H_real_s = np.zeros(basis_vecs_s.shape[0], dtype=np.complex128)
            energies_s = self.post_wfc.energies[s]
    
            for ikpt in range(nkpts_irr):
                E_k = energies_s[ikpt]
                valid_bands = np.where(E_k <= max_h_energy)[0]
    
                C_k = self.fetch_iao_coeffs(s, ikpt)
                if C_k is None or len(C_k) == 0 or len(valid_bands) == 0:
                    continue
    
                # Reconstruct H_k without high virtual plane-wave continuum noise
                A_val = C_k[valid_bands, :]
                E_val = E_k[valid_bands]
                H_k_val = A_val.conj().T @ (E_val[:, None] * A_val)
    
                # Star-folded FBZ phase factor for interatomic displacement vectors to preserve spatial symmetry
                star_kvecs = kpts_cart_full[full_to_irr == ikpt]
                phases = np.mean(np.exp(-1j * (basis_vecs_s @ star_kvecs.T)), axis=1)
    
                H_mapped = H_k_val[home_basis_pairs_s[:, 0], home_basis_pairs_s[:, 1]]
                H_real_s += k_weights[ikpt] * H_mapped * phases
    
            # Pre-combine D*H point-wise for this spin channel
            D_H_real_dict[s] = D_s_vec * np.real(H_real_s)
    
        # ---------------------------------------------------------------------
        # 3. CALCULATE P: PARTIAL DENSITY MATRIX
        # ---------------------------------------------------------------------
        def population_callback(ispin, ikpt, weight, **kwargs):
            C_k = self.fetch_iao_coeffs(ispin, ikpt)
            if C_k is None or len(C_k) == 0:
                return [None]
    
            home_basis_pairs_s = home_basis_pairs_dict.get(ispin, np.empty((0, 2), dtype=np.intp))
            if len(home_basis_pairs_s) == 0:
                return [np.zeros(C_k.shape[0])]
    
            C_mu = C_k[:, home_basis_pairs_s[:, 0]]
            C_nu = C_k[:, home_basis_pairs_s[:, 1]]
    
            # Unphased density matrix element in Bloch basis
            P = C_mu.conj() * C_nu  # Shape: (nbands, N_pairs_s)
    
            D_H = D_H_real_dict[ispin]
            band_pop = np.real(np.sum(P * D_H, axis=1))
    
            band_pop *= weight
            if negative_cohp:
                band_pop = -band_pop
    
            return [band_pop]
    
        # ---------------------------------------------------------------------
        # EXECUTE SPECTRAL ENGINE
        # ---------------------------------------------------------------------
        smeared = self.post_wfc._execute_spectral_engine(
            num_metrics=1,
            spin_channel=spin_channel,
            eval_callback=population_callback,
        )[0]
    
        if cumulative:
            smeared = cumulative_trapezoid(smeared, self.post_wfc.energy_grid, initial=0)
    
        if return_plot:
            pdos_dict = {"rCOHP": smeared}
            return self.post_wfc._generate_property_plot(
                plot_curves=pdos_dict,
                x_label="-rCOHP" if negative_cohp else "rCOHP",
                plot_range=plot_range,
                subplots=False,
            )
    
        return smeared
    