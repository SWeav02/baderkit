
from pathlib import Path
from typing import Literal
import json
import warnings

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


from baderkit.post_wfc.base_env import PostWFC
from baderkit.post_wfc.projection.all_electron_dataset import AESpecies
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
        self._build_iaos()

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
            C_orth_k = self.fetch_iao_coeffs(ispin, ikpt)
            orbital_weights = np.abs(C_orth_k) ** 2

            band_weights = []
            for _, indices in target_groups:
                if len(indices) > 0:
                    w = np.sum(orbital_weights[:, indices], axis=1) * weight
                else:
                    w = np.zeros(self.nbands)
                band_weights.append(w)

            return band_weights

        smeared = self.post_wfc._execute_spectral_engine(
            num_metrics=len(target_groups),
            spin_channel=spin_channel,
            eval_callback=pdos_callback,
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
    
        Parameters
        ----------
        spin_channel : int, default=-1
            Spin channel index (-1 for total/both, 0 for spin up, 1 for spin down).
        cumulative : bool, default=False
            If True, computes integrated DOS (number of states).
        return_plot : bool, default=False
            If True, returns a plot object comparing Total DOS, IAO DOS, and the Difference.
        plot_range : tuple[float, float], optional
            (E_min, E_max) energy range for plotting.
    
        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing:
            - `'total_dos'`: Exact total DOS from the underlying calculation.
            - `'iao_dos'`: Summed projected DOS spanned by the complete IAO basis.
            - `'difference'`: Residual/spillage DOS (`total_dos - iao_dos`).
        """
        def iao_dos_callback(ispin, ikpt, weight, **kwargs):
            # C_orth_k shape: (nbands, nbasis)
            C_orth_k = self.fetch_iao_coeffs(ispin, ikpt)
            # Sum norm over all IAO basis functions for each band
            band_iao_weight = np.sum(np.abs(C_orth_k) ** 2, axis=1) * weight
            return [band_iao_weight]
    
        smeared = self.post_wfc._execute_spectral_engine(
            num_metrics=1,
            spin_channel=spin_channel,
            eval_callback=iao_dos_callback,
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
    
        Parameters
        ----------
        spin_channel : int, default=-1
            Spin channel index (-1 for total/both, 0 for spin up, 1 for spin down).
        as_percent : bool, default=True
            If True, computes percentage spillage: ((Total DOS - IAO DOS) / Total DOS) * 100.
            If False, computes absolute spillage: (Total DOS - IAO DOS).
        return_plot : bool, default=False
            If True, returns a plot object showing spillage vs energy.
        plot_range : tuple[float, float], optional
            (E_min, E_max) energy range for plotting.
        tol : float, default=1e-8
            Minimum Total DOS threshold below which percentage spillage is set to 0.0%
            to avoid division-by-zero artifacts.
    
        Returns
        -------
        np.ndarray | matplotlib.figure.Figure
            1D numpy array containing spillage values or a Matplotlib Figure if `return_plot=True`.
        """
        def iao_dos_callback(ispin, ikpt, weight, **kwargs):
            # C_orth_k shape: (nbands, nbasis)
            C_orth_k = self.fetch_iao_coeffs(ispin, ikpt)
            # Sum norm over all IAO basis functions for each band
            band_iao_weight = np.sum(np.abs(C_orth_k) ** 2, axis=1) * weight
            return [band_iao_weight]
    
        smeared = self.post_wfc._execute_spectral_engine(
            num_metrics=1,
            spin_channel=spin_channel,
            eval_callback=iao_dos_callback,
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
    ) -> np.ndarray:
        """Retrieves IAO expansion coefficients A(k) directly from the FBZ dataset."""
        if ikpt_is_fbz:
            ikpt_idx = ikpt
        else:
            if hasattr(self.post_wfc, "irr_to_full_map"):
                ikpt_idx = self.post_wfc.irr_to_full_map[ikpt]
            else:
                ikpt_idx = np.where(self.post_wfc.full_to_irr_map == ikpt)[0][0]

        with h5py.File(self._iao_file, "r") as file:
            C_k = file["A_coeffs"][ispin, ikpt_idx]

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
        if hasattr(self, "atomic_rotations") and self.atomic_rotations is not None:
            rotations = np.array(self.atomic_rotations)
        else:
            with h5py.File(self._iao_file, "r") as file:
                if "atomic_rotations" in file:
                    rotations = file["atomic_rotations"][:]
                else:
                    # Default fallback: Identity matrices if frame alignment was skipped
                    rotations = np.tile(np.eye(3, dtype=np.float64), (len(self.structure), 1, 1))
    
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
        
    def _build_iaos(self, temperature: float = 300.0) -> None:
        """
        Executes the complete 3-pass IAO construction pipeline:
          1. Pass 1: Computes initial unaligned IAOs to extract overlap tensor M_a.
          2. Pass 2: Optimizes per-atom canonical local coordinate rotations R_a.
          3. Pass 3: Constructs frame-aligned IAOs and persists full dataset to HDF5.

        Parameters
        ----------
        temperature : float, optional
            Smearing temperature in Kelvin (default is 300.0 K).
        """
        rprint("\n" + "=" * 80)
        rprint("[bold green]          INITIATING CANONICAL IAO GENERATION PIPELINE          [/bold green]")
        rprint("=" * 80)

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

        # Pass 3: Construct frame-aligned IAOs using optimized rotations & save to HDF5
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

    def _optimize_atomic_orientations(
        self, 
        M_k_list: list[np.ndarray], 
        deg_tol: float = 1e-4
    ) -> list[np.ndarray]:
        """
        Computes optimal canonical 3x3 SO(3) local coordinate rotations R_a for each atom.
    
        Uses closed-form hierarchical sub-diagonalization across three strict tiers:
          1. Primary: Eigendecomposition of total M_3x3 accumulated across p-subshells.
          2. Secondary: Subspace tie-breaking via CrystalNN neighbor shell tensor K_geom.
          3. Tertiary: Subspace tie-breaking via an orthonormalized unit-cell lattice frame 
             tensor K_cell (guarantees zero angular skew and strict covariance).
    
        Atoms containing strictly s-orbitals bypass optimization and return identity.
    
        Parameters
        ----------
        M_k_list : list[np.ndarray]
            List of k-resolved overlap tensors M_a(s, k) from Pass 1.
        deg_tol : float, optional
            Tolerance threshold for grouping degenerate eigenvalues (default is 1e-4).
    
        Returns
        -------
        atomic_rotations : list[np.ndarray]
            List of length natoms containing 3x3 SO(3) rotation matrices R_a.
        """
        structure = self.structure
        atom_positions = structure.cart_coords
        atomic_rotations = []
    
        # -------------------------------------------------------------------------
        # Tertiary Anchor: Build Orthonormal Lattice Tensor K_cell (Gram-Schmidt)
        # -------------------------------------------------------------------------
        lat_mat = structure.lattice.matrix  # Rows are a1, a2, a3
        
        # Construct strictly orthonormal frame {e1, e2, e3} from lattice vectors
        e1 = lat_mat[0] / np.linalg.norm(lat_mat[0])
        e2_proj = lat_mat[1] - np.dot(lat_mat[1], e1) * e1
        e2 = e2_proj / np.linalg.norm(e2_proj)
        e3 = np.cross(e1, e2)
        norm_e3 = np.linalg.norm(e3)
        if norm_e3 > 1e-12:
            e3 /= norm_e3
    
        # K_cell built from orthogonal e_i has e1, e2, e3 as exact principal axes (zero skew)
        K_cell = 3.0 * np.outer(e1, e1) + 2.0 * np.outer(e2, e2) + 1.0 * np.outer(e3, e3)
        K_cell = 0.5 * (K_cell + K_cell.T)
    
        # Initialize CrystalNN while suppressing pymatgen oxidation state warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cnn = CrystalNN(weighted_cn=True)
    
        rprint("\n" + "=" * 80)
        rprint("[bold blue]PASS 2: OPTIMIZING CANONICAL LOCAL ATOMIC ROTATIONS[/bold blue]")
        rprint("=" * 80)
    
        for atom_idx, local_basis in enumerate(self.basis_map):
            subshells = self._get_atom_subshells(local_basis)
    
            # Pure s-orbital atoms require no rotation
            if all(l == 0 for l in subshells):
                atomic_rotations.append(np.eye(3, dtype=np.float64))
                rprint(
                    f" • Atom {atom_idx:3d} ({structure[atom_idx].species_string}): "
                    "Pure s-basis -> Rotation bypassed (R_a = I)"
                )
                continue
    
            # ---------------------------------------------------------------------
            # Step 1: Average M_a over spin and k-points and extract M_3x3
            # ---------------------------------------------------------------------
            M_a = M_k_list[atom_idx]  # Shape: (nspin, nkpoints_full, nbasis_atom, nbasis_atom)
            M_avg = np.mean(M_a.real, axis=(0, 1))
            M_avg = 0.5 * (M_avg + M_avg.T)
    
            M_3x3 = np.zeros((3, 3), dtype=np.float64)
            cursor = 0
            for l in subshells:
                dim = 2 * l + 1
                if l == 1:
                    p_slice = slice(cursor, cursor + 3)
                    M_3x3 += M_avg[p_slice, p_slice]
                cursor += dim
    
            # ---------------------------------------------------------------------
            # Step 2: Build Neighbor Shell Alignment Tensor via CrystalNN (K_geom)
            # ---------------------------------------------------------------------
            K_geom = np.zeros((3, 3), dtype=np.float64)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                nn_info = cnn.get_nn_info(structure, atom_idx)
    
            for entry in nn_info:
                neighbor_site = entry["site"]
                vec = neighbor_site.coords - atom_positions[atom_idx]
                r = np.linalg.norm(vec)
                if r > 1e-5:
                    u_vec = vec / r
                    weight = entry.get("weight", 1.0)
                    K_geom += weight * np.outer(u_vec, u_vec)
    
            K_geom = 0.5 * (K_geom + K_geom.T)
    
            # ---------------------------------------------------------------------
            # Step 3: Exact Closed-Form Hierarchical Sub-Diagonalization
            # ---------------------------------------------------------------------
            # Tier 1: Diagonalize Primary Tensor M_3x3
            vals_M, U = np.linalg.eigh(M_3x3)
            idx_sort = np.argsort(vals_M)[::-1]
            vals_M = vals_M[idx_sort]
            U = U[:, idx_sort]
    
            # Partition into degenerate eigenvalue clusters
            clusters_M = []
            i = 0
            while i < len(vals_M):
                cluster = [i]
                j = i + 1
                while j < len(vals_M) and abs(vals_M[i] - vals_M[j]) < deg_tol:
                    cluster.append(j)
                    j += 1
                clusters_M.append(cluster)
                i = j
    
            U_canon = U.copy()
    
            # Tier 2: Diagonalize Secondary Tensor K_geom within M_3x3 degenerate subspaces
            for cluster in clusters_M:
                if len(cluster) == 1:
                    continue
    
                idx = np.array(cluster)
                U_sub = U_canon[:, idx]  # Shape: (3, k)
    
                K_sub_geom = U_sub.T @ K_geom @ U_sub
                vals_K, R_K = np.linalg.eigh(K_sub_geom)
                idx_K = np.argsort(vals_K)[::-1]
                vals_K = vals_K[idx_K]
                R_K = R_K[:, idx_K]
    
                # Partition into remaining degenerate clusters of K_geom
                clusters_K = []
                m = 0
                while m < len(vals_K):
                    sub_cluster = [m]
                    n = m + 1
                    while n < len(vals_K) and abs(vals_K[m] - vals_K[n]) < deg_tol:
                        sub_cluster.append(n)
                        n += 1
                    clusters_K.append(sub_cluster)
                    m = n
    
                R_combined = R_K.copy()
    
                # Tier 3: Diagonalize Orthonormal K_cell within remaining degenerate subspaces
                for sub_cluster in clusters_K:
                    if len(sub_cluster) == 1:
                        continue
    
                    sub_idx = np.array(sub_cluster)
                    U_sub_sub = U_sub @ R_K[:, sub_idx]
                    K_sub_cell = U_sub_sub.T @ K_cell @ U_sub_sub
    
                    vals_cell, R_cell = np.linalg.eigh(K_sub_cell)
                    idx_cell = np.argsort(vals_cell)[::-1]
                    R_cell = R_cell[:, idx_cell]
    
                    R_combined[:, sub_idx] = R_K[:, sub_idx] @ R_cell
    
                U_canon[:, idx] = U_sub @ R_combined
    
            # ---------------------------------------------------------------------
            # Step 4: Enforce Deterministic Sign Phase & SO(3) Right-Handedness
            # ---------------------------------------------------------------------
            for col in (0, 1):
                max_row = np.argmax(np.abs(U_canon[:, col]))
                if U_canon[max_row, col] < 0.0:
                    U_canon[:, col] *= -1.0
    
            U_canon[:, 2] = np.cross(U_canon[:, 0], U_canon[:, 1])
            norm_col2 = np.linalg.norm(U_canon[:, 2])
            if norm_col2 > 1e-12:
                U_canon[:, 2] /= norm_col2
    
            atomic_rotations.append(U_canon)
    
            rprint(
                f" • Atom {atom_idx:3d} ({structure[atom_idx].species_string}): "
                f"Optimized local rotation R_a det = {np.linalg.det(U_canon):+.4f}"
            )
    
        rprint("=" * 80 + "\n")
        return atomic_rotations
    
    def _project_system(
        self, 
        temperature: float = 300.0, 
        atomic_rotations: list[np.ndarray] | None = None
    ) -> list[np.ndarray] | None:
        """
        Constructs IAOs directly on the Full Brillouin Zone (FBZ) mesh using Fermi-Dirac 
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
            containing the k-resolved overlap matrices M_a(k, s) = S21_a(k, s) @ A_a(k, s).
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
    
        # Initialize k-resolved overlap tensor list if running Pass 1
        M_k_list = None
        if atomic_rotations is None:
            M_k_list = [
                np.zeros((nspin, nkpoints_full, end_b - start_b, end_b - start_b), dtype=np.complex128)
                for start_b, end_b in basis_offsets
            ]
    
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
                if h5_file is not None:
                    k_group = grid_group.create_group(f"k_{ikpt_full}")
                    k_group.create_dataset("K_vecs", data=K_vecs, compression="lzf")
    
                    dset_C_pw = k_group.create_dataset(
                        "C_iao_pw", shape=(nspin, nbasis, n_gvecs), dtype=np.complex128, compression="lzf"
                    )
                    dset_P_paw = k_group.create_dataset(
                        "P_iao_paw", shape=(nspin, nbasis, total_paw_channels), dtype=np.complex128, compression="lzf"
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
                    # Apply SO(3) Atomic Rotations to S21 (if atomic_rotations passed)
                    ###################################################################
                    if atomic_rotations is not None:
                        for atom_idx in range(len(self.basis_map)):
                            start_b, end_b = basis_offsets[atom_idx]
                            D_a = wigner_D_blocks[atom_idx]
                            # Apply block-diagonal rotation D_a(R_a) directly to S21
                            S21[start_b:end_b, :] = D_a @ S21[start_b:end_b, :]
    
                    S12 = S21.conj().T
    
                    ###################################################################
                    # P_12 and P_21
                    ###################################################################
                    P12 = S12
                    P21 = S21
    
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
                    # Store k-Resolved Overlap Tensor M_a(k, s) (Pass 1)
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
                        dset_H[ispin, ikpt_full] = H_IAO
    
                        # Contract plane waves over band index: C_iao_pw = A.T @ psi_ps.T
                        # Resulting shape: (nbasis, n_gvecs)
                        dset_C_pw[ispin] = A.T @ psi_ps.T
    
                        # Contract PAW projector overlaps over band index: P_iao_paw = A.T @ paw_psi_ps_all
                        # Resulting shape: (nbasis, total_paw_channels)
                        dset_P_paw[ispin] = A.T @ paw_psi_ps_all
    
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
    
    ###########################################################################
    # IAO Evaluation
    ###########################################################################
    
    def _evaluate_pw_sum(
        self,
        grid_rel_3d: np.ndarray,
        K_all: np.ndarray,
        V_all: np.ndarray,
        grid_size: tuple[int, int, int],
        eps: float = 1e-6,
        sigma_factor: float = 0.85,  # Smooth roll-off parameter
    ) -> np.ndarray:
        """
        Evaluates plane-wave Fourier sums onto a 3D grid using FINUFFT Type-1 (NUFFT3D1)
        with super-Gaussian apodization to eliminate Gibbs ringing.
        """
        steps = grid_rel_3d[1, 1, 1] - grid_rel_3d[0, 0, 0]
    
        # Calculate grid center offsets
        offsets = grid_rel_3d[0, 0, 0] + (np.array(grid_size) / 2.0) * steps
        targets_raw = K_all * steps  # Shape: (N, 3)
    
        # 1. Nyquist filter
        nyquist_mask = np.all(np.abs(targets_raw) <= np.pi, axis=1)
        K_valid = K_all[nyquist_mask]
        V_valid = V_all[nyquist_mask]
        targets_valid = targets_raw[nyquist_mask].T
    
        # 2. Gaussian Apodization Filter to eliminate Gibbs ringing
        # Suppresses coefficients near the Nyquist limit (|K * step| -> pi) smoothly
        k_norm_sq = np.sum((targets_valid.T / np.pi) ** 2, axis=1)
        window = np.exp(-((k_norm_sq / sigma_factor) ** 4))  # Super-Gaussian window
    
        # 3. Apply phase modulation and window
        phase_offset = K_valid @ offsets
        c_coeffs = np.ascontiguousarray(
            V_valid * window * np.exp(1j * phase_offset), dtype=np.complex128
        )
    
        # 4. NUFFT Execution
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
    
    def evaluate_iao_on_grid(
        self,
        atom_idx: int,
        orbital_identifier: str | int,
        box_size_angstrom: tuple[float, float, float] = (8.0, 8.0, 8.0),
        grid_size: tuple[int, int, int] = (60, 60, 60),
        spin_channel: int = 0,
        include_paw_aug: bool = True,
        filename: str | Path | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluates a frame-aligned Intrinsic Atomic Orbital (IAO) on a 3D real-space grid
        using pre-contracted HDF5 grid data.
        """
        structure = self.structure
        site_coords = structure[atom_idx].coords
        inv_lattice = structure.lattice.inv_matrix
        lattice_matrix = structure.lattice.matrix
    
        # Resolve exact global IAO basis index via self.fetch_iao_coeffs
        c_sample = self.fetch_iao_coeffs(
            ispin=spin_channel, ikpt=0, atom_idx=atom_idx, orbital_identifier=orbital_identifier, ikpt_is_fbz=True
        )
        A_sample = self.fetch_iao_coeffs(ispin=spin_channel, ikpt=0, ikpt_is_fbz=True)
        iao_idx = int(np.argmin(np.linalg.norm(A_sample - c_sample[:, None], axis=0)))
    
        # =========================================================================
        # STEP 1: Real-Space Grid Construction
        # =========================================================================
        steps = np.array(box_size_angstrom) / np.array(grid_size)
        axes = [(np.arange(n) - (n - 1) / 2.0) * s for n, s in zip(grid_size, steps)]
        grid_rel_3d = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        grid_cart_3d = grid_rel_3d + site_coords
        grid_cart_flat = grid_cart_3d.reshape(-1, 3)
    
        # Open HDF5 archive containing contracted IAO grid fields
        with h5py.File(self._iao_file, "r") as file:
            paw_offsets = json.loads(file.attrs["paw_offsets_json"])
            kpts_cart = file["kpoints_cart"][:]
            atomic_rotations = file["atomic_rotations"][:] if "atomic_rotations" in file else None
            n_kpts_full = len(kpts_cart)
            w_fbz = 1.0 / n_kpts_full
    
            # =========================================================================
            # STEP 2: PAW Augmentation Setup
            # =========================================================================
            paw_aug_data = []
    
            if include_paw_aug:
                box_radius = max(box_size_angstrom) / 2.0
                for a_idx, site in enumerate(structure):
                    atom_basis = self.atom_bases[site.specie.symbol]
                    paw_sp = atom_basis.paw_species
                    rcut = atom_basis.max_paw_cutoff
    
                    if np.linalg.norm(site.coords - site_coords) > (rcut + box_radius + 0.5):
                        continue
    
                    diff_cart = grid_cart_flat - site.coords
                    cand_indices = np.where(np.all(np.abs(diff_cart) <= rcut + 0.5, axis=1))[0]
                    if cand_indices.size == 0:
                        continue
    
                    dr_frac = diff_cart[cand_indices] @ inv_lattice
                    dr_cart_min = (dr_frac - np.round(dr_frac)) @ lattice_matrix
                    inside = np.linalg.norm(dr_cart_min, axis=1) <= rcut
    
                    if np.any(inside):
                        # Rotate displacement vectors into the atom's local frame
                        R_a = atomic_rotations[a_idx] if atomic_rotations is not None else np.eye(3)
                        dr_local = dr_cart_min[inside] @ R_a
    
                        fields = paw_sp.evaluate_basis_fields(
                            dr_local, compute_gradients=False
                        )
                        start_ch, end_ch = paw_offsets[a_idx]
                        paw_aug_data.append(
                            {
                                "active_indices": cand_indices[inside],
                                "delta_phi": fields.phi_ae - fields.phi_ps,
                                "c_P_a_total": np.zeros(
                                    end_ch - start_ch, dtype=np.complex128
                                ),
                                "start_ch": start_ch,
                                "end_ch": end_ch,
                            }
                        )
    
            # =========================================================================
            # STEP 3: Brillouin Zone Integration & FBZ Accumulation
            # =========================================================================
            all_K_vecs, all_V_coeffs = [], []
            grid_group = file["grid_data"]
    
            for ikpt_full in range(n_kpts_full):
                k_group = grid_group[f"k_{ikpt_full}"]
                K_cart = k_group["K_vecs"][:]
                C_pw = k_group["C_iao_pw"][spin_channel, iao_idx, :]
    
                phase_K = np.exp(1j * (K_cart @ site_coords))
    
                all_K_vecs.append(K_cart)
                all_V_coeffs.append(w_fbz * C_pw * phase_K)
    
                if include_paw_aug and paw_aug_data:
                    P_paw = k_group["P_iao_paw"][spin_channel, iao_idx, :]
                    for item in paw_aug_data:
                        P_a = P_paw[item["start_ch"] : item["end_ch"]]
                        item["c_P_a_total"] += w_fbz * P_a
    
        K_all = np.vstack(all_K_vecs)
        V_all = np.concatenate(all_V_coeffs)
    
        # =========================================================================
        # STEP 4: FINUFFT Plane-Wave Summation & PAW Corrections
        # =========================================================================
        phi_grid_flat = self._evaluate_pw_sum(
            grid_rel_3d, K_all, V_all, grid_size=grid_size
        )
    
        for item in paw_aug_data:
            phi_grid_flat[item["active_indices"]] += (
                item["delta_phi"] @ item["c_P_a_total"]
            )
    
        # =========================================================================
        # STEP 5: Cube Export & Result Formatting
        # =========================================================================
        data = np.real(phi_grid_flat).reshape(grid_size)
    
        if filename is not None:
            symbol = structure[atom_idx].specie.symbol
            voxel_vectors = np.diag(steps)
            comment = f"IAO Atom {atom_idx} ({symbol}) - {orbital_identifier}"
            write_cube(
                filename, structure, data, grid_cart_3d[0, 0, 0], voxel_vectors, comment
            )
    
        return data, grid_cart_3d
    
    def evaluate_iao_real_space(
        self,
        r_point: tuple[float, float, float] | np.ndarray,
        sites: list | None = None,
        spin_channel: int = -1,
        coords_are_cartesian: bool = False,
    ) -> dict[int, np.ndarray]:
        """Evaluates frame-aligned Intrinsic Atomic Orbitals (IAOs) in real space at arbitrary 
        coordinate(s) r_point by Fourier-transforming the Bloch IAO representations across the 
        Full Brillouin Zone using FINUFFT grid evaluation and 3D cubic spline interpolation.
    
        Parameters
        ----------
        r_point : tuple[float, float, float] | np.ndarray
            Cartesian or fractional coordinates of shape (3,) or (N_points, 3).
        sites : list | None, optional
            List of atom indices whose IAO basis functions to evaluate (default is all atoms).
        spin_channel : int, optional
            Spin channel index (-1 for all spin channels, or 0/1).
        coords_are_cartesian : bool, optional
            If True, r_point is treated as Cartesian Ångström coordinates; otherwise fractional.
    
        Returns
        -------
        phi_iao_real_dict : dict[int, np.ndarray]
            Dictionary mapping spin channel indices to real-space IAO values of shape 
            (n_eval_basis, N_points) or (n_eval_basis,) for single points.
        """
        structure = self.structure
        inv_lattice = structure.lattice.inv_matrix
        lattice_matrix = structure.lattice.matrix
    
        # Parse input coordinates into Cartesian (N_pts, 3)
        r_cart = np.asarray(r_point, dtype=np.float64)
        if not coords_are_cartesian:
            r_cart = r_cart @ lattice_matrix
    
        is_single_point = (r_cart.ndim == 1)
        if is_single_point:
            r_cart = r_cart[None, :]  # Shape: (1, 3)
    
        N_pts = r_cart.shape[0]
    
        # Resolve atom indices and target IAO basis mapping
        if sites is None:
            sites = list(range(len(structure)))
    
        basis_map = []
        for atom_idx in sites:
            if atom_idx in self.atom_to_basis_indices:
                unit_indices = [int(x) for x in self.atom_to_basis_indices[atom_idx]]
            else:
                unit_indices = [int(x) for x in self.atom_to_basis_indices[str(atom_idx)]]
            basis_map.extend(unit_indices)
    
        basis_map = np.array(basis_map, dtype=np.intp)
        n_eval_basis = len(basis_map)
    
        # =========================================================================
        # STEP 1: Adaptive Bounding Box & Regular Grid Setup
        # =========================================================================
        r_min = np.min(r_cart, axis=0)
        r_max = np.max(r_cart, axis=0)
        center_coords = (r_min + r_max) / 2.0
    
        span = r_max - r_min
        max_rcut = max(self.atom_bases[site.specie.symbol].max_paw_cutoff for site in structure)
        box_padding = max_rcut + 2.0  # Safety buffer for PAW spheres and spline boundary conditions
    
        box_size_angstrom = tuple(np.maximum(span + 2 * box_padding, 8.0))
        spacing = 0.13  # Grid spacing in Ångströms
        grid_size = tuple(int(np.ceil(s / spacing)) for s in box_size_angstrom)
    
        steps = np.array(box_size_angstrom) / np.array(grid_size)
        axes_rel = [(np.arange(n) - (n - 1) / 2.0) * s for n, s in zip(grid_size, steps)]
        grid_rel_3d = np.stack(np.meshgrid(*axes_rel, indexing="ij"), axis=-1)
        grid_cart_3d = grid_rel_3d + center_coords
        grid_cart_flat = grid_cart_3d.reshape(-1, 3)
    
        # Absolute Cartesian axes for 3D Interpolator
        grid_axes_cart = [
            axes_rel[0] + center_coords[0],
            axes_rel[1] + center_coords[1],
            axes_rel[2] + center_coords[2],
        ]
    
        # =========================================================================
        # STEP 2: PAW Augmentation Setup on Regular Grid (Frame-Aligned)
        # =========================================================================
        with h5py.File(self._iao_file, "r") as file:
            nspin = int(file.attrs["nspin"])
            paw_offsets = json.loads(file.attrs["paw_offsets_json"])
            kpts_cart = file["kpoints_cart"][:]
            atomic_rotations = file["atomic_rotations"][:] if "atomic_rotations" in file else None
            n_kpts_full = len(kpts_cart)
            w_fbz = 1.0 / n_kpts_full
            grid_group = file["grid_data"]
    
            if spin_channel == -1:
                spins = list(range(nspin))
            else:
                spins = [spin_channel]
    
            paw_aug_data = []
            box_radius = max(box_size_angstrom) / 2.0
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
    
                dr_frac = diff_cart[cand_indices] @ inv_lattice
                dr_cart_min = (dr_frac - np.round(dr_frac)) @ lattice_matrix
                inside = np.linalg.norm(dr_cart_min, axis=1) <= rcut
    
                if np.any(inside):
                    # Rotate displacement vectors into atom a_idx's local canonical frame
                    R_a = atomic_rotations[a_idx] if atomic_rotations is not None else np.eye(3)
                    dr_local = dr_cart_min[inside] @ R_a
    
                    fields = paw_sp.evaluate_basis_fields(
                        dr_local, compute_gradients=False
                    )
                    start_ch, end_ch = paw_offsets[a_idx]
                    paw_aug_data.append(
                        {
                            "active_indices": cand_indices[inside],
                            "delta_phi": fields.phi_ae - fields.phi_ps,
                            "start_ch": start_ch,
                            "end_ch": end_ch,
                        }
                    )
    
            phi_iao_real_dict = {s: np.zeros((n_eval_basis, N_pts), dtype=np.float64) for s in spins}
    
            # =========================================================================
            # STEP 3: BZ Fourier Summation (FINUFFT) & Spline Interpolation
            # =========================================================================
            for ispin in spins:
                for b_i, iao_idx in enumerate(basis_map):
                    all_K_vecs = []
                    all_V_coeffs = []
    
                    c_P_a_total_dict = {
                        item["start_ch"]: np.zeros(item["end_ch"] - item["start_ch"], dtype=np.complex128)
                        for item in paw_aug_data
                    }
    
                    # Contract plane waves and PAW overlaps over Full Brillouin Zone
                    for ikpt_full in range(n_kpts_full):
                        k_group = grid_group[f"k_{ikpt_full}"]
                        K_cart = k_group["K_vecs"][:]
                        C_pw = k_group["C_iao_pw"][ispin, iao_idx, :]
    
                        # Phase shift for origin offset at center_coords
                        phase_K = np.exp(1j * (K_cart @ center_coords))
    
                        all_K_vecs.append(K_cart)
                        all_V_coeffs.append(w_fbz * C_pw * phase_K)
    
                        if paw_aug_data:
                            P_paw = k_group["P_iao_paw"][ispin, iao_idx, :]
                            for item in paw_aug_data:
                                P_a = P_paw[item["start_ch"] : item["end_ch"]]
                                c_P_a_total_dict[item["start_ch"]] += w_fbz * P_a
    
                    K_all = np.vstack(all_K_vecs)
                    V_all = np.concatenate(all_V_coeffs)
    
                    # FINUFFT Type-1/2 fast plane-wave grid evaluation
                    phi_grid_flat = self._evaluate_pw_sum(
                        grid_rel_3d, K_all, V_all, grid_size=grid_size
                    )
    
                    # Add PAW radial corrections on regular grid points
                    for item in paw_aug_data:
                        c_P_a = c_P_a_total_dict[item["start_ch"]]
                        phi_grid_flat[item["active_indices"]] += item["delta_phi"] @ c_P_a
    
                    grid_data_real = np.real(phi_grid_flat).reshape(grid_size)
    
                    # Interpolate from 3D regular grid to sample coordinates
                    interpolator = RegularGridInterpolator(
                        grid_axes_cart, grid_data_real, method="cubic", bounds_error=False, fill_value=0.0
                    )
                    phi_iao_real_dict[ispin][b_i, :] = interpolator(r_cart)
    
        if is_single_point:
            return {s: phi_arr[:, 0] for s, phi_arr in phi_iao_real_dict.items()}
        else:
            return phi_iao_real_dict
    
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
        """
        structure = self.structure
        spins, spin_weight = self.post_wfc._get_spin_channels_weights(spin_channel)

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

        # 2. Define population callback using constant H_{A,B}(R) and k-dependent density matrix
        def population_callback(ispin, ikpt_full, weight, **kwargs):
            C_k = self.fetch_iao_coeffs(ispin, ikpt_full, ikpt_is_fbz=True)
            if C_k is None or len(C_k) == 0:
                return [None]

            C_A = C_k[:, basis_A]  # Shape: (nbands, n_A)
            C_B = C_k[:, basis_B]  # Shape: (nbands, n_B)

            H_R = H_AB_R[ispin]    # Shape: (n_A, n_B)
            phase = fbz_phases_plus[ikpt_full]

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
            label = f"COHP {site_A_name}-{site_B_name}"
            if np.any(cell_translation):
                label += f" R={cell_translation}"

            pdos_dict = {label: smeared}
            return self.post_wfc._generate_property_plot(
                plot_curves=pdos_dict,
                x_label="-COHP" if negative_cohp else "COHP",
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
    