
from pathlib import Path
import psutil

import h5py
import numpy as np
from rich import print as rprint
from rich.progress import track
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import eigh, inv, sqrtm

from baderkit.post_wfc.base_env import PostWFC
from baderkit.post_wfc.projection.all_electron_dataset import AESpecies
from baderkit.post_wfc.wfc_numba import evaluate_real_harmonics_multi
from baderkit.toolkit.grid import Grid
import finufft

def write_cube(
    filename: str | Path,
    structure,
    data: np.ndarray,
    origin_cart: np.ndarray,
    voxel_vectors: np.ndarray,
    comment: str = "IAO Orbital Grid",
) -> None:
    """Writes 3D volumetric grid data and atomic structure to a Gaussian Cube file.

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
    """
    # Angstrom to Bohr conversion factor for Gaussian Cube format
    ANG_TO_BOHR = 1.8897261245650618
    nx, ny, nz = data.shape
    n_atoms = len(structure)

    # Convert coordinates and step vectors from Angstroms to Bohr
    origin_bohr = origin_cart * ANG_TO_BOHR
    voxels_bohr = voxel_vectors * ANG_TO_BOHR

    with open(filename, "w") as f:
        # Lines 1 & 2: Comments
        f.write(f"{comment}\n")
        f.write("Generated for VESTA visualization\n")

        # Line 3: Number of atoms + origin (in Bohr)
        f.write(
            f"{n_atoms:5d} {origin_bohr[0]:12.6f} {origin_bohr[1]:12.6f} {origin_bohr[2]:12.6f}\n"
        )

        # Lines 4-6: Grid dimensions + voxel step vectors (in Bohr)
        f.write(
            f"{nx:5d} {voxels_bohr[0, 0]:12.6f} {voxels_bohr[0, 1]:12.6f} {voxels_bohr[0, 2]:12.6f}\n"
        )
        f.write(
            f"{ny:5d} {voxels_bohr[1, 0]:12.6f} {voxels_bohr[1, 1]:12.6f} {voxels_bohr[1, 2]:12.6f}\n"
        )
        f.write(
            f"{nz:5d} {voxels_bohr[2, 0]:12.6f} {voxels_bohr[2, 1]:12.6f} {voxels_bohr[2, 2]:12.6f}\n"
        )

        # Atomic positions (Atomic number, charge=0.0, X, Y, Z in Bohr)
        for site in structure:
            z_num = getattr(site.specie, "Z", getattr(site.specie, "number", 1))
            pos_bohr = site.coords * ANG_TO_BOHR
            f.write(
                f"{z_num:5d} {0.0:12.6f} {pos_bohr[0]:12.6f} {pos_bohr[1]:12.6f} {pos_bohr[2]:12.6f}\n"
            )

        # Volumetric scalar data: 6 values per line in (ix, iy, iz) loop order
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
        self._project_system()

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
            basis_atom_indices = []
            global_basis_idx = 0

            for atom_idx, basis in enumerate(self.basis_map):
                for basis_idx in range(len(basis.angular_momenta)):
                    # Extract principal quantum number n if available
                    n_val = None
                    if hasattr(basis, "principal_quantum_numbers") and basis.principal_quantum_numbers is not None:
                        n_val = basis.principal_quantum_numbers[basis_idx]
                    elif hasattr(basis, "n_quantum") and basis.n_quantum is not None:
                        n_val = basis.n_quantum[basis_idx]
                    elif hasattr(basis, "n") and basis.n is not None:
                        n_val = basis.n[basis_idx]

                    all_bases.append({
                        "atom_idx": atom_idx,
                        "l": basis.angular_momenta[basis_idx],
                        "m": basis.magnetic_quantum_numbers[basis_idx],
                        "n": n_val,
                        "q_radial_spline": basis.q_radial_splines[basis_idx],
                    })
                    atom_to_basis_indices[atom_idx].append(global_basis_idx)
                    basis_atom_indices.append(atom_idx)
                    global_basis_idx += 1

            self._all_bases = all_bases
            self._atom_to_basis_indices = atom_to_basis_indices
            self._basis_atom_indices = basis_atom_indices
        return self._all_bases

    @property
    def atom_to_basis_indices(self):
        if getattr(self, "_atom_to_basis_indices", None) is None:
            self.all_bases  # noqa: B018
        return self._atom_to_basis_indices

    @property
    def basis_atom_indices(self):
        if getattr(self, "_basis_atom_indices", None) is None:
            self.all_bases  # noqa: B018
        return self._basis_atom_indices

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

    def fetch_iao_coeffs(self, ispin, ikpt):
        with h5py.File(self._iao_file, "r") as file:
            return file["A_coeffs"][ispin, ikpt]

    def fetch_mo_coeffs(self, ispin, ikpt):
        with h5py.File(self._iao_file, "r") as file:
            return file["mo_coeffs"][ispin, ikpt]

    def fetch_iao_hamiltonian(self, ispin, ikpt):
        with h5py.File(self._iao_file, "r") as file:
            return file["H"][ispin, ikpt]

    def fetch_iao_overlap(self, ispin, ikpt):
        """Fetches the IAO overlap matrix S for a given spin and k-point."""
        with h5py.File(self._iao_file, "r") as file:
            return file["S"][ispin, ikpt]
        
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
    
        energies = self.energies.ravel()
        spillages = self._spillage.ravel()
        
        spins, spin_weight = self.post_wfc._get_spin_channels_weights(-1)
    
        # Expand k-point weights across spins and bands
        weights = np.tile(
            np.repeat(self.post_wfc.kpoint_weights * spin_weight, self.nbands),
            self.nspin
        )
    
        # Filter to unoccupied states at or above Fermi level (E >= 0.0 eV)
        mask = energies >= 0.0
        if not np.any(mask):
            max_safe = float(np.max(self.energies))
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
    
    def _project_system(self, temperature: float = 300.0):
        """
        Constructs IAOs using Fermi-Dirac continuous occupations for enhanced 
        projection stability (especially in metallic systems) and eigh-based 
        inverse square roots for matrix orthogonalization.
    
        Parameters
        ----------
        temperature : float, optional
            Smearing temperature in Kelvin (default is 300.0 K).
            Set to 0.0 K for step-function occupations at E <= E_F.
        """
        structure = self.structure
        atom_positions = structure.cart_coords
        nspin = self.nspin
        nkpoints = self.nkpoints
        nbands = self.nbands
        volume = structure.volume  # Omega
        nbasis = self.nbasis
        atom_offsets = self.post_wfc._get_atom_channel_offsets()
    
        # Calculate smearing width sigma = k_B * T in eV
        # k_B = 8.617333262145e-5 eV/K
        k_B = 8.617333262145e-5
        sigma = k_B * temperature
    
        # Initialize spillage array
        self._spillage = np.zeros((nspin, nkpoints, nbands), dtype=np.float64)
    
        rprint("\n" + "=" * 80)
        rprint("[bold green]          STARTING PROJECTION          [/bold green]")
        rprint("=" * 80)
        rprint(f"[bold white]System Dimensions:[/bold white] Spin={nspin}, k-points={nkpoints}, Bands={nbands}")
        rprint(f"[bold white]Basis Dimensions :[/bold white] Total={nbasis}")
        rprint(f"[bold white]Fermi Level (E_F):[/bold white] {self.efermi:.4f} eV")
        rprint(f"[bold white]Smearing Temp (T):[/bold white] {temperature:.1f} K (sigma = {sigma:.4f} eV)")
        rprint(f"[bold white]Unit Cell Volume :[/bold white] {volume:.6f} Å^3")
    
        rprint("\n" + "=" * 80)
        rprint("[bold blue]INFO: Executing Reciprocal Projections & PAW Augmentation[/bold blue]")
        rprint("=" * 80)
    
        # Diagnostic trackers
        max_energy_err = 0.0
        max_charge_err = 0.0
        max_trace_err = 0.0
        max_cond_num_raw = 0.0
        max_cond_num_final = 0.0
    
        with h5py.File(self._iao_file, "w") as file:
            dset_H = file.create_dataset(
                "H",
                shape=(nspin, nkpoints, nbasis, nbasis),
                dtype=np.complex128,
                chunks=(nspin, 1, nbasis, nbasis),
                compression="lzf",
            )
    
            dset_A = file.create_dataset(
                "A_coeffs",
                shape=(nspin, nkpoints, nbands, nbasis),
                dtype=np.complex128,
                chunks=(nspin, 1, nbands, nbasis),
                compression="lzf",
            )
    
            # Loop over k points
            for ikpt in track(range(nkpoints), description="[bold blue]Constructing IAOs...", total=nkpoints):
    
                K_vecs = self.post_wfc.fetch_gvectors(ikpt, return_cart=True, add_k=True)
    
                local_basis_matrices = []
                local_partial_overlaps = []
    
                # collect <chi|phi_aug-phi_ps> and <chi|chi>
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
    
                for ispin in range(nspin):
    
                    ###################################################################
                    # S_12 and S_21
                    ###################################################################
                    # Get pseudo part of Psi
                    psi_ps = self.post_wfc.fetch_psi(ikpt=ikpt, ispin=ispin, iband=np.arange(nbands)).T
                    # calculate <chi | psi_ps> (pseudo part of projection)
                    chi_psi_ps = local_basis_matrix.conj() @ psi_ps
    
                    # get projector overlaps <p | psi_ps>
                    paw_psi_ps_all = self.post_wfc.fetch_projector_overlaps(ikpt=ikpt, ispin=ispin, iband=np.arange(nbands))
    
                    # Get augmentation part
                    chi_psi_aug = []
                    for atom_idx, paw_overlap in enumerate(local_partial_overlaps):
                        start_ch, end_ch = atom_offsets[atom_idx]
                        paw_psi_ps = paw_psi_ps_all[:, start_ch:end_ch]
                        chi_psi_aug.append(np.dot(paw_overlap, paw_psi_ps.T))
                    chi_psi_aug = np.vstack(chi_psi_aug)
    
                    # Construct S21/S12
                    S21 = chi_psi_ps + chi_psi_aug
                    S12 = S21.conj().T
    
                    ###################################################################
                    # P_12 and P_21
                    ###################################################################
                    P12 = S12
                    P21 = S21
    
                    ###################################################################
                    # Get Occupations with Fermi-Dirac Smearing
                    ###################################################################
                    E = self.energies[ispin, ikpt]
                    
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
                    
                        # SVD-based orthogonalization on C_tilde_raw directly
                        U_c, s_c, Vh_c = np.linalg.svd(C_tilde_raw, full_matrices=False)
                        C_tilde = U_c @ Vh_c  # Equivalent to C_tilde_raw @ (C_tilde_raw^\dagger C_tilde_raw)^{-1/2}
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
                    
                    # Direct SVD Löwdin Orthogonalization on A_raw
                    # A_raw has shape (nbands, nbasis) -> U is (nbands, nbasis), Vh is (nbasis, nbasis)
                    U_a, s_a, Vh_a = np.linalg.svd(A_raw, full_matrices=False)
                    A = U_a @ Vh_a
                    A_T = A.conj().T
                    
                    # Measure FINAL condition number after orthogonalization (Should be 1.0)
                    S_final = A_T @ A
                    cond_num_final = np.linalg.cond(S_final)
                    
                    # Track max values
                    max_cond_num_raw = max(max_cond_num_raw, cond_num_raw)
                    max_cond_num_final = max(max_cond_num_final, cond_num_final)
    
                    ###################################################################
                    # Spillage Calculation
                    ###################################################################
                    spillage_k = 1.0 - np.sum(np.abs(A) ** 2, axis=1)
                    self._spillage[ispin, ikpt] = np.maximum(0.0, spillage_k)
    
                    ###################################################################
                    # H_IAO and S_IAO construction
                    ###################################################################
                    H_IAO = A_T @ (E[:, None] * A)
                    S_IAO = A_T @ A
    
                    ###################################################################
                    # Save Results to disk
                    ###################################################################
                    dset_A[ispin, ikpt] = A
                    dset_H[ispin, ikpt] = H_IAO
    
                    ###################################################################
                    # Verify Results
                    ###################################################################
                    # 1. Total Integrated Charge Check
                    # Q_IAO = Tr(A^\dagger O A) must equal sum(f_n) = n_occ_eff
                    IAO_density = A.conj().T @ (f[:, None] * A)  # A^\dagger @ O @ A
                    q_iao = np.trace(IAO_density).real
                    q_err = np.abs(q_iao - n_occ_eff)
                    max_charge_err = max(max_charge_err, q_err)
                    
                    # 2. Depolarized Space Trace Check
                    # Tr(O_tilde) must equal rank(C_tilde) = n_active
                    t_err = np.abs(np.trace(O_tilde).real - n_active)
                    max_trace_err = max(max_trace_err, t_err)
                    
                    # 3. Raw vs Final Condition Number
                    cond_num_raw = np.linalg.cond(S_IAO)
                    max_cond_num_raw= max(max_cond_num_raw, cond_num_raw)
                    
                    # 4. Energy Recovery Check (for states with f_n > 0.5)
                    if n_occ_eff > nbasis:
                        rprint(
                            f"[bold red]WARNING (ikpt={ikpt}, ispin={ispin}):[/bold red] Effective occupied bands ({n_occ_eff:.2f}) > nbasis ({nbasis})! "
                            "Atomic basis is too small to span the occupied space."
                        )
                        max_energy_err = float("nan")
                    else:
                        eigvals_gen = eigh(H_IAO, np.eye(nbasis), eigvals_only=True)  # S_IAO_final is Identity
                        check_mask = f > 0.5
                        n_check = int(np.sum(check_mask))
                        if n_check > 0:
                            E_check = np.sort(E[check_mask])
                            iao_check = np.sort(eigvals_gen[:n_check])
                            e_err = np.max(np.abs(iao_check - E_check))
                            max_energy_err = max(max_energy_err, e_err)
    
        # Evaluate energy range cutoff using helper method
        self.max_safe_energy, safe_range_str = self._evaluate_spillage()
    
        # Print final summary
        status_energy = "[bold green]PASSED[/bold green]" if max_energy_err < 1e-6 else "[bold red]FAILED[/bold red]"
        status_charge = "[bold green]PASSED[/bold green]" if max_charge_err < 1e-6 else "[bold red]FAILED[/bold red]"
        status_trace = "[bold green]PASSED[/bold green]" if max_trace_err < 1e-6 else "[bold red]FAILED[/bold red]"
        status_cond_raw = "[bold green]PASSED[/bold green]" if max_cond_num_raw < 1e8 else "[bold yellow]MODERATE[/bold yellow]"
        status_cond_final = "[bold green]PASSED[/bold green]" if max_cond_num_final < 1e8 else "[bold yellow]MODERATE[/bold yellow]"
        
        rprint("\n" + "=" * 80)
        rprint("[bold green]          IAO PROJECTION DIAGNOSTIC SUMMARY          [/bold green]")
        rprint("=" * 80)
        rprint(f" • [bold white]Occupied Energy Recovery Max Err :[/bold white] {max_energy_err:.2e} eV  [{status_energy}]")
        rprint(f" • [bold white]Integrated Charge Max Err (Q_IAO):[/bold white] {max_charge_err:.2e}     [{status_charge}]")
        rprint(f" • [bold white]Projector Rank Trace Max Err     :[/bold white] {max_trace_err:.2e}     [{status_trace}]")
        rprint(f" • [bold white]Raw Basis Max Condition Num κ(S_raw)   :[/bold white] {max_cond_num_raw:.2e}     [{status_cond_raw}]")
        rprint(f" • [bold white]Final Basis Max Condition Num κ(S_final):[/bold white] {max_cond_num_final:.6f}     [{status_cond_final}")
        rprint(
            f" • [bold white]Safe Energy Range (spillage < {self.spillage_cutoff:.0%}):[/bold white] [bold cyan]{safe_range_str}[/bold cyan]"
        )
        rprint("=" * 80 + "\n")
    
    def _evaluate_pw_sum(
        self,
        grid_rel_3d: np.ndarray,
        K_all: np.ndarray,
        V_all: np.ndarray,
        grid_size: tuple[int, int, int],
        eps: float = 1e-6,
    ) -> np.ndarray:
        """Evaluates plane waves on a uniform 3D grid using FINUFFT Type 1
        (nufft3d1).
    
        Maps non-uniform plane waves K_all onto the regular real-space grid.
        """
        nx, ny, nz = grid_size
    
        # 1. Extract grid spacing and origin reference coordinates
        dx = grid_rel_3d[:, 0, 0, 0]
        dy = grid_rel_3d[0, :, 0, 1]
        dz = grid_rel_3d[0, 0, :, 2]
    
        step_x = dx[1] - dx[0]
        step_y = dy[1] - dy[0]
        step_z = dz[1] - dz[0]
    
        # Reference origin for centered mode ordering (modeord=1)
        x_offset = dx[0] + (nx // 2) * step_x
        y_offset = dy[0] + (ny // 2) * step_y
        z_offset = dz[0] + (nz // 2) * step_z
    
        # 2. Modulate plane wave coefficients with the origin phase shift
        phase_offset = (
            K_all[:, 0] * x_offset + K_all[:, 1] * y_offset + K_all[:, 2] * z_offset
        )
        c_coeffs = np.ascontiguousarray(
            V_all * np.exp(1j * phase_offset), dtype=np.complex128
        )
    
        # 3. Scale wavevectors to non-uniform coordinates in [-pi, pi]
        # Modulo wrapping ensures points lie strictly within FINUFFT's required domain
        x_target = np.ascontiguousarray(
            (K_all[:, 0] * step_x + np.pi) % (2 * np.pi) - np.pi, dtype=np.float64
        )
        y_target = np.ascontiguousarray(
            (K_all[:, 1] * step_y + np.pi) % (2 * np.pi) - np.pi, dtype=np.float64
        )
        z_target = np.ascontiguousarray(
            (K_all[:, 2] * step_z + np.pi) % (2 * np.pi) - np.pi, dtype=np.float64
        )
    
        # 4. Execute Type 1 NUFFT (Non-uniform sources -> Uniform 3D grid modes)
        phi_3d = finufft.nufft3d1(
            x_target,
            y_target,
            z_target,
            c_coeffs,
            n_modes=(nx, ny, nz),
            isign=1,
            modeord=1,
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
        structure = self.structure
        site_coords = structure[atom_idx].coords
        nx, ny, nz = grid_size
        Lx, Ly, Lz = box_size_angstrom
        inv_lattice = structure.lattice.inv_matrix
        lattice_matrix = structure.lattice.matrix
        recip_matrix = structure.lattice.reciprocal_lattice.matrix
    
        # =========================================================================
        # STEP 1: Target Global Basis & Subshell Resolution
        # =========================================================================
        basis_indices = self.atom_to_basis_indices[atom_idx]
        symbol = structure[atom_idx].specie.symbol
        local_basis = self.atom_bases[symbol]
    
        if isinstance(orbital_identifier, int):
            match_idx = orbital_identifier
        else:
            orbital_names = getattr(
                local_basis,
                "orbital_labels",
                [str(i) for i in range(len(basis_indices))],
            )
            match_idx = next(
                (
                    i
                    for i, name in enumerate(orbital_names)
                    if str(orbital_identifier).lower() in str(name).lower()
                ),
                None,
            )
            if match_idx is None:
                raise ValueError(
                    f"Orbital '{orbital_identifier}' not found for atom {symbol}({atom_idx})."
                )
    
        target_basis_idx = basis_indices[match_idx]
        l_target = local_basis.angular_momenta[match_idx]
        n_m = 2 * l_target + 1
    
        if l_target > 0:
            if hasattr(local_basis, "magnetic_quantum_numbers"):
                m_arr = local_basis.magnetic_quantum_numbers
                m_val = m_arr[match_idx]
                m_offset = (
                    m_val + l_target
                    if m_val < 0
                    else (m_val if m_val < n_m else m_val + l_target)
                )
                subshell_start = match_idx - int(m_offset)
                shell_mask = list(range(subshell_start, subshell_start + n_m))
            else:
                start = match_idx
                while (
                    start > 0
                    and local_basis.angular_momenta[start - 1] == l_target
                    and (match_idx - (start - 1)) < n_m
                ):
                    start -= 1
                shell_mask = list(range(start, start + n_m))
    
            shell_global_indices = [basis_indices[i] for i in shell_mask]
            target_in_shell_idx = shell_mask.index(match_idx)
    
        # =========================================================================
        # STEP 2: Real-Space Grid Construction
        # =========================================================================
        dx = np.linspace(-Lx / 2.0, Lx / 2.0, nx)
        dy = np.linspace(-Ly / 2.0, Ly / 2.0, ny)
        dz = np.linspace(-Lz / 2.0, Lz / 2.0, nz)
        mesh_x, mesh_y, mesh_z = np.meshgrid(dx, dy, dz, indexing="ij")
    
        grid_rel_3d = np.stack([mesh_x, mesh_y, mesh_z], axis=-1)
        grid_cart_3d = grid_rel_3d + site_coords
        grid_cart_flat = grid_cart_3d.reshape(-1, 3)
    
        # =========================================================================
        # STEP 3: PAW Augmentation Grid Setup (OPTIMIZED: Bounding Box Pre-filter)
        # =========================================================================
        atom_offsets = self.post_wfc._get_atom_channel_offsets()
        paw_aug_data = []
        box_radius = max(Lx, Ly, Lz) / 2.0
    
        if include_paw_aug:
            for a_idx, site in enumerate(structure):
                atom_basis = self.atom_bases[site.specie.symbol]
                paw_sp = atom_basis.paw_species
                rcut = atom_basis.max_paw_cutoff
    
                dist_to_center = np.linalg.norm(site.coords - site_coords)
                if dist_to_center > (rcut + box_radius + 0.5):
                    continue
    
                # FAST PRE-FILTER: Axis-aligned bounding box mask before inv_lattice calculation
                diff_cart = grid_cart_flat - site.coords
                aabb_mask = np.all(np.abs(diff_cart) <= rcut + 0.5, axis=1)
    
                if not np.any(aabb_mask):
                    continue
    
                active_candidate_indices = np.where(aabb_mask)[0]
                dr_cart_cand = diff_cart[active_candidate_indices]
    
                # Minimum image convention only on candidate grid points
                dr_frac = dr_cart_cand @ inv_lattice
                dr_frac_min = dr_frac - np.round(dr_frac)
                dr_cart_min = dr_frac_min @ lattice_matrix
                dists = np.linalg.norm(dr_cart_min, axis=1)
    
                inside = dists <= rcut
                if np.any(inside):
                    active_indices = active_candidate_indices[inside]
                    fields = paw_sp.evaluate_basis_fields(
                        dr_cart_min[inside], compute_gradients=False
                    )
                    delta_phi = fields.phi_ae - fields.phi_ps
                    start_ch, end_ch = atom_offsets[a_idx]
                    paw_aug_data.append(
                        {
                            "a_idx": a_idx,
                            "start_ch": start_ch,
                            "end_ch": end_ch,
                            "active_indices": active_indices,
                            "delta_phi": delta_phi,
                            "c_P_a_total": np.zeros(
                                end_ch - start_ch, dtype=np.complex128
                            ),
                        }
                    )
    
        # =========================================================================
        # STEP 4: BZ Accumulation & IBZ Caching
        # =========================================================================
        kpts_cart_full = self.post_wfc.kpoints_cart_full
        full_to_irr = self.post_wfc.full_to_irr_map
        kpoint_rotations = self.post_wfc.kpoint_rotations
        n_kpts_full = len(kpts_cart_full)
        w_fbz = 1.0 / n_kpts_full
        all_bands = np.arange(self.nbands)
    
        psi_cache, gvec_cache, paw_cache, D_matrix_cache = {}, {}, {}, {}
    
        all_K_vecs = []
        all_V_coeffs = []
    
        for ikpt_full in range(n_kpts_full):
            ikpt_irr = full_to_irr[ikpt_full]
            k_vec = kpts_cart_full[ikpt_full]
            R_recip = kpoint_rotations[ikpt_full]
    
            if ikpt_irr not in psi_cache:
                psi_cache[ikpt_irr] = self.post_wfc.fetch_psi(
                    ikpt=ikpt_irr, ispin=spin_channel, iband=all_bands
                )
                gvec_cache[ikpt_irr] = self.post_wfc.fetch_gvectors(
                    ikpt=ikpt_irr, return_cart=False
                )
                if include_paw_aug and paw_aug_data:
                    paw_cache[ikpt_irr] = self.post_wfc.fetch_projector_overlaps(
                        ikpt=ikpt_irr, ispin=spin_channel, iband=all_bands
                    )
    
            psi_ps = psi_cache[ikpt_irr]
            g_int_irr = gvec_cache[ikpt_irr]
            C_k = self.fetch_iao_coeffs(spin_channel, ikpt_irr)
    
            if l_target == 0:
                c_target = C_k[:, target_basis_idx]
            else:
                R_key = R_recip.tobytes()
                if R_key not in D_matrix_cache:
                    D_matrix_cache[R_key] = self._get_real_sph_rotation_matrix(
                        l_target, R_recip
                    )
                D_l = D_matrix_cache[R_key]
                c_shell = C_k[:, shell_global_indices]
                c_target = (c_shell @ D_l.T)[:, target_in_shell_idx]
    
            v_k = c_target @ psi_ps
            g_int_full = g_int_irr @ R_recip.T
            g_cart_full = g_int_full @ recip_matrix
    
            K_cart = k_vec[None, :] + g_cart_full
            phase_G = np.exp(1j * (g_cart_full @ site_coords))
            V_k = w_fbz * v_k * phase_G
    
            all_K_vecs.append(K_cart)
            all_V_coeffs.append(V_k)
    
            if include_paw_aug and paw_aug_data:
                phase_target_shift = np.exp(-1j * np.dot(site_coords, k_vec))
                paw_overlaps_all = paw_cache[ikpt_irr]
                for item in paw_aug_data:
                    P_a = paw_overlaps_all[:, item["start_ch"] : item["end_ch"]]
                    c_P_a = c_target @ P_a
                    item["c_P_a_total"] += w_fbz * c_P_a * phase_target_shift
    
        K_all = np.vstack(all_K_vecs)
        V_all = np.concatenate(all_V_coeffs)
    
        # =========================================================================
        # STEP 5: Ultra-Fast FINUFFT Plane-Wave Evaluation
        # =========================================================================
        # Pass grid_rel_3d (4D array) to _evaluate_pw_sum so step sizes can be indexed
        phi_grid_flat = self._evaluate_pw_sum(
            grid_rel_3d, K_all, V_all, grid_size=grid_size
        )
    
        # Add PAW sphere augmentations
        for item in paw_aug_data:
            phi_grid_flat[item["active_indices"]] += (
                item["delta_phi"] @ item["c_P_a_total"]
            )
    
        # =========================================================================
        # STEP 6: Cube Export
        # =========================================================================
        data = np.real(phi_grid_flat).reshape(nx, ny, nz)
    
        if filename is not None:
            origin_cart = grid_cart_3d[0, 0, 0]
            voxel_vectors = np.array([
                [dx[1] - dx[0], 0.0, 0.0],
                [0.0, dy[1] - dy[0], 0.0],
                [0.0, 0.0, dz[1] - dz[0]],
            ])
            comment = f"IAO Atom {atom_idx} ({symbol}) - {orbital_identifier}"
            write_cube(
                filename, structure, data, origin_cart, voxel_vectors, comment
            )
    
        return data, grid_cart_3d
    
        
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
        Uses star phase folding over the Full Brillouin Zone to preserve exact IBZ spatial symmetry.
        """
        structure = self.structure
        spins, spin_weight = self.post_wfc._get_spin_channels_weights(spin_channel)
    
        # 1. Fetch IAO basis indices for sites A and B
        basis_A = self.atom_to_basis_indices[site_idx_A]
        basis_B = self.atom_to_basis_indices[site_idx_B]
    
        # Real-space lattice translation vector R_lattice = n1*a1 + n2*a2 + n3*a3
        R_cell = np.asarray(cell_translation, dtype=np.float64)
        R_lattice = R_cell @ structure.lattice.matrix  # Shape: (3,)
    
        # Precompute star-averaged FBZ phase factor per IBZ k-point to guarantee point-group symmetry for R != 0
        has_translation = np.any(cell_translation)
        if has_translation:
            kpts_cart_full = self.post_wfc.kpoints_cart_full
            full_to_irr = self.post_wfc.full_to_irr_map
            full_phases = np.exp(-1j * (kpts_cart_full @ R_lattice))
            
            nkpts_irr = self.post_wfc.nkpoints
            ibz_phases = np.array([
                np.mean(full_phases[full_to_irr == ikpt_irr])
                for ikpt_irr in range(nkpts_irr)
            ], dtype=np.complex128)
    
        # 2. Define population evaluation callback
        def population_callback(ispin, ikpt, weight, **kwargs):
            # Fetch IAO band coefficients C_k of shape (nbands, nbasis)
            C_k = self.fetch_iao_coeffs(ispin, ikpt)
            if C_k is None or len(C_k) == 0:
                return [None]
    
            # Fetch Hamiltonian matrix in IAO basis at this k-point
            H_k = self.fetch_iao_hamiltonian(ispin=ispin, ikpt=ikpt)
            if H_k is None or len(H_k) == 0:
                return [np.zeros(C_k.shape[0])]
    
            # Extract sub-block H_{mu, nu}(k) for mu in A, nu in B
            H_AB = H_k[np.ix_(basis_A, basis_B)]  # Shape: (n_A, n_B)
    
            # Apply star-folded lattice phase shift for R != 0
            if has_translation:
                H_AB = H_AB * ibz_phases[ikpt]
    
            # Slice coefficients for sites A and B
            C_A = C_k[:, basis_A]  # Shape: (nbands, n_A)
            C_B = C_k[:, basis_B]  # Shape: (nbands, n_B)
    
            # Matrix contraction: Sum_{mu, nu} C_{j, mu}^* H_{mu, nu} C_{j, nu}
            band_pop = np.real(np.sum((C_A.conj() @ H_AB) * C_B, axis=1))  # Shape: (nbands,)
    
            # Adjust for k-point weight and apply -COHP sign convention
            band_pop *= weight
            if negative_cohp:
                band_pop = -band_pop
    
            return [band_pop]
    
        # 3. Execute spectral engine to smear band populations onto energy grid
        smeared = self.post_wfc._execute_spectral_engine(
            num_metrics=1,
            spin_channel=spin_channel,
            eval_callback=population_callback,
        )[0]
    
        if cumulative:
            smeared = cumulative_trapezoid(smeared, self.post_wfc.energy_grid, initial=0)
    
        if return_plot:
            site_A_name = f"{structure[site_idx_A].specie.symbol}({site_idx_A})"
            site_B_name = f"{structure[site_idx_B].specie.symbol}({site_idx_B})"
            label = f"COHP {site_A_name}-{site_B_name}"
            if has_translation:
                label += f" R={cell_translation}"
    
            pdos_dict = {label: smeared}
            return self.post_wfc._generate_property_plot(
                plot_curves=pdos_dict,
                x_label="-COHP" if negative_cohp else "COHP",
                plot_range=plot_range,
                subplots=False,
            )
    
        return smeared

    def evaluate_iao_real_space(
        self,
        r_point: tuple[float, float, float] | np.ndarray,
        sites: list | None = None,
        spin_channel: int = -1,
        coords_are_cartesian: bool = False,
    ) -> dict[int, np.ndarray]:
        """
        Evaluates orthogonal Intrinsic Atomic Orbitals (IAOs) in real space at coordinate r_point
        by Fourier-transforming the Bloch IAO representations across the Full Brillouin Zone,
        rotating reciprocal vectors according to the IBZ symmetry star.
        """
        structure = self.structure
        if coords_are_cartesian:
            r_cart = np.asarray(r_point, dtype=np.float64)
        else:
            r_cart = np.asarray(r_point, dtype=np.float64) @ structure.lattice.matrix
    
        if sites is None:
            sites = [i for i in structure]
    
        spins, _ = self.post_wfc._get_spin_channels_weights(spin_channel)
    
        # Determine basis index mapping and Cartesian site positions R_basis
        basis_map = []
        R_basis_list = []
        for site in sites:
            atom_idx = site.index
            unit_indices = self.atom_to_basis_indices[atom_idx]
            basis_map.extend(unit_indices)
            R_basis_list.append(np.tile(site.coords, (len(unit_indices), 1)))
        basis_map = np.array(basis_map, dtype=np.intp)
        R_basis = np.vstack(R_basis_list)
    
        n_eval_basis = len(basis_map)
        phi_iao_real_dict = {s: np.zeros(n_eval_basis, dtype=np.complex128) for s in spins}
    
        # k-point mapping and rotation arrays from PostWFC
        kpts_cart_full = self.post_wfc.kpoints_cart_full
        full_to_irr = self.post_wfc.full_to_irr_map
        kpoint_cart_rotations = self.post_wfc.kpoint_cart_rotations
        n_kpts_full = len(kpts_cart_full)
        w_fbz = 1.0 / n_kpts_full
    
        atom_offsets = self.post_wfc._get_atom_channel_offsets()
        all_bands = np.arange(self.nbands)
    
        for ikpt_full in range(n_kpts_full):
            ikpt_irr = full_to_irr[ikpt_full]
            k_vec = kpts_cart_full[ikpt_full]
            R_cart = kpoint_cart_rotations[ikpt_full]
    
            # Fetch IBZ Cartesian K-vectors (k + G) and rotate to FBZ position
            K_vecs_irr = self.post_wfc.fetch_gvectors(ikpt_irr, return_cart=True, add_k=True)
            K_vecs_full = K_vecs_irr @ R_cart.T
            
            pw_phase = np.exp(1j * (K_vecs_full @ r_cart)) / np.sqrt(structure.volume)
            phases = np.exp(-1j * (R_basis @ k_vec))
    
            for ispin in spins:
                # 1. Pseudo wavefunctions psi_ps(r) at r_cart
                psi_ps = self.post_wfc.fetch_psi(ikpt=ikpt_irr, ispin=ispin, iband=all_bands)
                psi_r_k = psi_ps @ pw_phase  # Shape: (nbands,)
    
                # 2. PAW sphere augmentations across sites
                paw_overlaps_all = self.post_wfc.fetch_projector_overlaps(
                    ikpt=ikpt_irr, ispin=ispin, iband=all_bands
                )
    
                for site in sites:
                    atom_idx = site.index
                    dr = r_cart - site.coords
    
                    local_basis = self.atom_bases[site.specie.symbol]
                    paw_sp = local_basis.paw_species
    
                    fields = paw_sp.evaluate_basis_fields(np.atleast_2d(dr), compute_gradients=False)
                    delta_phi_i = (fields.phi_ae - fields.phi_ps).squeeze(axis=0)
    
                    start_ch, end_ch = atom_offsets[atom_idx]
                    P_a = paw_overlaps_all[:, start_ch:end_ch]
    
                    psi_r_k += P_a @ delta_phi_i
    
                # 3. Transform Bloch band states to Bloch IAOs
                C_k = self.fetch_iao_coeffs(ispin, ikpt_irr)  # Shape: (nbands, nbasis)
                phi_iao_k = C_k.T @ psi_r_k  # Shape: (nbasis,)
    
                # 4. Phase shift to site position R_basis and average over FBZ
                phi_iao_real_dict[ispin] += w_fbz * (phi_iao_k[basis_map] * phases)
    
        return {s: np.real(phi_arr) for s, phi_arr in phi_iao_real_dict.items()}
    
    
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
    