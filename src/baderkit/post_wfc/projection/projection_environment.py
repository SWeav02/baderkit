
from pathlib import Path

import h5py
import numpy as np
from rich import print as rprint
from rich.progress import track
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import eigh, inv, sqrtm

from baderkit.post_wfc.base_env import PostWFC
from baderkit.post_wfc.projection.all_electron_dataset import AESpecies
from baderkit.post_wfc.wfc_numba import evaluate_real_harmonics_multi


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
        spillage_cutoff: float = 0.02,
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
    # PDOS Methods
    ###########################################################################

    def _get_target_basis_indices(
        self,
        site: int | str | list[int | str] | None = None,
        orbitals: str | int | list[str | int] | None = None,
    ) -> list[int]:
        """Resolves target sites and orbital specifiers into global basis indices."""
        def _parse_sites(s_spec):
            if s_spec is None:
                return list(range(len(self.structure)))
            if isinstance(s_spec, int):
                return [s_spec]
            if isinstance(s_spec, str):
                if s_spec.isdigit():
                    return [int(s_spec)]
                return [
                    i for i, site in enumerate(self.structure)
                    if site.species_string.lower() == s_spec.lower() or site.specie.symbol.lower() == s_spec.lower()
                ]
            res = []
            for item in s_spec:
                res.extend(_parse_sites(item))
            return list(dict.fromkeys(res))

        atom_indices = _parse_sites(site)
        basis_labels = self._get_basis_labels()
        matched = []

        orb_specs = [orbitals] if isinstance(orbitals, (int, str)) else (list(orbitals) if orbitals else None)

        for a_idx in atom_indices:
            for g_idx in self.atom_to_basis_indices[a_idx]:
                if orb_specs is None:
                    matched.append(g_idx)
                else:
                    b_info = self.all_bases[g_idx]
                    lbl = basis_labels[g_idx]
                    if any(self._is_orbital_match(b_info, lbl, g_idx, spec) for spec in orb_specs):
                        matched.append(g_idx)

        return matched

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

    def _parse_overlap_spec(
        self,
        spec: str | int | tuple,
    ) -> tuple[list[int], tuple[int, int, int], str]:
        """Parses a single overlap specifier into (basis_indices, image_tuple, clean_label)."""
        image = (0, 0, 0)

        if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[1], (tuple, list)):
            spec_str, image = spec[0], tuple(spec[1])
        elif isinstance(spec, str) and ":" in spec:
            spec_str, img_str = spec.split(":", 1)
            spec_str = spec_str.strip()
            img_clean = img_str.strip().strip("[]()")
            image = tuple(int(x.strip()) for x in img_clean.split(","))
        else:
            spec_str = spec

        site = None
        orbitals = None

        if isinstance(spec_str, int):
            site = spec_str
        elif isinstance(spec_str, str):
            if "-" in spec_str and not spec_str.startswith("-"):
                parts = spec_str.split("-", 1)
                site = parts[0]
                orbitals = parts[1]
            else:
                if spec_str.isdigit() or any(
                    spec_str.lower() == s.species_string.lower() or spec_str.lower() == s.specie.symbol.lower()
                    for s in self.structure
                ):
                    site = spec_str
                else:
                    orbitals = spec_str

        basis_indices = self._get_target_basis_indices(site=site, orbitals=orbitals)

        img_suffix = f": [{image[0]}, {image[1]}, {image[2]}]" if any(i != 0 for i in image) else ""
        label = f"{spec_str}{img_suffix}"

        return basis_indices, image, label
    
    def evaluate_basis_at_point(
        self,
        r_point: tuple[float, float, float] | np.ndarray,
    ) -> np.ndarray:
        """
        Evaluates all basis functions chi_mu(r) at a Cartesian real-space point r = (x, y, z).

        Returns
        -------
        np.ndarray
            1D array of shape (nbasis,) containing basis evaluations chi_mu(r).
        """
        r_pt = np.asarray(r_point, dtype=float)
        chi_vals = np.zeros(self.nbasis, dtype=complex)

        for g_idx, b_info in enumerate(self.all_bases):
            atom_idx = b_info["atom_idx"]
            R_atom = self.structure.cart_coords[atom_idx]
            
            # Format displacement vector as 2D array of shape (1, 3)
            d_vec = (r_pt - R_atom)[None, :]
            r_dist = np.linalg.norm(d_vec)

            l = b_info["l"]
            m = b_info["m"]
            q_spline = b_info["q_radial_spline"]

            # Evaluate radial spline and real spherical harmonic Y_lm
            r_val = np.nan_to_num(q_spline(r_dist),0.0)
            y_lm_array, _ = evaluate_real_harmonics_multi(l, m, d_vec)
            
            chi_vals[g_idx] = r_val * y_lm_array[0]
        return chi_vals

    ###########################################################################
    # COOP / COHP / rCOOP / rCOHP Core Engine
    ###########################################################################

    def get_real_space_crystal_orbital_population(
        self,
        # Basic
        r_point: tuple[float, float, float] | np.ndarray,
        pop_type: str = "cohp",

        # Filters
        cumulative: bool = False,
        spin_channel: int = -1,

        # Plotting
        return_plot: bool = False,
        plot_range: tuple[float, float] | None = None,
        negative_cohp: bool = True,
        **kwargs,
    ) -> dict[str, np.ndarray] | np.ndarray:
        """
        Calculates COOP, COHP, rCOOP, or rCOHP using precalculated orthogonal IAO coefficients
        mapped back into the unorthogonalized basis to prevent inter-body tail folding.
        """
        # Check which type to use (coop or cohp)
        pop_type = pop_type.lower()
        if pop_type not in ("coop", "cohp"):
            raise ValueError("pop_type must be either 'coop' or 'cohp'")

        # Get point
        chi_r = self.evaluate_basis_at_point(r_point)
        W_r = np.outer(chi_r.conj(), chi_r)

        def population_callback(ispin, ikpt, weight, **kwargs):
            # get coefficients
            C_k = self.fetch_iao_coeffs(ispin, ikpt)
            # Get overlap or hamiltonian matrix at this kpoint
            M_k = self.fetch_iao_overlap(ispin, ikpt) if pop_type == "coop" else self.fetch_iao_hamiltonian(ispin, ikpt)
            if not len(M_k):
                return [None]
            
            M_eff = M_k * W_r
            # Zero out diagonal terms
            np.fill_diagonal(M_eff, 0.0)
            
            # Sum off-diagonal parts and multiply by k-point weight
            band_pop = np.zeros(self.nbands)
            for band_idx in range(self.nbands):
                current_sum = np.sum(C_k[band_idx, :].conj() * C_k[band_idx, :] * M_eff)
                band_pop[band_idx] = current_sum
            
            band_pop *= weight

            if pop_type == "cohp" and negative_cohp:
                band_pop = -band_pop
            return [band_pop]

        metric_name = f"r{pop_type.capitalize()}"
        
        # Get spectral value
        smeared = np.zeros((self.nbands,), dtype=float)  # Initialize with zeros for debugging
        smeared = self.post_wfc._execute_spectral_engine(
            num_metrics=1,
            spin_channel=spin_channel,
            eval_callback=population_callback,
        )[0]

        if cumulative:
            smeared = cumulative_trapezoid(smeared, self.post_wfc.energy_grid)

        if return_plot:
            pdos_dict = {
                metric_name: smeared
            }

            plot = self.post_wfc._generate_property_plot(
                plot_curves=pdos_dict,
                x_label=metric_name,
                plot_range=plot_range,
                subplots=False,
            )
            
            return plot

        return smeared
    
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
        cumulative: bool = False,
        return_plot: bool = False,
        plot_range: tuple[float, float] | None = None,
        tol: float = 1e-8,
    ) -> np.ndarray | dict[str, np.ndarray]:
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
        cumulative : bool, default=False
            If True, computes integrated DOS spillage.
        return_plot : bool, default=False
            If True, returns a plot object showing spillage vs energy.
        plot_range : tuple[float, float], optional
            (E_min, E_max) energy range for plotting.
        tol : float, default=1e-8
            Minimum Total DOS threshold below which percentage spillage is set to 0.0%
            to avoid division-by-zero artifacts.
    
        Returns
        -------
        np.ndarray
            1D numpy array containing the spillage values aligned with `self.energy_grid`.
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
    
        diff = np.maximum(0.0, total_dos - iao_dos)
    
        if as_percent:
            with np.errstate(divide="ignore", invalid="ignore"):
                spillage = np.where(total_dos > tol, (diff / total_dos) * 100.0, 0.0)
                spillage = np.clip(spillage, 0.0, 100.0)
            units_label = "Spillage (%)"
        else:
            spillage = diff
            units_label = "Spillage (States)" if cumulative else "Spillage (States / eV)"
    
        if return_plot:
            prefix = "Integrated " if cumulative else ""
            x_label = f"{prefix}{units_label}"
    
            plot_curves = {
                f"{prefix}IAO {units_label}": spillage,
            }
    
            return self.post_wfc._generate_property_plot(
                plot_curves=plot_curves,
                x_label=x_label,
                plot_range=plot_range,
                subplots=False,
            )
    
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
        
    def _project_system(self):
        # We call the AE wavefunctions, Psi, our B1 basis
        # We want to form Bloch IAOs using a minimal atomic basis, Chi or B2
        
        structure = self.structure
        atom_positions = structure.cart_coords
        nspin = self.nspin
        nkpoints = self.nkpoints
        nbands = self.nbands
        volume = structure.volume # Omega
        nbasis = self.nbasis
        atom_offsets = self.post_wfc._get_atom_channel_offsets()
        spillage_cutoff = self.spillage_cutoff
        
        # Initialize spillage array: shape (nspin, nkpoints, nbands)
        self._spillage = np.zeros((nspin, nkpoints, nbands), dtype=np.float64)
        
        rprint("\n" + "="*80)
        rprint("[bold green]          STARTING PROJECTION          [/bold green]")
        rprint("="*80)
        rprint(f"[bold white]System Dimensions:[/bold white] Spin={nspin}, k-points={nkpoints}, Bands={nbands}")
        rprint(f"[bold white]Unit Cell Volume (Omega):[/bold white] {volume:.6f} Å^3")
        
        rprint("\n" + "="*80)
        rprint("[bold blue]INFO: Executing Reciprocal Projections & PAW Augmentation[/bold blue]")
        rprint("="*80)
        
        # Diagnostic trackers
        max_energy_err = 0.0
        max_trace_err = 0.0
        max_cond_num = 0.0
        
        if self.nspin == 1:
            target_n_occ = round(self.total_charge / 2.0)
        else:
            target_n_occ = round(self.total_charge / 2.0)
        
        with h5py.File(self._iao_file, "w") as file:
            dset_H = file.create_dataset(
                "H",
                shape=(nspin, nkpoints, nbasis, nbasis),
                dtype=np.complex128,
                chunks=(nspin, 1, nbasis, nbasis),
                compression="lzf",
            )
            
            dset_S = file.create_dataset(
                "S",
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
                    # |psi_ps>
                    psi_ps = self.post_wfc.fetch_psi(ikpt=ikpt, ispin=ispin, iband=np.arange(nbands)).T
                    # <chi | psi_ps>
                    chi_psi_ps = local_basis_matrix.conj() @ psi_ps
                    # <p | psi_ps> (separated by atoms)
                    paw_psi_ps_all = self.post_wfc.fetch_projector_overlaps(ikpt=ikpt, ispin=ispin, iband=np.arange(nbands))
                    
                    # <chi | psi_aug> (summed over atoms)
                    chi_psi_aug = []
                    for atom_idx, paw_overlap in enumerate(local_partial_overlaps):
                        # Get projector overlap for this atom
                        start_ch, end_ch = atom_offsets[atom_idx]
                        paw_psi_ps = paw_psi_ps_all[:, start_ch:end_ch]
                        chi_psi_aug.append(np.dot(paw_overlap, paw_psi_ps.T))
                    # combine into full <chi | psi_aug>
                    chi_psi_aug = np.vstack(chi_psi_aug)
                    # sum <chi | psi>
                    S21 = chi_psi_ps + chi_psi_aug
                    S12 = S21.conj().T
                    ###################################################################
                    # P_12 and P_21
                    ###################################################################
                    P12 = S12
                    P21 = S21
                    
                    energy_order = np.argsort(self.energies[ispin, ikpt])
                    
                    occ_mask = np.zeros(self.nbands, dtype=bool)
                    occ_mask[energy_order[:target_n_occ]] = True
                    
                    ###################################################################
                    # C_tilde
                    ###################################################################
                    P21_occ = P21[:, occ_mask]
                    C_tilde_raw = P12 @ P21_occ
                    
                    # Löwdin orthogonalization
                    ctc = C_tilde_raw.conj().T @ C_tilde_raw
                    ctc_inv_sqrt = inv(sqrtm(ctc))
                    C_tilde = C_tilde_raw @ ctc_inv_sqrt
                    
                    ###################################################################
                    # Projectors O and O_tilde
                    ###################################################################
                    O = np.diag(occ_mask.astype(float))
                    O_tilde = C_tilde @ C_tilde.conj().T
                    
                    ###################################################################
                    # IAO Coefficients A
                    ###################################################################
                    I_bands = np.eye(nbands)
                    A = (I_bands + 2*(O @ O_tilde) - O_tilde - O) @ P12
                    A_T = A.conj().T
    
                    # Löwdin orthogonalization
                    S_IAO = A.conj().T @ A
                    S_IAO_inv_sqrt = inv(sqrtm(S_IAO))
                    A = A @ S_IAO_inv_sqrt
                    A_T = A.conj().T
    
                    ###################################################################
                    # Spillage Calculation
                    ###################################################################
                    # Fraction of state m spanned by IAOs is sum_mu |A_{m, mu}|^2
                    spillage_k = 1.0 - np.sum(np.abs(A)**2, axis=1)
                    
                    # Explicitly zero spillage for occupied states & clamp numerical noise
                    spillage_k[occ_mask] = 0.0
                    spillage_k = np.maximum(0.0, spillage_k)
                    
                    self._spillage[ispin, ikpt] = spillage_k
    
                    ###################################################################
                    # E, H_IAO, S_IAO
                    ###################################################################
                    E = self.energies[ispin, ikpt]
                    H_IAO = A_T @ (E[:, None] * A)
                    S_IAO = A_T @ A
                    
                    ###################################################################
                    # Save Results to disk
                    ###################################################################
                    dset_A[ispin, ikpt] = A
                    dset_H[ispin, ikpt] = H_IAO
                    dset_S[ispin, ikpt] = S_IAO
                    
                    ###################################################################
                    # Verify Results
                    ###################################################################
                    if target_n_occ > nbasis:
                        rprint(f"[bold red]WARNING (ikpt={ikpt}, ispin={ispin}):[/bold red] target_n_occ ({target_n_occ}) > nbasis ({nbasis})! "
                               "Atomic basis is too small to span the occupied space.")
                        max_energy_err = float("nan")
                    else:
                        eigvals_gen = eigh(H_IAO, S_IAO, eigvals_only=True)
                        E_occ = np.sort(E[occ_mask])
                        iao_occ = np.sort(eigvals_gen[:target_n_occ])
                        
                        e_err = np.max(np.abs(iao_occ - E_occ))
                        max_energy_err = max(max_energy_err, e_err)
                    
                    t_err = np.abs(np.trace(O_tilde).real - target_n_occ)
                    max_trace_err = max(max_trace_err, t_err)
                    
                    cond_num = np.linalg.cond(S_IAO)
                    max_cond_num = max(max_cond_num, cond_num)
    
        # Determine maximum safe energy threshold across all spin/k-point channels
        unsafe_energies = []
        for ispin in range(nspin):
            for ikpt in range(nkpoints):
                E_k = self.energies[ispin, ikpt]
                spill_k = self._spillage[ispin, ikpt]
                
                # Sort states by energy
                order = np.argsort(E_k)
                sorted_E = E_k[order]
                sorted_spill = spill_k[order]
                
                # Find first state exceeding spillage cutoff
                exceeded = np.where(sorted_spill > spillage_cutoff)[0]
                if len(exceeded) > 0:
                    unsafe_energies.append(sorted_E[exceeded[0]])
    
        if unsafe_energies:
            max_safe_energy = np.min(unsafe_energies)
            safe_range_str = f"Up to {max_safe_energy:.4f} eV"
        else:
            max_safe_energy = np.max(self.energies)
            safe_range_str = f"> {max_safe_energy:.4f} eV (All states within cutoff)"
    
        # Print final summary
        status_energy = "[bold green]PASSED[/bold green]" if max_energy_err < 1e-6 else "[bold red]FAILED[/bold red]"
        status_trace  = "[bold green]PASSED[/bold green]" if max_trace_err < 1e-6 else "[bold red]FAILED[/bold red]"
        if max_cond_num < 1e8:
            status_cond = "[bold green]PASSED[/bold green]"
        elif max_cond_num < 1e12:
            status_cond = "[bold yellow]MODERATE[/bold yellow]"
        else:
            status_cond = "[bold red]ILL-CONDITIONED[/bold red]"
        
        rprint("\n" + "=" * 80)
        rprint("[bold green]          IAO PROJECTION DIAGNOSTIC SUMMARY          [/bold green]")
        rprint("=" * 80)
        rprint(f" • [bold white]Occupied Energy Recovery Max Err :[/bold white] {max_energy_err:.2e} eV  [{status_energy}]")
        rprint(f" • [bold white]Projector Trace Max Err          :[/bold white] {max_trace_err:.2e}     [{status_trace}]")
        rprint(f" • [bold white]Max Basis Condition Number κ(S)  :[/bold white] {max_cond_num:.2f}     [{status_cond}]")
        rprint(f" • [bold white]Safe Energy Range (spillage < {spillage_cutoff:.0%}):[/bold white] [bold cyan]{safe_range_str}[/bold cyan]")
        rprint("=" * 80 + "\n")