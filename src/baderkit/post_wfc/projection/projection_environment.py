# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import inv, sqrtm, eigh
from rich.progress import track
import h5py

from rich import print as rprint

from baderkit.post_wfc.projection.all_electron_dataset import AESpecies

from baderkit.post_wfc.base_env import PostWFC

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
        **kwargs
    ):
        # Register post_wfc as the reference state context link
        if post_wfc is None:
            post_wfc = PostWFC(**kwargs)

        self.post_wfc = post_wfc
        self._meta = post_wfc._meta
        self.directory = post_wfc.directory
        
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
                    all_bases.append({
                        "atom_idx": atom_idx,
                        "l": basis.angular_momenta[basis_idx],
                        "m": basis.magnetic_quantum_numbers[basis_idx],
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
            self.all_bases
        return self._atom_to_basis_indices

    @property
    def basis_atom_indices(self):
        if getattr(self, "_basis_atom_indices", None) is None:
            self.all_bases
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
    def _iao_file(self):
        # NOTE: kept as property to update if user changes directory variable
        return self.directory / "iao.h5"
    
    ###########################################################################
    # PDOS Methods
    ###########################################################################
    def get_projected_density_of_states(
        self,
        atoms: int | list[int] | None = None,
        spin_channel: int = -1,
        cumulative: bool = False,
        return_plot: bool = False,
        plot_range: tuple[float, float] = None,
    ) -> dict[int, np.ndarray] | np.ndarray:
        """Calculates the Atom-Projected Density of States (PDOS) using precalculated orthogonalized IAOs."""
        # Parse target atom indices
        if atoms is None:
            target_atoms = list(range(len(self.structure)))
        elif isinstance(atoms, int):
            target_atoms = [atoms]
        else:
            target_atoms = list(atoms)
    
        # Pre-build orbital index masks for each target atom
        atom_masks = {
            atom_idx: self.atom_to_basis_indices[atom_idx]
            for atom_idx in target_atoms
        }
    
        def pdos_callback(ispin, ikpt, weight, **kwargs):
            # Fetch precalculated orthogonalized coefficients directly from disk
            C_orth_k = self.fetch_orth_iao_coeffs(ispin, ikpt)  # shape: (nbands, nbasis)
            orbital_weights = np.abs(C_orth_k) ** 2             # shape: (nbands, nbasis)
    
            # Sum orbital weights per target atom
            band_weights = []
            for atom_idx in target_atoms:
                orb_indices = atom_masks[atom_idx]
                if len(orb_indices) > 0:
                    w_atom = np.sum(orbital_weights[:, orb_indices], axis=1) * weight
                else:
                    w_atom = np.zeros(self.nbands)
                band_weights.append(w_atom)
    
            return band_weights
    
        # Execute spectral engine (num_metrics = number of target atoms)
        smeared = self.post_wfc._execute_spectral_engine(
            num_metrics=len(target_atoms),
            spin_channel=spin_channel,
            eval_callback=pdos_callback,
        )
    
        # Assemble atom_idx -> pdos dictionary
        pdos_dict = {}
        for i, atom_idx in enumerate(target_atoms):
            atom_pdos = smeared[i]
            if cumulative:
                atom_pdos = cumulative_trapezoid(atom_pdos, self.post_wfc.energy_grid, initial=0)
            pdos_dict[atom_idx] = atom_pdos
            
        if return_plot:
            prefix = "Integrated " if cumulative else ""
            x_label = "States" if cumulative else "States / eV"
            
            # get tdos
            tdos= self.post_wfc.get_density_of_states(
                 spin_channel=spin_channel,
                 cumulative=cumulative,
                 return_plot=False,
                 )
    
            plot_curves = {
                "total": tdos
                }
            for atom_idx in target_atoms:
                symbol = self.structure.labels[atom_idx]
                label = f"{prefix}PDOS ({symbol} #{atom_idx})"
                plot_curves[label] = pdos_dict[atom_idx]
    
            return self.post_wfc._generate_property_plot(
                plot_curves=plot_curves, x_label=x_label, plot_range=plot_range, subplots=False
            )
    
        if isinstance(atoms, int):
            return pdos_dict[atoms]
    
        return pdos_dict

    ###########################################################################
    # Helper Functions
    ###########################################################################
    
    def fetch_iao_coeffs(self, ispin, ikpt):
        with h5py.File(self._iao_file, "r") as file:
            return file["A_coeffs"][ispin, ikpt]
    
    def fetch_orth_iao_coeffs(self, ispin, ikpt):
        with h5py.File(self._iao_file, "r") as file:
            return file["C_orth"][ispin, ikpt]
    
    def fetch_mo_coeffs(self, ispin, ikpt):
        with h5py.File(self._iao_file, "r") as file:
            return file["mo_coeffs"][ispin, ikpt]
    
    def fetch_iao_hamiltonian(self, ispin, ikpt):
        with h5py.File(self._iao_file, "r") as file:
            return file["H"][ispin, ikpt]
        
    ###########################################################################
    # Primary Projection Functions
    ###########################################################################
    def _load_bases(self):
        """Parses NPZ binaries and filters out target elements matching cell contents."""
        unique_elements = set(site.specie.symbol for site in self.structure)
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
            target_n_occ = int(round(self.total_charge / 2.0))
        else:
            target_n_occ = int(round(self.total_charge / 2.0))
        
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
    
            dset_C_orth = file.create_dataset(
                "C_orth",
                shape=(nspin, nkpoints, nbands, nbasis),
                dtype=np.complex128,
                chunks=(nspin, 1, nbands, nbasis),
                compression="lzf",
            )
    
            # Store references on projection_env for direct access during spectral tasks
            self.dset_H = dset_H
            self.dset_S = dset_S
            self.dset_A = dset_A
            self.dset_C_orth = dset_C_orth
            
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
                    psi_ps = self.post_wfc.fetch_psi(ikpt=ikpt, ispin=ispin, iband=np.arange(nbands)).T
                    chi_psi_ps = local_basis_matrix.conj() @ psi_ps
                    paw_psi_ps_all = self.post_wfc.fetch_projector_overlaps(ikpt=ikpt, ispin=ispin, iband=np.arange(nbands))
                    
                    chi_psi_aug = np.zeros_like(chi_psi_ps)
                    for atom_idx, paw_overlap in enumerate(local_partial_overlaps):
                        start_ch, end_ch = atom_offsets[atom_idx]
                        paw_psi_ps = paw_psi_ps_all[:, start_ch:end_ch]
                        chi_psi_aug += np.dot(paw_overlap, paw_psi_ps.T)
                    
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
                    
                    ctc = C_tilde_raw.conj().T @ C_tilde_raw
                    ctc_inv_sqrt = inv(sqrtm(ctc))
                    C_tilde = C_tilde_raw @ ctc_inv_sqrt
                    
                    ###################################################################
                    # Projectors O and O_tilde
                    ###################################################################
                    O = np.diag(occ_mask.astype(float))
                    O_tilde = C_tilde @ C_tilde.conj().T
                    
                    ###################################################################
                    # Unorthogonalized IAO Coefficients A
                    ###################################################################
                    I_bands = np.eye(nbands)
                    A = (I_bands + 2*(O @ O_tilde) - O_tilde - O) @ P12
                    A_T = A.conj().T
                    
                    ###################################################################
                    # E, H_IAO, S_IAO, and C_orth
                    ###################################################################
                    E = self.energies[ispin, ikpt]
                    H_IAO = A_T @ (E[:, None] * A)
                    S_IAO = A_T @ A
    
                    # Löwdin orthogonalization in IAO space: C_orth = A @ S^(-1/2)
                    eigvals_S, eigvecs_S = eigh(S_IAO)
                    eigvals_S = np.maximum(eigvals_S, 1e-12)
                    S_inv_sqrt = eigvecs_S @ np.diag(1.0 / np.sqrt(eigvals_S)) @ eigvecs_S.conj().T
                    C_orth = A @ S_inv_sqrt
                    
                    ###################################################################
                    # Save Results to disk
                    ###################################################################
                    dset_A[ispin, ikpt] = A
                    dset_C_orth[ispin, ikpt] = C_orth
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
        rprint("=" * 80 + "\n")
                    