# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import inv, sqrtm
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
    def projection_coefficients(self):
        if getattr(self,"_projection_coefficients",None) is None:
            self._project_system()
        return self._projection_coefficients
    
    @property
    def atom_contributions(self):
        if getattr(self,"_atom_contributions",None) is None:
            self._process_pdos()
        return self._atom_contributions
    
    @property
    def atom_pdos(self):
        if getattr(self, "_atom_pdos", None) is None:
            self._atom_pdos = self.get_atom_projected_density_of_states()
        return self._atom_pdos
    
    @property
    def orbital_pdos(self):
        if getattr(self, "_orbital_pdos", None) is None:
            self._orbital_pdos = self.get_atom_projected_density_of_states()
        return self._orbital_pdos
    
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
                    global_basis_idx += 1
            self._all_bases = all_bases
            self._atom_to_basis_indices = atom_to_basis_indices
        return self._all_bases
    
    @property
    def atom_to_basis_indices(self):
        if getattr(self, "_atom_to_basis_indices", None) is None:
            self.all_bases
        return self._atom_to_basis_indices
    
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
    def get_atom_projected_density_of_states(
        self, 
        spin_channel=-1, 
        use_occupancies=False,
        return_plot=False,
        plot_range=None,
    ) -> dict:
        """
        Computes the atom-resolved Projected Density of States (PDOS) by applying
        the designated smearing method to the orbital projection projection_coefficients.
        """
        coeffs = self._projection_coefficients  # Maps to our updated spin-first array                 
        basis_map = self.all_bases
        num_atoms = len(self.structure)

        # Pre-allocate array for channel intensities: shape (num_atoms, nspin, nkpoints, nbands)
        atom_weights = np.zeros((num_atoms, self.nspin, self.nkpoints, self.nbands), dtype=np.float64)
        
        atom_indices = np.array([b['atom_idx'] for b in basis_map], dtype=np.int32)
        proj_intensity = np.abs(coeffs) ** 2       

        for i_atom in range(num_atoms):
            mask = (atom_indices == i_atom)
            if np.any(mask):
                atom_weights[i_atom] = np.sum(proj_intensity[:, :, mask, :], axis=2)

        # Delegate execution context to the unified smearing helper
        smeared_data = self._compute_smeared_channels(
            channel_weights=atom_weights,
            spin_channel=spin_channel,
            use_occupancies=use_occupancies
        )

        energy_grid = self.energy_grid
        pdos_data = {
            "total": np.zeros(len(energy_grid), dtype=np.float64)
        }
        for i_atom in range(num_atoms):
            pdos_data[i_atom] = smeared_data[i_atom]
            pdos_data["total"] += smeared_data[i_atom]

        if return_plot:
            plot_curves = {}
            for i_atom in range(num_atoms):
                symbol = self.structure[i_atom].specie.symbol
                plot_curves[f"Atom {i_atom} ({symbol})"] = pdos_data[i_atom]
            return self._generate_dos_plot(
                total_dos=pdos_data["total"],
                plot_curves=plot_curves,
                plot_range=plot_range
            )

        return pdos_data

    def get_orbital_character_projected_density_of_states(
        self,
        spin_channel=-1,
        use_occupancies=False,
        return_plot=False,
        plot_range=None,
    ) -> dict:
        """
        Computes the orbital-resolved Projected Density of States (PDOS) decomposed
        by both atomic site location index and angular momentum character (s, p, d, f).
        """
        coeffs = self._projection_coefficients  # Maps to our updated spin-first array
        basis_map = self.all_bases
        
        l_symbols = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
        
        # Identify active combination tracks (atom_idx, l) present across the system basis pool
        channels = sorted(list(set((b['atom_idx'], b['l']) for b in basis_map)))
        num_channels = len(channels)
        
        channel_weights = np.zeros((num_channels, self.nspin, self.nkpoints, self.nbands), dtype=np.float64)
        
        atom_indices = np.array([b['atom_idx'] for b in basis_map], dtype=np.int32)
        l_indices = np.array([b['l'] for b in basis_map], dtype=np.int32)
        proj_intensity = np.abs(coeffs) ** 2
        
        for idx, (i_atom, l_val) in enumerate(channels):
            mask = (atom_indices == i_atom) & (l_indices == l_val)
            if np.any(mask):
                # FIXED: Slice axis 2 explicitly and sum over axis 2 (the basis dimension)
                channel_weights[idx] = np.sum(proj_intensity[:, :, mask, :], axis=2)

        # Delegate execution context to the unified smearing helper
        smeared_data = self._compute_smeared_channels(
            channel_weights=channel_weights,
            spin_channel=spin_channel,
            use_occupancies=use_occupancies
        )
        
        energy_grid = self.energy_grid
        pdos_data = {
            "total": np.zeros(len(energy_grid), dtype=np.float64)
        }
        
        plot_curves = {}
        for idx, (i_atom, l_val) in enumerate(channels):
            l_char = l_symbols.get(l_val, f"l={l_val}")
            key = f"atom_{i_atom}_{l_char}"
            
            pdos_data[key] = smeared_data[idx]
            pdos_data["total"] += smeared_data[idx]
            
            symbol = self.structure[i_atom].specie.symbol
            plot_curves[f"Atom {i_atom} ({symbol}) - {l_char}"] = smeared_data[idx]
            
        if return_plot:
            return self._generate_dos_plot(
                total_dos=pdos_data["total"],
                plot_curves=plot_curves,
                plot_range=plot_range
            )
            
        return pdos_data

    ###########################################################################
    # Helper Functions
    ###########################################################################
    def _process_pdos(self):
        """
        Normalizes individual atom PDOS arrays so the occupied states integrate exactly
        to each atom's valence count, then computes the cumulative total cell charge profile.
        """
        # Get PDOS
        pdos_data = self.atom_pdos
        
        # get energies and energy vs. total charge
        normalized_data = {}
        
        total = pdos_data["total"]
        total_cum = cumulative_trapezoid(total, initial=0)
        nonzero = total_cum > 0.0
        
        normalized = []
        charge_data = []
        # Get normalized pdos for each atom
        for i in range(len(self.structure)):
            spectrum = pdos_data[i]
            spectrum_cum = cumulative_trapezoid(spectrum, initial=0)
            # get fraction of total
            spectrum_norm = np.zeros_like(spectrum_cum)
            spectrum_norm[nonzero] = spectrum_cum[nonzero] / total_cum[nonzero]
            normalized.append(spectrum_norm)
            charge_data.append(spectrum_norm * self.total_charge_grid)
            
        normalized_data["normalized"] = np.vstack(normalized)
        normalized_data["partial_charge"] = np.vstack(charge_data)

        self._atom_contributions = normalized_data        
        
    def fetch_iao_coeffs(self, ispin, ikpt):
        with h5py.File(self._iao_file, "r") as file:
            return file["A_coeffs"][ispin, ikpt]
        
    def fetch_mo_ceoffs(self, ispin, ikpt):
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
        # Required parts:
            # 1. C - coefficients of Psi, simply occupations
            # 2. S1 - Self overlap of Psi, equivalent to identity matrix
            # 3. S2 - Self overlap of Chi, precalculated
            # 4. S12 - Overlap of Chi/Psi
                # 4.1 - pseudo overlap of Chi/Psi
                # 4.2 - Overlap of Chi and partial waves, precalculated
                # 4.3 - Overlap of PAW projectors, p, and Psi
        
        structure = self.structure
        atom_positions = structure.cart_coords
        nspin = self.nspin
        nkpoints = self.nkpoints
        nbands = self.nbands
        volume = structure.volume # Omega
        nbasis = self.nbasis
        
        rprint("\n" + "="*80)
        rprint("[bold green]         STARTING PROJECTION          [/bold green]")
        rprint("="*80)
        rprint(f"[bold white]System Dimensions:[/bold white] Spin={nspin}, k-points={nkpoints}, Bands={nbands}")
        rprint(f"[bold white]Unit Cell Volume (Omega):[/bold white] {volume:.6f} Å^3")
        
        rprint("\n" + "="*80)
        rprint("[bold blue]INFO: Executing Reciprocal Projections & PAW Augmentation[/bold blue]")
        rprint("="*80)
        
        with h5py.File(self._iao_file, "w") as file:
            dset_H = file.create_dataset(
                "H",
                shape=(nspin, nkpoints, nbasis, nbasis),
                dtype=np.complex128,
                chunks=(nspin, 1, nbasis, nbasis), # chunk along k points for faster query
                compression="lzf",  # gzip: better compression, slower
            )
            
            dset_A = file.create_dataset(
                "A_coeffs",
                shape=(nspin, nkpoints, nbands, nbasis),
                dtype=np.complex128,
                chunks=(nspin, 1, nbasis, nbasis), # chunk along k points for faster query
                compression="lzf",
            )
            
            dset_C = file.create_dataset(
                "mo_coeffs",
                shape=(nspin, nkpoints, nbasis, nbasis),
                dtype=np.complex128,
                chunks=(nspin, 1, nbasis, nbasis), # chunk along k points for faster query
                compression="lzf",
            )
            # Loop over k points
            for ikpt in track(range(nkpoints), description="[bold blue]Constructing IAOs...", total=nkpoints):
                
                # get K vectors (k + G)
                K_vecs = self.post_wfc.fetch_gvectors(ikpt, return_cart=True, add_k=True)
                # get weight of this kpoint            
                k_weight = self.post_wfc.kpoint_weights[ikpt]
                
                # create lists to store basis matrices
                local_basis_matrices = []
                local_partial_overlaps = []
                
                for atom_idx, local_basis in enumerate(self.basis_map):
                    # get phase
                    spatial_phase = np.exp(-1j * np.dot(K_vecs, atom_positions[atom_idx]))
                    
                    # get basis matrix
                    local_basis_matrices.append(
                        local_basis.evaluate_q_functions(
                            K_vecs, 
                            atom_positions[atom_idx],
                            spatial_phase=spatial_phase,
                            )    
                    )
                    
                    local_partial_overlaps.append(local_basis.basis_aug_overlaps)
                # combine to single matrix for efficiency (shape: nbasis,nKvecs)
                local_basis_matrix = np.vstack(local_basis_matrices)
                local_partial_overlap_matrix = np.vstac(local_partial_overlaps)
                
                for ispin in range(nspin):
                    # read coefficients (shape: nbands, ngvecs)
                    
                    ###################################################################
                    # S_12 and S_21
                    ###################################################################
                    # We precalculate Psi_ps. Read from disk 
                    # shape: ngvecs, nbands
                    psi_ps = self.post_wfc.fetch_psi(ikpt=ikpt, ispin=ispin, iband=np.arange(nbands)).T
                    
                    # <Chi | Psi_ps>
                    # shape: nbasis, nbands
                    chi_psi_ps = local_basis_matrix.conj() @ psi_ps
                    
                    # <p | Psi_ps>
                    # We also precalculate the paw projector overlaps, 
                    # shape: nbands, nprojectors
                    paw_psi_ps = self.post_wfc.fetch_projector_overlaps(ikpt=ikpt, ispin=ispin, iband=np.arange(nbands))
                    
                    # <chi | phi_diff>
                    breakpoint()

                    # Overlap S21 = <Chi | Psi_ps> dot <p | Psi_ps>
                    # (nbasis, ngvecs) @ (ngvecs, nbands) -> (nbasis, nbands)
                    S21 = np.dot(chi_psi_ps, paw_psi_ps)
            
                    # 3. Transpose to get S12: shape (nbands, nbasis)
                    S12 = S21.T
                    ###################################################################
                    # P_12 and P_21
                    ###################################################################
                    # Both P_12 and P_21 are simplified by our choice of Psi as our
                    # B1 basis and an orthonormal (assuming negligible cross-atom overlap)
                    # NAO basis for B2.
                    # P_12 = S_1^-1 S_12 = S_12
                    P12 = S12
                    # P_21 = S_2^-1 S_21 = S_21
                    P21 = S21
                    
                    # 1. Determine occupied band indices
                    # (For insulators/semiconductors occ > 1e-5; for metals use threshold or formal valence count)
                    occ_mask = self.occupancies[ispin, ikpt] > 1e-5
                    breakpoint()
                    
                    ###################################################################
                    # C_tilde (Shape: nbands x n_occ)
                    ###################################################################
                    # Slice S21 for occupied bands only -> shape: (nbasis, n_occ)
                    P21_occ = P21[:, occ_mask]
                    
                    # P12 is (nbands, nbasis), S21_occ is (nbasis, n_occ)
                    # C_tilde_raw shape: (nbands, n_occ)
                    # NOTE: C_occ would just be an identity matrix in our case
                    C_tilde_raw = P12 @ P21_occ
                    
                    # ctc matrix(n_occ x n_occ)
                    ctc = C_tilde_raw.conj().T @ C_tilde_raw
                    ctc_inv_sqrt = inv(sqrtm(ctc))
                    
                    # C_tilde shape: (nbands, n_occ)
                    C_tilde = C_tilde_raw @ ctc_inv_sqrt
                    
                    ###################################################################
                    # Projectors O and O_tilde (Shape: nbands x nbands)
                    ###################################################################
                    # In the band basis, O is a diagonal matrix of band occupancies
                    O = np.diag(occ_mask.astype(float))
                    
                    # Depolarized occupied projector O_tilde = C_tilde @ C_tilde.T
                    O_tilde = C_tilde @ C_tilde.T
                    
                    ###################################################################
                    # Unorthogonalized IAO Coefficients A (Shape: nbands x nbasis)
                    ###################################################################
                    I_bands = np.eye(nbands)
                    # A = [ 1 + 2*(O @ O_tilde) - O_tilde - O ] @ P12
                    A = (I_bands + 2*(O @ O_tilde) - O_tilde - O) @ P12
                    A_T = A.conj().T
                    
                    ###################################################################
                    # E and H_IAO
                    ###################################################################
                    E = self.energies[ispin, ikpt]
                    H_IAO = A_T @ (E[:, None] * A)
                    
                    ###################################################################
                    # C_IAO
                    ###################################################################
                    eigvals_iao, C_IAO = np.linalg.eigh(H_IAO)
                    
                    ###################################################################
                    # Save Results to disk
                    ###################################################################
                    dset_A[ispin, ikpt] = A * k_weight
                    dset_C[ispin, ikpt] = C_IAO * k_weight
                    dset_H[ispin, ikpt] = H_IAO * k_weight