# -*- coding: utf-8 -*-

from pathlib import Path
import json

from pymatgen.core import Element
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator, CubicSpline
from scipy.integrate import simpson
from baderkit.post_wfc.paw.paw_dataset import PAWSpecies
from baderkit.post_wfc.base_basis import BaseSpecies

from baderkit.post_wfc.wfc_numba import evaluate_real_harmonics_multi

@dataclass
class AESpecies(BaseSpecies):
    """
    Standardized, code-agnostic data representation of atomic core reconstruction 
    parameters. Overcomplete basis channels are canonically orthogonalized and sorted 
    by energy on initialization to yield clear, independent radial charge densities.
    All calculations and quantities are handled natively in Angstrom and eV units.
    """
    
    # BASIC INFORMATION
    
    basis: str = field(default_factory=None)
    """The basis set used to generate the reference"""
    
    functional: str = field(default_factory=None)
    """The XC functional used to generate the reference"""

    unrestricted: bool = field(default_factory=False)
    """Whether or not this reference is unrestricted (spin-polarized)"""
    
    primitives: dict = field(default_factory=None)
    """The primitive basis functions grouped by angular momentum"""
    
    # PARENT REFERENCE CONTEXT
    paw_species: PAWSpecies | None = None
    """The pseudopotential this species maps onto"""
    
    # QUANTUM NUMBERS

    spin_channels: NDArray = field(default_factory=lambda: np.empty(0, dtype=np.int8))
    """1D array mapping spin channel projections (0 for alpha/restricted, 1 for beta)."""

    # STATE RECONSTRUCTIONS & SLICING
    state_vectors: NDArray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    """2D array of shape (states, flat_basis) containing pre-diagonalized linear combination coefficients."""

    l_slices: dict = field(default_factory=dict)
    """Dictionary mapping angular momentum l values to their index slices within state_vectors."""

    # WAVEFUNCTIONS & CHARGE DENSITIES
    radial_functions: NDArray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    """2D array of shape (channels, grid) containing normalized real-space wavefunctions R_nl(r)."""

    q_radial_functions: NDArray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    """2D array of shape (channels, grid) containing normalized reciprocal-space wavefunctions R_nl(r)."""

    radial_rho: NDArray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    """2D array of shape (channels, grid) containing radial charge density (in Angstrom^-3)."""
    
    radial_tau: NDArray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    """2D array of shape (channels, grid) containing radial kinetic energy density (in Angstrom^-5)."""

    # CORE RECONSTRUCTION OVERLAPS
    ae_overlaps: NDArray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    """2D array of shape (basis_channels, paw_channels) containing the overlap between this basis and the PAW all-electron partial waves."""

    ps_overlaps: NDArray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    """2D array of shape (basis_channels, paw_channels) containing the overlap between this basis and the PAW pseudo partial waves"""

    core_correction_overlaps: NDArray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    """2D array of shape (basis_channels, paw_channels) containing the overlap between this basis and the PAW partial waves needed for correcting the smooth wavefunction"""

    # 1D CONTINUOUS SPLINES
    radial_splines: list = field(default_factory=list)
    """List of scipy.interpolate.CubicSpline objects representing 1D real-space wavefunctions R_nl(r)."""

    q_radial_splines: list = field(default_factory=list)
    """List of scipy.interpolate.CubicSpline objects representing 1D reciprocal-space NAO profiles."""
    
    @property
    def max_occupancies(self):
        if getattr(self, "_max_occupancies", None) is None:
            self._max_occupancies = self.get_occupancies()
        return self._max_occupancies
    
    @property
    def cumulative_occupancies(self):
        if getattr(self, "_cumulative_occupancies", None) is None:
            self._cumulative_occupancies = np.cumulative_sum(self.max_occupancies, include_initial=True)
        return self._cumulative_occupancies
    
    @property
    def cumulative_radial_rho(self):
        if getattr(self, "_cumulative_radial_rho", None) is None:
            self._cumulative_radial_rho = np.cumulative_sum(self.max_occupancies[:, None] * self.radial_rho, axis=0, include_initial=True)
        return self._cumulative_radial_rho
    
    @property
    def cumulative_radial_tau(self):
        if getattr(self, "_cumulative_radial_tau", None) is None:
            self._cumulative_radial_tau = np.cumulative_sum(self.max_occupancies[:, None] * self.radial_tau, axis=0, include_initial=True)
        return self._cumulative_radial_tau
    
    @property
    def radial_rho_interpolator(self):
        if getattr(self, "_radial_rho_interpolator", None) is None:
            self._radial_rho_interpolator = RegularGridInterpolator((self.cumulative_occupancies, self.radial_grid), self.cumulative_radial_rho, method='linear')
        return self._radial_rho_interpolator
    
    @property
    def radial_tau_interpolator(self):
        if getattr(self, "_radial_tau_interpolator", None) is None:
            self._radial_tau_interpolator = RegularGridInterpolator((self.cumulative_occupancies, self.radial_grid), self.cumulative_radial_tau, method='linear')
        return self._radial_tau_interpolator
    
    def __post_init__(self):
        """
        Generates the radial rho, kinetic energy density, and NAO radial splines on initialization
        only if they are not already provided (bypasses recalculation for valence subsets).
        """
        self.radial_rho, self.radial_tau = self._calculate_radial_densities(alpha=1.0)
        
        self._generate_radial_functions()

        self._calculate_correction_overlap()
        
    def get_occupancies(self, min_electrons: float = None, max_electrons: float = None) -> NDArray:
        """
        Generates a 1D occupancy array of shape (n_channels,) containing the 
        number of electrons allocated to each subshell channel within the specified 
        electron window [min_electrons, max_electrons] using the Aufbau principle.
        """
        max_per_state = 1.0 if self.unrestricted else 2.0
        
        if min_electrons is None:
            min_electrons = 0
        if max_electrons is None:
            max_electrons = len(self.angular_momenta) * max_per_state
        
        if min_electrons > max_electrons:
            raise ValueError("min_electrons cannot be greater than max_electrons.")
        
        n_channels = len(self.eigenvalues)
        occs = np.zeros(n_channels, dtype=np.float64)
        
        prev_count = 0.0
        for idx in range(n_channels):
            # Increment current_count
            current_count = prev_count + max_per_state
            overlap = max(0.0, min(current_count, max_electrons) - max(prev_count, min_electrons))
            occs[idx] = overlap
            prev_count = current_count
                
        return occs
    
    def get_radial_charge_density(self, min_electrons: float, max_electrons: float) -> NDArray:
        """
        Computes the total isotropic radial charge density profiles across a range of total electron counts.
        """
        occupancies = self.get_occupancies(min_electrons, max_electrons)
        total_rho = np.sum(occupancies * self.radial_rho.T, axis=1)
        return total_rho

    def get_radial_kinetic_energy_density(self, min_electrons: float, max_electrons: float) -> NDArray:
        """
        Computes the total isotropic radial positive-definite kinetic energy density (KED) profiles 
        across a range of total electron counts.
        """
        occupancies = self.get_occupancies(min_electrons, max_electrons)
        total_tau = np.sum(occupancies * self.radial_tau.T, axis=1)
        return total_tau

    @classmethod
    def from_file(
            cls,
            filename: str | Path,
            paw_species: PAWSpecies | Path | None = None,
            energy_range: tuple | None = None,
            energy_tol: float = 0.1,
            cutoff_radius: float = 8.0,
            g_cutoff_radius: float = 15.0,
            ):
        """
        Parses compressed linear state-vector references natively stored in Angstrom and eV units.
        Expands degenerate l-shells into individual (2l + 1) magnetic m-states.
        """
        file_path = Path(filename)
        if not file_path.exists():
            raise FileNotFoundError(f"Missing analytical basis binary for element: {file_path}")
            
        if isinstance(paw_species, Path) or isinstance(paw_species, str):
            paw_species = PAWSpecies.from_filename(paw_species)
            
        data = np.load(file_path)
        metadata = json.loads(str(data["metadata"]))
        
        element = metadata["element"]
        element_py = Element(element)
        matrix_dims = metadata["matrix_layout_dimensions"]
        
        # Precompute slicing masks
        l_slices = {}
        curr_offset = 0
        for l_str in sorted(matrix_dims.keys(), key=int):
            l_val = int(l_str)
            dim = matrix_dims[l_str]
            l_slices[l_val] = (curr_offset, curr_offset + dim)
            curr_offset += dim
    
        # Build our grids to exactly match the PAW potential's
        radial_grid, real_is_log, q_radial_grid, q_is_log = cls._build_radial_grid(
            paw_species, cutoff_radius, g_cutoff_radius
        )
        
        # Remove states outside system bounds
        if paw_species is not None:
            keep_indices = cls._prune_basis(
                paw_species, 
                basis_data=data, 
                energy_range=energy_range,
                tol=energy_tol,
                )
        else:
            keep_indices = [i for i in range(len(data["occupancies"]))]
        
        # Replicate degenerate radial states across magnetic quantum numbers (m)
        kept_l = data["angular_momenta"][keep_indices]
        kept_n = data["principal_quantum_numbers"][keep_indices]
        kept_energies = data["energies"][keep_indices]
        kept_occupancies = data["occupancies"][keep_indices]
        kept_spin = data["spin_channels"][keep_indices]
        kept_states = data["packed_state_vectors"][keep_indices, :]
        
        expanded_l = []
        expanded_n = []
        expanded_m = []
        expanded_energies = []
        expanded_occupancies = []
        expanded_spin = []
        expanded_states = []
        
        for i in range(len(keep_indices)):
            l_val = int(kept_l[i])
            # Replicate 2l + 1 times for magnetic components: -l <= m <= l
            for m_val in range(-l_val, l_val + 1):
                expanded_l.append(l_val)
                expanded_n.append(kept_n[i])
                expanded_m.append(m_val)
                expanded_energies.append(kept_energies[i])
                # We retain the core occupancy per channel or distribute it if preferred.
                # Usually we keep the base metadata of the parent state.
                expanded_occupancies.append(kept_occupancies[i])
                expanded_spin.append(kept_spin[i])
                expanded_states.append(kept_states[i])
                
        # Re-pack back into numpy arrays
        expanded_l = np.array(expanded_l, dtype=np.int32)
        expanded_n = np.array(expanded_n, dtype=np.int32)
        expanded_m = np.array(expanded_m, dtype=np.int32)
        expanded_energies = np.array(expanded_energies, dtype=np.float64)
        expanded_occupancies = np.array(expanded_occupancies, dtype=np.float64)
        expanded_spin = np.array(expanded_spin, dtype=np.int32)
        expanded_states = np.vstack(expanded_states)
        
        # Instantiate instance with the explicit PAW parent context bound
        return cls(
            name=f"{element}_{metadata['functional']}",
            element=element,
            Z=float(element_py.Z),
            basis=metadata.get("basis", None),
            functional=metadata["functional"],
            unrestricted=int(np.max(data["spin_channels"])) > 0,
            primitives=cls._flatten_basis_primitives(metadata["basis_primitives"]),
            radial_grid=radial_grid, 
            q_radial_grid=q_radial_grid,
            real_is_log=real_is_log,
            q_is_log=q_is_log,
            paw_species=paw_species,
            angular_momenta=expanded_l,
            principal_quantum_numbers=expanded_n,
            magnetic_quantum_numbers=expanded_m,
            eigenvalues=expanded_energies, 
            reference_occupations=expanded_occupancies,
            spin_channels=expanded_spin,
            state_vectors=expanded_states,
            l_slices=l_slices,
        )
        
    ###########################################################################
    # Utility Functions
    ###########################################################################
    
    @staticmethod
    def _flatten_basis_primitives(basis_primitives):
        """Flattens nested primitive structures into flat continuous 1D arrays and offset pointers."""
        contractions_by_l = {}
        for l_str, bas_list in basis_primitives.items():
            l = int(l_str)
            exps_list = []
            coeffs_list = []
            g_coeffs_list = []
            offsets = [0]
            
            for bas_data in bas_list:
                exps = np.array(bas_data["exponents"], dtype=np.float64)
                coeffs_mat = np.array(bas_data["coefficients"], dtype=np.float64)
                g_coeffs_mat = np.array(bas_data["g_coefficients"], dtype=np.float64)
                nctr = coeffs_mat.shape[1]
                
                for c in range(nctr):
                    exps_list.extend(exps)
                    coeffs_list.extend(coeffs_mat[:, c])
                    g_coeffs_list.extend(g_coeffs_mat[:, c])
                    offsets.append(offsets[-1] + len(exps))
                    
            contractions_by_l[l] = {
                "exps": np.array(exps_list, dtype=np.float64),
                "coeffs": np.array(coeffs_list, dtype=np.float64),
                "g_coeffs": np.array(g_coeffs_list, dtype=np.float64),
                "offsets": np.array(offsets, dtype=np.int32)
            }
        return contractions_by_l
    
    @staticmethod
    def _prune_basis(
        paw_species: PAWSpecies,
        basis_data: dict,
        energy_range: tuple | None,
        tol: float = 0.1,
        ):
        # BELOW PSEUDO-CORE
        Z = paw_species.Z
        
        occupancies = np.where(basis_data["occupancies"] > 1e-4, basis_data["occupancies"], 0.0)
        occupied_indices = np.flip(np.where(occupancies>0)[0])
        
        # loop over indices in reverse until we match pseudopotential
        accumulated_charge = 0.0
        for idx in occupied_indices:
            l = basis_data["angular_momenta"][idx]
            accumulated_charge += occupancies[idx] * (2 * l + 1)
            if accumulated_charge >= Z - 1e-4:
                min_idx = idx
                break
            
        if energy_range is None:
            e_min = basis_data["energies"][min_idx]
            e_max = basis_data["energies"].max()
        else:
            e_min, e_max = energy_range
        
        # ABOVE SYSTEM ENERGY RANGE
        # adjust to systems lowest energy
        atom_energies = basis_data["energies"] + (e_min - basis_data["energies"][min_idx])
        # get mask where basis is below max allowed energy
        valid_bases = atom_energies <= (e_max * (1+tol))
        valid_bases[:min_idx] = False
        
        return np.where(valid_bases)[0]
        
    @staticmethod
    def _build_radial_grid(
        paw_species: PAWSpecies | None,
        cutoff_radius: float = 8.0,
        g_cutoff_radius: float = 15.0,
    ) -> tuple:
        """
        Constructs a radial coordinate mesh grid matching the related PAWSpecies
        grids, but extended out to the cutoff radii.
        """
        # -------------------------------------------------------------
        # Fallback when no PAWSpecies is provided (e.g., isolated atom)
        # -------------------------------------------------------------
        if paw_species is None:
            # Fine logarithmic real-space grid (typical starting point ~ 1e-5 to avoid r=0 divergence)
            r_start = 1e-5
            r_points = 2000
            real_grid = np.geomspace(r_start, cutoff_radius, num=r_points)
            real_is_log = True
            
            # Fine linear reciprocal-space grid
            q_points = 2000
            q_grid = np.linspace(0.0, g_cutoff_radius, num=q_points)
            q_is_log = False
            
            return real_grid, real_is_log, q_grid, q_is_log
    
        # -------------------------------------------------------------
        # Standard path using PAWSpecies grids
        # -------------------------------------------------------------
        paw_grid = paw_species.radial_grid
        g_paw_grid = paw_species.q_radial_grid
        
        result_grids = []
        
        for grid, cutoff in zip((paw_grid, g_paw_grid), (cutoff_radius, g_cutoff_radius)):
        
            if cutoff <= grid[-1]:
                result_grids.append(grid.copy())
                # We still need to determine if the existing grid is logarithmic or linear
                # so we can append the correct boolean flag to match the return structure
                r1, r2, r3 = grid[-1], grid[-2], grid[-3]
                ratio1 = r1 / r2 if r2 != 0.0 else 1.0
                ratio2 = r2 / r3 if r3 != 0.0 else 1.0
                is_log = abs(ratio1 - ratio2) / ratio1 < 1e-4
                result_grids.append(is_log)
                continue
            
            # Extract spacing variables dynamically from grid boundary
            r1, r2, r3 = grid[-1], grid[-2], grid[-3]
            ratio1 = r1 / r2 if r2 != 0.0 else 1.0
            ratio2 = r2 / r3 if r3 != 0.0 else 1.0
            diff1 = r1 - r2
            
            extended_points = list(grid)
            current_r = r1
            
            # If log ratio is highly linear (logarithmic coordinate grid style)
            if abs(ratio1 - ratio2) / ratio1 < 1e-4:
                is_log = True
                while current_r < cutoff:
                    current_r *= ratio1
                    extended_points.append(current_r)
            else:
                is_log = False
                # Linear grid extension
                while current_r < cutoff:
                    current_r += diff1
                    extended_points.append(current_r)
                    
            result_grids.append(np.array(extended_points, dtype=np.float64))
            result_grids.append(is_log)
    
        return tuple(result_grids)

    def _generate_radial_functions(self) -> None:
        """
        Generates radial coordinates, normalized real-space wavefunctions R_nl(r), 
        and pre-calculates the exact analytical reciprocal-space G-splines directly 
        on the VASP radial grid coordinate system.
        """
        real_grid = self.radial_grid
        q_grid = self.q_radial_grid
        
        num_grid = len(real_grid)
        num_q_grid = len(q_grid)
        num_states = len(self.eigenvalues)
        
        # Pre-detect if the real-space grid is logarithmic
        log_spacing = np.diff(np.log(real_grid))
        is_logarithmic = self.real_is_log
        
        # Initialize arrays and lists to store the generated NAO representations
        self.radial_functions = np.zeros((num_states, num_grid), dtype=np.float64)
        self.q_radial_functions = np.zeros((num_states, num_q_grid), dtype=np.float64)
        self.q_radial_splines = []
        self.radial_splines = []
        
        # Construct radial basis functions
        for idx in range(num_states):
            l = self.angular_momenta[idx]
            # Safety fallback if we don't have this angular momentum
            if l not in self.primitives:
                self.q_radial_splines.append(CubicSpline(q_grid, np.zeros_like(q_grid), extrapolate=False))
                self.radial_splines.append(CubicSpline(real_grid, np.zeros_like(real_grid), extrapolate=False))
                
                self.alphas_states.append(np.array([1.0], dtype=np.float64))
                self.g_coeffs_states.append(np.array([0.0], dtype=np.float64))
                continue
                
            # Get basis exponents and coefficients
            c_data = self.primitives[l]
            exps = c_data["exps"]
            coeffs = c_data["coeffs"]
            offsets = c_data["offsets"]
            
            # Extract state vector coefficients
            start_l, end_l = self.l_slices[l]
            c_state = self.state_vectors[idx, start_l:end_l]
            dim = len(c_state)
            
            # Combine GTO primitives according to state vector coefficients
            alphas_state = []
            coeffs_state = []
            for p in range(dim):
                start = offsets[p]
                end = offsets[p+1]
                alphas_state.extend(exps[start:end])
                coeffs_state.extend(c_state[p] * coeffs[start:end])
                
            alphas_state = np.array(alphas_state, dtype=np.float64)
            coeffs_state = np.array(coeffs_state, dtype=np.float64)
            
            # Evaluate real-space radial wavefunction: R(r) = r^l * sum_i c_i e^{-alpha_i r^2}
            r_pow_l = real_grid ** l
            raw_R = np.zeros(num_grid, dtype=np.float64)
            for alpha, c in zip(alphas_state, coeffs_state):
                raw_R += c * np.exp(-alpha * (real_grid ** 2))
            raw_R *= r_pow_l
            
            # Compute normalization integral: int_0^inf R(r)^2 r^2 dr
            if is_logarithmic:
                # Integrand: R(r)^2 * r^3 over d(ln r)
                norm_integral = simpson(y=(raw_R ** 2) * (real_grid ** 3), dx=log_spacing[0])
            else:
                # Integrand: R(r)^2 * r^2 over dr
                norm_integral = simpson(y=(raw_R ** 2) * (real_grid ** 2), x=real_grid)
            
            # Apply normalization factor to real-space radial wavefunctions
            if norm_integral > 1e-12:
                norm_factor = np.sqrt(norm_integral)
                R_normalized = raw_R / norm_factor
            else:
                norm_factor = 1.0
                R_normalized = raw_R
                
            self.radial_functions[idx, :] = R_normalized
    
            # Construct exact analytical spherical Bessel transform in G-space
            phi_q = np.zeros_like(q_grid)
            for alpha, c in zip(alphas_state, coeffs_state):
                amplitude_factor = (np.pi / alpha) ** 1.5
                shape_factor = (q_grid ** l) / ((2.0 * alpha) ** l)
                decay_factor = np.exp(-(q_grid ** 2) / (4.0 * alpha))
                
                primitive_g_space = amplitude_factor * shape_factor * decay_factor
                
                # Apply cusp boundary conditions at q = 0 for l > 0
                if l > 0:
                    primitive_g_space[q_grid < 1e-12] = 0.0
                    
                phi_q += c * primitive_g_space
                
            phi_q_normalized = phi_q / norm_factor
            
            self.q_radial_functions[idx, :] = phi_q_normalized
            
        # Pre-compile physical 1D splines
        self.radial_splines = self._create_1d_splines(real_grid, self.radial_functions)
        self.q_radial_splines = self._create_1d_splines(q_grid, self.q_radial_functions)
            
    def _calculate_correction_overlap(self):
        # get grid. Matches PAW grid but extends beyond it.
        grid = self.radial_grid
        grid_sq = grid ** 2
        grid_cu = grid ** 3
        
        # Detect if the grid is logarithmic
        is_logarithmic = self.real_is_log
        if is_logarithmic:
            spacing = np.diff(np.log(grid))
        else:
            spacing = np.diff(grid)
        dx = spacing[0]
        
        # Get local basis radial functions and PAW partial waves
        local_functions = self.radial_functions
        paw_ae_waves = self.paw_species.all_electron_partial_waves
        paw_ps_waves = self.paw_species.pseudo_partial_waves
        
        # Get the cutoff radii for each paw channel
        r_cs = self.paw_species.paw_cutoffs
        
        # Initialize containers (orbital_index, projector_index)
        self.core_correction_overlaps = np.empty((len(local_functions), len(paw_ae_waves)))
        
        # Loop over each basis function
        for alpha, chi_alpha in enumerate(local_functions):
            l_a = self.angular_momenta[alpha]
            m_a = self.magnetic_quantum_numbers[alpha]
            
            # Loop over the PAW partial wave channels (indexed by projector index i)
            for i, phi_ae in enumerate(paw_ae_waves):
                l_b = self.paw_species.angular_momenta[i]
                m_b = self.paw_species.magnetic_quantum_numbers[i]
                
                # enforce kronecker delta
                if l_a != l_b or m_a != m_b:
                    self.core_correction_overlaps[alpha, i] = 0.0
                    continue
                
                # Get cutoff radius and mask
                r_c = r_cs[i]
                mask = np.where(grid < r_c)[0]
                
                # cut down to maks size
                chi_cut = chi_alpha[mask]
                phi_ps = paw_ps_waves[i][mask]
                phi_ae = paw_ae_waves[i][mask]
                
                # evaluate <chi|phi_ae> and <chi|phi_ps>:
                if is_logarithmic:
                    # Log-grid integration: Integrate r^3 * chi * phi d(ln r)
                    # NOTE: We use r^3 because dr = r * d(ln r)
                    grid_cut = grid_cu[mask]
                else:
                    # Linear-grid integration: Integrate r^2 * chi * phi dr
                    grid_cut = grid_sq[mask]
                    
                # <chi|phi> = integral(r^2*chi(r)*phi(r)dr)
                diff_integrand = grid_cut * chi_cut * (phi_ae-phi_ps)
                
                diff_overlap = simpson(y=diff_integrand, dx=dx)
                
                # Store results
                self.core_correction_overlaps[alpha, i] = diff_overlap
                
    def _calculate_radial_densities(self, alpha: float) -> tuple[NDArray, NDArray]:
        """
        Evaluates the radial charge density (rho) and positive-definite kinetic energy density (tau) 
        profiles natively in Angstrom units for a given primitive exponent scaling multiplier (alpha).
        Calculations strictly enforce spherical averaging over the full angular harmonic space.
        """
        real_grid = self.radial_grid
        
        num_states = len(self.eigenvalues)
        num_grid = len(self.radial_grid)
        radial_rho = np.zeros((num_states, num_grid), dtype=np.float64)
        radial_tau = np.zeros((num_states, num_grid), dtype=np.float64)
        
        for idx in range(num_states):
            l = self.angular_momenta[idx]
            if l in self.primitives:
                c_data = self.primitives[l]
                exps = alpha * c_data["exps"] 
                coeffs = c_data["coeffs"]
                offsets = c_data["offsets"]
                
                # Directly slice the linear state coefficients out of the flat database row
                start_l, end_l = self.l_slices[l]
                c_state = self.state_vectors[idx, start_l:end_l]
                
                dim = len(c_state)
                phi = np.zeros((dim, num_grid), dtype=np.float64)
                
                r_pow_l = real_grid ** l
                r_pow_l_plus_1 = real_grid ** (l + 1)
                
                # Analytical accumulation vectors to avoid division-by-zero errors at r=0
                sum_0_matrix = np.zeros((dim, num_grid), dtype=np.float64)
                sum_1_matrix = np.zeros((dim, num_grid), dtype=np.float64)
                
                for p in range(dim):
                    start = offsets[p]
                    end = offsets[p+1]
                    
                    sum_0 = np.zeros(num_grid, dtype=np.float64)
                    sum_1 = np.zeros(num_grid, dtype=np.float64)
                    for k in range(start, end):
                        exp_factor = np.exp(-exps[k] * (real_grid ** 2))
                        sum_0 += coeffs[k] * exp_factor
                        sum_1 += coeffs[k] * exps[k] * exp_factor
                    
                    sum_0_matrix[p, :] = sum_0
                    sum_1_matrix[p, :] = sum_1
                    phi[p, :] = r_pow_l * sum_0
                
                # Pure state vector transformations 
                phi_state = np.dot(c_state, phi)
                sum_0_state = np.dot(c_state, sum_0_matrix)
                sum_1_state = np.dot(c_state, sum_1_matrix)
                
                # Evaluate derivatives and kinetic components analytically.
                if l == 0:
                    phi_prime_state = -2.0 * real_grid * sum_1_state
                    raw_tau = 0.5 * (phi_prime_state * phi_prime_state)
                else:
                    r_pow_l_minus_1 = real_grid ** (l - 1)
                    phi_prime_state = l * r_pow_l_minus_1 * sum_0_state - 2.0 * r_pow_l_plus_1 * sum_1_state
                    phi_state_over_r = r_pow_l_minus_1 * sum_0_state
                    raw_tau = 0.5 * ((phi_prime_state * phi_prime_state) + l * (l + 1) * (phi_state_over_r * phi_state_over_r))
                
                # Scale by 1/(4*pi) to obtain the perfectly isotropic spherical average per electron
                rho_final = (phi_state * phi_state) / (4.0 * np.pi)
                tau_final = raw_tau / (4.0 * np.pi)
                
                # Normalize density profiles accurately inside r-space
                integrated_charge = np.trapezoid(4.0 * np.pi * (self.radial_grid ** 2) * rho_final, self.radial_grid)
                
                if integrated_charge > 1e-6:
                    rho_final /= integrated_charge
                    tau_final /= integrated_charge
                
                radial_rho[idx, :] = rho_final
                radial_tau[idx, :] = tau_final
                
        return radial_rho, radial_tau