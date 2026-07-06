# -*- coding: utf-8 -*-

from pathlib import Path
import json

from pymatgen.core import Element
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

from baderkit.post_wfc.pseudopotentials.paw_dataset import PAWSpecies


@dataclass
class AESpecies:
    """
    Standardized, code-agnostic data representation of atomic core reconstruction 
    parameters. Overcomplete basis channels are canonically orthogonalized and sorted 
    by energy on initialization to yield clear, independent radial charge densities.
    """
    # --- Fields WITHOUT default values first ---
    name: str
    """Name of this all electron reference"""
    
    element: str
    """Chemical element symbol (e.g., 'Ca')."""
    
    Z: float
    """Total electrons in this pseudopotential"""
    
    basis: str
    """The basis set used to generate the reference"""
    
    functional: str
    """The XC functional used to generate the reference"""

    unrestricted: bool
    """Whether or not this reference is unrestricted (spin-polarized)"""
    
    primitives: dict
    """The primitive basis functions grouped by angular momentum"""
    
    radial_grid: NDArray
    """1D array containing the radial coordinate mesh grid points, r."""
    
    # --- Fields WITH default values second ---
    paw_species: PAWSpecies | None = None
    """The pseudopotential this species maps onto"""
    
    angular_momenta: NDArray = field(default_factory=lambda: np.empty(0, dtype=np.int_))
    """1D array of orbital angular momentum quantum numbers, l, for each merged active channel."""
    
    eigenvalues: NDArray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    """1D array of atomic reference state energy eigenvalues for each merged channel."""
    
    reference_occupations: NDArray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    """1D array of reference atomic occupations for each merged channel."""

    density_matrices: list = field(default_factory=list)
    """List of dictionaries containing the full symmetric density matrices for each merged channel."""

    # --- Fields initialized during post_init ---
    radial_rho: NDArray = field(init=False)
    """2D array of shape (channels, grid) containing radial charge density."""
    
    radial_tau: NDArray = field(init=False)
    """2D array of shape (channels, grid) containing radial kinetic energy density."""
    
    def __post_init__(self):
        """
        Generates the radial rho and kinetic energy density matrices on initialization.
        If a PAWSpecies dataset is provided, triggers an isolated optimization helper 
        method to calibrate the analytical primitive exponents first.
        """
        # Execute exponent refinement if a PAW validation baseline is present
        # if self.paw_species is not None:
        #     self._optimize_exponents()
            
        # Final evaluation pass mapping fields onto the active grid space
        self.radial_rho, self.radial_tau = self._compute_radial_densities(alpha=1.0)

    def _optimize_exponents(self):
        """
        Executes a regularized linear algebraic projection of the PAW core partial waves 
        directly onto the original contracted basis functions of the AE reference.
        Preserves the original canonical state definitions and energy levels intact, 
        ensuring missing valence states (like 4s) or virtual manifolds are retained.
        """
        r = self.radial_grid
        BOHR_TO_ANGSTROM = 0.529177210903
        r_bohr = r / BOHR_TO_ANGSTROM
        
        # Interpolate raw PAW wavefunctions onto master radial grid to align coordinates
        r_paw = self.paw_species.radial_grid
        num_paw_channels = self.paw_species.all_electron_partial_waves.shape[0]
        
        paw_u_interp = np.zeros((num_paw_channels, len(r)), dtype=np.float64)
        for i in range(num_paw_channels):
            paw_u_interp[i, :] = np.interp(r, r_paw, self.paw_species.all_electron_partial_waves[i])
            
        unique_l = np.unique(self.angular_momenta)
        
        # Coordinate weighting function to favor the core and accept valence variance
        W = np.exp(-2.0 * r) / (r + 1e-6)
        
        print("\n" + "="*80)
        print("     STARTING AE-RETAINED INVERTED CHANGE-OF-BASIS PROJECTION")
        print("="*80)
        
        for l in unique_l:
            if l not in self.primitives:
                continue
                
            # Extract pristine contracted basis descriptors from the loaded AE reference
            c_data = self.primitives[l]
            exps = c_data["exps"]
            coeffs = c_data["coeffs"]
            offsets = c_data["offsets"]
            dim = len(offsets) - 1
            
            # Evaluate the original contracted basis functions Phi_p(r) on the active mesh
            phi_contracted = np.zeros((dim, len(r)), dtype=np.float64)
            r_pow_l = r_bohr ** l
            for p in range(dim):
                start = offsets[p]
                end = offsets[p+1]
                sum_0 = np.zeros(len(r), dtype=np.float64)
                for k in range(start, end):
                    sum_0 += coeffs[k] * np.exp(-exps[k] * (r_bohr ** 2))
                phi_contracted[p, :] = r_pow_l * sum_0
                
            # Compute Contracted-Contracted Overlap Matrix S 
            S = np.zeros((dim, dim), dtype=np.float64)
            for p1 in range(dim):
                for p2 in range(dim):
                    S[p1, p2] = np.trapezoid(W * phi_contracted[p1, :] * phi_contracted[p2, :], r)
                    
            S_reg = S + 1e-9 * np.eye(dim)
            
            # Isolate occupied core channels inside the PAW potential
            paw_indices = np.where((self.paw_species.angular_momenta == l) & (self.paw_species.reference_occupations > 1e-4))[0]
            paw_indices = paw_indices[np.argsort(self.paw_species.eigenvalues[paw_indices])]
            
            # Map them directly onto the lowest matching core states of the AE reference
            ae_indices = np.where(self.angular_momenta == l)[0]
            ae_sorted = ae_indices[np.argsort(self.eigenvalues[ae_indices])]
            
            shell_name = {0: 's', 1: 'p', 2: 'd', 3: 'f'}.get(l, '?')
            print(f"\n[Shell l={l} ({shell_name}-orbitals)]")
            print(f"  -> Total AE reference states defined: {len(ae_sorted)}")
            print(f"  -> Active PAW core channels mapped:    {len(paw_indices)}")
            
            # Run state-by-state projection for overlapping core domains
            for pair_idx, paw_idx in enumerate(paw_indices):
                if pair_idx >= len(ae_sorted):
                    print(f"    [!] WARNING: Extra PAW channel {paw_idx} exceeds AE state capacities. Skipping.")
                    break
                    
                ae_state_idx = ae_sorted[pair_idx]
                u_target = paw_u_interp[paw_idx]
                
                # Transform target wavefunction back to amplitude: phi = u / r
                with np.errstate(divide='ignore', invalid='ignore'):
                    phi_target = np.where(r > 1e-12, u_target / r, 0.0)
                if r[0] < 1e-10:
                    phi_target[0] = phi_target[1]
                    
                # Compute Cross-Overlap vector against original contracted functions
                B = np.zeros(dim, dtype=np.float64)
                for p in range(dim):
                    B[p] = np.trapezoid(W * phi_contracted[p, :] * phi_target, r)
                    
                C_state = np.linalg.solve(S_reg, B)
                
                # Update density matrix blocks for this state to align with PAW configurations
                for active_l in unique_l:
                    self.density_matrices[ae_state_idx][active_l] = np.zeros_like(self.density_matrices[ae_state_idx].get(active_l, np.zeros((1,1))))
                
                self.density_matrices[ae_state_idx][l] = np.outer(C_state, C_state)
                
                # Evaluation sanity diagnostics
                u_fit = r * np.dot(C_state, phi_contracted)
                max_err = np.max(np.abs(u_target - u_fit))
                print(f"    * Core Pair {pair_idx} -> Realigned AE State Index {ae_state_idx} using PAW Channel {paw_idx}")
                print(f"        Energy: {self.eigenvalues[ae_state_idx]:.4f} eV | Core Fit Max Abs Error: {max_err:.4e}")
                
            # Print status of protected unmapped states (like the 4s valence shell)
            if len(ae_sorted) > len(paw_indices):
                print(f"  --> Preserving {len(ae_sorted) - len(paw_indices)} unmapped valence/virtual states in their original form:")
                for unmapped_idx in ae_sorted[len(paw_indices):]:
                    print(f"        * Retained AE State Index {unmapped_idx} | Energy: {self.eigenvalues[unmapped_idx]:.4f} eV | Occ Baseline: {self.reference_occupations[unmapped_idx]:.2f}")
                    
        print("\n" + "="*80)
        print("     DIAGNOSTICS COMPLETE - SYSTEM GEOMETRY LOCKED TO AE LAYOUT")
        print("="*80 + "\n")
        
    def _compute_radial_densities(self, alpha: float) -> tuple[NDArray, NDArray]:
        """
        Evaluates the radial charge density (rho) and kinetic energy density (tau) 
        profiles for a given primitive exponent scaling multiplier (alpha).
        """
        BOHR_TO_ANGSTROM = 0.529177210903
        r_bohr = self.radial_grid / BOHR_TO_ANGSTROM
        
        num_states = len(self.eigenvalues)
        num_grid = len(self.radial_grid)
        radial_rho = np.zeros((num_states, num_grid), dtype=np.float64)
        radial_tau = np.zeros((num_states, num_grid), dtype=np.float64)
        
        for idx, d_blocks in enumerate(self.density_matrices):
            l = self.angular_momenta[idx]
            if l in self.primitives:
                c_data = self.primitives[l]
                exps = alpha * c_data["exps"] # Apply trial scaling factor
                coeffs = c_data["coeffs"]
                offsets = c_data["offsets"]
                D_l = d_blocks[l]
                
                # Clip non-physical negative eigenvalues to exactly zero
                if D_l.shape[0] > 0:
                    vals, vecs = np.linalg.eigh(D_l)
                    vals = np.maximum(vals, 0.0) 
                    D_l = vecs @ np.diag(vals) @ vecs.T
                
                dim = D_l.shape[0]
                phi = np.zeros((dim, num_grid), dtype=np.float64)
                phi_prime = np.zeros((dim, num_grid), dtype=np.float64)
                
                # Precompute r^l and r^{l+1} components
                r_pow_l = r_bohr ** l
                r_pow_l_plus_1 = r_bohr ** (l + 1)
                
                for p in range(dim):
                    start = offsets[p]
                    end = offsets[p+1]
                    
                    # Compute the two necessary primitive sums
                    sum_0 = np.zeros(num_grid, dtype=np.float64)
                    sum_1 = np.zeros(num_grid, dtype=np.float64)
                    for k in range(start, end):
                        exp_factor = np.exp(-exps[k] * (r_bohr ** 2))
                        sum_0 += coeffs[k] * exp_factor
                        sum_1 += coeffs[k] * exps[k] * exp_factor
                    
                    phi[p, :] = r_pow_l * sum_0
                    
                    # phi'(r) = l/r * phi(r) - 2 * r^{l+1} * sum_1
                    if l == 0:
                        phi_prime[p, :] = -2.0 * r_bohr * sum_1
                    else:
                        phi_prime[p, :] = (l / r_bohr) * phi[p, :] - 2.0 * r_pow_l_plus_1 * sum_1
                
                # Compute radial charge density: rho = sum Dpq * phip * phiq
                raw_rho = np.einsum('pq,pi,qi->i', D_l, phi, phi)
                rho_final = raw_rho / (4.0 * np.pi * (BOHR_TO_ANGSTROM ** 3))
                
                # Compute radial kinetic energy density
                angular_term = np.zeros(num_grid, dtype=np.float64)
                if l > 0:
                    angular_term = (l * (l + 1) / (r_bohr ** 2))
                
                term1 = np.einsum('pq,pi,qi->i', D_l, phi_prime, phi_prime)
                term2 = np.einsum('pq,pi,qi->i', D_l, phi * angular_term, phi)
                raw_tau = 0.5 * (term1 + term2)
                tau_final = raw_tau / (4.0 * np.pi * (BOHR_TO_ANGSTROM ** 5))
                
                # Integrate the raw radial charge density over the spherical grid
                integrated_charge = np.trapezoid(4.0 * np.pi * (self.radial_grid ** 2) * rho_final, self.radial_grid)
                
                # Normalize both arrays to a single electron representation
                if integrated_charge > 1e-6:
                    rho_final /= integrated_charge
                    tau_final /= integrated_charge
                
                radial_rho[idx, :] = rho_final
                radial_tau[idx, :] = tau_final
                
        return radial_rho, radial_tau
            
    @property
    def max_occupations(self) -> NDArray:
        """
        1D array containing the maximum allowed electron occupation for each channel.
        If unrestricted, the capacity per spin channel is (2l + 1).
        If restricted, the capacity is 2 * (2l + 1) = (4l + 2).
        """
        if self.unrestricted:
            return 2 * self.angular_momenta + 1.0
        return 4 * self.angular_momenta + 2.0

    def get_valence_dataset(self, paw_species: PAWSpecies) -> "AESpecies":
        """
        Extracts a subset of the dataset containing only valence and virtual states, 
        dropping any underlying core states. 
        
        Accumulation starts from the highest energy occupied states downwards until 
        the requested valence charge value `Z` is reached. All virtual states are preserved.
        """
        Z = paw_species.Z
        # 1. Isolate occupied states
        occupied_indices = np.where(self.reference_occupations > 1e-4)[0]
        
        # 2. Sort occupied states descending by energy (valence down to deep core)
        sorted_occupied = occupied_indices[np.argsort(self.eigenvalues[occupied_indices])[::-1]]
        
        valence_indices = []
        accumulated_charge = 0.0
        
        for idx in sorted_occupied:
            valence_indices.append(idx)
            accumulated_charge += self.reference_occupations[idx]
            if accumulated_charge >= Z - 1e-4:
                break
                
        # 3. Always preserve unphysical/physical stable virtual manifolds
        virtual_indices = np.where(self.reference_occupations <= 1e-4)[0]
        
        # 4. Merge pools and re-sort ascending by energy to maintain proper canonical structure
        keep_indices = sorted(list(valence_indices) + list(virtual_indices), key=lambda idx: self.eigenvalues[idx])
        
        # 5. Build filtered properties
        filtered_density_matrices = [self.density_matrices[i] for i in keep_indices]
        
        return AESpecies(
            name=f"{self.name}_valence",
            element=self.element,
            Z=float(Z),
            basis=self.basis,
            functional=self.functional,
            unrestricted=self.unrestricted,
            primitives=self.primitives,
            radial_grid=self.radial_grid,
            paw_species=paw_species,
            angular_momenta=self.angular_momenta[keep_indices],
            eigenvalues=self.eigenvalues[keep_indices],
            reference_occupations=self.reference_occupations[keep_indices],
            density_matrices=filtered_density_matrices
        )
    
    def get_occupancies(self, min_electrons: float, max_electrons: float) -> NDArray:
        """
        Generates a 1D occupancy array of shape (n_channels,) containing the 
        number of electrons allocated to each channel within the specified 
        electron window [min_electrons, max_electrons] using the Aufbau principle.
        """
        if min_electrons > max_electrons:
            raise ValueError("min_electrons cannot be greater than max_electrons.")
        
        n_channels = len(self.eigenvalues)
        max_occ = self.max_occupations
        occs = np.zeros(n_channels, dtype=np.float64)
        
        prev_count = 0.0
        for idx in range(n_channels):
            current_count = prev_count + max_occ[idx]
            
            # Calculate the exact electron slice overlapping this channel's energy capacity
            overlap = max(0.0, min(current_count, max_electrons) - max(prev_count, min_electrons))
            occs[idx] = overlap
            
            prev_count = current_count
                
        return occs

    def get_radial_charge_density(self, min_electrons: float, max_electrons: float) -> NDArray:
        """
        Computes the total radial charge density profiles across a range of total electron counts.
        
        Returns:
            A 2D numpy array of shape (n_configurations, n_grid) where each row represents 
            the total radial charge density profile for that electron count step.
        """
        # Get electron occupancies of each state
        occupancies = self.get_occupancies(min_electrons, max_electrons)
        # Get weighted sum
        total_rho = np.sum(occupancies * self.radial_rho.T, axis=1)
        
        return total_rho

    def get_radial_kinetic_energy_density(self, min_electrons: float, max_electrons: float) -> NDArray:
        """
        Computes the total radial positive-definite kinetic energy density (KED) profiles 
        across a range of total electron counts.
        
        Returns:
            A 2D numpy array of shape (n_configurations, n_grid) where each row represents 
            the total radial KED profile for that electron count step.
        """
        # Get electron occupancies of each state
        occupancies = self.get_occupancies(min_electrons, max_electrons)
        
        # Get weighted sum
        total_tau = np.sum(occupancies * self.radial_tau.T, axis=1)
        
        return total_tau

    @classmethod
    def from_file(
            cls,
            filename: str | Path,
            paw_species: PAWSpecies = None,
            cutoff_radius: float = 10.0,
            grid_points: int = 2000):
        """
        Parses compressed references generated by generate_references.py, automatically
        merging degenerate spatial orientation manifolds into unique consolidated channels.
        """
        file_path = Path(filename)
        if not file_path.exists():
            raise FileNotFoundError(f"Missing analytical basis binary for element: {file_path}")
            
        data = np.load(file_path)
        metadata = json.loads(str(data["metadata"]))
        
        # Metadata parsing
        element = metadata["element"]
        element_py = Element(element)
        functional = metadata["functional"]
        basis = metadata.get("basis", None)
        Z = float(element_py.Z)
        
        spin_channels = data["spin_channels"]
        unrestricted = int(np.max(spin_channels)) > 0
        
        # Load values
        raw_eigenvalues = data["energies"]
        raw_occupations = data["occupancies"]
        raw_occupations[raw_occupations < 1e-16] = 0.0
        
        packed_d = data["packed_d_matrices"]
        matrix_dims = metadata["matrix_layout_dimensions"]
        primitives = cls._flatten_basis_primitives(metadata["basis_primitives"])
        
        all_d_blocks = [cls._unpack_triu_matrices(packed_d[i], matrix_dims) for i in range(len(raw_eigenvalues))]
        
        raw_states = []
        for i in range(len(raw_eigenvalues)):
            d_map = all_d_blocks[i]
            l_assigned = max(d_map.keys(), key=lambda l: np.max(np.abs(d_map[l])))
            raw_states.append({
                'energy': raw_eigenvalues[i],
                'occupancy': raw_occupations[i],
                'spin': spin_channels[i],
                'l': l_assigned,
                'd_blocks': d_map
            })
            
        # --- Merge Identical Channels (Spherical Degeneracy Grouping) ---
        merged_states = []
        visited = set()
        
        for i in range(len(raw_states)):
            if i in visited:
                continue
                
            group = [raw_states[i]]
            visited.add(i)
            
            for j in range(i + 1, len(raw_states)):
                if j not in visited:
                    if (raw_states[j]['l'] == raw_states[i]['l'] and 
                        raw_states[j]['spin'] == raw_states[i]['spin'] and 
                        abs(raw_states[j]['energy'] - raw_states[i]['energy']) < 1e-4):
                        group.append(raw_states[j])
                        visited.add(j)
            
            avg_energy = sum(g['energy'] for g in group) / len(group)
            total_occupancy = sum(g['occupancy'] for g in group)
            
            merged_d_blocks = {}
            for l_str, dim in matrix_dims.items():
                l_val = int(l_str)
                merged_d_blocks[l_val] = np.zeros((dim, dim), dtype=np.float64)
                
            for g in group:
                for l_val, mat in g['d_blocks'].items():
                    merged_d_blocks[l_val] += mat
                    
            merged_states.append({
                'energy': avg_energy,
                'occupancy': total_occupancy,
                'spin': raw_states[i]['spin'],
                'l': raw_states[i]['l'],
                'd_blocks': merged_d_blocks
            })
            
        merged_states.sort(key=lambda x: x['energy'])
        
        # --- Dynamic Grid Reconstruction Layout ---
        if paw_species is not None:
            paw_grid = paw_species.radial_grid
            if cutoff_radius <= paw_grid[-1]:
                radial_grid = paw_grid.copy()
            else:
                # Track the local grid spacing trends at the outer tail
                diff1 = paw_grid[-1] - paw_grid[-2]
                diff2 = paw_grid[-2] - paw_grid[-3]
                ratio1 = paw_grid[-1] / paw_grid[-2] if paw_grid[-2] != 0.0 else 1.0
                ratio2 = paw_grid[-2] / paw_grid[-3] if paw_grid[-3] != 0.0 else 1.0
                
                extended_points = list(paw_grid)
                current_r = paw_grid[-1]
                
                # Check variance of difference vs ratio to detect exponential/geometric vs linear mesh
                if abs(ratio1 - ratio2) / ratio1 < abs(diff1 - diff2) / diff1:
                    # Logarithmic/Geometric expansion matching VASP POTCAR conventions
                    while current_r < cutoff_radius:
                        current_r *= ratio1
                        extended_points.append(current_r)
                else:
                    # Linear extension layout
                    while current_r < cutoff_radius:
                        current_r += diff1
                        extended_points.append(current_r)
                        
                radial_grid = np.array(extended_points, dtype=np.float64)
        else:
            radial_grid = np.geomspace(1e-10, cutoff_radius, grid_points)
        
        result = cls(
            name=f"{element}_{functional}",
            element=element,
            Z=Z,
            basis=basis,
            functional=functional,
            unrestricted=unrestricted,
            primitives=primitives,
            radial_grid=radial_grid,
            paw_species=None,
            angular_momenta=np.array([m['l'] for m in merged_states], dtype=np.int_),
            eigenvalues=np.array([m['energy'] for m in merged_states], dtype=np.float64),
            reference_occupations=np.array([m['occupancy'] for m in merged_states], dtype=np.float64),
            density_matrices=[m['d_blocks'] for m in merged_states]
        )
        if paw_species is not None:
            return result.get_valence_dataset(paw_species)
        return result

    @staticmethod
    def _unpack_triu_matrices(packed_vector, matrix_dims):
        """Reconstructs full symmetric density matrices from packed upper-triangles."""
        d_blocks = {}
        current_idx = 0
        for l_str in sorted(matrix_dims.keys(), key=int):
            l = int(l_str)
            dim = matrix_dims[l_str]
            size = (dim * (dim + 1)) // 2
            vec = packed_vector[current_idx:current_idx + size]
            current_idx += size
            
            mat = np.zeros((dim, dim))
            iu = np.triu_indices(dim)
            mat[iu] = vec
            mat = mat + mat.T - np.diag(np.diag(mat))
            d_blocks[l] = mat
        return d_blocks

    @staticmethod
    def _flatten_basis_primitives(basis_primitives):
        """Flattens nested primitive structures into flat continuous 1D arrays and offset pointers."""
        contractions_by_l = {}
        for l_str, bas_list in basis_primitives.items():
            l = int(l_str)
            exps_list = []
            coeffs_list = []
            offsets = [0]
            
            for bas_data in bas_list:
                exps = np.array(bas_data["exponents"], dtype=np.float64)
                coeffs_mat = np.array(bas_data["coefficients"], dtype=np.float64)
                nctr = coeffs_mat.shape[1]
                
                for c in range(nctr):
                    exps_list.extend(exps)
                    coeffs_list.extend(coeffs_mat[:, c])
                    offsets.append(offsets[-1] + len(exps))
                    
            contractions_by_l[l] = {
                "exps": np.array(exps_list, dtype=np.float64),
                "coeffs": np.array(coeffs_list, dtype=np.float64),
                "offsets": np.array(offsets, dtype=np.int32)
            }
        return contractions_by_l