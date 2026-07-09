#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import json
from pathlib import Path
import numpy as np
from pyscf import lib
from scipy.special import gamma

# Physical Constants
HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.5291772109
BOHR_SQ = BOHR_TO_ANG ** 2

OUTPUT_PATH = Path.home() / Path("github/baderkit/src/baderkit/post_wfc/projection/bases/dyall")

def compress_checkpoint_to_radial_basis(root_dir, filename_pattern="chkpt.pbe"):
    """
    Loops over folders, extracts raw orbital coefficients, groups them into
    spherically symmetric subshells (n, l), orthogonalizes them via an 
    energy-ordered Gram-Schmidt sweep over the contracted AO overlap matrix,
    and saves flat linear state vector coefficients along with principal quantum numbers.
    """
    root_path = Path(root_dir)
    chkpt_files = list(root_path.rglob(filename_pattern))
    
    if not chkpt_files:
        print(f"No checkpoint files matching '{filename_pattern}' found in {root_path}.")
        return

    print(f"Found {len(chkpt_files)} checkpoint files to process.\n")

    for chkpt_path in chkpt_files:
        try:
            mol = lib.chkfile.load_mol(str(chkpt_path))
            scf_rec = lib.chkfile.load(str(chkpt_path), 'scf')
            
            element = mol.atom_symbol(0)
            
            # Extract raw quantities from the checkpoint file
            mo_energy_raw = scf_rec['mo_energy']
            mo_coeff = scf_rec['mo_coeff']
            mo_occ = scf_rec['mo_occ']
            
            is_unrestricted = isinstance(mo_energy_raw, tuple) or (isinstance(mo_energy_raw, np.ndarray) and mo_energy_raw.ndim == 2)
            
            # Build the Atomic Orbital (AO) Index Map to Radial Contractions
            ao_map = []
            l_counts = {}
            ao_idx = 0
            
            for bas_id in range(mol.nbas):
                l = mol.bas_angular(bas_id)
                nctr = mol.bas_nctr(bas_id)
                start_p = l_counts.get(l, 0)
                
                for c in range(nctr):
                    p = start_p + c
                    for m in range(2 * l + 1):
                        ao_map.append((ao_idx, l, p, m))
                        ao_idx += 1
                l_counts[l] = start_p + nctr

            # Collect all individual atomic states (Alpha and Beta)
            raw_states = []
            if is_unrestricted:
                for ispin, spin_label in enumerate([0, 1]):
                    for iband in range(len(mo_energy_raw[ispin])):
                        raw_states.append({
                            'energy': float(mo_energy_raw[ispin][iband] * HARTREE_TO_EV),
                            'coeff': mo_coeff[ispin][:, iband],
                            'occ': float(mo_occ[ispin][iband]),
                            'spin': spin_label
                        })
            else:
                for iband in range(len(mo_energy_raw)):
                    raw_states.append({
                        'energy': float(mo_energy_raw[iband] * HARTREE_TO_EV),
                        'coeff': mo_coeff[:, iband],
                        'occ': float(mo_occ[iband]),
                        'spin': 0
                    })

            # FIX: Removed the unphysical `max_c > 3.0` shortcut constraint entirely.
            # True high-energy virtual states (like the 5s orbital) are safely kept,
            # while mathematical overcompleteness is handled by our Pass 3 Gram-Schmidt rank engine.
            active_states = []
            for state in raw_states:
                active_states.append(state)

            # Determine the dominant l channel for each active state
            for state in active_states:
                l_weights = {}
                for ao_idx, ao_l, p, ao_m in ao_map:
                    l_weights[ao_l] = l_weights.get(ao_l, 0.0) + state['coeff'][ao_idx]**2
                state['dominant_l'] = max(l_weights, key=l_weights.get)

            unique_subshells = []
            remaining_states = list(active_states)

            # Cluster degenerate states into unified structural shells based on energy and l symmetry
            while remaining_states:
                base_state = remaining_states.pop(0)
                base_energy = base_state['energy']
                base_l = base_state['dominant_l']
                base_spin = base_state['spin']
                
                subshell_group = [base_state]
                i = 0
                while i < len(remaining_states):
                    check_state = remaining_states[i]
                    if (check_state['dominant_l'] == base_l and 
                        check_state['spin'] == base_spin and 
                        abs(check_state['energy'] - base_energy) < 1e-3):
                        subshell_group.append(remaining_states.pop(i))
                    else:
                        i += 1
                        
                avg_occ = float(np.mean([s['occ'] for s in subshell_group]))
                unique_subshells.append({
                    'energy': base_energy,
                    'l': base_l,
                    'spin': base_spin,
                    'occ': avg_occ,
                    'states': subshell_group,
                    'c_p_pure': None  # To be filled by SVD
                })

            # Sort subshells strictly by ascending energy
            unique_subshells.sort(key=lambda x: x['energy'])

            # =================================================================
            # FIX UPGRADE: ENERGETIC ORDER ASSIGNMENT PRE-PRUNING
            # =================================================================
            # Map principal quantum numbers (n) sequentially using the complete sorted unpruned sequence.
            # Standard quantum mechanical spectroscopic convention starts shell index at n = l + 1
            n_counters = {}  # Maps (spin, l) -> current principal quantum number n
            for subshell in unique_subshells:
                s_chan = subshell['spin']
                l_chan = subshell['l']
                s_l_key = (s_chan, l_chan)
                if s_l_key not in n_counters:
                    n_counters[s_l_key] = l_chan + 1
                else:
                    n_counters[s_l_key] += 1
                subshell['n'] = n_counters[s_l_key]

            # --- PASS 1: Extract Base Radial Shell Profiles via SVD ---
            for subshell in unique_subshells:
                state_l = subshell['l']
                n_contract = l_counts[state_l]
                
                feature_vectors = []
                for state in subshell['states']:
                    coeff = state['coeff']
                    matrix_l = np.zeros((n_contract, 2 * state_l + 1))
                    for ao_idx, ao_l, p, ao_m in ao_map:
                        if ao_l == state_l:
                            matrix_l[p, ao_m] = coeff[ao_idx]
                    for m in range(2 * state_l + 1):
                        feature_vectors.append(matrix_l[:, m])
                        
                feature_matrix = np.column_stack(feature_vectors)
                U, S, Vt = np.linalg.svd(feature_matrix, full_matrices=False)
                c_p_pure = U[:, 0]
                
                if np.sum(c_p_pure) < 0:
                    c_p_pure = -c_p_pure
                subshell['c_p_pure'] = c_p_pure

            # --- PASS 2: Exact Contracted AO Overlap Matrix Generation ---
            contracted_defs = {}  # l -> list of {'exps': arr, 'coeffs': arr}
            for bas_id in range(mol.nbas):
                l = mol.bas_angular(bas_id)
                if l not in contracted_defs:
                    contracted_defs[l] = []
                exps_bohr = mol.bas_exp(bas_id)
                exps_ang = exps_bohr / BOHR_SQ
                coeffs_mat = mol.bas_ctr_coeff(bas_id)
                gto_norms = np.array([mol.gto_norm(l, alpha) for alpha in exps_bohr], dtype=np.float64)
                normalized_coeffs_mat = coeffs_mat * gto_norms[:, np.newaxis] * (BOHR_TO_ANG ** -(l + 1.5))
                
                nctr = normalized_coeffs_mat.shape[1]
                for c in range(nctr):
                    contracted_defs[l].append({
                        'exps': exps_ang,
                        'coeffs': normalized_coeffs_mat[:, c]
                    })

            S_contracted_blocks = {}
            for l, def_list in contracted_defs.items():
                n_contract = len(def_list)
                S_mat = np.zeros((n_contract, n_contract), dtype=np.float64)
                gamma_factor = gamma(l + 1.5)
                
                for p in range(n_contract):
                    def_p = def_list[p]
                    for q in range(n_contract):
                        def_q = def_list[q]
                        s_val = 0.0
                        for k in range(len(def_p['exps'])):
                            for j in range(len(def_q['exps'])):
                                A = def_p['exps'][k] + def_q['exps'][j]
                                integral = 0.5 * (A ** -(l + 1.5)) * gamma_factor
                                s_val += def_p['coeffs'][k] * def_q['coeffs'][j] * integral
                        S_mat[p, q] = s_val
                S_contracted_blocks[l] = S_mat

            # --- PASS 3: Energy-Ordered Radial Gram-Schmidt Sweep ---
            final_retained_subshells = []
            
            for l_channel in sorted(l_counts.keys()):
                S_metric = S_contracted_blocks[l_channel]
                l_subshells = [s for s in unique_subshells if s['l'] == l_channel]
                
                orthogonalized_vectors = []
                for idx, subshell in enumerate(l_subshells):
                    v = subshell['c_p_pure'].copy()
                    
                    # Project out components belonging to all lower-energy core/valence shells
                    for u in orthogonalized_vectors:
                        proj = float(u.T @ S_metric @ v) / float(u.T @ S_metric @ u)
                        v -= proj * u
                        
                    # Re-normalize the state back to exact unit norm within the contracted space
                    norm = np.sqrt(float(v.T @ S_metric @ v))
                    
                    # Strict threshold: if the remaining norm is too small, the state is redundant
                    if norm > 1e-5:
                        v /= norm
                        orthogonalized_vectors.append(v)
                        subshell['c_p_pure'] = v  
                        final_retained_subshells.append(subshell)
                    else:
                        logging.info(f"Skipping redundant virtual subshell: Element {element}, l={l_channel}, Energy={subshell['energy']:.2f} eV (Linear Singularity)")

            # Overwrite unique_subshells with only the linearly independent states
            unique_subshells = final_retained_subshells
            unique_subshells.sort(key=lambda x: x['energy'])
            
            # --- PASS 4: Linear State Array Packing and Matrix Layout ---
            num_subshells = len(unique_subshells)
            print(f"[{element}] --- Shell-Driven Subshell Compression Profile ---")
            print(f"  * Total Raw Input States:      {len(raw_states)}")
            # print(f"  * Noise Virtuals Pruned:       {dropped_count}")
            print(f"  * Orthonormalized Subshells:   {num_subshells}")
            print(f"-------------------------------------------------------")
            
            l_symbols = {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g'}
            for ishell, subshell in enumerate(unique_subshells):
                l_sym = l_symbols.get(subshell['l'], f"l={subshell['l']}")
                print(f"  Subshell {ishell:2d}: {subshell['n']}{l_sym} | Energy: {subshell['energy']:10.4f} eV | Occ: {subshell['occ']:.4f} | Spin: {subshell['spin']}")
            print(f"-------------------------------------------------------")

            flat_vector_size = sum(l_counts.values())
            energies = np.zeros(num_subshells, dtype=np.float64)
            occupancies = np.zeros(num_subshells, dtype=np.float64)
            spin_channels = np.zeros(num_subshells, dtype=np.int8)
            magnetic_quantum_numbers = np.zeros(num_subshells, dtype=np.int_)
            angular_momenta = np.zeros(num_subshells, dtype=np.int_)
            principal_quantum_numbers = np.zeros(num_subshells, dtype=np.int_)
            packed_state_vectors = np.zeros((num_subshells, flat_vector_size), dtype=np.float64)

            for ishell, subshell in enumerate(unique_subshells):
                state_l = subshell['l']
                energies[ishell] = subshell['energy']
                occupancies[ishell] = subshell['occ']
                spin_channels[ishell] = subshell['spin']
                angular_momenta[ishell] = state_l
                magnetic_quantum_numbers[ishell] = 0
                principal_quantum_numbers[ishell] = subshell['n']
                
                state_vector = []
                for l in sorted(l_counts.keys()):
                    mat_dim = l_counts[l]
                    c_p_vector = np.zeros(mat_dim, dtype=np.float64)
                    if l == state_l:
                        c_p_vector[:] = subshell['c_p_pure']
                    state_vector.extend(c_p_vector.tolist())
                    
                packed_state_vectors[ishell, :] = state_vector

            # Build Header with Primitives scaled to Angstrom units
            basis_primitives = {}
            for bas_id in range(mol.nbas):
                l = mol.bas_angular(bas_id)
                l_str = str(l)
                if l_str not in basis_primitives:
                    basis_primitives[l_str] = []
                
                exps_bohr = mol.bas_exp(bas_id)
                exps_ang = exps_bohr / BOHR_SQ
                
                coeffs_mat = mol.bas_ctr_coeff(bas_id)
                gto_norms = np.array([mol.gto_norm(l, alpha) for alpha in exps_bohr], dtype=np.float64)
                
                absolute_coeffs_bohr = coeffs_mat * gto_norms[:, np.newaxis]
                normalized_coeffs_mat = absolute_coeffs_bohr * (BOHR_TO_ANG ** -(l + 1.5))
                
                g_prefactors = (np.pi / exps_ang) ** 1.5 * (1.0 / (2.0 * exps_ang)) ** l
                g_coeffs_mat = normalized_coeffs_mat * g_prefactors[:, np.newaxis]
                
                basis_primitives[l_str].append({
                    "exponents": exps_ang.tolist(),
                    "coefficients": normalized_coeffs_mat.tolist(),
                    "g_coefficients": g_coeffs_mat.tolist()
                })
                
            metadata = {
                "file_format": "SphericalRadialWavefunction_LinearVector_NPZ",
                "element": element,
                "functional": "pbe",
                "matrix_layout_dimensions": {str(l): n for l, n in l_counts.items()},
                "basis_primitives": basis_primitives,
                "basis": mol.basis,
                "units": {"energy": "eV", "length": "Angstrom"}
            }

            if not OUTPUT_PATH.exists():
                OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

            output_path = OUTPUT_PATH / f"{element}.npz"
            np.savez_compressed(
                output_path,
                energies=energies,
                occupancies=occupancies,
                spin_channels=spin_channels,
                angular_momenta=angular_momenta,
                magnetic_quantum_numbers=magnetic_quantum_numbers,
                principal_quantum_numbers=principal_quantum_numbers,
                packed_state_vectors=packed_state_vectors,
                metadata=json.dumps(metadata, indent=2)
            )
            print(f"Successfully generated distinct orthonormal footprint: {output_path}\n")

        except Exception as e:
            import traceback
            print(f"Failed to process {chkpt_path}. Error: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    compress_checkpoint_to_radial_basis(Path.cwd())