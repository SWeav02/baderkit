#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import numpy as np
from pyscf import lib

# Physical Constants
HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.5291772109
BOHR_SQ = BOHR_TO_ANG ** 2

OUTPUT_PATH = Path.home() / Path("github/baderkit/src/baderkit/post_wfc/projection/bases/dyall")

def compress_checkpoint_to_radial_basis(root_dir, filename_pattern="chkpt.pbe"):
    """
    Loops over folders, extracts raw orbital coefficients, transforms units
    to Materials Science standards (eV, Angstrom), filters unphysical virtuals,
    and saves the flat linear state vector coefficients.
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

            # Collect and Flatten All Individual States (Alpha and Beta) using raw coefficients
            raw_states = []
            if is_unrestricted:
                for ispin, spin_label in enumerate([0, 1]):
                    for iband in range(len(mo_energy_raw[ispin])):
                        raw_states.append({
                            'energy': float(mo_energy_raw[ispin][iband] * HARTREE_TO_EV), # Convert energy to eV
                            'coeff': mo_coeff[ispin][:, iband],
                            'occ': float(mo_occ[ispin][iband]),
                            'spin': spin_label
                        })
            else:
                for iband in range(len(mo_energy_raw)):
                    raw_states.append({
                        'energy': float(mo_energy_raw[iband] * HARTREE_TO_EV), # Convert energy to eV
                        'coeff': mo_coeff[:, iband],
                        'occ': float(mo_occ[iband]),
                        'spin': 0
                    })

            # Sort states by their newly converted energies
            raw_states.sort(key=lambda x: x['energy'])

            # Filter out unphysical virtual states using the original threshold criteria
            initial_count = len(raw_states)
            active_states = []
            dropped_count = 0
            
            for state in raw_states:
                if state['occ'] < 1e-4:
                    max_c = np.max(np.abs(state['coeff']))
                    if max_c > 3.0:
                        dropped_count += 1
                        continue 
                active_states.append(state)
                
            num_states = len(active_states)
            print(f"[{element}] --- State Vector Compression Profile (Units: eV, Angstrom) ---")
            print(f"  * Total Raw Input States:      {initial_count}")
            print(f"  * Linear Dep States Filtered:  {dropped_count}")
            print(f"  * Total Compressed States:     {num_states}")
            print(f"-------------------------------------------------------")

            # Allocate flat state vector arrays
            flat_vector_size = sum(l_counts.values())
            energies = np.zeros(num_states, dtype=np.float64)
            occupancies = np.zeros(num_states, dtype=np.float64)
            spin_channels = np.zeros(num_states, dtype=np.int8)
            magnetic_quantum_numbers = np.zeros(num_states, dtype=np.int_)
            angular_momenta = np.zeros(num_states, dtype=np.int_)
            packed_state_vectors = np.zeros((num_states, flat_vector_size), dtype=np.float64)

            # =================================================================
            # Direct state vector mapping via Singular Value Decomposition
            # =================================================================
            for istate, state in enumerate(active_states):
                energies[istate] = state['energy']
                occupancies[istate] = state['occ']
                spin_channels[istate] = state['spin']
                
                coeff = state['coeff']
                
                # 1. Map out the absolute orbital weight across all available angular channels
                l_weights = {}
                for ao_idx, ao_l, p, ao_m in ao_map:
                    l_weights[ao_l] = l_weights.get(ao_l, 0.0) + coeff[ao_idx]**2
                state_l = max(l_weights, key=l_weights.get)
                
                # 2. Extract the contraction coefficient matrix for the dominant l subspace
                n_contract = l_counts[state_l]
                matrix_l = np.zeros((n_contract, 2 * state_l + 1))
                
                for ao_idx, ao_l, p, ao_m in ao_map:
                    if ao_l == state_l:
                        matrix_l[p, ao_m] = coeff[ao_idx]
                        
                # 3. Use SVD to compress all angular channels back into a single clean radial vector
                U, S, Vt = np.linalg.svd(matrix_l, full_matrices=False)
                c_p_pure = U[:, 0] * S[0]
                
                # 4. Identify the dominant magnetic orientation to preserve metadata mapping
                m_weights = np.sum(matrix_l**2, axis=0)
                state_m = np.argmax(m_weights)
                
                # Align signs with the original dominant component column to prevent phase inversion
                if np.dot(matrix_l[:, state_m], c_p_pure) < 0:
                    c_p_pure = -c_p_pure
                    
                angular_momenta[istate] = state_l
                magnetic_quantum_numbers[istate] = state_m
                
                # 5. Distribute the complete reconstructed radial array into the flat state vector layout
                state_vector = []
                for l in sorted(l_counts.keys()):
                    mat_dim = l_counts[l]
                    c_p_vector = np.zeros(mat_dim, dtype=np.float64)
                    
                    if l == state_l:
                        # Feed the full, clean radial profile into the target state row
                        c_p_vector[:] = c_p_pure
                        
                    state_vector.extend(c_p_vector.tolist())
                    
                packed_state_vectors[istate, :] = state_vector

            # =================================================================
            # Build Header with Primitives scaled to Angstrom units (FIXED)
            # =================================================================
            basis_primitives = {}
            for bas_id in range(mol.nbas):
                l = mol.bas_angular(bas_id)
                l_str = str(l)
                if l_str not in basis_primitives:
                    basis_primitives[l_str] = []
                
                # Exponents scale inversely with the square of the length unit
                exps_bohr = mol.bas_exp(bas_id)
                exps_ang = exps_bohr / BOHR_SQ
                
                # Fetch raw contraction coefficients (coefficients of normalized primitives)
                coeffs_mat = mol.bas_ctr_coeff(bas_id)
                
                # FIXED: Compute the primitive GTO normalization factors in Bohr units
                gto_norms = np.array([mol.gto_norm(l, alpha) for alpha in exps_bohr], dtype=np.float64)
                
                # Reconstruct absolute coefficients of unnormalized primitives in Bohr space
                absolute_coeffs_bohr = coeffs_mat * gto_norms[:, np.newaxis]
                
                # Apply the coordinate transformation scaling factor to normalize cleanly in Angstroms
                normalized_coeffs_mat = absolute_coeffs_bohr * (BOHR_TO_ANG ** -(l + 1.5))
                
                # Compute G-space analytic prefactors for the Fourier Transform of the primitives
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
                packed_state_vectors=packed_state_vectors,
                metadata=json.dumps(metadata, indent=2)
            )
            print(f"Successfully generated footprint: {output_path}\n")

        except Exception as e:
            print(f"Failed to process {chkpt_path}. Error: {str(e)}")

if __name__ == "__main__":
    compress_checkpoint_to_radial_basis(Path.cwd())