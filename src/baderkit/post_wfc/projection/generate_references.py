#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import numpy as np
from pyscf import lib, gto

OUTPUT_PATH = Path.home() / Path("github/baderkit/src/baderkit/post_wfc/all_electron_references")

def compress_checkpoint_to_radial_basis(root_dir, filename_pattern="chkpt.pbe"):
    """
    Loops over folders using pathlib, extracts raw orbital coefficients, determines
    their explicit l and m quantum numbers, and saves the flat linear state vector
    coefficients directly into {element}.npz—eliminating density matrices entirely.
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
            mo_energy = scf_rec['mo_energy']
            mo_coeff = scf_rec['mo_coeff']
            mo_occ = scf_rec['mo_occ']
            
            is_unrestricted = isinstance(mo_energy, tuple) or (isinstance(mo_energy, np.ndarray) and mo_energy.ndim == 2)
            
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

            # Collect and Flatten All Individual States (Alpha and Beta)
            raw_states = []
            if is_unrestricted:
                for ispin, spin_label in enumerate([0, 1]):
                    for iband in range(len(mo_energy[ispin])):
                        raw_states.append({
                            'energy': float(mo_energy[ispin][iband]),
                            'coeff': mo_coeff[ispin][:, iband],
                            'occ': float(mo_occ[ispin][iband]),
                            'spin': spin_label
                        })
            else:
                for iband in range(len(mo_energy)):
                    raw_states.append({
                        'energy': float(mo_energy[iband]),
                        'coeff': mo_coeff[:, iband],
                        'occ': float(mo_occ[iband]),
                        'spin': 0
                    })

            raw_states.sort(key=lambda x: x['energy'])

            # Filter out unphysical virtual states showing instabilities
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
            print(f"[{element}] --- State Vector Compression Profile ---")
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

            # Direct state vector mapping
            for istate, state in enumerate(active_states):
                energies[istate] = state['energy']
                occupancies[istate] = state['occ']
                spin_channels[istate] = state['spin']
                
                coeff = state['coeff']
                
                # Determine the dominant l and m component for this state
                dominant_ao_idx = np.argmax(np.abs(coeff))
                _, state_l, _, state_m = ao_map[dominant_ao_idx]
                
                angular_momenta[istate] = state_l
                magnetic_quantum_numbers[istate] = state_m
                
                # Map the raw coefficients directly onto the flat contraction structure
                state_vector = []
                for l in sorted(l_counts.keys()):
                    mat_dim = l_counts[l]
                    c_p_vector = np.zeros(mat_dim, dtype=np.float64)
                    
                    if l == state_l:
                        for ao_idx, ao_l, p, ao_m in ao_map:
                            if ao_l == state_l and ao_m == state_m:
                                c_p_vector[p] = coeff[ao_idx]
                                
                    state_vector.extend(c_p_vector.tolist())
                
                packed_state_vectors[istate, :] = state_vector

            # Build Header with Baked G-space Transform Coefficients
            basis_primitives = {}
            for bas_id in range(mol.nbas):
                l = mol.bas_angular(bas_id)
                l_str = str(l)
                if l_str not in basis_primitives:
                    basis_primitives[l_str] = []
                
                exps = mol.bas_exp(bas_id)
                coeffs_mat = mol.bas_ctr_coeff(bas_id)
                prim_norms = gto.gto_norm(l, exps)
                normalized_coeffs_mat = coeffs_mat * prim_norms[:, np.newaxis]
                
                g_prefactors = (np.pi / exps) ** 1.5 * (1.0 / (2.0 * exps)) ** l
                g_coeffs_mat = normalized_coeffs_mat * g_prefactors[:, np.newaxis]
                
                basis_primitives[l_str].append({
                    "exponents": exps.tolist(),
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
            }

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