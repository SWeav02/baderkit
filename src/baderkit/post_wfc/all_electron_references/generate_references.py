#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import numpy as np
from pyscf import lib, gto

OUTPUT_PATH = Path.home() / Path("github/baderkit/src/baderkit/post_wfc/all_electron_references")

def compress_checkpoint_to_radial_basis(root_dir, filename_pattern="chkpt.r2scan"):
    """
    Loops over folders using pathlib, extracts spin-polarized orbital coefficients, 
    enforces spherical symmetry by averaging degenerate shell occupations, filters out
    unphysical virtual states with linear dependency explosions, and packs 
    the analytical upper-triangle Reduced Radial Density Matrices into {element}.npz.
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
            
            # 2. Build the Atomic Orbital (AO) Index Map to Radial Contractions
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

            same_lm_pairs = {l: [] for l in l_counts}
            for i_ao, l_i, p_i, m_i in ao_map:
                for j_ao, l_j, p_j, m_j in ao_map:
                    if l_i == l_j and m_i == m_j:
                        same_lm_pairs[l_i].append((i_ao, j_ao, p_i, p_j))

            # 3. Collect and Flatten All Individual States (Alpha and Beta)
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

            # 4. Enforce Spherical Symmetry via Degeneracy Smearing Redistribution
            smeared_states = []
            i = 0
            while i < len(raw_states):
                current_spin = raw_states[i]['spin']
                current_energy = raw_states[i]['energy']
                
                group = [raw_states[i]]
                j = i + 1
                while j < len(raw_states) and raw_states[j]['spin'] == current_spin and abs(raw_states[j]['energy'] - current_energy) < 1e-4:
                    group.append(raw_states[j])
                    j += 1
                
                avg_occupancy = sum(g['occ'] for g in group) / len(group)
                
                for g in group:
                    smeared_states.append({
                        'energy': g['energy'],
                        'coeff': g['coeff'],
                        'occ': avg_occupancy,
                        'spin': g['spin']
                    })
                i = j

            # INTERCEPT HERE: Drop virtual states showing linear dependency instabilities
            initial_count = len(smeared_states)
            filtered_states = []
            dropped_count = 0
            
            for state in smeared_states:
                # Screen unoccupied manifold states
                if state['occ'] < 1e-4:
                    max_c = np.max(np.abs(state['coeff']))
                    if max_c > 3.0:
                        dropped_count += 1
                        continue # Exclude this unphysical ghost state
                filtered_states.append(state)
                
            smeared_states = filtered_states
            remaining_total = len(smeared_states)
            remaining_virtuals = sum(1 for s in smeared_states if s['occ'] < 1e-4)
            remaining_occupied = remaining_total - remaining_virtuals
            
            # Print detailed sanity-check breakdown of basis function counts
            print(f"[{element}] --- Basis Manifold Reduction Profile ---")
            print(f"  * Total Raw Input States:      {initial_count}")
            print(f"  * Linear Dep States Filtered:  {dropped_count}")
            print(f"  * Total Retained Pool States:  {remaining_total}")
            print(f"    - Stable Occupied Baseline:  {remaining_occupied}")
            print(f"    - Stable Virtual Continuum:  {remaining_virtuals}")
            print(f"-------------------------------------------------------")

            flat_d_size = sum((n * (n + 1)) // 2 for n in l_counts.values())
            
            num_states = len(smeared_states)
            energies = np.zeros(num_states, dtype=np.float64)
            occupancies = np.zeros(num_states, dtype=np.float64)
            spin_channels = np.zeros(num_states, dtype=np.int8)
            packed_d_matrices = np.zeros((num_states, flat_d_size), dtype=np.float32)

            # 5. Extract and Pack Reduced Radial Density Matrices Analytical Upper Triangles
            for istate, state in enumerate(smeared_states):
                energies[istate] = state['energy']
                occupancies[istate] = state['occ']
                spin_channels[istate] = state['spin']
                
                coeff = state['coeff']
                state_vector = []
                
                for l in sorted(l_counts.keys()):
                    mat_dim = l_counts[l]
                    D_l = np.zeros((mat_dim, mat_dim), dtype=float)
                    
                    for i_ao, j_ao, p_i, p_j in same_lm_pairs[l]:
                        D_l[p_i, p_j] += coeff[i_ao] * coeff[j_ao]
                    
                    iu = np.triu_indices(mat_dim)
                    state_vector.extend(D_l[iu].tolist())
                
                packed_d_matrices[istate, :] = state_vector

            # 6. Build Basis Set Primitive Header with Pre-Applied PySCF Normalization
            basis_primitives = {}
            for bas_id in range(mol.nbas):
                l = mol.bas_angular(bas_id)
                l_str = str(l)
                if l_str not in basis_primitives:
                    basis_primitives[l_str] = []
                
                exps = mol.bas_exp(bas_id)
                coeffs_mat = mol.bas_ctr_coeff(bas_id)
                
                # Fetch PySCF's exact analytical primitive normalization constants
                prim_norms = gto.gto_norm(l, exps)
                
                # Bake the primitive normalization directly into the contraction weights
                normalized_coeffs_mat = coeffs_mat * prim_norms[:, np.newaxis]
                
                basis_primitives[l_str].append({
                    "exponents": exps.tolist(),
                    "coefficients": normalized_coeffs_mat.tolist()
                })

            metadata = {
                "file_format": "SphericalRadialWavefunction_DMatrix_NPZ",
                "element": element,
                "functional": "r2scan",
                "matrix_layout_dimensions": {str(l): n for l, n in l_counts.items()},
                "basis_primitives": basis_primitives
            }

            output_path = OUTPUT_PATH / f"{element}.npz"
            np.savez_compressed(
                output_path,
                energies=energies,
                occupancies=occupancies,
                spin_channels=spin_channels,
                packed_d_matrices=packed_d_matrices,
                metadata=json.dumps(metadata, indent=2)
            )
            print(f"Successfully generated compact footprint: {output_path}\n")

        except Exception as e:
            print(f"Failed to process {chkpt_path}. Error: {str(e)}")

if __name__ == "__main__":
    compress_checkpoint_to_radial_basis(Path.cwd())