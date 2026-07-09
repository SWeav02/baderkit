#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import numpy as np

def analyze_all_electron_reference(npz_path: str | Path):
    path = Path(npz_path)
        
    print("=" * 80)
    print(f"     BADERKIT ALL-ELECTRON REFERENCE VALIDATION SUITE: {path.name}     ")
    print("=" * 80)
    
    # 1. Load raw archives
    data = np.load(path, allow_pickle=True)
    energies = data["energies"]
    occupancies = data["occupancies"]
    spin_channels = data["spin_channels"]
    angular_momenta = data["angular_momenta"]
    mag_nums = data["magnetic_quantum_numbers"]
    state_vectors = data["packed_state_vectors"]
    metadata = json.loads(str(data["metadata"]))
    
    element = metadata.get("element", "Unknown")
    basis_name = metadata.get("basis", "Unknown")
    units = metadata.get("units", {})
    basis_primitives = metadata["basis_primitives"]
    
    print(f"Chemical Element: {element}")
    print(f"Source Basis Set: {basis_name}")
    print(f"Configured Units: Length={units.get('length')}, Energy={units.get('energy')}")
    print(f"Total States in Footprint: {len(energies)}")
    
    # 2. Parse and flatten primitive contractions exactly like all_electron_dataset.py
    primitives = {}
    l_slices = {}
    current_slice_idx = 0
    
    # Reconstruct the internal map definitions grouped by angular momentum 'l'
    for l_str, bas_list in basis_primitives.items():
        l = int(l_str)
        exps_list = []
        coeffs_list = []
        offsets = [0]
        
        start_idx = current_slice_idx
        for bas_data in bas_list:
            # Load exponents and coefficients array shapes
            exps = np.array(bas_data["exponents"], dtype=np.float64)
            coeffs_mat = np.array(bas_data["coefficients"], dtype=np.float64)
            nctr = coeffs_mat.shape[1]
            
            for c in range(nctr):
                exps_list.extend(exps)
                coeffs_list.extend(coeffs_mat[:, c])
                offsets.append(offsets[-1] + len(exps))
                current_slice_idx += 1
                
        end_idx = current_slice_idx
        l_slices[l] = (start_idx, end_idx)
        
        primitives[l] = {
            "exps": np.array(exps_list, dtype=np.float64),
            "coeffs": np.array(coeffs_list, dtype=np.float64),
            "offsets": offsets
        }

    # 3. Establish a pristine, ultra-fine log-radial mesh grid for integration checks
    # Covers from 1e-5 Angstrom out to a safe 10.0 Angstrom core cutoff boundary
    r_grid = np.geomspace(1e-5, 10.0, 2000)
    num_grid = len(r_grid)
    
    print("\nReconstructing continuous radial spaces and calculating mutual overlap matrices...")
    print(f"  -> Evaluation Mesh: {num_grid} logarithmic coordinates spanning [{r_grid[0]:.1e} to {r_grid[-1]:.1f}] Å")
    
    # Initialize container matrices to evaluate physical configurations
    num_states = len(energies)
    wavefunctions = np.zeros((num_states, num_grid))
    
    for idx in range(num_states):
        l = angular_momenta[idx]
        if l not in primitives:
            continue
            
        c_data = primitives[l]
        exps = c_data["exps"]
        coeffs = c_data["coeffs"]
        offsets = c_data["offsets"]
        
        # Slicing state coefficients using the global l-channel map boundaries
        start_l, end_l = l_slices[l]
        c_state = state_vectors[idx, start_l:end_l]
        dim = len(c_state)
        
        phi = np.zeros((dim, num_grid))
        r_pow_l = r_grid ** l
        
        for p in range(dim):
            start = offsets[p]
            end = offsets[p+1]
            sum_0 = np.zeros(num_grid)
            
            for k in range(start, end):
                sum_0 += coeffs[k] * np.exp(-exps[k] * (r_grid ** 2))
            phi[p, :] = r_pow_l * sum_0
            
        # Linearly combine the evaluated primitive values into the full radial state
        wavefunctions[idx, :] = np.dot(c_state, phi)

    # 4. Compute and evaluate the overlap matrix elements grouped by (Spin, l) channels
    # S_ij = \int_0^\infty R_i(r) * R_j(r) * r^2 * dr
    unique_spins = np.unique(spin_channels)
    unique_ls = np.unique(angular_momenta)
    
    print("\n" + "-"*80)
    print("                      DETAILED CHANNEL-BY-CHANNEL PROFILE                    ")
    print("-"*80)
    print(f"{'State ID':<10} {'Spin':<6} {'l':<4} {'m':<4} {'Energy (eV)':<12} {'Occupancy':<10} {'Self-Overlap':<14}")
    
    unhealthy_states_count = 0
    
    for idx in range(num_states):
        r_sq_element = (wavefunctions[idx, :] ** 2) * (r_grid ** 2)
        self_overlap = np.trapezoid(r_sq_element, r_grid)
        
        status_flag = ""
        if not (0.99 <= self_overlap <= 1.01):
            status_flag = " [⚠️ INF-ERR]"
            unhealthy_states_count += 1
            
        print(f"{idx:<10} {spin_channels[idx]:<6} {angular_momenta[idx]:<4} {mag_nums[idx]:<4} "
              f"{energies[idx]:>11.4f} {occupancies[idx]:>10.4f} {self_overlap:>14.6f}{status_flag}")

    print("\n" + "-"*80)
    print("                     MUTUAL ORTHOGONALITY & SUB-BLOCK HEALTH                  ")
    print("-"*80)
    
    for s in unique_spins:
        for l in unique_ls:
            # Isolate channel blocks sharing identical symmetry channels
            channel_indices = np.where((spin_channels == s) & (angular_momenta == l))[0]
            if len(channel_indices) == 0:
                continue
                
            wfc_block = wavefunctions[channel_indices, :]
            num_block = len(channel_indices)
            
            # Compute mutual overlaps across the sub-block configuration matrix
            S_block = np.zeros((num_block, num_block))
            for i in range(num_block):
                for j in range(num_block):
                    integrand = wfc_block[i, :] * wfc_block[j, :] * (r_grid ** 2)
                    S_block[i, j] = np.trapezoid(integrand, r_grid)
                    
            eigenvals = np.linalg.eigvalsh(S_block)
            min_ev, max_ev = np.min(eigenvals), np.max(eigenvals)
            condition_num = max_ev / max(min_ev, 1e-15)
            
            # Cross-orthogonality check: average off-diagonal absolute error
            if num_block > 1:
                off_diag_mask = ~np.eye(num_block, dtype=bool)
                avg_off_diagonal = np.mean(np.abs(S_block[off_diag_mask]))
            else:
                avg_off_diagonal = 0.0
                
            print(f"Channel Sub-Block (Spin={s}, l={l}): Size={num_block}x{num_block}")
            print(f"  * Eigenvalue Range   : [{min_ev:.4e} to {max_ev:.4f}]")
            print(f"  * Condition Number  : {condition_num:.4e}")
            print(f"  * Avg Off-Diag Error : {avg_off_diagonal:.4e}")
            
            if condition_num > 1e6:
                print("    ⚠️  WARNING: High condition number indicates near-singular linear dependency features.")
            else:
                print("    ✅ Channel matrix is well-conditioned and healthy.")

    print("=" * 80)
    if unhealthy_states_count == 0:
        print("🎉 SUCCESS: All electron references are perfectly normalized and stable!")
    else:
        print(f"❌ CRITICAL FAILURE: Detected {unhealthy_states_count} un-normalized or corrupted orbital fields.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    analyze_all_electron_reference("/home/Sam/github/baderkit/src/baderkit/post_wfc/projection/bases/dyall/Ca.npz")