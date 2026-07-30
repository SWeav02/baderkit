# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
from baderkit.post_wfc.paw.paw_dataset import PAWSpecies

def _parse_float_array(lines: list[str], start_idx: int, target_count: int | None = None) -> tuple[np.ndarray, int]:
    """Reads contiguous floating point numbers from lines starting at start_idx.
    
    Stops when target_count numbers are collected or when a text section header is encountered.
    Returns the concatenated numpy array and the index of the next unparsed line.
    """
    vals = []
    curr = start_idx
    total_count = 0
    
    while curr < len(lines):
        raw_line = lines[curr].split('#')[0].split('!')[0].strip()
        if not raw_line:
            curr += 1
            continue
            
        tokens = raw_line.split()
        # Check if line contains non-numeric text tokens (header line)
        has_alpha_words = any(
            any(c.isalpha() and c.lower() not in ['e', 'd'] for c in tok)
            for tok in tokens
        )
        if has_alpha_words:
            break
            
        try:
            line_vals = np.fromstring(" ".join(tokens), sep=' ')
            if len(line_vals) == 0:
                break
            vals.append(line_vals)
            total_count += len(line_vals)
            curr += 1
            
            if target_count is not None and total_count >= target_count:
                break
        except Exception:
            break
            
    if vals:
        arr = np.concatenate(vals)
        if target_count is not None:
            arr = arr[:target_count]
        return arr, curr
    else:
        return np.empty(0, dtype=np.float64), curr


def parse_vasp_potcar(directory: Path | str, verbose: bool = False) -> dict:
    """Parses a VASP POTCAR file, extracting real-space projector functions, 
    partial wavefunctions, the DION coupling strength matrix, principal quantum numbers,
    and frozen core density data with core quantum numbers.
    """
    potcar_path = Path(directory)
    if potcar_path.is_dir():
        potcar_path = potcar_path / "POTCAR"

    if not potcar_path.is_file():
        raise FileNotFoundError(f"Target POTCAR file not found at: {potcar_path}")
        
    master_dataset = {}
    
    with open(potcar_path, 'r') as f:
        content = f.read()
        
    # Split concatenated multi-element entries by VASP's official block delimiter
    raw_species_blocks = [b.strip() for b in content.split("End of Dataset") if b.strip()]

    for block in raw_species_blocks:
        lines = block.split('\n')
        if not lines or not lines[0].strip():
            continue
            
        # --- 1. Pass A: Scrape Metadata, Grid Boundaries, and NDATA/PSMAXN ---
        name = lines[0].strip()
        z_val = float(lines[1].strip())
        
        element = "Unknown"
        rcut_global = None
        ndata = None
        num_points = 0
        angular_momenta = []
        cutoff_radii = []
        psmaxn = None     # Maximum reciprocal grid limit
        qcut_val = None   # Global reciprocal space cutoff QCUT
        
        title_line = name
        if "PAW_" in title_line:
            tokens = title_line.split()
            if len(tokens) >= 2:
                element = tokens[1].split('_')[0]
                
        angular_energies = []
        atomic_config = [] # Keeps track of parsed tuples: (n, l, E, occ)
        in_atomic_config = False
        
        for idx, line in enumerate(lines):
            line_stripped = line.strip()
            
            if "atomic configuration" in line_stripped.lower():
                in_atomic_config = True
                continue
                
            if in_atomic_config:
                t_tokens = line_stripped.split()
                if len(t_tokens) == 5:
                    try:
                        n_atom = int(t_tokens[0])
                        l_atom = int(t_tokens[1])
                        occ_atom = float(t_tokens[3])
                        E_atom = float(t_tokens[4])
                        atomic_config.append((n_atom, l_atom, E_atom, occ_atom))
                    except ValueError:
                        pass
                elif t_tokens and not t_tokens[0].isdigit() and "description" not in line_stripped.lower():
                    in_atomic_config = False

            if line_stripped.lower() == "description":
                in_atomic_config = False
                table_idx = idx + 2  # Skip text label row
                while table_idx < len(lines):
                    t_tokens = lines[table_idx].strip().split()
                    if t_tokens and t_tokens[0].isdigit():
                        angular_momenta.append(int(t_tokens[0]))
                        
                        try:
                            angular_energies.append(float(t_tokens[1]))
                        except (ValueError, IndexError):
                            angular_energies.append(0.0)
                            
                        if len(t_tokens) >= 4:
                            try:
                                cutoff_radii.append(float(t_tokens[3]))
                            except ValueError:
                                cutoff_radii.append(rcut_global)
                        else:
                            cutoff_radii.append(rcut_global)
                        table_idx += 1
                    else:
                        break
            elif "RCORE" in line_stripped:
                rcut_global = float(line_stripped.split('=')[1].split()[0])
            elif "QCUT" in line_stripped:
                parts = line_stripped.replace('=', ' ').replace(';', ' ').split()
                try:
                    qcut_idx = parts.index("QCUT")
                    qcut_val = float(parts[qcut_idx + 1])
                except (ValueError, IndexError):
                    pass
            elif "NDATA" in line_stripped:
                clean_line = line_stripped.replace('=', ' ').replace(';', ' ')
                tokens = clean_line.split()
                if len(tokens) >= 2:
                    try:
                        ndata = int(tokens[1])
                    except ValueError:
                        pass
            elif line_stripped.lower().startswith("paw radial sets"):
                next_tokens = lines[idx + 1].strip().split()
                if next_tokens and next_tokens[0].isdigit():
                    num_points = int(next_tokens[0])
            
            clean_line = line_stripped.split('!')[0].split('#')[0].strip()
            tokens = clean_line.split()
            if len(tokens) == 2 and tokens[1] in ["T", "F", ".TRUE.", ".FALSE."]:
                try:
                    val = float(tokens[0])
                    if val > 0.0:
                        psmaxn = val
                except ValueError:
                    pass

        # --- 2. Pass B: Targeted Numerical Array Collection ---
        radial_grid = None
        raw_projectors = []
        pseudo_partial_waves = []
        all_electron_partial_waves = []
        dion_matrix_raw = None
        
        ae_core_charge_density = None
        ps_core_charge_density = None
        ae_core_kinetic_density = None
        ps_core_kinetic_density = None
        
        line_idx = 0
        in_radial_sets = False  
        
        while line_idx < len(lines):
            line_lbl = lines[line_idx].strip().lower()
            norm_lbl = line_lbl.replace('-', ' ')
            
            if line_lbl.startswith("paw radial sets"):
                in_radial_sets = True
                line_idx += 1
                continue
            
            if in_radial_sets and line_lbl.startswith("grid"):
                radial_grid, line_idx = _parse_float_array(lines, line_idx + 1, target_count=num_points)
                continue

            # AE Core Charge Density ("core charge-density")
            elif in_radial_sets and "core charge" in norm_lbl and "pseudized" not in norm_lbl:
                ae_core_charge_density, line_idx = _parse_float_array(lines, line_idx + 1, target_count=num_points)
                continue

            # PS Core Charge Density ("core charge-density (pseudized)")
            elif in_radial_sets and "core charge" in norm_lbl and "pseudized" in norm_lbl:
                ps_core_charge_density, line_idx = _parse_float_array(lines, line_idx + 1, target_count=num_points)
                continue

            # AE Core Kinetic Energy Density ("kinetic energy-density")
            elif in_radial_sets and "kinetic" in norm_lbl and "pseudized" not in norm_lbl:
                ae_core_kinetic_density, line_idx = _parse_float_array(lines, line_idx + 1, target_count=num_points)
                continue

            # PS Core Kinetic Energy Density ("mkinetic energy-density pseudized")
            elif in_radial_sets and "kinetic" in norm_lbl and "pseudized" in norm_lbl:
                ps_core_kinetic_density, line_idx = _parse_float_array(lines, line_idx + 1, target_count=num_points)
                continue

            elif line_lbl.startswith("reciprocal space part"):
                proj, line_idx = _parse_float_array(lines, line_idx + 1, target_count=ndata)
                raw_projectors.append(proj)
                continue
                
            elif in_radial_sets and line_lbl.startswith("pseudo wavefunction"):
                wave, line_idx = _parse_float_array(lines, line_idx + 1, target_count=num_points)
                pseudo_partial_waves.append(wave)
                continue
                
            elif in_radial_sets and line_lbl.startswith("ae wavefunction"):
                wave, line_idx = _parse_float_array(lines, line_idx + 1, target_count=num_points)
                all_electron_partial_waves.append(wave)
                continue

            elif line_lbl == "dion":
                try:
                    dion_size = int(lines[line_idx + 1].strip())
                    dion_rows = []
                    for r_idx in range(dion_size):
                        row_vals = np.fromstring(lines[line_idx + 2 + r_idx].strip(), sep=' ')
                        dion_rows.append(row_vals)
                    dion_matrix_raw = np.array(dion_rows, dtype=np.float64)
                    line_idx += 2 + dion_size
                    continue
                except Exception:
                    pass
                line_idx += 1
                continue
                
            line_idx += 1

        # --- 3. Pass C: Core Alignment and Master Synchronization Validation ---
        n_channels = len(angular_momenta)
        reference_counts = []
        principal_quantum_numbers = []
        l_counters = {}
        matched_atomic_indices = set()
        
        for l_desc, E_desc in zip(angular_momenta, angular_energies):
            matched_n = None
            matched_occ = 0.0
            best_idx = None
            min_e_diff = float('inf')
            
            for idx_cfg, (n_atom, l_atom, E_atom, occ_atom) in enumerate(atomic_config):
                if l_desc == l_atom:
                    e_diff = abs(E_desc - E_atom)
                    if e_diff < min_e_diff:
                        min_e_diff = e_diff
                        best_idx = idx_cfg
            
            if best_idx is not None and min_e_diff <= 5.0:
                matched_n = atomic_config[best_idx][0]
                matched_occ = atomic_config[best_idx][3]
                matched_atomic_indices.add(best_idx)
            
            if matched_n is None:
                if l_desc not in l_counters:
                    l_counters[l_desc] = l_desc + 1
                else:
                    l_counters[l_desc] += 1
                matched_n = l_counters[l_desc]
                
            principal_quantum_numbers.append(matched_n)
            reference_counts.append(matched_occ)

        core_n = []
        core_l = []
        core_energies = []
        core_occupations = []
        
        for idx_cfg, (n_atom, l_atom, E_atom, occ_atom) in enumerate(atomic_config):
            if idx_cfg not in matched_atomic_indices:
                core_n.append(n_atom)
                core_l.append(l_atom)
                core_energies.append(E_atom)
                core_occupations.append(occ_atom)

        while len(all_electron_partial_waves) < n_channels:
            all_electron_partial_waves.append(np.zeros(num_points))
        while len(pseudo_partial_waves) < n_channels:
            pseudo_partial_waves.append(np.zeros(num_points))
        while len(raw_projectors) < n_channels:
            raw_projectors.append(np.zeros(ndata))
        while len(cutoff_radii) < n_channels:
            cutoff_radii.append(rcut_global)
            
        if dion_matrix_raw is None or dion_matrix_raw.shape != (n_channels, n_channels):
            dion_matrix_raw = np.zeros((n_channels, n_channels), dtype=np.float64)

        q_radial_grid = np.linspace(0, psmaxn if psmaxn is not None else 1.0, ndata if ndata is not None else 1)
        qcut_base = qcut_val if qcut_val is not None else (psmaxn if psmaxn is not None else 15.0)

        expanded_ang_moms = []
        expanded_ms = []
        expanded_principal_quantum_numbers = []
        expanded_cutoff_radii = []
        expanded_q_cutoff_radii = []
        expanded_ae_partial_waves = []
        expanded_ps_partial_waves = []
        expanded_raw_projectors = []
        expanded_eigenvalues = []
        expanded_ref_counts = []
        expanded_channel_map = []
        
        for idx in range(n_channels):
            l = angular_momenta[idx]
            n_p = principal_quantum_numbers[idx]
            rc = cutoff_radii[idx]
            ae_wave = all_electron_partial_waves[idx]
            ps_wave = pseudo_partial_waves[idx]
            proj_1d = raw_projectors[idx]
            energy = angular_energies[idx] if idx < len(angular_energies) else 0.0
            total_count = reference_counts[idx] if idx < len(reference_counts) else 0.0
            sub_channel_occ = total_count / (2 * l + 1)
            
            for m in range(-l, l + 1):
                expanded_channel_map.append((idx, l, m))
                expanded_ang_moms.append(l)
                expanded_ms.append(m)
                expanded_principal_quantum_numbers.append(n_p)
                expanded_cutoff_radii.append(rc)
                expanded_q_cutoff_radii.append(qcut_base)
                expanded_eigenvalues.append(energy)
                expanded_ref_counts.append(sub_channel_occ)
                expanded_ae_partial_waves.append(ae_wave)
                expanded_ps_partial_waves.append(ps_wave)
                expanded_raw_projectors.append(proj_1d)
        
        num_expanded = len(expanded_ang_moms)
        expanded_dion = np.zeros((num_expanded, num_expanded), dtype=np.float64)
        
        for A in range(num_expanded):
            idx_A, l_A, m_A = expanded_channel_map[A]
            for B in range(num_expanded):
                idx_B, l_B, m_B = expanded_channel_map[B]
                if l_A == l_B and m_A == m_B:
                    expanded_dion[A, B] = dion_matrix_raw[idx_A, idx_B]
                    
        if radial_grid is not None and len(radial_grid) > 2:
            real_is_log = not np.allclose(np.diff(radial_grid), radial_grid[1] - radial_grid[0], rtol=1e-4)
        else:
            real_is_log = False

        q_is_log = False
        r_safe = np.where(radial_grid > 0, radial_grid, 1e-12)
        ae_waves = np.array(expanded_ae_partial_waves) / r_safe
        ps_waves = np.array(expanded_ps_partial_waves) / r_safe

        species_obj = PAWSpecies(
            source="vasp",
            name=name,
            element=element,
            Z=z_val,
            radial_grid=radial_grid,
            q_radial_grid=q_radial_grid,
            real_is_log=real_is_log,
            q_is_log=q_is_log,
            paw_cutoffs=np.array(expanded_cutoff_radii, dtype=float),
            q_paw_cutoffs=np.array(expanded_q_cutoff_radii, dtype=float),
            max_paw_cutoff=rcut_global if rcut_global is not None else 0.0,
            principal_quantum_numbers=np.array(expanded_principal_quantum_numbers, dtype=int),
            angular_momenta=np.array(expanded_ang_moms, dtype=int),
            magnetic_quantum_numbers=np.array(expanded_ms, dtype=int),
            all_electron_partial_waves=ae_waves,
            pseudo_partial_waves=ps_waves,
            q_projectors=np.array(expanded_raw_projectors),
            eigenvalues=np.array(expanded_eigenvalues, dtype=float),
            reference_occupations=np.array(expanded_ref_counts, dtype=float),
            
            # --- CORE INFORMATION ---
            core_charge_density=ae_core_charge_density,
            ps_core_charge_density=ps_core_charge_density,
            core_kinetic_density=ae_core_kinetic_density,
            ps_core_kinetic_density=ps_core_kinetic_density,
            core_principal_quantum_numbers=np.array(core_n, dtype=int),
            core_angular_momenta=np.array(core_l, dtype=int),
            core_eigenvalues=np.array(core_energies, dtype=float),
            core_occupations=np.array(core_occupations, dtype=float),
        )

        if verbose:
            print(f"[POTCAR Parser Diagnostic - {element}]")
            print(f"  AE Core Charge:  {ae_core_charge_density is not None} (size: {len(ae_core_charge_density) if ae_core_charge_density is not None else 0})")
            print(f"  PS Core Charge:  {ps_core_charge_density is not None} (size: {len(ps_core_charge_density) if ps_core_charge_density is not None else 0})")
            print(f"  AE Core Kinetic: {ae_core_kinetic_density is not None} (size: {len(ae_core_kinetic_density) if ae_core_kinetic_density is not None else 0})")
            print(f"  PS Core Kinetic: {ps_core_kinetic_density is not None} (size: {len(ps_core_kinetic_density) if ps_core_kinetic_density is not None else 0})")
            print(f"  Core States Parsed: {len(core_n)} (n: {core_n}, l: {core_l})")

        master_dataset[element] = species_obj
        
    return master_dataset