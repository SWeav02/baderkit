# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
from baderkit.post_wfc.paw.paw_dataset import PAWSpecies

def parse_vasp_potcar(directory: Path | str) -> dict:
    """
    Parses a VASP POTCAR file, extracting real-space projector functions, 
    partial wavefunctions, the DION coupling strength matrix, and the 
    principal quantum numbers.
    
    Parameters
    ----------
    potcar_path : Path | str
        The path to the POTCAR file containing the PAW datasets.
        
    Returns
    -------
    dict
        A dictionary mapping element symbols directly to their simplified PAWSpecies data blocks.
    """
    potcar_path = Path(directory) / "POTCAR"
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
        psmaxn = None     # Automatically holds the maximum reciprocal grid limit
        qcut_val = None   # Global reciprocal space cutoff QCUT
        
        title_line = name
        if "PAW_" in title_line:
            tokens = title_line.split()
            if len(tokens) >= 2:
                element = tokens[1].split('_')[0]
                
        # Map out the exact sequence indices of the upcoming channel states
        angular_energies = []
        atomic_config = [] # Keeps track of parsed tuples: (n, l, E, occ)
        in_atomic_config = False
        
        for idx, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Context switch to parse reference configurations
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
                    # Switched to another block
                    in_atomic_config = False

            if line_stripped.lower() == "description":
                in_atomic_config = False
                table_idx = idx + 2  # Skip text label row
                while table_idx < len(lines):
                    t_tokens = lines[table_idx].strip().split()
                    if t_tokens and t_tokens[0].isdigit():
                        angular_momenta.append(int(t_tokens[0]))
                        
                        # Parse energy eigenvalue
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
            
            # Scan for the unlabeled line containing: [PSMAXN] [Logical Flag]
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
        
        line_idx = 0
        in_radial_sets = False  
        
        while line_idx < len(lines):
            line_lbl = lines[line_idx].strip().lower()
            
            if line_lbl.startswith("paw radial sets"):
                in_radial_sets = True
                line_idx += 1
                continue
            
            # Extract spatial grid points
            if in_radial_sets and line_lbl.startswith("grid"):
                vals_list = []
                total_count = 0

                while total_count < num_points and (line_idx + 1) < len(lines):
                    line_idx += 1
                    cleaned_str = " ".join(lines[line_idx].split('#')[0].split())
                    if not cleaned_str:
                        continue
                    try:
                        line_vals = np.fromstring(cleaned_str, sep=' ')
                    except ValueError:
                        valid_tokens = []
                        for t in cleaned_str.split():
                            try:
                                float(t)
                                valid_tokens.append(t)
                            except ValueError:
                                break
                        if not valid_tokens:
                            continue
                        line_vals = np.fromstring(" ".join(valid_tokens), sep=' ')
                    
                    vals_list.append(line_vals)
                    total_count += len(line_vals)

                radial_grid = np.concatenate(vals_list)[:num_points]
                
                if len(radial_grid) > 1:
                    dr = np.zeros_like(radial_grid)
                    dr[0] = radial_grid[1] - radial_grid[0]
                    dr[1:-1] = 0.5 * (radial_grid[2:] - radial_grid[:-2])
                    dr[-1] = radial_grid[-1] - radial_grid[-2]
                line_idx += 1
                continue
                
            # Extract RECIPROCAL space projectors bounded strictly by NDATA
            elif line_lbl.startswith("reciprocal space part"):
                vals_list = []
                total_count = 0
                while total_count < ndata and (line_idx + 1) < len(lines):
                    line_idx += 1
                    cleaned_str = " ".join(lines[line_idx].split('#')[0].split())
                    if not cleaned_str:
                        continue
                    if any(t in cleaned_str for t in ["PAW_PBE", "Ca_sv_GW"]):
                        continue
                    try:
                        line_vals = np.fromstring(cleaned_str, sep=' ')
                    except ValueError:
                        valid_tokens = []
                        for t in cleaned_str.split():
                            try:
                                float(t)
                                valid_tokens.append(t)
                            except ValueError:
                                break
                        if not valid_tokens:
                            continue
                        line_vals = np.fromstring(" ".join(valid_tokens), sep=' ')
                    
                    if len(line_vals) > 0:
                        vals_list.append(line_vals)
                        total_count += len(line_vals)
                
                if vals_list:
                    raw_projectors.append(np.concatenate(vals_list)[:ndata])
                line_idx += 1
                continue
                
            # Extract smooth pseudo partial waves (Real Space)
            elif in_radial_sets and line_lbl.startswith("pseudo wavefunction"):
                vals_list = []
                total_count = 0
                while total_count < num_points and (line_idx + 1) < len(lines):
                    line_idx += 1
                    cleaned_str = " ".join(lines[line_idx].split('#')[0].split())
                    if not cleaned_str:
                        continue
                    try:
                        line_vals = np.fromstring(cleaned_str, sep=' ')
                    except ValueError:
                        valid_tokens = []
                        for t in cleaned_str.split():
                            try:
                                float(t)
                                valid_tokens.append(t)
                            except ValueError:
                                break
                        if not valid_tokens:
                            continue
                        line_vals = np.fromstring(" ".join(valid_tokens), sep=' ')
                        
                    vals_list.append(line_vals)
                    total_count += len(line_vals)
                    
                pseudo_partial_waves.append(np.concatenate(vals_list)[:num_points])
                line_idx += 1
                continue
                
            # Extract true oscillatory all-electron partial waves (Real Space)
            elif in_radial_sets and line_lbl.startswith("ae wavefunction"):
                vals_list = []
                total_count = 0
                while total_count < num_points and (line_idx + 1) < len(lines):
                    line_idx += 1
                    cleaned_str = " ".join(lines[line_idx].split('#')[0].split())
                    if not cleaned_str:
                        continue
                    try:
                        line_vals = np.fromstring(cleaned_str, sep=' ')
                    except ValueError:
                        valid_tokens = []
                        for t in cleaned_str.split():
                            try:
                                float(t)
                                valid_tokens.append(t)
                            except ValueError:
                                break
                        if not valid_tokens:
                            continue
                        line_vals = np.fromstring(" ".join(valid_tokens), sep=' ')
                        
                    vals_list.append(line_vals)
                    total_count += len(line_vals)
                    
                all_electron_partial_waves.append(np.concatenate(vals_list)[:num_points])
                line_idx += 1
                continue

            # Extract local coupling strength matrix (DION)
            elif line_lbl == "dion":
                try:
                    dion_size = int(lines[line_idx + 1].strip())
                    dion_rows = []
                    for r_idx in range(dion_size):
                        row_vals = np.fromstring(lines[line_idx + 2 + r_idx].strip(), sep=' ')
                        dion_rows.append(row_vals)
                    dion_matrix_raw = np.array(dion_rows, dtype=np.float64)
                except Exception:
                    pass
                line_idx += 1
                continue
                
            line_idx += 1

        # --- 3. Pass C: Core Alignment and Master Synchronization Validation ---
        n_channels = len(angular_momenta)
        
        # Cross-reference Description channels against Atomic configurations using energy matching
        energy_tol = 1e-2
        reference_counts = []
        principal_quantum_numbers = []
        l_counters = {}
        
        for l_desc, E_desc in zip(angular_momenta, angular_energies):
            matched_n = None
            matched_occ = 0.0
            for n_atom, l_atom, E_atom, occ_atom in atomic_config:
                if l_desc == l_atom and abs(E_desc - E_atom) <= energy_tol:
                    matched_n = n_atom
                    matched_occ = occ_atom
                    break
            
            # Smart fallback if the valence configuration has no direct energy match
            if matched_n is None:
                if l_desc not in l_counters:
                    l_counters[l_desc] = l_desc + 1
                else:
                    l_counters[l_desc] += 1
                matched_n = l_counters[l_desc]
                
            principal_quantum_numbers.append(matched_n)
            reference_counts.append(matched_occ)

        # Pad raw blocks if any fields were omitted or under-parsed
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

        # --- GENERATE UNIFORM LINEAR RECIPROCAL GRID UP TO PSMAXN ---
        q_radial_grid = np.linspace(0, psmaxn if psmaxn is not None else 1.0, ndata if ndata is not None else 1)
        qcut_base = qcut_val if qcut_val is not None else (psmaxn if psmaxn is not None else 15.0)

        # --- Simultaneous (l, m) Expansion Loop ---
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
        
        # Track channel indices mappings for coupling DION expansion
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
            
            # Spherically average the total shell occupancy across the 2l + 1 sub-channels
            sub_channel_occ = total_count / (2 * l + 1)
            
            # Expand into 2l + 1 magnetic sub-channels
            for m in range(-l, l + 1):
                expanded_channel_map.append((idx, l, m))
                
                expanded_ang_moms.append(l)
                expanded_ms.append(m)
                expanded_principal_quantum_numbers.append(n_p)
                expanded_cutoff_radii.append(rc)
                expanded_q_cutoff_radii.append(qcut_base)
                expanded_eigenvalues.append(energy)
                expanded_ref_counts.append(sub_channel_occ)
                
                # Each magnetic sub-channel shares the same 1D radial profile
                expanded_ae_partial_waves.append(ae_wave)
                expanded_ps_partial_waves.append(ps_wave)
                expanded_raw_projectors.append(proj_1d)
        
        # --- Expand the DION matrix coupling matching channels ---
        num_expanded = len(expanded_ang_moms)
        expanded_dion = np.zeros((num_expanded, num_expanded), dtype=np.float64)
        
        for A in range(num_expanded):
            idx_A, l_A, m_A = expanded_channel_map[A]
            for B in range(num_expanded):
                idx_B, l_B, m_B = expanded_channel_map[B]
                # DION matrix terms couple channels only when they share the same l and m
                if l_A == l_B and m_A == m_B:
                    expanded_dion[A, B] = dion_matrix_raw[idx_A, idx_B]
                    
        # Check if grids are logarithmic or linear
        if radial_grid is not None and len(radial_grid) > 2:
            # If the spacing between points is not uniform, it's a logarithmic/non-linear grid
            real_is_log = not np.allclose(np.diff(radial_grid), radial_grid[1] - radial_grid[0], rtol=1e-4)
        else:
            real_is_log = False

        # q_radial_grid is explicitly generated above using np.linspace, so it is always linear
        q_is_log = False

        # Instantiate the PAWSpecies object with fully expanded NumPy arrays
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
            
            # FIX: Divide by radial grid to keep consistent metrics
            all_electron_partial_waves=np.array(expanded_ae_partial_waves)/radial_grid,
            pseudo_partial_waves=np.array(expanded_ps_partial_waves)/radial_grid,
            
            q_projectors=np.array(expanded_raw_projectors),
            eigenvalues=np.array(expanded_eigenvalues, dtype=float),
            reference_occupations=np.array(expanded_ref_counts, dtype=float)
        )

        master_dataset[element] = species_obj
        
    return master_dataset

