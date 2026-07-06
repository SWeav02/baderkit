# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
from .paw_dataset import PAWSpecies

def parse_vasp_potcar(potcar_path: Path | str) -> dict:
    """
    Parses a VASP POTCAR file, extracting the reciprocal space projector functions
    and dynamically attaching the uniform linear wavevector grid generated from the 
    unlabeled internal VASP PSMAXN parameters for continuous reciprocal spline interpolation.
    
    Parameters
    ----------
    potcar_path : Path | str
        The path to the POTCAR file containing the PAW datasets.
        
    Returns
    -------
    dict
        A dictionary mapping element symbols directly to their simplified PAWSpecies data blocks.
    """
    potcar_path = Path(potcar_path)
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
        
        title_line = name
        if "PAW_" in title_line:
            tokens = title_line.split()
            if len(tokens) >= 2:
                element = tokens[1].split('_')[0]
                
        # Map out the exact sequence indices of the upcoming channel states
        angular_energies = []
        atomic_config = [] # Keeps track of parsed tuples: (l, E, occ)
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
                        l_atom = int(t_tokens[1])
                        E_atom = float(t_tokens[3])
                        occ_atom = float(t_tokens[4])
                        atomic_config.append((l_atom, E_atom, occ_atom))
                    except ValueError:
                        pass
                elif t_tokens and not t_tokens[0].isdigit() and "description" not in line_stripped.lower():
                    pass

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
                
            # Extract smooth pseudo partial waves
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
                
            # Extract true oscillatory all-electron partial waves
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
                
            line_idx += 1

        # --- 3. Pass C: Core Alignment and Master Synchronization Validation ---
        n_channels = len(angular_momenta)
        
        # Cross-reference Description channels against Atomic configurations using energy matching
        energy_tol = 1e-2
        reference_counts = []
        for l_desc, E_desc in zip(angular_momenta, angular_energies):
            matched_occ = 0.0
            for l_atom, E_atom, occ_atom in atomic_config:
                if l_desc == l_atom and abs(E_desc - E_atom) <= energy_tol:
                    matched_occ = occ_atom
                    break
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
            
        # --- GENERATE UNIFORM LINEAR RECIPROCAL GRID UP TO PSMAXN ---
        q_linear_grid = np.linspace(0, psmaxn if psmaxn is not None else 1.0, ndata if ndata is not None else 1)
        
        # --- Simultaneous (l, m) Expansion Loop ---
        expanded_ang_moms = []
        expanded_ms = []
        expanded_cutoff_radii = []
        expanded_ae_partial_waves = []
        expanded_ps_partial_waves = []
        expanded_raw_projectors = []
        expanded_eigenvalues = []
        expanded_ref_counts = []
        
        for idx in range(n_channels):
            l = angular_momenta[idx]
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
                expanded_ang_moms.append(l)
                expanded_ms.append(m)
                expanded_cutoff_radii.append(rc)
                expanded_eigenvalues.append(energy)
                expanded_ref_counts.append(sub_channel_occ)
                
                # Each magnetic sub-channel shares the same 1D radial profile
                expanded_ae_partial_waves.append(ae_wave)
                expanded_ps_partial_waves.append(ps_wave)
                expanded_raw_projectors.append(proj_1d)
        
        # Instantiate the PAWSpecies object with fully expanded NumPy arrays
        species_obj = PAWSpecies(
            source="vasp",
            name=name,
            element=element,
            Z=z_val,
            radial_grid=radial_grid if radial_grid is not None else np.empty(0),
            angular_momenta=np.array(expanded_ang_moms, dtype=int),
            magnetic_nums=np.array(expanded_ms, dtype=int),
            all_electron_partial_waves=np.array(expanded_ae_partial_waves),
            pseudo_partial_waves=np.array(expanded_ps_partial_waves),
            cutoff_radii=np.array(expanded_cutoff_radii, dtype=float),
            max_cutoff_radius=rcut_global if rcut_global is not None else 0.0,
            q_linear_grid=q_linear_grid,
            reciprocal_projectors=np.array(expanded_raw_projectors),
            eigenvalues=np.array(expanded_eigenvalues, dtype=float),
            reference_occupations=np.array(expanded_ref_counts, dtype=float)
        )

        master_dataset[element] = species_obj
        
    return master_dataset