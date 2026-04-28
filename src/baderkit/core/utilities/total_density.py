# -*- coding: utf-8 -*-
from pymatgen.core.periodic_table import Element
import numpy as np
from baderkit.core import Grid
from numba import njit, prange

@njit(cache=True)
def interp1d_numba(x, y, xq):
    """
    Linear interpolation of y(x) at query points xq.

    Parameters
    ----------
    x : 1D array (sorted ascending)
    y : 1D array (same length as x)
    xq : scalar or 1D array of query points

    Returns
    -------
    Interpolated values at xq
    """
    n = len(x)

    # Extrapolation (left)
    if xq <= x[0]:
        yq = y[0]


    # Extrapolation (right)
    if xq >= x[-1]:
        yq = y[-1]

    # Binary search for interval
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (hi + lo) // 2
        if x[mid] <= xq:
            lo = mid
        else:
            hi = mid

    # Linear interpolation
    x0, x1 = x[lo], x[lo + 1]
    y0, y1 = y[lo], y[lo + 1]

    t = (xq - x0) / (x1 - x0)
    yq = y0 + t * (y1 - y0)

    return yq

@njit(cache=True, parallel=True)
def sum_total_charge(
    site_frac_coords,
    site_cores,
    site_z_cores,
    matrix,
    nx, ny, nz,
    dist_cutoff=5.0,
):

    shape = np.array((nx, ny, nz), dtype=np.float64)
    core_grid = np.zeros((nx, ny, nz), dtype=np.float64)
    weight_grid = np.zeros((nx, ny, nz), dtype=np.float64)

    n_atoms = len(site_frac_coords)

    # --- Compute cell lengths (approx, from matrix rows) ---
    ax = np.sqrt(matrix[0,0]**2 + matrix[0,1]**2 + matrix[0,2]**2)
    by = np.sqrt(matrix[1,0]**2 + matrix[1,1]**2 + matrix[1,2]**2)
    cz = np.sqrt(matrix[2,0]**2 + matrix[2,1]**2 + matrix[2,2]**2)

    # --- Determine how many periodic images are needed ---
    max_tx = int(np.ceil(dist_cutoff / ax))
    max_ty = int(np.ceil(dist_cutoff / by))
    max_tz = int(np.ceil(dist_cutoff / cz))

    for a in prange(n_atoms):

        atom_frac = site_frac_coords[a]
        Z_core = site_z_cores[a]
        r, rho = site_cores[a]

        # --- Loop over relevant periodic images ---
        for tx in range(-max_tx, max_tx + 1):
            for ty in range(-max_ty, max_ty + 1):
                for tz in range(-max_tz, max_tz + 1):

                    # image position in fractional coords
                    img_fx = atom_frac[0] + tx
                    img_fy = atom_frac[1] + ty
                    img_fz = atom_frac[2] + tz

                    # --- Convert cutoff to fractional bounding box ---
                    # conservative bounding box
                    fx_min = img_fx - dist_cutoff / ax
                    fx_max = img_fx + dist_cutoff / ax
                    fy_min = img_fy - dist_cutoff / by
                    fy_max = img_fy + dist_cutoff / by
                    fz_min = img_fz - dist_cutoff / cz
                    fz_max = img_fz + dist_cutoff / cz

                    # convert to grid indices
                    i_min = max(0, int(np.floor(fx_min * nx)))
                    i_max = min(nx-1, int(np.ceil(fx_max * nx)))
                    j_min = max(0, int(np.floor(fy_min * ny)))
                    j_max = min(ny-1, int(np.ceil(fy_max * ny)))
                    k_min = max(0, int(np.floor(fz_min * nz)))
                    k_max = min(nz-1, int(np.ceil(fz_max * nz)))

                    # --- Loop only over nearby grid points ---
                    for i in range(i_min, i_max + 1):
                        fx = i / shape[0]

                        dx = fx - img_fx

                        for j in range(j_min, j_max + 1):
                            fy = j / shape[1]

                            dy = fy - img_fy

                            for k in range(k_min, k_max + 1):
                                fz = k / shape[2]

                                dz = fz - img_fz

                                # --- Convert to Cartesian ---
                                cx = dx*matrix[0,0] + dy*matrix[1,0] + dz*matrix[2,0]
                                cy = dx*matrix[0,1] + dy*matrix[1,1] + dz*matrix[2,1]
                                cz_ = dx*matrix[0,2] + dy*matrix[1,2] + dz*matrix[2,2]

                                dist = np.sqrt(cx*cx + cy*cy + cz_*cz_)

                                if dist > dist_cutoff:
                                    continue

                                val = interp1d_numba(r, rho, dist)

                                core_grid[i, j, k] += val * Z_core
                                weight_grid[i, j, k] += Z_core

    # --- Normalize per grid point ---
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                if weight_grid[i, j, k] > 0.0:
                    core_grid[i, j, k] /= weight_grid[i, j, k]

    # --- Global normalization ---
    n_grid = nx * ny * nz
    total_Z = site_z_cores.sum()

    core_sum = core_grid.sum()
    if core_sum > 0:
        core_grid *= (n_grid * total_Z / core_sum)

    return core_grid

# @njit(cache=True, parallel=True)
# def sum_total_charge(
#     site_frac_coords,
#     site_cores,
#     site_z_cores,
#     matrix,
#     nx, ny, nz,
#     dist_cutoff=5.0,
#         ):

#     shape = np.array((nx,ny,nz), dtype=np.float64)
#     n_grid = nx*ny*nz
#     core_grid = np.zeros((nx,ny,nz))
    
#     transforms = np.empty((27,3), dtype=np.int64)
#     idx=0
#     for i in (-1,0,1):
#         for j in (-1,0,1):
#             for k in (-1,0,1):
#                 transforms[idx] = (i,j,k)
#                 idx+=1
    
#     for i in prange(nx):
#         for j in range(ny):
#             for k in range(nz):
#                 frac_coord = np.array((i,j,k),dtype=np.float64) / shape
#                 total_Z = 0
#                 for idx, (r, rho) in enumerate(site_cores):
#                     atom_frac = site_frac_coords[idx]
#                     Z_core = site_z_cores[idx]
#                     # check each transform
#                     for trans in transforms:
#                         trans_frac = atom_frac + trans
#                         diff = frac_coord - trans_frac
#                         # convert to cart
#                         diff = diff @ matrix
#                         dist = np.linalg.norm(diff)
#                         if dist > dist_cutoff:
#                             continue
                        
#                         # get contribution from this atom
#                         val = interp1d_numba(r, rho, dist)
                
#                         # normalize to required number of core electrons
#                         core_grid[i,j,k] += val*Z_core
#                         total_Z += Z_core
#                 if total_Z:
#                     core_grid[i,j,k] /= total_Z
#     # normalize
#     core_grid = core_grid * (n_grid * site_z_cores.sum() / core_grid.sum())
    
#     return core_grid

def create_total_chgcar(chgcar_path, potcar_path, output_path):
    """
    Adds PAW core electron densities to a CHGCAR and writes the total density.

    Parameters
    ----------
    chgcar_path : str
        Path to the input CHGCAR file.
    potcar_path : str
        Path to the POTCAR file containing core densities.
    output_path : str
        Path for the output CHGCAR file with total density.
    """
    # -----------------------------
    # Load CHGCAR
    # -----------------------------
    chgcar = Grid.from_vasp(chgcar_path)
    structure = chgcar.structure
    grid_valence = np.array(chgcar.total, dtype=float)
    nx, ny, nz = grid_valence.shape

    # -----------------------------
    # Parse POTCAR core densities
    # -----------------------------
    def parse_paw_core_density(potcar_file):
        core_data, Z_dict = {}, {}
        with open(potcar_file, "r") as f:
            all_text = f.read()
            potcar_lines = [i.splitlines() for i in all_text.split("End of Dataset")]

        for lines in potcar_lines:
            if not lines[0].strip():
                continue
            
            r = rho = None
            element = lines[0].split()[1].split("_")[0].strip()
            ZVAL = float(lines[1].strip())
            Z = Element(element).Z
            
            i = 2
            while i < len(lines):
                line = lines[i].strip()

                if "grid" == line:
                    i += 1
                    r = []
                    while i < len(lines):
                        try:
                            r.extend([float(x) for x in lines[i].strip().split()])
                        except ValueError:
                            break
                        i += 1
                    r = np.array(r)
                if "core charge-density" in line and "(partial)" not in line:
                    i += 1
                    rho = []
                    while i < len(lines):
                        try:
                            rho.extend([float(x) for x in lines[i].strip().split()])
                        except ValueError:
                            break
                        i += 1
                    rho = np.array(rho)
                    if np.all(rho==0):
                        continue
                    core_data[element] = (r, rho)
                    if Z is not None and ZVAL is not None:
                        Z_dict[element] = Z - ZVAL
                    break
                i += 1
        return core_data, Z_dict

    paw_core, Z_dict = parse_paw_core_density(potcar_path)
    if not paw_core:
        print("No core density blocks found in POTCAR")
        chgcar.write_vasp(output_path)
        return
    
    site_cores = [paw_core.get(i.specie.symbol, (np.empty((0)), np.empty((0)))) for i in structure]
    site_z_cores = np.array([Z_dict.get(i.specie.symbol, 0.0) for i in structure])
    site_frac_coords = structure.frac_coords
    matrix = chgcar.matrix

    # -----------------------------
    # Build core density grid
    # -----------------------------
    core_grid = sum_total_charge(
        site_frac_coords,
        site_cores,
        site_z_cores,
        matrix,
        nx, ny, nz,
            )
    # -----------------------------
    # Combine valence and core, write output
    # -----------------------------
    chgcar.total += core_grid
    chgcar.write_vasp(output_path)

    print("Done.")
    print(f"Total electrons: {chgcar.total.sum()/(nx*ny*nz):.6f}")
    print(f"Output written to {output_path}")