import numpy as np
from scipy.interpolate import interp1d
from baderkit.core import Grid
import re

# -----------------------------
# User inputs
# -----------------------------
CHGCAR_FILE = "CHGCAR"
POTCAR_FILE = "POTCAR"
OUTPUT_FILE = "CHGCAR_total"

# -----------------------------
# Load CHGCAR
# -----------------------------
chgcar = Grid.from_vasp(CHGCAR_FILE)
structure = chgcar.structure
grid_valence = np.array(chgcar.total, dtype=float)
nx, ny, nz = grid_valence.shape
ngrid = nx * ny * nz
lattice = structure.lattice

# Cartesian grid
x = np.linspace(0, 1, nx, endpoint=False)
y = np.linspace(0, 1, ny, endpoint=False)
z = np.linspace(0, 1, nz, endpoint=False)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
frac_grid = np.stack([X, Y, Z], axis=-1)
cart_grid = lattice.get_cartesian_coords(frac_grid.reshape(-1, 3)).reshape(nx, ny, nz, 3)

# -----------------------------
# Parse POTCAR core densities safely
# -----------------------------
def parse_paw_core_density(potcar_file):
    core_data = {}
    Z_dict = {}     # atomic number
    ZVAL_dict = {}  # valence electrons

    # read file
    with open(potcar_file, "r") as f:
        all_text = f.read()
        # split to different POTCARS
        potcar_lines = [i.splitlines() for i in all_text.split("End of Dataset")]

    for lines in potcar_lines:
        i = 0
        element = None
        Z = None
        ZVAL = None
        r = None
        rho = None
        while i < len(lines):
            line = lines[i].strip()
    
            # Element name
            if line.startswith("VRHFIN"):
                element = line.split("=")[1].split()[0]
    
            # Atomic number and valence from POMASS/ZVAL
            if "POMASS" in line:
                m = re.search(r"ZVAL\s*=\s*([\d\.]+)", line)
                if m:
                    ZVAL = float(m.group(1))
                if element is not None:
                    try:
                        from pymatgen.core.periodic_table import Element as pmgElement
                        Z = pmgElement(element).Z
                    except:
                        Z = int(ZVAL + 10)  # fallback
    
                    
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
    
    
            # Core charge-density (main, not partial)
            if "core charge-density" in line and "(partial)" not in line:
                i += 1
                rho = []
                while i < len(lines):
                    try:
                        rho.extend([float(x) for x in lines[i].strip().split()])
                    except ValueError:
                        break
                    i += 1
                rho= np.array(rho)
                
                # normalize rho to correct Z
                Z_core = Z-ZVAL
                total_check = np.trapezoid(rho * r**2, r)
                rho *= nx*ny*nz*Z_core/total_check
    
                core_data[element] = (r, rho)
                if Z is not None and ZVAL is not None:
                    Z_dict[element] = Z
                    ZVAL_dict[element] = ZVAL
                break
    
            i += 1

    return core_data, Z_dict, ZVAL_dict

paw_core, Z_dict, ZVAL_dict = parse_paw_core_density(POTCAR_FILE)
if not paw_core:
    raise RuntimeError("No core density blocks found in POTCAR!")

# -----------------------------
# Interpolation
# -----------------------------
interp_core = {}
for el, (r_arr, rho_arr) in paw_core.items():
    interp_core[el] = interp1d(r_arr, rho_arr, bounds_error=False, fill_value=0.0)

# -----------------------------
# Build core density grid
# -----------------------------
core_grid = np.zeros_like(grid_valence)
for site in structure:
    el = site.specie.symbol
    if el not in interp_core:
        continue

    atom_pos = site.coords
    diff = cart_grid - atom_pos
    frac_diff = lattice.get_fractional_coords(diff)
    frac_diff -= np.round(frac_diff)  # minimum image
    diff = lattice.get_cartesian_coords(frac_diff)
    r = np.linalg.norm(diff, axis=-1)

    rho_vals = interp_core[el](r.flatten()).reshape(nx, ny, nz)
    core_grid += rho_vals

# -----------------------------
# Normalize to correct total electrons
# -----------------------------
expected_core_electrons = 0
for site in structure:
    el = site.specie.symbol
    if el in paw_core and el in Z_dict and el in ZVAL_dict:
        r_arr, rho_arr = paw_core[el]
        expected_core_electrons += Z_dict[el] - ZVAL_dict[el]

# actual from 3D grid
actual_core_electrons = core_grid.sum()
core_grid *= expected_core_electrons*(nx*ny*nz) / actual_core_electrons

# -----------------------------
# Combine and write
# -----------------------------
total_density = grid_valence + core_grid
chgcar.total = total_density
chgcar.write_vasp(OUTPUT_FILE)

print("Done.")
print(f"Expected core electrons: {expected_core_electrons:.6f}")
print(f"Core integrated on grid: {core_grid.sum()/(nx*ny*nz):.6f}")
print(f"Output written to {OUTPUT_FILE}")