# -*- coding: utf-8 -*-

from pymatgen.core import Structure
import numpy as np
import os

input_poscar = "POSCAR"

# Load structure
struct = Structure.from_file(input_poscar)

# Choose rotation axis ('x', 'y', or 'z')
axis = [0, 0, 1]  # z-axis

# Rotation center at oxygen
center = struct[1].coords

# Loop over rotation angles (0° to 180° exclusive, step 15°)
for angle in range(0, 180, 15):
    rotated_struct = struct.copy()
    rotated_struct.rotate_sites(
        indices=range(len(struct)),
        theta=np.radians(angle),
        axis=axis,
        anchor=center
    )
    
    # Make directory for this rotation
    os.makedirs(f"{angle:03d}", exist_ok=True)
    
    # Write POSCAR
    rotated_struct.to(fmt="poscar", filename=os.path.join(f"{angle:03d}", "POSCAR"))

