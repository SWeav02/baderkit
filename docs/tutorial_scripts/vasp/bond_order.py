# -*- coding: utf-8 -*-
import math
from baderkit import Grid
from baderkit.elf_analysis import ElfLabeler

# generate CHGCAR_sum file
core_grid = Grid.from_vasp("AECCAR0")
val_grid = Grid.from_vasp("AECCAR2")
total = core_grid.linear_add(val_grid)
total.write_vasp("CHGCAR_sum")

# load labelers
labeler = ElfLabeler.from_vasp(
    charge_filename="CHGCAR",
    reference_filename="ELFCAR",
    total_charge_filename="CHGCAR_sum",
    pseudopotential_filename="POTCAR"
    )

# get chemical features and each basins charge
features = labeler.basin_types
charges = labeler.elf_bader.basin_charges

# get indices that are covalent
covalent = [i for i, j in enumerate(features) if j == "covalent bond"]

# print bond order rounded up
for idx in covalent:
    print(f"Basin {idx} Bond Order: {math.ceil(charges[idx]/2)}")