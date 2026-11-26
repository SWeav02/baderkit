# -*- coding: utf-8 -*-

from baderkit import Grid
from baderkit.elf_analysis import ElfLabeler

# generate CHGCAR_sum file
core_grid = Grid.from_vasp("AECCAR0")
val_grid = Grid.from_vasp("AECCAR2")
total = core_grid.linear_add(val_grid)
total.write_vasp("CHGCAR_sum")

# load labelers
labeler = ElfLabeler.from_vasp(
    charge_grid="CHGCAR",
    reference_grid="ELFCAR",
    total_charge_grid="CHGCAR_sum",
    pseudopotential_filename="POTCAR",
)

# get chemical features and each basins charge
features = labeler.basin_types
charges = labeler.elf_bader.basin_charges

# get index of covalent bond
covalent_idx = features.index("covalent bond")

# print bond order
print(f"Bond Order: {charges[covalent_idx]/2}")
