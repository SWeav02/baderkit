# -*- coding: utf-8 -*-

from baderkit import Grid
from baderkit.elf_analysis import Badelf

# generate CHGCAR_sum file
core_grid = Grid.from_vasp("AECCAR0")
val_grid = Grid.from_vasp("AECCAR2")
total = core_grid.linear_add(val_grid)
total.write_vasp("CHGCAR_sum")

# create badelf object
badelf = Badelf.from_vasp(
    charge_grid="CHGCAR",
    reference_grid="ELFCAR",
    total_charge_grid="CHGCAR_sum",
    pseudopotential_filename="POTCAR"
    )

electride_structure = badelf.nna_structure
electrides_per_formula = badelf.nnas_per_reduced_formula
electride_dimensionality = badelf.nna_dimensionality

# structure including electride site
print(f"Electride Structure: {electride_structure}")

# print electron counts
print(f"Electron Count: {electrides_per_formula}")

# print dimensionality
print(f"Electride Dimensionality: {electride_dimensionality}")