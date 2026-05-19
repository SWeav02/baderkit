# -*- coding: utf-8 -*-

from baderkit import Grid
from baderkit.elf_analysis import ElfRadii

# generate CHGCAR_sum file
core_grid = Grid.from_vasp("AECCAR0")
val_grid = Grid.from_vasp("AECCAR2")
total = core_grid.linear_add(val_grid)
total.write_vasp("CHGCAR_sum")

# load radii method
elf_radii = ElfRadii.from_vasp(
    charge_grid="CHGCAR",
    reference_grid="ELFCAR",
    total_charge_grid="CHGCAR_sum",
    pseudopotential_filename="POTCAR"
    )

# get the radius of Na
elf_radius = elf_radii.atom_radii[0]
shannon_radius = elf_radii.structure[0].specie.average_ionic_radius

# print results
print(f"ELF Radius: {round(elf_radius,2)} ang")
print(f"Shannon Radius: {shannon_radius}")