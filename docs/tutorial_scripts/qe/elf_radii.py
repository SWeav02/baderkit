# -*- coding: utf-8 -*-

from baderkit.elf_analysis import ElfRadii

# load radii method
elf_radii = ElfRadii.from_cube(
    charge_filename="chg.cube",
    reference_filename="elf.cube",
    total_charge_filename="tot_chg.cube",
)

# get the radius of Na
elf_radius = elf_radii.atom_radii[0]
shannon_radius = elf_radii.structure[0].specie.average_ionic_radius

# print results
print(f"ELF Radius: {round(elf_radius,2)} ang")
print(f"Shannon Radius: {shannon_radius}")
