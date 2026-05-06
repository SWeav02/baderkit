# -*- coding: utf-8 -*-

from baderkit.elf_analysis import Badelf

# create badelf object
badelf = Badelf.from_vasp(
    charge_filename="CHGCAR",
    reference_filename="ELFCAR",
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