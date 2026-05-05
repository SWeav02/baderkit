# -*- coding: utf-8 -*-

from baderkit.elf_analysis import Badelf

# create badelf object
badelf = Badelf.from_vasp(
    charge_filename="CHGCAR",
    reference_filename="ELFCAR",
    pseudopotential_filename="POTCAR"
    )

# structure including electride site
print(f"Electride Structure: {badelf.nna_structure}")

# print electron counts
print(f"Electron Count: {badelf.nnas_per_reduced_formula}")

# print dimensionality
print(f"Electride Dimensionality: {badelf.nna_dimensionality}")