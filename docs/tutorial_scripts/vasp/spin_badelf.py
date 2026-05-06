# -*- coding: utf-8 -*-

from baderkit import Grid
from baderkit.elf_analysis import Badelf

# load polarized grids
polarized_charge = Grid.from_vasp("CHGCAR", total_only=False)
polarized_elf = Grid.from_vasp("ELFCAR", total_only=False)

# split to spin up/down
charge_up, charge_down = polarized_charge.split_to_spin()
elf_up, elf_down = polarized_elf.split_to_spin()

# create badelf up/down
badelf_up = Badelf(
    charge_grid=charge_up,
    reference_grid=elf_up,
    valence_counts={
        "Fe": 16
        }
    )
badelf_down = Badelf(
    charge_grid=charge_down,
    reference_grid=elf_down,
    valence_counts={
        "Fe": 16
        }
    )

# print useful info
metal_bonds_up = badelf_up.nnas_per_reduced_formula
metal_bonds_down = badelf_down.nnas_per_reduced_formula

# print electron counts
print(f"Spin-up metal bond population: {metal_bonds_up}")
print(f"Spin-down metal bond population: {metal_bonds_down}")