# -*- coding: utf-8 -*-

from baderkit import Grid
from baderkit.elf_analysis import Badelf
from baderkit.global_numba.elf_construction import compute_elf_from_grid

# load grids
charge_up = Grid.from_xsf("chg_up.xsf")
charge_down = Grid.from_xsf("chg_down.xsf")
ked_up = Grid.from_xsf("kin_up.xsf")
ked_down = Grid.from_xsf("kin_down.xsf")

# calculate the ELF
elf_up = compute_elf_from_grid(
    charge_grid=charge_up,
    ked_grid=ked_up,
)
elf_down = compute_elf_from_grid(
    charge_grid=charge_down,
    ked_grid=ked_down,
)

# create badelf up/down
badelf_up = Badelf(charge_grid=charge_up, reference_grid=elf_up, spin=True)
badelf_down = Badelf(charge_grid=charge_down, reference_grid=elf_down, spin=True)

# print useful info
metal_bonds_up = badelf_up.nnas_per_reduced_formula
metal_bonds_down = badelf_down.nnas_per_reduced_formula

# print electron counts
print(f"Spin-up metal bond population: {metal_bonds_up}")
print(f"Spin-down metal bond population: {metal_bonds_down}")
