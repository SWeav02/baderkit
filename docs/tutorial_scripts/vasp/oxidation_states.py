# -*- coding: utf-8 -*-

from baderkit import Grid, Bader

# write total charge density
core_grid = Grid.from_vasp("AECCAR0")
val_grid = Grid.from_vasp("AECCAR2")
total = core_grid.linear_add(val_grid)
total.write_vasp("CHGCAR_sum")

# create bader object
bader = Bader.from_vasp(
    charge_grid="CHGCAR",
    total_charge_grid="CHGCAR_sum",
    pseudopotential_filename="POTCAR"
    )

# print oxidation states
print(bader.oxidation_states)