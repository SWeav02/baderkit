# -*- coding: utf-8 -*-

from baderkit import Bader

# create bader object
bader = Bader.from_cube(
    charge_filename="chg.cube",
    total_charge_filename="tot_chg.cube",
    vacuum_tol=False,
    persistence_tol=0.2,
    )

# print oxidation states
print(bader.oxidation_states)