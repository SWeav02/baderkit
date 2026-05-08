# -*- coding: utf-8 -*-

from baderkit.elf_analysis import ElfLabeler

# load labelers
labeler = ElfLabeler.from_cube(
    charge_filename="chg.cube",
    reference_filename="elf.cube",
    total_charge_filename="tot_chg.cube",
    )

# get chemical features and each basins charge
features = labeler.basin_types
charges = labeler.elf_bader.basin_charges

# get index of covalent bond
covalent_idx = features.index("covalent bond")

# print bond order
print(f"Bond Order: {charges[covalent_idx]/2}")