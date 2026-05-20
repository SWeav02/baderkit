# -*- coding: utf-8 -*-

from pathlib import Path

from baderkit.elf_analysis import ElfLabeler

folders = [Path("SiO2"), Path("SiSe2")]

# lists to hold bond information
bond_types = []
bond_polarities = []

for folder in folders:

    # load labeler
    labeler = ElfLabeler.from_cube(
        charge_filename=folder / "chg.cube",
        reference_filename=folder / "elf.cube",
        total_charge_filename=folder / "tot_chg.cube",
    )

    # get the first basin corresponding to a bond
    basin_type = None
    basin_idx = None
    for idx, i in enumerate(labeler.basin_types):
        if "bond" in i:
            basin_type = i
            basin_idx = idx
            break

    # get the bond polarity
    bond_polarity = labeler.overlap.polarization_indexes[basin_idx]

    # add bond types and polarities to our lists
    bond_types.append(basin_type)
    bond_polarities.append(bond_polarity)

# print our bond information to console
for system, bond, polarity in zip(folders, bond_types, bond_polarities):
    # print the polarity and basin type
    print(f"{system.name} Bond Polarity: {polarity} -> {bond}")
