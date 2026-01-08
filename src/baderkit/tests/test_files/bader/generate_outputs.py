# -*- coding: utf-8 -*-

from pathlib import Path

from baderkit.core import Bader
from baderkit.core.bader.methods import Method

path = Path(".").resolve()
parent = path.parent
bader = Bader.from_vasp(parent / "CHGCAR")
for method in [i.value for i in Method]:
    subfolder = path / method
    bader.method = method
    bader.write_json(subfolder / "bader.json")
    bader.write_atom_tsv(subfolder / "bader_atoms.tsv")
    bader.write_basin_tsv(subfolder / "bader_basins.tsv")
