# -*- coding: utf-8 -*-

from pathlib import Path

from baderkit.core import SpinBadelf

path = Path(".").resolve()
parent = path.parent

for method in ["badelf", "voronelf", "zero-flux"]:
    badelf = SpinBadelf.from_vasp(
        charge_file=parent / "CHGCAR",
        reference_file=parent / "ELFCAR",
        method=method,
    )
    subfolder = path / method
    badelf.write_json(subfolder / "badelf.json", write_spin=True)
    badelf.write_atom_tsv(subfolder / "badelf_atoms.tsv", write_spin=True)
