# -*- coding: utf-8 -*-

from pathlib import Path

from baderkit.elf_analysis import SpinBadelf

path = Path(".").resolve()
parent = path.parent

for method in ["badelf", "voronelf", "zero-flux"]:
    badelf = SpinBadelf.from_vasp(
        charge_filename=parent / "CHGCAR",
        reference_filename=parent / "ELFCAR",
        partition_method=method,
    )
    subfolder = path / method
    badelf.write_json(subfolder / "badelf.json")
    badelf.write_atom_tsv(subfolder / "badelf_atoms.tsv")
