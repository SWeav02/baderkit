# -*- coding: utf-8 -*-

from pathlib import Path

from baderkit import Bader
from baderkit.bader.methods import BaderMethod
from baderkit.elf_analysis import Badelf, BasinOverlap, ElfLabeler, ElfRadii
from baderkit.elf_analysis.badelf.methods import BadelfMethod

path = Path(".")

# Bader
bader = Bader.from_vasp(path / "CHGCAR")
bader_path = path / "bader"
for method in [i.value for i in BaderMethod]:
    subfolder = bader_path / method
    bader.method = method
    bader.write_json(subfolder / "bader.json")
    bader.write_atom_tsv(subfolder / "bader_atoms.tsv")
    bader.write_basin_tsv(subfolder / "bader_basins.tsv")

# BadELF
badelf_path = path / "badelf"
for method in [i.value for i in BadelfMethod]:
    badelf = Badelf.from_vasp(
        charge_grid=path / "CHGCAR",
        reference_grid=path / "ELFCAR",
        partition_method=method,
    )
    subfolder = badelf_path / method
    badelf.write_json(subfolder / "badelf.json")
    badelf.write_atom_tsv(subfolder / "badelf_atoms.tsv")

# Labeler
labeler_path = path / "elf_labeler"
labeler = ElfLabeler.from_vasp(
    charge_grid=path / "CHGCAR",
    reference_grid=path / "ELFCAR",
    partition_method=method,
)
labeler.write_json(labeler_path / "labeler.json")

# Radii
radii_path = path / "elf_radii"
radii = ElfRadii.from_vasp(
    charge_grid=path / "CHGCAR",
    reference_grid=path / "ELFCAR",
    partition_method=method,
)
radii.write_json(radii_path / "radii.json")

# Overlap
overlap_path = path / "overlap"
overlap = BasinOverlap.from_vasp(
    charge_grid=path / "CHGCAR",
    reference_grid=path / "ELFCAR",
    partition_method=method,
)
overlap.write_json(overlap_path / "overlap.json")
