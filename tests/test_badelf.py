# -*- coding: utf-8 -*-
"""
This file contains a series of tests for the core functionality of the BadELF class.
"""

from pathlib import Path

import pytest

from baderkit.elf_analysis import Badelf
from baderkit.elf_analysis.badelf.methods import BadelfMethod

TEST_FOLDER = Path(__file__).parent / "test_files"
TEST_CHGCAR = TEST_FOLDER / "CHGCAR"
TEST_ELFCAR = TEST_FOLDER / "ELFCAR"
TEST_BADELF_FOLDER = TEST_FOLDER / "badelf"
TEST_CHGCAR_CUBE = TEST_FOLDER / "CHGCAR.cube"
TEST_CHGCAR_HDF5 = TEST_FOLDER / "CHGCAR.hdf5"


def test_read_badelf_from_file():
    # test default read ins
    badelf = Badelf.from_vasp(
        charge_grid=TEST_CHGCAR, reference_grid=TEST_ELFCAR, total_only=False
    )

    assert badelf.charge_grid.diff is not None


def test_writing_badelf(tmp_path):
    # read in badelf
    badelf = Badelf.from_vasp(charge_grid=TEST_CHGCAR, reference_grid=TEST_ELFCAR)

    # write results files
    badelf.write_json(tmp_path / "badelf.json")
    badelf.write_atom_tsv(tmp_path / "badelf_atoms.tsv")

    # Try writing results
    badelf.write_atom_volumes([0], filename=tmp_path / "ELFCAR")
    badelf.write_atom_volumes_sum([0], filename=tmp_path / "ELFCAR")

    assert Path(tmp_path / "badelf.json").exists()
    assert Path(tmp_path / "badelf_atoms.tsv").exists()
    assert Path(tmp_path / "ELFCAR_a0").exists()
    assert Path(tmp_path / "ELFCAR_asum").exists()


@pytest.mark.parametrize(
    "method",
    [i.value for i in BadelfMethod],
)
def test_running_badelf_methods(tmp_path, method):
    badelf = Badelf.from_vasp(
        charge_grid=TEST_CHGCAR,
        reference_grid=TEST_ELFCAR,
        partition_method=method,
    )
    with open(TEST_BADELF_FOLDER / method / "badelf.json", "r") as file:
        expected_json = file.read()

    with open(TEST_BADELF_FOLDER / method / "badelf_atoms.tsv", "r") as file:
        expected_atom_results = file.read()

    # write results to temp file then compare outputs
    badelf.write_json(tmp_path / "badelf.json")
    badelf.write_atom_tsv(tmp_path / "badelf_atoms.tsv")

    # read in results and compare
    with open(tmp_path / "badelf.json", "r") as file:
        json_results = file.read()

    with open(tmp_path / "badelf_atoms.tsv", "r") as file:
        atom_results = file.read()

    # make sure we find the electride site
    assert badelf.num_nnas == 1

    assert json_results == expected_json

    assert atom_results == expected_atom_results
