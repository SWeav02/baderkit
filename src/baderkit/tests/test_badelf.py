# -*- coding: utf-8 -*-
"""
This file contains a series of tests for the core functionality of the BadELF class.
"""
from pathlib import Path

import pytest

from baderkit.core import Badelf, SpinBadelf

TEST_FOLDER = Path(__file__).parent / "test_files"
TEST_CHGCAR = TEST_FOLDER / "CHGCAR"
TEST_ELFCAR = TEST_FOLDER / "ELFCAR"
TEST_BADELF_FOLDER = TEST_FOLDER / "badelf"
TEST_CHGCAR_CUBE = TEST_FOLDER / "CHGCAR.cube"
TEST_CHGCAR_HDF5 = TEST_FOLDER / "CHGCAR.hdf5"


def test_read_badelf_from_file():
    # test default read ins
    badelf = Badelf.from_vasp(
        charge_file=TEST_CHGCAR,
        reference_file=TEST_ELFCAR,
    )

    badelf = SpinBadelf.from_vasp(charge_file=TEST_CHGCAR, reference_file=TEST_ELFCAR)
    assert badelf.charge_grid.diff is not None


def test_writing_badelf(tmp_path):
    # read in badelf
    badelf = SpinBadelf.from_vasp(charge_file=TEST_CHGCAR, reference_file=TEST_ELFCAR)

    # write results files
    badelf.write_json(tmp_path / "badelf.json", write_spin=True)
    badelf.write_atom_tsv(tmp_path / "badelf_atoms.tsv", write_spin=True)

    # Try writing results
    badelf.badelf_up.write_atom_volumes([0], directory=tmp_path)
    badelf.badelf_down.write_atom_volumes_sum([0], directory=tmp_path)

    assert Path(tmp_path / "badelf.json").exists()
    assert Path(tmp_path / "badelf_atoms.tsv").exists()
    assert Path(tmp_path / "ELFCAR_a0").exists()
    assert Path(tmp_path / "ELFCAR_asum").exists()


@pytest.mark.parametrize(
    "method",
    ["badelf", "voronelf", "zero-flux"],
)
def test_running_badelf_methods(tmp_path, method):
    badelf = SpinBadelf.from_vasp(
        charge_file=TEST_CHGCAR,
        reference_file=TEST_ELFCAR,
        method=method,
    )
    with open(TEST_BADELF_FOLDER / method / "badelf.json", "r") as file:
        expected_json = file.read()
    with open(TEST_BADELF_FOLDER / method / "badelf_up.json", "r") as file:
        expected_json_up = file.read()
    with open(TEST_BADELF_FOLDER / method / "badelf_down.json", "r") as file:
        expected_json_down = file.read()

    with open(TEST_BADELF_FOLDER / method / "badelf_atoms.tsv", "r") as file:
        expected_atom_results = file.read()
    with open(TEST_BADELF_FOLDER / method / "badelf_atoms_up.tsv", "r") as file:
        expected_atom_results_up = file.read()
    with open(TEST_BADELF_FOLDER / method / "badelf_atoms_down.tsv", "r") as file:
        expected_atom_results_down = file.read()

    # write results to temp file then compare outputs
    badelf.write_json(tmp_path / "badelf.json", write_spin=True)
    badelf.write_atom_tsv(tmp_path / "badelf_atoms.tsv", write_spin=True)

    # read in results and compare
    with open(tmp_path / "badelf.json", "r") as file:
        json_results = file.read()
    with open(tmp_path / "badelf_up.json", "r") as file:
        json_results_up = file.read()
    with open(tmp_path / "badelf_down.json", "r") as file:
        json_results_down = file.read()

    with open(tmp_path / "badelf_atoms.tsv", "r") as file:
        atom_results = file.read()
    with open(tmp_path / "badelf_atoms_up.tsv", "r") as file:
        atom_results_up = file.read()
    with open(tmp_path / "badelf_atoms_down.tsv", "r") as file:
        atom_results_down = file.read()

    # make sure we find the electride site
    assert badelf.nelectrides == 1

    assert json_results == expected_json
    assert json_results_up == expected_json_up
    assert json_results_down == expected_json_down

    assert atom_results == expected_atom_results
    assert atom_results_up == expected_atom_results_up
    assert atom_results_down == expected_atom_results_down
