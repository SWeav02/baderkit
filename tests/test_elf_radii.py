# -*- coding: utf-8 -*-
"""
This file contains a series of tests for the core functionality of the ElfRadii class.
"""

from pathlib import Path

from baderkit.elf_analysis import ElfRadii

TEST_FOLDER = Path(__file__).parent / "test_files"
TEST_CHGCAR = TEST_FOLDER / "CHGCAR"
TEST_ELFCAR = TEST_FOLDER / "ELFCAR"
TEST_BADELF_FOLDER = TEST_FOLDER / "elf_radii"
TEST_CHGCAR_CUBE = TEST_FOLDER / "CHGCAR.cube"
TEST_CHGCAR_HDF5 = TEST_FOLDER / "CHGCAR.hdf5"


def test_read_radii_from_file():
    # test default read ins
    radii = ElfRadii.from_vasp(
        charge_grid=TEST_CHGCAR, reference_grid=TEST_ELFCAR, total_only=False
    )

    assert radii.charge_grid.diff is not None


def test_writing_radii(tmp_path):
    # read in radii
    radii = ElfRadii.from_vasp(charge_grid=TEST_CHGCAR, reference_grid=TEST_ELFCAR)

    # write results files
    radii.write_json(tmp_path / "radii.json")

    assert Path(tmp_path / "radii.json").exists()
