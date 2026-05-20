# -*- coding: utf-8 -*-
"""
This file contains a series of tests for the core functionality of the BasinOverlap class.
"""

from pathlib import Path

from baderkit.elf_analysis import BasinOverlap

TEST_FOLDER = Path(__file__).parent / "test_files"
TEST_CHGCAR = TEST_FOLDER / "CHGCAR"
TEST_ELFCAR = TEST_FOLDER / "ELFCAR"
TEST_BADELF_FOLDER = TEST_FOLDER / "elf_overlap"
TEST_CHGCAR_CUBE = TEST_FOLDER / "CHGCAR.cube"
TEST_CHGCAR_HDF5 = TEST_FOLDER / "CHGCAR.hdf5"


def test_read_overlap_from_file():
    # test default read ins
    overlap = BasinOverlap.from_vasp(
        charge_filename=TEST_CHGCAR, reference_filename=TEST_ELFCAR, total_only=False
    )

    assert overlap.charge_grid.diff is not None


def test_writing_overlap(tmp_path):
    # read in overlap
    overlap = BasinOverlap.from_vasp(
        charge_filename=TEST_CHGCAR, reference_filename=TEST_ELFCAR
    )

    # write results files
    overlap.write_json(tmp_path / "overlap.json")

    assert Path(tmp_path / "overlap.json").exists()
