# -*- coding: utf-8 -*-
"""
This file contains a series of tests for the core functionality of the BasinOverlap class.
"""

import json
from pathlib import Path

from baderkit.elf_analysis import BasinOverlap

from .base import assert_nested_equal

TEST_FOLDER = Path(__file__).parent / "test_files"
TEST_CHGCAR = TEST_FOLDER / "CHGCAR"
TEST_ELFCAR = TEST_FOLDER / "ELFCAR"
TEST_OVERLAP_FOLDER = TEST_FOLDER / "overlap"
TEST_CHGCAR_CUBE = TEST_FOLDER / "CHGCAR.cube"
TEST_CHGCAR_HDF5 = TEST_FOLDER / "CHGCAR.hdf5"


def test_read_overlap_from_file():
    # test default read ins
    overlap = BasinOverlap.from_vasp(
        charge_grid=TEST_CHGCAR, reference_grid=TEST_ELFCAR, total_only=False
    )

    assert overlap.charge_grid.diff is not None


def test_writing_overlap(tmp_path):
    # read in overlap
    overlap = BasinOverlap.from_vasp(
        charge_grid=TEST_CHGCAR, reference_grid=TEST_ELFCAR
    )

    # write results files
    overlap.write_json(tmp_path / "overlap.json")

    assert Path(tmp_path / "overlap.json").exists()


def test_running_overlap(tmp_path):
    overlap = BasinOverlap.from_vasp(
        charge_grid=TEST_CHGCAR, reference_grid=TEST_ELFCAR
    )
    with open(TEST_OVERLAP_FOLDER / "overlap.json", "r") as file:
        expected_json = json.load(file)

    # write results to temp file then compare outputs
    overlap.write_json(tmp_path / "overlap.json")

    # read in results and compare
    with open(tmp_path / "overlap.json", "r") as file:
        json_results = json.load(file)

    # Tolerant JSON comparison
    assert_nested_equal(json_results, expected_json)
