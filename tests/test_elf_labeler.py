# -*- coding: utf-8 -*-
"""
This file contains a series of tests for the core functionality of the ElfLabeler class.
"""

import json
from pathlib import Path

from baderkit.elf_analysis import ElfLabeler

from .base import assert_nested_equal

TEST_FOLDER = Path(__file__).parent / "test_files"
TEST_CHGCAR = TEST_FOLDER / "CHGCAR"
TEST_ELFCAR = TEST_FOLDER / "ELFCAR"
TEST_ELF_LABELER_FOLDER = TEST_FOLDER / "elf_labeler"
TEST_CHGCAR_CUBE = TEST_FOLDER / "CHGCAR.cube"
TEST_CHGCAR_HDF5 = TEST_FOLDER / "CHGCAR.hdf5"


def test_read_labeler_from_file():
    # test default read ins
    labeler = ElfLabeler.from_vasp(
        charge_grid=TEST_CHGCAR, reference_grid=TEST_ELFCAR, total_only=False
    )

    assert labeler.charge_grid.diff is not None


def test_writing_labeler(tmp_path):
    # read in labeler
    labeler = ElfLabeler.from_vasp(charge_grid=TEST_CHGCAR, reference_grid=TEST_ELFCAR)

    # write results files
    labeler.write_json(tmp_path / "labeler.json")
    labeler.write_features_by_type(
        "non-nuclear attractor", filename=tmp_path / "ELFCAR"
    )

    assert Path(tmp_path / "labeler.json").exists()
    assert Path(tmp_path / "ELFCAR_Xmc").exists()


def test_running_labeler(tmp_path):
    labeler = ElfLabeler.from_vasp(charge_grid=TEST_CHGCAR, reference_grid=TEST_ELFCAR)
    with open(TEST_ELF_LABELER_FOLDER / "labeler.json", "r") as file:
        expected_json = json.load(file)

    # write results to temp file then compare outputs
    labeler.write_json(tmp_path / "labeler.json")

    # read in results and compare
    with open(tmp_path / "labeler.json", "r") as file:
        json_results = json.load(file)

    # Tolerant JSON comparison
    assert_nested_equal(json_results, expected_json)
