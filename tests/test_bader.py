# -*- coding: utf-8 -*-
"""
This file contains a series of tests for the core functionality of the Bader methods.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from baderkit import Bader, Grid
from baderkit.bader.methods import BaderMethod

from .base import assert_nested_equal

TEST_FOLDER = Path(__file__).parent / "test_files"
TEST_BADER_FOLDER = TEST_FOLDER / "bader"
TEST_CHGCAR = TEST_FOLDER / "CHGCAR"
TEST_CHGCAR_CUBE = TEST_FOLDER / "CHGCAR.cube"
TEST_CHGCAR_HDF5 = TEST_FOLDER / "CHGCAR.hdf5"


def test_instance_bader_from_grid():
    # try reading the grid with vasp method
    grid = Grid.from_vasp(TEST_CHGCAR, total_only=False)
    assert grid.diff is not None
    # try reading the grid with dynamic method
    grid = Grid.from_dynamic(TEST_CHGCAR, total_only=False)
    assert grid.diff is not None
    # try to make bader object
    bader = Bader(charge_grid=grid, reference_grid=grid)
    assert bader.reference_grid.diff is not None


def test_read_bader_from_file():
    # test default read ins
    # vasp
    bader = Bader.from_dynamic(TEST_CHGCAR, total_only=False)
    assert bader.charge_grid.diff is not None
    # cube
    bader = Bader.from_dynamic(TEST_CHGCAR_CUBE)
    assert bader.charge_grid.total is not None
    # hdf5
    bader = Bader.from_dynamic(TEST_CHGCAR_HDF5)
    assert bader.charge_grid.diff is not None
    # test reading in reference file
    bader = Bader.from_dynamic(charge_grid=TEST_CHGCAR, reference_grid=TEST_CHGCAR)
    assert bader.reference_grid.total is not None


def test_writing_bader(tmp_path):
    # read in bader
    bader = Bader.from_dynamic(TEST_CHGCAR, method="ongrid")

    # write results files
    bader.write_json(tmp_path / "bader.json")
    bader.write_atom_tsv(tmp_path / "bader_atoms.tsv")
    bader.write_basin_tsv(tmp_path / "bader_basins.tsv")

    # Try writing results
    bader.write_atom_volumes([0], filename=tmp_path / "CHGCAR")
    bader.write_atom_volumes_sum([0], filename=tmp_path / "CHGCAR")
    bader.write_basin_volumes([0], filename=tmp_path / "CHGCAR")
    bader.write_basin_volumes_sum([0], filename=tmp_path / "CHGCAR")
    assert Path(tmp_path / "bader.json").exists()
    assert Path(tmp_path / "bader_atoms.tsv").exists()
    assert Path(tmp_path / "bader_basins.tsv").exists()
    assert Path(tmp_path / "CHGCAR_a0").exists()
    assert Path(tmp_path / "CHGCAR_b0").exists()
    assert Path(tmp_path / "CHGCAR_asum").exists()
    assert Path(tmp_path / "CHGCAR_bsum").exists()


@pytest.mark.parametrize(
    "method",
    [i.value for i in BaderMethod],
)
def test_running_bader_methods(tmp_path, method):
    bader = Bader.from_dynamic(TEST_CHGCAR, method=method)

    assert len(np.where(bader.maxima_basin_labels == len(bader.structure))[0]) == 0

    counts = {
        "weight": 66632,
        "ongrid": 72545,
        "neargrid": 67199,
        "neargrid-weight": 67199,
    }
    assert len(np.where(bader.maxima_basin_labels == 2)[0]) == counts[method]

    # Expected outputs
    with open(TEST_BADER_FOLDER / method / "bader.json") as f:
        expected_json = json.load(f)

    with open(TEST_BADER_FOLDER / method / "bader_atoms.tsv") as f:
        expected_atom_results = f.read()

    with open(TEST_BADER_FOLDER / method / "bader_basins.tsv") as f:
        expected_basin_results = f.read()

    # Write outputs
    bader.write_json(tmp_path / "bader.json")
    bader.write_atom_tsv(tmp_path / "bader_atoms.tsv")
    bader.write_basin_tsv(tmp_path / "bader_basins.tsv")

    # Read generated outputs
    with open(tmp_path / "bader.json") as f:
        json_results = json.load(f)

    with open(tmp_path / "bader_atoms.tsv") as f:
        atom_results = f.read()

    with open(tmp_path / "bader_basins.tsv") as f:
        basin_results = f.read()

    # Tolerant JSON comparison
    assert_nested_equal(json_results, expected_json)

    # Exact text comparisons
    assert atom_results == expected_atom_results
    assert basin_results == expected_basin_results


def test_hill_climb_with_masked_peak():
    """
    Regression test for segfault when hill climbing in init_by_approx_persistence
    encounters a higher-valued voxel masked out by vacuum_mask.
    """
    from baderkit import Structure

    shape = (12, 12, 12)
    ref_data = np.ones(shape, dtype=np.float64)
    # Local maximum at (6, 6, 6)
    ref_data[6, 6, 6] = 10.0
    # Higher peak at (6, 6, 8) (2 voxels away, reachable by transforms2)
    ref_data[6, 6, 8] = 50.0
    ref_data[6, 6, 7] = 5.0

    # Charge grid has negative values at (6, 6, 8) so vacuum_mask is True there
    chg_data = ref_data.copy()
    chg_data[6, 6, 8] = -1.0

    structure = Structure(
        lattice=np.eye(3) * 10.0,
        species=["Si"],
        frac_coords=np.array([[0.5, 0.5, 0.5]]),
    )
    charge_grid = Grid(structure=structure, data={"total": chg_data})
    reference_grid = Grid(structure=structure, data={"total": ref_data})

    bader = Bader(charge_grid=charge_grid, reference_grid=reference_grid)
    assert len(bader.maxima_frac) >= 1
    assert np.isfinite(bader.basin_charges).all()
