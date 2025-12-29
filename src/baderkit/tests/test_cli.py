# -*- coding: utf-8 -*-

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from baderkit.command_line.base import baderkit_app

TEST_FOLDER = Path(__file__).parent / "test_files"
TEST_CHGCAR = TEST_FOLDER / "CHGCAR"
TEST_ELFCAR = TEST_FOLDER / "ELFCAR"

runner = CliRunner()


def test_sum():
    # create a temporary folder to run the command in
    with runner.isolated_filesystem():
        # copy CHGCAR over to temp path
        shutil.copyfile(TEST_CHGCAR, "CHGCAR")
        # run sum
        result = runner.invoke(app=baderkit_app, args=["sum", "CHGCAR", "CHGCAR"])
        assert result.exit_code == 0
        assert Path("CHGCAR_sum").exists()


def test_split():
    # create a temporary folder to run the command in
    with runner.isolated_filesystem():
        # copy CHGCAR over to temp path
        shutil.copyfile(TEST_CHGCAR, "CHGCAR")
        # run split
        result = runner.invoke(app=baderkit_app, args=["split", "CHGCAR"])
        assert result.exit_code == 0
        assert Path("CHGCAR_up").exists()


def test_convert():
    # NOTE: This also tests the load/write functions for each method
    # create a temporary folder to run the command in
    with runner.isolated_filesystem():
        # copy CHGCAR over to temp path
        shutil.copyfile(TEST_CHGCAR, "CHGCAR")
        # run convert from vasp to hdf5
        result = runner.invoke(
            app=baderkit_app, args=["convert", "CHGCAR", "CHGCAR.hdf5", "hdf5"]
        )
        assert result.exit_code == 0
        assert Path("CHGCAR.hdf5").exists()
        # run convert from hdf5 to cube
        result = runner.invoke(
            app=baderkit_app, args=["convert", "CHGCAR.hdf5", "CHGCAR.cube", "cube"]
        )
        assert result.exit_code == 0
        assert Path("CHGCAR.cube").exists()
        # run convert from cube to vasp
        result = runner.invoke(
            app=baderkit_app, args=["convert", "CHGCAR.cube", "CHGCAR_vasp", "vasp"]
        )
        assert result.exit_code == 0
        assert Path("CHGCAR_vasp").exists()


def test_bader():
    # create a temporary folder to run the command in
    with runner.isolated_filesystem():
        # copy CHGCAR over to temp path
        shutil.copyfile(TEST_CHGCAR, "CHGCAR")
        # run sum
        result = runner.invoke(
            app=baderkit_app,
            args=[
                "run",
                "CHGCAR",
                "-ref",
                "CHGCAR",
                "-m",
                "ongrid",
                "-f",
                "vasp",
                "-p",
                "sel_atoms",
                "[0]",
            ],
        )
        time.sleep(0)
        assert result.exit_code == 0
        assert Path("CHGCAR_a0").exists()


def test_badelf():
    # create a temporary folder to run the command in
    with runner.isolated_filesystem():
        # copy CHGCAR/ELFCAR over to temp path
        shutil.copyfile(TEST_CHGCAR, "CHGCAR")
        shutil.copyfile(TEST_ELFCAR, "ELFCAR")
        # run sum
        result = runner.invoke(
            app=baderkit_app,
            args=[
                "badelf",
                "CHGCAR",
                "ELFCAR",
                "-m",
                "zero-flux",
                "-s",
                "-p",
                "sel_atoms",
                "[0]",
            ],
        )
        time.sleep(0)
        assert result.exit_code == 0
        assert Path("ELFCAR_a0_up").exists()


def test_labeler():
    # create a temporary folder to run the command in
    with runner.isolated_filesystem():
        # copy CHGCAR/ELFCAR over to temp path
        shutil.copyfile(TEST_CHGCAR, "CHGCAR")
        shutil.copyfile(TEST_ELFCAR, "ELFCAR")
        # run sum
        result = runner.invoke(
            app=baderkit_app,
            args=[
                "label",
                "CHGCAR",
                "ELFCAR",
                "-s",
                "-p",
                "sel_feat",
                "[metallic]",
            ],
        )
        time.sleep(0)
        assert result.exit_code == 0
        assert Path("ELFCAR_metallic_fsum_up").exists()


# TODO: I couldn't get this to run and be headless, so for now I'm going to
# try and remember to always run this myself
# def test_gui():
#     # ensure Qt runs headless
#     os.environ["BADERKIT_TEST"] = "1"

#     result = runner.invoke(app=baderkit_app, args=["gui"])
#     assert result.exit_code == 0
