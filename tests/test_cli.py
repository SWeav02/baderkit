# -*- coding: utf-8 -*-

import shutil
import time
from pathlib import Path

from typer.testing import CliRunner

from baderkit.ui.cli.base import baderkit_app

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
        # run convert from cube to xsf
        result = runner.invoke(
            app=baderkit_app, args=["convert", "CHGCAR.cube", "CHGCAR.xsf", "xsf"]
        )
        assert result.exit_code == 0
        assert Path("CHGCAR.xsf").exists()
        # run convert from xsf to vasp
        result = runner.invoke(
            app=baderkit_app, args=["convert", "CHGCAR.xsf", "CHGCAR_vasp", "vasp"]
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
                "bader",
                "CHGCAR",
                "-tot",
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
                "-p",
                "sel_atoms",
                "[0]",
            ],
        )
        time.sleep(0)
        assert result.exit_code == 0
        assert Path("ELFCAR_a0").exists()


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
                "-p",
                "sel",
                "metallic bond",
            ],
        )
        time.sleep(0)
        assert result.exit_code == 0
        assert Path("ELFCAR_Xm").exists()


def test_radii():
    # create a temporary folder to run the command in
    with runner.isolated_filesystem():
        # copy CHGCAR/ELFCAR over to temp path
        shutil.copyfile(TEST_CHGCAR, "CHGCAR")
        shutil.copyfile(TEST_ELFCAR, "ELFCAR")
        # run sum
        result = runner.invoke(
            app=baderkit_app,
            args=[
                "radii",
                "CHGCAR",
                "ELFCAR",
            ],
        )
        time.sleep(0)
        assert result.exit_code == 0


def test_overlap():
    # create a temporary folder to run the command in
    with runner.isolated_filesystem():
        # copy CHGCAR/ELFCAR over to temp path
        shutil.copyfile(TEST_CHGCAR, "CHGCAR")
        shutil.copyfile(TEST_ELFCAR, "ELFCAR")
        # run sum
        result = runner.invoke(
            app=baderkit_app,
            args=[
                "overlap",
                "CHGCAR",
                "ELFCAR",
            ],
        )
        time.sleep(0)
        assert result.exit_code == 0


# TODO: I couldn't get this to run and be headless, so for now I'm going to
# try and remember to always run this myself
# def test_gui():
#     # ensure Qt runs headless
#     os.environ["BADERKIT_TEST"] = "1"

#     result = runner.invoke(app=baderkit_app, args=["gui"])
#     assert result.exit_code == 0
