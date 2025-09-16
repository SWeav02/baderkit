# -*- coding: utf-8 -*-

import os
import shutil
import time
from pathlib import Path
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from baderkit.command_line.base import baderkit_app

TEST_FOLDER = Path(__file__).parent / "test_files"
TEST_CHGCAR = TEST_FOLDER / "CHGCAR"

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


def test_run():
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
                "0",
            ],
        )
        time.sleep(0)
        assert result.exit_code == 0
        assert Path("CHGCAR_a0").exists()



@pytest.mark.timeout(30)
def test_gui():
    # Ensure Qt runs headless
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"

    # Run the CLI command
    # Use sys.executable to ensure the same Python environment
    proc = subprocess.Popen(
        [sys.executable, "-m", "baderkit", "gui"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Give it some time to start
        time.sleep(20)

        # Check if process is still running
        assert proc.poll() is None, "GUI process exited prematurely"

    finally:
        # Clean up: terminate process
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()

# def test_webapp(tmp_path, monkeypatch):
#     # copy CHGCAR over to temp path
#     shutil.copyfile(TEST_CHGCAR, tmp_path / "CHGCAR")
#     # get path to streamlit app
#     current_file = Path(__file__).resolve()
#     webapp_path = (
#         current_file.parent.parent / "plotting" / "web_gui" / "streamlit" / "webapp.py"
#     )
#     # set environment variables
#     os.environ["CHARGE_FILE"] = "CHGCAR"
#     os.environ["BADER_METHOD"] = "ongrid"
#     os.environ["REFINE_METHOD"] = "single"
#     os.environ["VACUUM_TOL"] = "0.001"
#     os.environ["NORMALIZE_VAC"] = "True"
#     os.environ["BASIN_TOL"] = "0.001"
#     # move into the tmp_directory
#     monkeypatch.chdir(tmp_path)
#     # run webapp
#     at = AppTest.from_file(webapp_path, default_timeout=30)
#     at.run()
#     if at.exception:
#         raise RuntimeError("Streamlit AppTest encountered an error") from at.exception
