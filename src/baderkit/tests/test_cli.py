# -*- coding: utf-8 -*-

import os
import shutil
import time
from pathlib import Path

from streamlit.testing.v1 import AppTest
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
        result = runner.invoke(
            app=baderkit_app, args=["tools", "sum", "CHGCAR", "CHGCAR"]
        )
        assert result.exit_code == 0
        assert Path("CHGCAR_sum").exists()


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


def test_webapp(tmp_path, monkeypatch):
    # copy CHGCAR over to temp path
    shutil.copyfile(TEST_CHGCAR, tmp_path / "CHGCAR")
    # get path to streamlit app
    current_file = Path(__file__).resolve()
    webapp_path = (
        current_file.parent.parent / "plotting" / "web_gui" / "streamlit" / "webapp.py"
    )
    # set environment variables
    os.environ["CHARGE_FILE"] = "CHGCAR"
    os.environ["BADER_METHOD"] = "ongrid"
    os.environ["REFINE_METHOD"] = "single"
    os.environ["VACUUM_TOL"] = "0.001"
    os.environ["NORMALIZE_VAC"] = "True"
    os.environ["BASIN_TOL"] = "0.001"
    # move into the tmp_directory
    monkeypatch.chdir(tmp_path)
    # run webapp
    at = AppTest.from_file(webapp_path, default_timeout=15)
    at.run()
    if at.exception:
        raise RuntimeError("Streamlit AppTest encountered an error") from at.exception
