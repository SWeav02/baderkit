# -*- coding: utf-8 -*-

import os
import shutil
import subprocess
import time
from pathlib import Path

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


def test_webapp(tmp_path):
    # copy CHGCAR over to temp path
    shutil.copyfile(TEST_CHGCAR, tmp_path / "CHGCAR")
    # get path to streamlit app
    current_file = Path(__file__).resolve()
    webapp_path = (
        current_file.parent.parent / "plotting" / "web_gui" / "streamlit" / "webapp.py"
    )
    args = [
        "streamlit",
        "run",
        str(webapp_path),
        "--server.headless",
        "true",
    ]
    # set environment variables
    os.environ["CHARGE_FILE"] = "CHGCAR"
    os.environ["BADER_METHOD"] = "ongrid"

    # Start streamlit
    proc = subprocess.Popen(
        args=args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=tmp_path,
        text=True,
        bufsize=1,  # line-buffered
    )

    try:
        # Let streamlit start then check that it's still running
        time.sleep(15)
        assert proc.poll() is None
    finally:
        # stop the process
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()

    # Get the output
    stdout, _ = proc.communicate()
    for word in ["Traceback", "Error", "Exception", "CRITICAL"]:
        assert word not in stdout, f"Found error keyword '{word}' in output:\n{stdout}"
