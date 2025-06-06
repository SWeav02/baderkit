# -*- coding: utf-8 -*-

"""
Defines the base 'baderkit' command that all other commands stem from.
"""

from enum import Enum
from pathlib import Path
import sys
import os
import subprocess

import typer
from streamlit import config as _config
from streamlit.web.bootstrap import run as _run

from baderkit.core import Bader

# from baderkit.command_line.run import run_app

baderkit_app = typer.Typer(rich_markup_mode="markdown")


@baderkit_app.callback(no_args_is_help=True)
def base_command():
    """
    This is the base command that all baderkit commands stem from
    """
    pass


@baderkit_app.command()
def version():
    """
    Prints the version of baderkit that is installed
    """
    import baderkit

    print(f"Installed version: v{baderkit.__version__}")


class Method(str, Enum):
    weight = "weight"
    hybrid_weight = "hybrid-weight"
    ongrid = "ongrid"


@baderkit_app.command()
def run(
    charge_file: Path = typer.Argument(
        ...,
        help="The path to the charge density file",
    ),
    reference_file: Path = typer.Option(
        None,
        "--reference_file",
        "-ref",
        help="The path to the reference file",
    ),
    method: Method = typer.Option(
        Method.weight,
        "--method",
        "-m",
        help="The method to use for separating bader basins",
        case_sensitive=False,
    ),
):
    """
    Runs a bader analysis on the provided files. File formats are automatically
    parsed based on the name. Current accepted files include VASP's CHGCAR/ELFCAR.
    """
    # instance bader
    bader = Bader.from_vasp(
        charge_filename=charge_file, reference_filename=reference_file, method=method
    )
    # write summary
    bader.write_results_summary()

    # TODO:
    # Add methods for printing basin and atom volumes

@baderkit_app.command()
def webapp():
    """
    Starts the web interface
    """
    # get this files path
    current_file = Path(__file__).resolve()
    # get relative path to streamlit app
    webapp_path = current_file.parent.parent / "streamlit" / "webapp.py"
    process = subprocess.Popen(
        ["streamlit", "run", str(webapp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Look for prompt and send blank input if needed
    for line in process.stdout:
        print(line, end="")  # Optional: show Streamlit output
        if "email" in line:
            process.stdin.write("\n")
            process.stdin.flush()
            break  # After this, Streamlit should proceed normally
    
    
    