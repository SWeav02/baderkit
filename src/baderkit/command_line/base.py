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
def webapp(
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
            "weight",
            "--method",
            "-m",
            help="The method to use for separating bader basins",
            case_sensitive=False,
        ),
        dev: bool = typer.Option(
            False,
            "--dev",
            "-d",
            help="Launches panel in development version",
            )
        ):
    """
    Starts the web interface
    """
    # get this files path
    current_file = Path(__file__).resolve()
    # get relative path to streamlit app
    webapp_path = current_file.parent.parent / "panel" / "webapp.py"
    # set environmental variables
    os.environ["CHARGE_FILE"] = str(charge_file)
    os.environ["BADER_METHOD"] = method
    
    if reference_file is not None:
        os.environ["REFERENCE_FILE"] = str(reference_file)
    
    args = [
        "panel",
        "serve",
        str(webapp_path),
        ]
    
    if dev:
        args.append("--dev")
    
    subprocess.run(
        args = args,
        check=True
    )


    
    
    