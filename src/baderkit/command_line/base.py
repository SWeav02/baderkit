# -*- coding: utf-8 -*-

"""
Defines the base 'baderkit' command that all other commands stem from.
"""

import typer
from enum import Enum
from pathlib import Path

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
            )
        ):
    """
    Runs a bader analysis on the provided files. File formats are automatically
    parsed based on the name. Current accepted files include VASP's CHGCAR/ELFCAR.
    """
    # instance bader
    bader = Bader.from_vasp(
        charge_filename=charge_file,
        reference_filename=reference_file,
        method=method
        )
    # write summary
    bader.write_results_summary()
    
    # TODO:
        # Add methods for printing basin and atom volumes

