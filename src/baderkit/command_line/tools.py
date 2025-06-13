# -*- coding: utf-8 -*-

import logging
import os
import subprocess
from enum import Enum
from pathlib import Path

import typer

tools_app = typer.Typer(rich_markup_mode="markdown")


@tools_app.callback(no_args_is_help=True)
def base_command():
    """
    A collection of tools for assisting in bader analysis
    """
    pass


@tools_app.command()
def sum(
    file1: Path = typer.Argument(
        ...,
        help="The path to the first file to sum",
    ),
    file2: Path = typer.Argument(
        ...,
        help="The path to the second file to sum",
    ),
):
    """
    A helper function for summing two grids. Note that the output is currently
    always a VASP file.
    """
    from baderkit.core import Grid

    # make sure files are paths
    file1 = Path(file1)
    file2 = Path(file2)
    logging.info(f"Summing files {file1.name} and {file2.name}")

    grid1 = Grid.from_dynamic(file1)
    grid2 = Grid.from_dynamic(file2)
    # sum grids
    summed_grid = Grid.sum_grids(grid1, grid2)
    # get name to use
    if "elf" in file1.name.lower():
        file_pre = "ELFCAR"
    else:
        file_pre = "CHGCAR"
    summed_grid.write_file(f"{file_pre}_sum")


class Method(str, Enum):
    weight = "weight"
    hybrid_weight = "hybrid-weight"
    ongrid = "ongrid"
    neargrid = "neargrid"


@tools_app.command()
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
    # dev: bool = typer.Option(
    #     False,
    #     "--dev",
    #     "-d",
    #     help="Launches panel in development version",
    #     )
):
    """
    Starts the web interface
    """
    # get this files path
    current_file = Path(__file__).resolve()
    # get relative path to streamlit app
    webapp_path = (
        current_file.parent.parent / "plotting" / "web_gui" / "streamlit" / "webapp.py"
    )
    # set environmental variables
    os.environ["CHARGE_FILE"] = str(charge_file)
    os.environ["BADER_METHOD"] = method

    if reference_file is not None:
        os.environ["REFERENCE_FILE"] = str(reference_file)

    args = [
        "streamlit",
        "run",
        str(webapp_path),
    ]

    process = subprocess.Popen(
        args=args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    # Look for prompt and send blank input if needed
    for line in process.stdout:
        print(line, end="")  # Optional: show Streamlit output
        if "email" in line:
            process.stdin.write("\n")
            process.stdin.flush()
            break  # After this, Streamlit should proceed normally
