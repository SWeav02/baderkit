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
        "--reference-file",
        "-ref",
        help="The path to the reference file",
    ),
    method: Method = typer.Option(
        Method.neargrid,
        "--method",
        "-m",
        help="The method to use for separating bader basins",
        case_sensitive=False,
    ),
    vacuum_tolerance: float = typer.Option(
        1.0e-03,
        "--vacuum-tolerance",
        "-vtol",
        help="The value below which a point will be considered part of the vacuum. By default the grid points are normalized by the structure's volume to accomodate VASP's charge format. This can be turned of with the --normalize-vacuum tag.",
    ),
    normalize_vacuum: bool = typer.Option(
        True,
        "--normalize-vacuum",
        "-nvac",
        help="Whether or not to normalize charge to the structure's volume when finding vacuum points.",
    ),
    basin_tolerance: float = typer.Option(
        1.0e-03,
        "--basin-tolerance",
        "-btol",
        help="The charge below which a basin won't be considered significant. Only significant basins will be written to the output file, but the charges and volumes are still assigned to the atoms.",
    ),
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
    os.environ["VACUUM_TOL"] = str(vacuum_tolerance)
    os.environ["NORMALIZE_VAC"] = str(normalize_vacuum)
    os.environ["BASIN_TOL"] = str(basin_tolerance)

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
