# -*- coding: utf-8 -*-

"""
Defines the base 'baderkit' command that all other commands stem from.
"""

import logging
import os
import subprocess
from enum import Enum
from pathlib import Path

import typer

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


# Enums for auto-fill options
class Method(str, Enum):
    weight = "weight"
    ongrid = "ongrid"
    neargrid = "neargrid"


class Format(str, Enum):
    vasp = "vasp"
    cube = "cube"


class PrintOptions(str, Enum):
    all_atoms = "all_atoms"
    sel_atoms = "sel_atoms"
    sum_atoms = "sum_atoms"
    all_basins = "all_basins"
    sel_basins = "sel_basins"
    sum_basins = "sum_basins"


@baderkit_app.command(no_args_is_help=True)
def run(
    charge_file: Path = typer.Argument(
        default=...,
        help="The path to the charge density file",
    ),
    reference_file: Path = typer.Option(
        None,
        "--reference_file",
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
    format: Format = typer.Option(
        None,
        "--format",
        "-f",
        help="The format of the files",
        case_sensitive=False,
    ),
    print: PrintOptions = typer.Option(
        None,
        "--print",
        "-p",
        help="Optional printing of atom or bader basins",
        case_sensitive=False,
    ),
    indices=typer.Argument(
        default=[],
        help="The indices used for print method. Can be added at the end of the call. For example: `baderkit run CHGCAR -p sel_basins 0 1 2`",
    ),
):
    """
    Runs a bader analysis on the provided files. File formats are automatically
    parsed based on the name. Current accepted files include VASP's CHGCAR/ELFCAR
    or .cube files.
    """
    from baderkit.core import Bader

    # instance bader
    bader = Bader.from_dynamic(
        charge_filename=charge_file,
        reference_filename=reference_file,
        method=method,
        format=format,
        vacuum_tol=vacuum_tolerance,
        normalize_vacuum=normalize_vacuum,
        basin_tol=basin_tolerance,
    )
    # write summary
    bader.write_results_summary()

    # write basins
    if indices is None:
        indices = []
    if print == "all_atoms":
        bader.write_all_atom_volumes()
    elif print == "all_basins":
        bader.write_all_basin_volumes()
    elif print == "sel_atoms":
        bader.write_atom_volumes(atom_indices=indices)
    elif print == "sel_basins":
        bader.write_basin_volumes(basin_indices=indices)
    elif print == "sum_atoms":
        bader.write_atom_volumes_sum(atom_indices=indices)
    elif print == "sum_basins":
        bader.write_basin_volumes_sum(basin_indices=indices)


@baderkit_app.command()
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


@baderkit_app.command()
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
