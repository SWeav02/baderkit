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

from baderkit.core.methods import Method
from baderkit.core.toolkit import Format

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


class PrintOptions(str, Enum):
    all_atoms = "all_atoms"
    sel_atoms = "sel_atoms"
    sum_atoms = "sum_atoms"
    all_basins = "all_basins"
    sel_basins = "sel_basins"
    sum_basins = "sum_basins"


def float_or_bool(value: str):
    """
    Function for parsing arguments that may be a bool or float
    """
    # Handle booleans
    if value.lower() in {"true", "t", "yes", "y"}:
        return True
    if value.lower() in {"false", "f", "no", "n"}:
        return False
    # Otherwise, try float
    try:
        return float(value)
    except ValueError:
        raise typer.BadParameter("Value must be a float or a boolean.")


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
    vacuum_tolerance: str = typer.Option(
        "1.0e-03",
        "--vacuum-tolerance",
        "-vtol",
        help="The value below which a point will be considered part of the vacuum. By default the grid points are normalized by the structure's volume to accomodate VASP's charge format. This can be turned of with the --normalize-vacuum tag. The vacuum can be ignored by setting this to `False`",
        callback=float_or_bool,
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


@baderkit_app.command(no_args_is_help=True)
def sum(
    file1: Path = typer.Argument(
        ...,
        help="The path to the first file to sum",
    ),
    file2: Path = typer.Argument(
        ...,
        help="The path to the second file to sum",
    ),
    output_name: Path = typer.Option(
        None,
        "--output-path",
        "-o",
        help="The path to write the summed grids to",
        case_sensitive=True,
    ),
    input_format: Format = typer.Option(
        Format.vasp,
        "--input-format",
        "-if",
        help="The input format of the files",
        case_sensitive=False,
    ),
    output_format: Format = typer.Option(
        Format.vasp,
        "--output-format",
        "-of",
        help="The output format of the files",
        case_sensitive=False,
    ),
):
    """
    A helper function for summing two grids.
    """
    from baderkit.core import Grid

    # make sure files are paths
    file1 = Path(file1)
    file2 = Path(file2)
    logging.info(f"Summing files {file1.name} and {file2.name}")

    if input_format == "vasp":
        grid1 = Grid.from_vasp(file1)
        grid2 = Grid.from_vasp(file2)
    elif input_format == "cube":
        grid1 = Grid.from_cube(file1)
        grid2 = Grid.from_cube(file2)

    shape1 = tuple(grid1.shape)
    shape2 = tuple(grid2.shape)
    assert (
        shape1 == shape2
    ), f"""
    Grids must have the same shape. {file1.name}: {shape1} differs from {file2.name}: {shape2}
    """
    # sum grids
    summed_grid = grid1.linear_add(grid2)
    # get name to use
    if output_name is None:
        if "elf" in file1.name.lower():
            file_pre = "ELFCAR"
        else:
            file_pre = "CHGCAR"
        output_name = f"{file_pre}_sum"
        if output_format == "cube":
            output_name += ".cube"
    # convert output to path
    output_name = Path(output_name)
    # write to file
    if output_format == "vasp":
        summed_grid.write_vasp(output_name)
    elif output_format == "cube":
        summed_grid.write_cube(output_name)


@baderkit_app.command(no_args_is_help=True)
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
    vacuum_tolerance: str = typer.Option(
        "1.0e-03",
        "--vacuum-tolerance",
        "-vtol",
        help="The value below which a point will be considered part of the vacuum. By default the grid points are normalized by the structure's volume to accomodate VASP's charge format. This can be turned of with the --normalize-vacuum tag. The vacuum can be ignored by setting this to `False`",
        callback=float_or_bool,
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
