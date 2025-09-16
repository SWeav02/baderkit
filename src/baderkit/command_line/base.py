# -*- coding: utf-8 -*-

"""
Defines the base 'baderkit' command that all other commands stem from.
"""

import logging
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
    output_path: Path = typer.Option(
        "CHGCAR_sum",
        "--output-path",
        "-o",
        help="The path to write the converted grid to",
        case_sensitive=True,
    ),
    input_format: Format = typer.Option(
        None,
        "--input-format",
        "-if",
        help="The input format of the file. If None, this will be guessed from the file.",
        case_sensitive=False,
    ),
    output_format: Format = typer.Option(
        None,
        "--output-format",
        "-of",
        help="The output format of the files. If None, the input format will be used.",
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

    # load grids dynamically
    grid1 = Grid.from_dynamic(file1, format=input_format, total_only=False)
    grid2 = Grid.from_dynamic(file2, format=input_format, total_only=False)

    shape1 = tuple(grid1.shape)
    shape2 = tuple(grid2.shape)
    assert (
        shape1 == shape2
    ), f"""
    Grids must have the same shape. {file1.name}: {shape1} differs from {file2.name}: {shape2}
    """
    # sum grids
    summed_grid = grid1.linear_add(grid2)
    # convert output to path
    output_path = Path(output_path)
    # write to file
    summed_grid.write(filename=output_path, output_format=output_format)


@baderkit_app.command(no_args_is_help=True)
def split(
    file: Path = typer.Argument(
        ...,
        help="The path to the file to split",
    ),
    output_up: Path = typer.Option(
        None,
        "--output-up",
        "-ou",
        help="The path to write the spin-up data to. If None, will append '_up' to the original file name.",
        case_sensitive=True,
    ),
    output_down: Path = typer.Option(
        None,
        "--output-down",
        "-od",
        help="The path to write the spin-down data to. If None, will append '_down' to the original file name.",
        case_sensitive=True,
    ),
    input_format: Format = typer.Option(
        None,
        "--input-format",
        "-if",
        help="The input format of the file. If None, this will be guessed from the file.",
        case_sensitive=False,
    ),
    output_format: Format = typer.Option(
        None,
        "--output-format",
        "-of",
        help="The output format of the files. If None, the input format will be used.",
        case_sensitive=False,
    ),
):
    """
    A helper function for splitting a spin polarized charge density or ELF to its
    spin-up and spin-down components.
    """
    from baderkit.core import Grid

    # make sure files are paths
    file = Path(file)
    logging.info(f"Splitting file {file.name}")

    # load grid dynamically
    grid = Grid.from_dynamic(file, format=input_format, total_only=False)

    if not grid.is_spin_polarized:
        raise Exception(
            """
            This method only splits files that contain both the total and difference
            charge densities like VASP's CHGCAR format.
            If you need help splitting a charge density or ELF in a different format, please
            open a discussion on our [github](https://github.com/SWeav02/baderkit/discussions) and
            we will try and add an example to our documentation.
            """
        )

    # split grid. raises errors internally
    spin_up, spin_down = grid.split_to_spin()
    # get name to use
    suffix = file.suffix
    filename = file.name.strip(suffix)
    if output_up is None:
        output_up = f"{filename}_up{suffix}"
    if output_down is None:
        output_down = f"{filename}_down{suffix}"
    # convert outputs to path objects
    output_up = Path(output_up)
    output_down = Path(output_down)
    # write to file
    spin_up.write(filename=output_up, output_format=output_format)
    spin_down.write(filename=output_down, output_format=output_format)


@baderkit_app.command(no_args_is_help=True)
def convert(
    file: Path = typer.Argument(
        ...,
        help="The path to the file to convert",
    ),
    output_path: Path = typer.Argument(
        ...,
        help="The path to write the summed grids to",
        case_sensitive=True,
    ),
    output_format: Format = typer.Argument(
        ...,
        help="The output format of the files",
        case_sensitive=False,
    ),
    input_format: Format = typer.Option(
        None,
        "--input-format",
        "-if",
        help="The input format of the file. If None, this will be guessed from the file.",
        case_sensitive=False,
    ),
):
    """
    Converts the provided file to another format.
    """
    from baderkit.core import Grid

    # make sure files are paths
    file = Path(file)
    logging.info(f"Converting file {file.name}")

    # load grid dynamically
    grid = Grid.from_dynamic(file, format=input_format)

    # write file
    grid.write(output_path, output_format)


@baderkit_app.command()
def gui():
    """
    Launches the BaderKit GUI application
    """
    try:
        import pyvista
        import pyvistaqt
        import qtpy
    except:
        logging.warning(
            "Baderkits GUI requires additional dependencies. Please run 'pip install baderkit\[gui]'"
        )
        return

    from baderkit.plotting.gui.main import run_app

    run_app()
