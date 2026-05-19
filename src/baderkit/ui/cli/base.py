# -*- coding: utf-8 -*-

"""
Defines the base 'baderkit' command that all other commands stem from.
"""

import logging
from enum import Enum
from pathlib import Path

import typer

from baderkit.bader.methods import Method
from baderkit.global_numba.file_parsers import Format

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
    sel_spec = "sel_spec"


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

###############################################################################
# Main Method Commands
###############################################################################

@baderkit_app.command(no_args_is_help=True)
def bader(
    charge_file: Path = typer.Argument(
        default=...,
        help="The path to the charge density file",
    ),
    total_charge_file: Path = typer.Option(
        None,
        "--total-charge-file",
        "-tot",
        help="The path to the total charge file",
    ),
    reference_file: Path = typer.Option(
        None,
        "--reference-file",
        "-ref",
        help="The path to the reference file",
    ),
    pseudopotential_file: list[Path] = typer.Option(
        None,
        "--pseudopotentials" "-pp",
        help="The path to pseudopotential files for calculating oxidation states. If None, the current directory will be searched for files with common pseudopotential names (POTCAR, .UPF, .xml). Multiple files can be specified by calling this parameter multiple times (e.g. -pp file1 -pp file2 etc.)",
    ),
    method: Method = typer.Option(
        Method.default,
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
    include_minima: bool = typer.Option(
        False,
        "--include-minima",
        "-im",
        help="Whether or not to include summary of Bader calculation performed on the minima (dual)",
    ),
    indices: str = typer.Argument(
        default="",
        help="The indices used for print method. Can be added at the end of the call. For example: `baderkit run CHGCAR -p sel_basins [1,2,3]`",
    ),
):
    """
    Runs a bader analysis on the provided files and writes the results to `bader.json`
    """
    from baderkit import Bader

    # instance bader
    bader = Bader.from_dynamic(
        charge_grid=charge_file,
        total_charge_grid=total_charge_file,
        reference_grid=reference_file,
        pseudopotential_filename=pseudopotential_file,
        method=method,
        format=format,
        vacuum_tol=vacuum_tolerance,
        normalize_vacuum=normalize_vacuum,
        basin_tol=basin_tolerance,
    )
    # write summary
    bader.write_json("bader.json")
    bader.write_atom_tsv("atom_summary.tsv")
    bader.write_basin_tsv("basin_summary.tsv")

    # convert indices from string to list
    try:
        indices = [int(i) for i in indices.strip("[] ").split(",")]
    except:
        indices = [i.strip() for i in indices.strip("[] ").split(",")]

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
    elif print == "sel_spec":
        for species in indices:
            bader.write_species_volume(species=species)


class BadelfPrintOptions(str, Enum):
    all_atoms = "all_atoms"
    sel_atoms = "sel_atoms"
    sum_atoms = "sum_atoms"
    sel_spec = "sel_spec"


class BadelfMethod(str, Enum):
    badelf = "badelf"
    voronelf = "voronelf"
    zero_flux = "zero-flux"


@baderkit_app.command(no_args_is_help=True)
def badelf(
    charge_file: Path = typer.Argument(
        default=...,
        help="The path to the charge density file",
    ),
    reference_file: Path = typer.Argument(
        default=...,
        help="The path to the reference file",
    ),
    total_charge_file: Path = typer.Option(
        None,
        "--total-charge-file",
        "-tot",
        help="The path to the total charge file",
    ),
    pseudopotential_file: list[Path] = typer.Option(
        None,
        "--pseudopotentials" "-pp",
        help="The path to pseudopotential files for calculating oxidation states. If None, the current directory will be searched for files with common pseudopotential names (POTCAR, .UPF, .xml). Multiple files can be specified by calling this parameter multiple times (e.g. -pp file1 -pp file2 etc.)",
    ),
    method: BadelfMethod = typer.Option(
        BadelfMethod.badelf,
        "--method",
        "-m",
        help="The method to use for separating atoms and electrides",
        case_sensitive=False,
    ),
    bader_method: Method = typer.Option(
        Method.default,
        "--bader-method",
        "-bm",
        help="The method to use for bader portions of the algorithm",
        case_sensitive=False,
    ),
    print: BadelfPrintOptions = typer.Option(
        None,
        "--print",
        "-p",
        help="Optional printing of atom basins",
        case_sensitive=False,
    ),
    indices: str = typer.Argument(
        default="",
        help="The indices used for print method. Can be added at the end of the call. For example: `baderkit run CHGCAR -p sel_basins [1,2,3]`",
    ),
):
    """
    Runs a BadELF analysis on the provided files and writes the results to `badelf.json`
    """

    from baderkit.elf_analysis import Badelf

    # instance bader
    badelf = Badelf.from_dynamic(
        charge_grid=charge_file,
        total_charge_grid=total_charge_file,
        reference_grid=reference_file,
        pseudopotential_filename=pseudopotential_file,
        partition_method=method,
        elf_labeler={"method": bader_method},
        # format=format,
    )
    # write summary
    badelf.write_json("badelf.json")

    # convert indices from string to list
    try:
        indices = [int(i) for i in indices.strip("[] ").split(",")]
    except:
        indices = [i.strip() for i in indices.strip("[] ").split(",")]

    # write basins
    if indices is None:
        indices = []
    if print == "all_atoms":
        badelf.write_all_atom_volumes()
    elif print == "sel_atoms":
        badelf.write_atom_volumes(atom_indices=indices)
    elif print == "sum_atoms":
        badelf.write_atom_volumes_sum(atom_indices=indices)
    elif print == "sel_spec":
        for species in indices:
            badelf.write_species_volume(species=species)


class LabelerPrintOptions(str, Enum):
    all_atoms = "all"
    sel_atoms = "sel"


@baderkit_app.command(no_args_is_help=True)
def label(
    charge_file: Path = typer.Argument(
        default=...,
        help="The path to the charge density file",
    ),
    reference_file: Path = typer.Argument(
        default=...,
        help="The path to the reference file",
    ),
    total_charge_file: Path = typer.Option(
        None,
        "--total-charge-file",
        "-tot",
        help="The path to the total charge file",
    ),
    pseudopotential_file: list[Path] = typer.Option(
        None,
        "--pseudopotentials" "-pp",
        help="The path to pseudopotential files for calculating oxidation states. If None, the current directory will be searched for files with common pseudopotential names (POTCAR, .UPF, .xml). Multiple files can be specified by calling this parameter multiple times (e.g. -pp file1 -pp file2 etc.)",
    ),
    method: Method = typer.Option(
        Method.default,
        "--method",
        "-m",
        help="The bader method to use for partitioning the ELF",
        case_sensitive=False,
    ),
    print: LabelerPrintOptions = typer.Option(
        None,
        "--print",
        "-p",
        help="Optional printing of chemical feature volumes",
        case_sensitive=False,
    ),
    features: str = typer.Argument(
        default="",
        help="The feature labels used for print method. Can be added at the end of the call. For example: `baderkit label CHGCAR ELFCAR -p sel_feat metallic`",
    ),
):
    """
    Labels the ELF features in the provided files and writes the results to
    `labeler.json`
    """

    from baderkit.elf_analysis import ElfLabeler

    # instance bader
    labeler = ElfLabeler.from_dynamic(
        charge_grid=charge_file,
        total_charge_grid=total_charge_file,
        reference_grid=reference_file,
        pseudopotential_filename=pseudopotential_file,
        method=method,
        # format=format,
    )
    # write summary
    labeler.write_json("labeler.json")

    labeler.label_structure.to("POSCAR_labeled", "POSCAR")

    # convert indices from string to list
    # write basins
    if features is None:
        features = "unknown"
    if print == "all":
        labeler.write_all_features()
    elif print == "sel":
        labeler.write_features_by_type(features)

@baderkit_app.command(no_args_is_help=True)
def overlap(
    charge_file: Path = typer.Argument(
        default=...,
        help="The path to the charge density file",
    ),
    reference_file: Path = typer.Argument(
        default=...,
        help="The path to the reference file",
    ),
    total_charge_file: Path = typer.Option(
        None,
        "--total-charge-file",
        "-tot",
        help="The path to the total charge file",
    ),
    pseudopotential_file: list[Path] = typer.Option(
        None,
        "--pseudopotentials" "-pp",
        help="The path to pseudopotential files for calculating oxidation states. If None, the current directory will be searched for files with common pseudopotential names (POTCAR, .UPF, .xml). Multiple files can be specified by calling this parameter multiple times (e.g. -pp file1 -pp file2 etc.)",
    ),
    method: Method = typer.Option(
        Method.default,
        "--method",
        "-m",
        help="The bader method to use for partitioning the ELF",
        case_sensitive=False,
    ),
):
    """
    Calculates the overlap between the ELF and QTAIM basins and writes the results to `overlap.json`.
    """

    from baderkit.elf_analysis import BasinOverlap

    # instance bader
    overlap = BasinOverlap.from_dynamic(
        charge_grid=charge_file,
        total_charge_grid=total_charge_file,
        reference_grid=reference_file,
        pseudopotential_filename=pseudopotential_file,
        method=method,
    )
    # write summary
    overlap.write_json("overlap.json")

@baderkit_app.command(no_args_is_help=True)
def radii(
    charge_file: Path = typer.Argument(
        default=...,
        help="The path to the charge density file",
    ),
    reference_file: Path = typer.Argument(
        default=...,
        help="The path to the reference file",
    ),
    total_charge_file: Path = typer.Option(
        None,
        "--total-charge-file",
        "-tot",
        help="The path to the total charge file",
    ),
    pseudopotential_file: list[Path] = typer.Option(
        None,
        "--pseudopotentials" "-pp",
        help="The path to pseudopotential files for calculating oxidation states. If None, the current directory will be searched for files with common pseudopotential names (POTCAR, .UPF, .xml). Multiple files can be specified by calling this parameter multiple times (e.g. -pp file1 -pp file2 etc.)",
    ),
    method: Method = typer.Option(
        Method.default,
        "--method",
        "-m",
        help="The bader method to use for partitioning the ELF",
        case_sensitive=False,
    ),
    include_nnas: bool = typer.Option(
        False,
        "--include-nnas",
        "-nna",
        help="Whether or not to treat non-nuclear attractors (metals, electrides, etc.) as quasi-atoms."
        )
):
    """
    Calculates the ionic/covalent radii for each atom in the system and writes the results to `radii.json`.
    """

    from baderkit.elf_analysis import ElfRadii

    # instance bader
    radii = ElfRadii.from_dynamic(
        charge_grid=charge_file,
        total_charge_grid=total_charge_file,
        reference_grid=reference_file,
        pseudopotential_filename=pseudopotential_file,
        method=method,
        include_nnas=include_nnas,
    )
    # write summary
    radii.write_json("radii.json")

###############################################################################
# Utility Commands
###############################################################################

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
    Sums two grid files.
    """
    from baderkit import Grid

    # make sure files are paths
    file1 = Path(file1)
    file2 = Path(file2)
    logging.info(f"Summing files {file1.name} and {file2.name}")

    # load grids dynamically
    grid1 = Grid.from_dynamic(file1, format=input_format, total_only=False)
    grid2 = Grid.from_dynamic(file2, format=input_format, total_only=False)

    shape1 = tuple(grid1.shape)
    shape2 = tuple(grid2.shape)
    assert shape1 == shape2, f"""
    Grids must have the same shape. {file1.name}: {shape1} differs from {file2.name}: {shape2}
    """
    # sum grids
    summed_grid = grid1.linear_add(grid2)
    # convert output to path
    output_path = Path(output_path)
    # write to file
    summed_grid.write(filename=output_path, output_format=output_format)


@baderkit_app.command(no_args_is_help=True)
def regrid(
    file: Path = typer.Argument(
        ...,
        help="The path to the file to sum",
    ),
    resolution: int = typer.Option(
        1200,
        "--resolution",
        "-r",
        help="The resolution in pts/A^3 to interpolate to",
        case_sensitive=True,
    ),
    output_path: Path = typer.Option(
        None,
        "--output-path",
        "-o",
        help="The path to write the converted grid to. If None, appends 'regrid' to input name",
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
    Creates a new grid file with a different voxel resolution. This is particularly useful for creating files with lower resolution for easier plotting.
    """
    from baderkit import Grid

    # make sure files are paths
    file = Path(file)
    logging.info(f"Regriding file {file.name} to a resolution of {resolution} pts/A^3")

    # load grids dynamically
    grid = Grid.from_dynamic(file, format=input_format, total_only=False)

    # regrid
    grid = grid.regrid(desired_resolution=resolution)

    # convert output to path
    if output_path is None:
        output_path = file.parent / (str(file.stem) + "_regrid")
    output_path = Path(output_path)
    # write to file
    grid.write(filename=output_path, output_format=output_format)


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
    Splits a spin polarized charge density or ELF to its spin-up and spin-down components.
    """
    from baderkit import Grid

    # make sure files are paths
    file = Path(file)
    logging.info(f"Splitting file {file.name}")

    # load grid dynamically
    grid = Grid.from_dynamic(file, format=input_format, total_only=False)

    if not grid.is_spin_polarized:
        raise Exception("""
            This method only splits files that contain both the total and difference
            charge densities like VASP's CHGCAR format.
            If you need help splitting a charge density or ELF in a different format, please
            open a discussion on our [github](https://github.com/SWeav02/baderkit/discussions) and
            we will try and add an example to our documentation.
            """)

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
def make_elf(
    charge_file: Path = typer.Argument(
        ...,
        help="The path to the file containing the charge density",
    ),
    kinetic_file: Path = typer.Argument(
        ...,
        help="The path to the file containing the kinetic energy density"
    ),
    use_spin: bool = typer.Option(
        False,
        "--use-spin",
        "-s",
        help="If set, corrects the prefactor for a single spin rather than the total charge-density"
        ),
    output_path: Path = typer.Option(
        None,
        "--output-path",
        "-o",
        help="The path to write the converted grid to. If None, defaults to ELFCAR or elf.suffix",
        case_sensitive=True,
    ),
    output_format: Format = typer.Option(
        None,
        "--output-format",
        "-of",
        help="The output format of the files. If None, the input format will be used.",
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
    Calculates the ELF from a charge density and kinetic energy density grid. This is useful, for example, for
    calculating the spin-polarized ELF from Quantum Espresso outputs. For spin-polarized systems, the `use-spin`
    parameter should be set to obtain proper values.

    !!! Warning
        While we have checked this method to confirm it nearly replicates QE's implementation, we
        generally suggest using you ab-initio code's ELF method when possible.
    """
    from baderkit import Grid
    from baderkit.global_numba.elf_construction import compute_elf_from_grid

    # make sure files are paths
    charge_file = Path(charge_file)
    kinetic_file = Path(kinetic_file)
    # load grid dynamically
    charge_grid = Grid.from_dynamic(charge_file, format=input_format)
    ked_grid = Grid.from_dynamic(kinetic_file, format=input_format)

    logging.info("Calculating ELF")

    elf_grid = compute_elf_from_grid(
        charge_grid=charge_grid,
        ked_grid=ked_grid,
        spin=use_spin,
        )

    if output_path is None:
        format = charge_grid.source_format

        if format == Format.vasp:
            output_path = Path("ELFCAR")
        else:
            output_path = Path("elf").with_suffix(format.suffix)

    # write file
    elf_grid.write(output_path, output_format)


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
    from baderkit import Grid

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
        import PyQt5
        import pyvista
        import pyvistaqt
        import qtpy
    except:
        logging.warning(
    r'Please run `pip install baderkit\[gui]` (or `pip install "baderkit\[gui]"` depending on the OS/shell).'
)
        return

    from baderkit.ui.gui.main import run_app

    run_app()