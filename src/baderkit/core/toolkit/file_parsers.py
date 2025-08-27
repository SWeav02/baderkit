#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from enum import Enum
from pathlib import Path

import numpy as np
from numba import njit
from numpy.typing import NDArray
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar


# We list all available import/export formats that we've worked on for consistency
# here
class Format(str, Enum):
    vasp = "vasp"
    cube = "cube"

    @property
    def writer(self):
        return {
            Format.vasp: "write_vasp",
            Format.cube: "write_cube",
        }[self]


def detect_format(filename: str | Path):
    filename = Path(filename)
    with open(filename, "r") as f:
        # skip the first two lines
        next(f)
        next(f)
        # The third line of a VASP CHGCAR/ELFCAR will have 3 values
        # corresponding to the first lattice vector. .cube files will
        # have 4 corresponding to the number of atoms and origin coords
        line_len = len(next(f).split())
        if line_len == 3:
            return Format.vasp
        elif line_len == 4:
            return Format.cube
        else:
            raise ValueError("File format not recognized.")


def read_vasp(filename: str | Path):
    filename = Path(filename)
    with open(filename, "r") as f:
        ###########################################################################
        # Read Structure
        ###########################################################################
        # Read header lines first
        next(f)  # line 0
        scale = float(next(f).strip())  # line 1

        lattice_matrix = (
            np.array([[float(x) for x in next(f).split()] for _ in range(3)]) * scale
        )

        atom_types = next(f).split()
        atom_counts = list(map(int, next(f).split()))
        total_atoms = sum(atom_counts)

        # Skip the 'Direct' or 'Cartesian' line
        next(f)

        coords = np.array(
            [list(map(float, next(f).split())) for _ in range(total_atoms)]
        )

        lattice = Lattice(lattice_matrix)
        atom_list = [
            elem
            for typ, count in zip(atom_types, atom_counts)
            for elem in [typ] * count
        ]
        structure = Structure(lattice=lattice, species=atom_list, coords=coords)

        ###########################################################################
        # Read FFT
        ###########################################################################
        # skip empty line
        next(f)
        # Read the rest of the file to avoid loop overhead
        rest = f.readlines()

    # get the dimensions of the grid
    fft_dim_str = rest[0]
    nx, ny, nz = map(int, fft_dim_str.split())
    ngrid = nx * ny * nz

    # get the number of lines that should exist for each set of data
    vals_per_line = len(rest[0].split())
    nlines = math.ceil(ngrid / vals_per_line)

    # Read datasets until end of file is reached
    all_dataset = []
    all_dataset_aug = {}
    i = 0
    n = 0
    while True:
        # get the remaining info without the dimension line
        rest = rest[i + 1 :]
        grid_lines = rest[:nlines]
        # add the data
        all_dataset.append(
            np.fromstring("".join(grid_lines), sep=" ", dtype=np.float64)
            .ravel()
            .reshape((nx, ny, nz), order="F")
        )
        # loop until the next line that lists grid dimensions or
        # the end of the file
        i = -1
        fft_dim_ints = tuple(map(int, fft_dim_str.split()))
        while i < len(rest):
            try:
                if tuple(map(int, rest[i].split())) == fft_dim_ints:
                    break
            except:
                pass
            i += 1
        # get data aug
        if i > 0:
            all_dataset_aug[n] = rest[:i]
        if len(rest[i:]) == 0:
            break
        n += 1

    # Check for magnetized density. Copied directly from PyMatGen
    if len(all_dataset) == 4:
        data = {
            "total": all_dataset[0],
            "diff_x": all_dataset[1],
            "diff_y": all_dataset[2],
            "diff_z": all_dataset[3],
        }
        data_aug = {
            "total": all_dataset_aug.get(0),
            "diff_x": all_dataset_aug.get(1),
            "diff_y": all_dataset_aug.get(2),
            "diff_z": all_dataset_aug.get(3),
        }

        # Construct a "diff" dict for scalar-like magnetization density,
        # referenced to an arbitrary direction (using same method as
        # pymatgen.electronic_structure.core.Magmom, see
        # Magmom documentation for justification for this)
        # TODO: re-examine this, and also similar behavior in
        # Magmom - @mkhorton
        # TODO: does CHGCAR change with different SAXIS?
        diff_xyz = np.array([data["diff_x"], data["diff_y"], data["diff_z"]])
        diff_xyz = diff_xyz.reshape((3, nx * ny * nz))
        ref_direction = np.array([1.01, 1.02, 1.03])
        ref_sign = np.sign(np.dot(ref_direction, diff_xyz))
        diff = np.multiply(np.linalg.norm(diff_xyz, axis=0), ref_sign)
        data["diff"] = diff.reshape((nx, ny, nz))

    elif len(all_dataset) == 2:
        data = {"total": all_dataset[0], "diff": all_dataset[1]}
        data_aug = {
            "total": all_dataset_aug.get(0),
            "diff": all_dataset_aug.get(1),
        }
    else:
        data = {"total": all_dataset[0]}
        data_aug = {"total": all_dataset_aug.get(0)}
    return structure, data, data_aug


@njit(cache=True)
def format_fortran(mant, exp):
    abs_exp = abs(exp)
    pre = " 0." if mant >= 0 else " -."
    if exp >= 0:
        if abs_exp < 10:
            pre_es = "E+0"
        else:
            pre_es = "E+"
    else:
        if abs_exp < 10:
            pre_es = "E-0"
        else:
            pre_es = "E-"
    return pre + str(mant) + pre_es + str(abs_exp)


@njit(cache=True)
def format_fortran_arr(mants, exps, line_len):
    formatted = []
    for m, e in zip(mants, exps):
        formatted.append(format_fortran(m, e))
    # return formatted
    if len(formatted) == 0:
        return ""
    else:
        rows = []
        for i in range(0, len(formatted), line_len):
            rows.append("".join(formatted[i : i + line_len]))

        return "\n".join(rows) + "\n"


def write_vasp_data(file, arr, chunk_lines=50, line_len=5):
    # calculate chunk size
    chunk_size = line_len * chunk_lines
    # flatten array in Fortran order (z fastest)
    flat = arr.ravel(order="F")
    # create placeholder for mantissa and exponent in fortran scientific notation
    mant = np.zeros_like(flat, dtype=float)
    exp = np.zeros_like(flat, dtype=int)
    # mask out places where value is 0
    nonzero = flat != 0
    # update exponent and mantissa arrays with appropriate values. Note we add 1 to
    # the exp for fortran formatting later and multiply the mant so it is an
    # integer with a length of 10 digits
    exp[nonzero] = np.floor(np.log10(np.abs(flat[nonzero]))) + 1
    mant[nonzero] = (flat[nonzero] / (10.0 ** exp[nonzero])) * 1e11
    mant = np.round(mant).astype(np.int64)

    for i in range(0, len(flat), chunk_size):
        formatted = format_fortran_arr(
            mant[i : i + chunk_size], exp[i : i + chunk_size], line_len
        )
        if formatted:
            file.write(formatted)


def write_vasp(
    filename: str | Path,
    grid,
    vasp4_compatible: bool = False,
) -> None:
    """
    This is largely borrowed from PyMatGen's write function, but attempts
    to speed things up by reducing python loops
    """
    filename = Path(filename)
    structure = grid.structure
    data = grid.data
    data_aug = grid.data_aug

    poscar = Poscar(structure)
    lattice_matrix = structure.lattice.matrix

    # Header lines
    lines = "Written by BaderKit\n"
    # Scale. Read method converts scale so this should always be 1.
    lines += "   1.00000000000000\n"
    # lattice matrix
    for vec in lattice_matrix:
        lines += f" {vec[0]:12.6f}{vec[1]:12.6f}{vec[2]:12.6f}\n"
    # atom symbols and counts
    if not vasp4_compatible:
        lines += "".join(f"{s:5}" for s in poscar.site_symbols) + "\n"
    lines += "".join(f"{x:6}" for x in poscar.natoms) + "\n"
    # atom coordinates
    lines += "Direct\n"
    for site in structure:
        dim, b, c = site.frac_coords
        lines += f"{dim:10.6f}{b:10.6f}{c:10.6f}\n"
    lines += " \n"

    # open file
    with open(filename, "w") as file:
        # write full header
        file.write(lines)
        # Write eahc FFT grid and aug data if it exists
        for key in ["total", "diff", "diff_x", "diff_y", "diff_z"]:
            arr = data.get(key, None)
            if arr is None:
                continue
            # grid dims
            nx, ny, nz = arr.shape
            file.write(f"{nx:6d}{ny:6d}{nz:6d}\n")

            # write to file
            write_vasp_data(file, arr)

            # augmentation info (raw text lines) - write all at once
            if key in data_aug and data_aug[key]:
                # ensure augmentation lines end with newline
                aug_lines = [
                    ln if ln.endswith("\n") else ln + "\n" for ln in data_aug[key]
                ]
                file.writelines(aug_lines)


def read_cube(
    filename: str | Path,
):
    filename = Path(filename)
    with open(filename, "r") as f:
        # Skip first two comment lines
        next(f)
        next(f)

        # Get number of ions and origin
        line = f.readline().split()
        nions = int(line[0])
        origin = np.array(line[1:], dtype=float)

        # Get lattice and grid shape info
        bohr_units = True
        shape = np.empty(3, dtype=int)
        lattice_matrix = np.empty((3, 3), dtype=float)
        for i in range(3):
            line = f.readline().split()
            npts_i = int(line[0])
            # A negative value indicates units are Ang. Positive is Bohr
            if npts_i < 0:
                bohr_units = False
                npts_i = -npts_i
            shape[i] = npts_i
            lattice_matrix[i] = np.array(line[1:], dtype=float)

        # Scale lattice_matrix to cartesian
        lattice_matrix *= shape[:, None]

        # Get atom info
        atomic_nums = np.empty(nions, dtype=int)
        ion_charges = np.empty(nions, dtype=float)
        atom_coords = np.empty((nions, 3), dtype=float)
        for i in range(nions):
            line = f.readline().split()
            atomic_nums[i] = int(line[0])
            ion_charges[i] = float(line[1])
            atom_coords[i] = np.array(line[2:], dtype=float)

        # convert to Angstrom
        if bohr_units:
            lattice_matrix /= 1.88973
            origin /= 1.88973
            atom_coords /= 1.88973
        # Adjust atom positions based on origin
        atom_coords -= origin

        # Create Structure object
        lattice = Lattice(lattice_matrix)
        structure = Structure(
            lattice=lattice,
            species=atomic_nums,
            coords=atom_coords,
            coords_are_cartesian=True,
        )

        # Read charge density
        ngrid = shape.prod()
        # Read all remaining numbers at once for efficiency
        rest = f.read()
    # get data from remaining lines
    volume = structure.volume
    if bohr_units:
        volume *= 1.88973**3
    data = {}
    data["total"] = (
        np.fromstring(rest, sep=" ", dtype=np.float64, count=ngrid)
        .ravel()
        .reshape(shape, order="F")
    ) * volume

    return structure, data, ion_charges, origin


def write_cube(
    filename: str | Path,
    grid,
    ion_charges: NDArray[float] | None = None,
    origin: NDArray[float] | None = None,
) -> None:
    """
    Write a Gaussian .cube file containing charge density.

    Parameters
    ----------
    filename
        Output filename (extension will be changed to .cube).
    ion_charges
        Iterable of length natoms of atomic partial charges / nuclear charges. If None, zeros used.
        (This corresponds to Fortran's ions%ion_chg.)
    origin
        3-element iterable for origin coordinates (cartesian, Angstrom). If None, defaults to (0,0,0).
        (This corresponds to chg%org_car in the Fortran.)

    """
    # normalize inputs and basic checks
    cube_path = Path(filename)
    # cube_path = cube_path.with_suffix(".cube")

    # get structure and grid info
    structure = grid.structure
    nx, ny, nz = grid.shape
    # adjust total by volume in bohr units
    total = grid.total / (structure.volume * 1.88973**3)

    natoms = len(structure)
    if ion_charges is None:
        ion_charges = np.zeros(natoms, dtype=float)
    else:
        ion_charges = np.array(ion_charges)

    if origin is None:
        origin = np.zeros(3, dtype=float)

    # compute voxel vectors
    voxel = grid.matrix / grid.shape[:, None]

    atomic_numbers = structure.atomic_numbers

    positions = structure.cart_coords

    # Convert everything to bohr units
    voxel *= 1.88973
    origin *= 1.88973
    positions *= 1.88973

    # write to file
    # generate header lines
    header = ""
    # header lines
    header += " Gaussian cube file\n"
    header += " Bader charge\n"
    # number of atoms and origin
    header += f"{natoms:5d}{origin[0]:12.6f}{origin[1]:12.6f}{origin[2]:12.6f}\n"
    # grid lines: npts and voxel vectors
    for i in range(3):
        header += f"{grid.shape[i]:5d}{voxel[i,0]:12.6f}{voxel[i,1]:12.6f}{voxel[i,2]:12.6f}\n"
    # atom lines
    for Z, q, pos in zip(atomic_numbers, ion_charges, positions):
        x, y, z = pos
        header += f"{int(Z):5d}{float(q):12.6f}{x:12.6f}{y:12.6f}{z:12.6f}\n"

    # get flat, then reshape to lines of the appropriate size
    flat = total.ravel(order="F")
    flat = flat.reshape((nx * ny, nz))

    with open(cube_path, "w", encoding="utf-8") as file:
        file.write(header)
        for line in flat:
            formatted = [f"{float(d):13.5E}" for d in line]
            if not formatted:
                continue
            # join 6 entries per line, then join lines and add final newline
            rows = ("".join(formatted[i : i + 6]) for i in range(0, len(formatted), 6))
            file.write("\n".join(rows) + "\n")
