#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import math
import mmap
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


def read_vasp(filename):
    path = Path(filename)
    ###########################################################################
    # Read Structure. Open in string read mode
    ###########################################################################
    with open(path, "r") as f:
        # Read header lines first
        f.readline()  # line 0
        scale = float(f.readline().strip())  # line 1

        lattice_matrix = (
            np.array([[float(x) for x in f.readline().split()] for _ in range(3)])
            * scale
        )

        atom_types = f.readline().split()
        atom_counts = list(map(int, f.readline().split()))
        total_atoms = sum(atom_counts)

        # Skip the 'Direct' or 'Cartesian' line
        f.readline()

        coords = np.array(
            [list(map(float, f.readline().split())) for _ in range(total_atoms)]
        )

        # Skip empty line
        f.readline()
        # get dimensions
        fft_dim_str = f.readline()
        fft_dim_bytes = fft_dim_str.encode("latin1").strip()
        # The next line is the start of the data. Record this point
        data_start_offset = f.tell()
        # Also get a single line of data
        data_line = f.readline()
        # data_line_bytes = data_line.encode("latin1")

        lattice = Lattice(lattice_matrix)
        atom_list = [
            elem
            for typ, count in zip(atom_types, atom_counts)
            for elem in [typ] * count
        ]
        structure = Structure(lattice=lattice, species=atom_list, coords=coords)

        nx, ny, nz = map(int, fft_dim_str.split())

    # Get the number of entries and how many there are per line
    nvals = nx * ny * nz
    vals_per_line = len(data_line.split())
    ###########################################################################
    # Read FFT Grids. Use byte read mode and mmap for faster read and lower memory
    ###########################################################################
    all_datasets = []
    all_datasets_aug = []
    with open(path, "rb") as fb:
        mm = mmap.mmap(fb.fileno(), 0, access=mmap.ACCESS_READ)
        pos = data_start_offset

        # move to the first line of data
        mm.seek(pos)
        # read a single line
        data_line = mm.readline()
        # determine how many extra bytes there are per line
        extra_bytes = len(data_line) % vals_per_line
        # get bytes per entry
        bytes_per_entry = (len(data_line) - extra_bytes) / vals_per_line
        # get total number of extra bytes
        line_bytes = math.ceil((nvals) / 5) * extra_bytes
        # get the total number of bytes per block
        nbytes_per_block = int((bytes_per_entry * nvals) + line_bytes)

        while pos < mm.size():
            # 1. slice exact byte window for this data set
            block_bytes = mm[pos : pos + nbytes_per_block]  # returns bytes (one copy)
            pos_block_end = pos + nbytes_per_block

            # 2) decode and parse with numpy
            # latin1 is the fastest 1:1 mapping decode
            text = block_bytes.decode("latin1")
            arr = np.fromstring(text, sep=" ", count=nvals, dtype=np.float64)
            if arr.size < nvals:
                # incomplete block or EOF
                logging.warn("End of file reached before expected")
                break

            # 3) reshape noting VASP's Fortran ordering
            grid = arr.reshape((nx, ny, nz), order="F")

            # Optional: make C-contiguous if you want to index in C-order quickly
            grid = np.ascontiguousarray(grid)

            # move pos to end of block
            pos = pos_block_end

            # 4) collect augmentation bytes (lines) until next numeric start
            aug_lines = []
            mm.seek(pos)
            while True:
                line = mm.readline()  # returns bytes (fast)
                if not line:
                    # End of file
                    pos = mm.size()
                    break
                # the end of the augment data is marked by a repeat of the grid shape
                if line.strip() == fft_dim_bytes:
                    pos = mm.tell()
                    break
                # otherwise, append line
                else:
                    aug_lines.append(line)
                    pos = mm.tell()
            augment = b"".join(aug_lines).decode("latin1")
            all_datasets.append(grid)
            all_datasets_aug.append(augment)

        mm.close()

    # Check for magnetized density. Copied directly from PyMatGen
    if len(all_datasets) == 4:
        data = {
            "total": all_datasets[0],
            "diff_x": all_datasets[1],
            "diff_y": all_datasets[2],
            "diff_z": all_datasets[3],
        }
        data_aug = {
            "total": all_datasets_aug[0],
            "diff_x": all_datasets_aug[1],
            "diff_y": all_datasets_aug[2],
            "diff_z": all_datasets_aug[3],
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

    elif len(all_datasets) == 2:
        data = {"total": all_datasets[0], "diff": all_datasets[1]}
        data_aug = {
            "total": all_datasets_aug[0],
            "diff": all_datasets_aug[1],
        }
    else:
        data = {"total": all_datasets[0]}
        data_aug = {"total": all_datasets[0]}

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
    # make sure file is a path object
    filename = Path(filename)
    ###########################################################################
    # Read Structure. Open file in string reading mode
    ###########################################################################
    with open(filename, "r") as f:
        # Skip first two comment lines
        f.readline()
        f.readline()

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

        # The next line is the start of the data. Get the exact byte position
        data_start_offset = f.tell()

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

    ###########################################################################
    # Read FFT Grids. Use byte read mode and mmap for faster read and lower memory
    ###########################################################################
    with open(filename, "rb") as fb:
        mm = mmap.mmap(fb.fileno(), 0, access=mmap.ACCESS_READ)
        # read the rest of the file
        block_bytes = mm[data_start_offset:]  # returns bytes (one copy)
        # decode and parse with numpy
        # latin1 is the fastest 1:1 mapping decode
        text = block_bytes.decode("latin1")
        arr = np.fromstring(text, sep=" ", count=ngrid, dtype=np.float64)

        if arr.size < ngrid:
            # incomplete block or EOF
            logging.warn("End of file reached before expected")
        # reshape noting VASP's Fortran ordering
        arr = arr.reshape(shape, order="F")
        arr = np.ascontiguousarray(arr)

        mm.close()

    # adjust data to vasp conventions and store in data dict
    volume = structure.volume
    if bohr_units:
        volume *= 1.88973**3
    data = {}
    data["total"] = arr * volume

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
    # TODO: update formatting to remove leading 0s.
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
