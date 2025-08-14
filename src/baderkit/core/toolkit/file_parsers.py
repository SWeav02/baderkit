#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pymatgen.core import Lattice, Structure


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
        fft_dim_str = next(f)
        nx, ny, nz = map(int, fft_dim_str.split())
        ngrid = nx * ny * nz

        # Read the rest of the file to avoid loop overhead
        rest = f.readlines()
    # get the number of lines that should exist for the first grid
    vals_per_line = len(rest[0].split())
    nlines = math.ceil(ngrid / vals_per_line)
    # get the lines corresponding to the first grid and the remaining lines after
    grid_lines = rest[:nlines]
    rest = rest[nlines:]
    # get the total array
    # load the first set of data
    data = {}
    data["total"] = (
        np.fromstring("".join(grid_lines), sep=" ", dtype=np.float64)
        .ravel()
        .reshape((nx, ny, nz), order="F")
    )
    # loop until the next line that lists grid dimensions
    i = -1
    fft_dim_ints = tuple(map(int, fft_dim_str.split()))
    while i < len(rest):
        try:
            if tuple(map(int, rest[i].split())) == fft_dim_ints:
                break
        except:
            pass
        i += 1
    # get the first augmentation set of lines
    data_aug = {"total": rest[:i]}
    # if we've reached the end of the file, return what we have here
    if len(rest[i:]) == 0:
        return structure, data, data_aug
    # get the remaining info without the dimension line
    rest = rest[i + 1 :]
    # get the second grid and remaining lines after
    grid_lines = rest[:nlines]
    # get diff data
    data["diff"] = (
        np.fromstring("".join(grid_lines), sep=" ", dtype=np.float64)
        .ravel()
        .reshape((nx, ny, nz), order="F")
    )
    data_aug["diff"] = rest[nlines:]
    return structure, data, data_aug


# TODO: Better write vasp


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
        atom_charges = np.empty(nions, dtype=float)
        atom_coords = np.empty((nions, 3), dtype=float)
        for i in range(nions):
            line = f.readline().split()
            atomic_nums[i] = int(line[0])
            atom_charges[i] = float(line[1])
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

    return structure, data


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
    filename = Path(filename)
    cube_path = filename.with_suffix(".cube")

    # get structure and grid info
    structure = grid.structure
    nx, ny, nz = grid.shape
    total = grid.total / structure.volume

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

    # Flatten in Fortran order (ix fastest outer, iz fastest inner)
    flat = total.ravel(order="F")

    # Pad to multiple of 6 to avoid edge-case logic
    pad = (-len(flat)) % 6
    if pad:
        flat = np.concatenate([flat, np.full(pad, " " * 13, dtype=flat.dtype)])

    # Reshape so each row is 6 values -> one line
    lines = flat.reshape(-1, 6)

    # generate header lines
    header = ""
    # header lines
    header += "Gaussian cube file\n"
    header += "Bader charge\n"
    # number of atoms and origin
    header += f"{natoms:5d}{origin[0]:12.6f}{origin[1]:12.6f}{origin[2]:12.6f}\n"
    # grid lines: npts and voxel vectors
    for i in range(3):
        header += f"{grid.shape[i]:5d}{voxel[i,0]:12.6f}{voxel[i,1]:12.6f}{voxel[i,2]:12.6f}\n"
    # atom lines
    for Z, q, pos in zip(atomic_numbers, ion_charges, positions):
        x, y, z = pos
        header += f"{int(Z):5d}{float(q):12.6f}{x:12.6f}{y:12.6f}{z:12.6f}\n"
    # write to file with numpy's np.savetxt
    np.savetxt(
        cube_path,
        lines,
        fmt="%13.5E",
        header=header,
        comments="",
    )
