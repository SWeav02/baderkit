#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from pathlib import Path

import numpy as np
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
