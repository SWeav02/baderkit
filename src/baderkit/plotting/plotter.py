# -*- coding: utf-8 -*-

"""
Defines a helper class for plotting Grids
"""
from itertools import product

import numpy as np
import pyvista as pv
from numpy.typing import NDArray
from pymatgen.core.structure import Site, Structure
from vtk import vtkMatrix4x4, vtkTransform

from baderkit.plotting.defaults import ATOM_COLORS


class StructurePlotter:
    """
    Plots a Pymatgen Structure object using pyvista
    """

    def __init__(
        self,
        structure: Structure,
        hidden_atoms: list[int] = [],
        radii: list | None = None,
        colors: list | None = None,
        show_lattice: bool = True,
        wrap_atoms: bool = True,
        lattice_thickness: float = 0.1,
        atom_metallicness: int = 0.0,
        show_axes: bool = True,
    ):
        # relabel structure
        structure.relabel_sites()
        self.structure = structure
        self.hidden_atoms = hidden_atoms
        self.show_lattice = show_lattice
        self.wrap_atoms = wrap_atoms
        self.lattice_thickness = lattice_thickness
        self.radii = radii
        self.colors = colors
        self.atom_metallicness = atom_metallicness
        self.show_axes = show_axes
        # if radii or colors are not provided we generate them
        # automatically
        if radii is None:
            self.radii = [s.specie.atomic_radius for s in structure]
        if colors is None:
            self.colors = [
                ATOM_COLORS.get(s.specie.symbol, (1, 1, 1)) for s in structure
            ]

    @staticmethod
    def get_edge_atom_fracs(frac_coord: NDArray, tol: float = 1e-08) -> NDArray:
        """
        Generates translationally equivalent atoms if coords are exactly on an edge
        of the lattice

        Parameters
        ----------
        frac_coord : NDArray
            The fractiona coordinates of a single atom to wrap.
        tol : float, optional
            The tolerance in fractional coords to consider an atom on an edge
            of the unit cell. The default is 1e-08.

        Returns
        -------
        NDArray
            The fractional coordinates of the atom wrapped at edges.

        """
        transforms = [
            [0, 1] if abs(x) < tol else [0, -1] if abs(x - 1) < tol else [0]
            for x in frac_coord
        ]

        shifts = set(product(*transforms))
        return [np.array(frac_coord) + np.array(shift) for shift in shifts]

    def get_site_mesh(self, site_idx: int):
        site = self.structure[site_idx]
        radius = self.radii[site_idx]
        frac_coords = site.frac_coords
        # wrap atom if on edge
        if self.wrap_atoms:
            all_frac_coords = self.get_edge_atom_fracs(frac_coords)
        else:
            all_frac_coords = [frac_coords]
        # convert to cart coords
        cart_coords = all_frac_coords @ self.structure.lattice.matrix
        # generate meshes for each atom
        spheres = []
        for cart_coord in cart_coords:
            spheres.append(
                pv.Sphere(
                    radius=radius * 0.3,
                    center=cart_coord,
                    theta_resolution=30,
                    phi_resolution=30,
                )
            )
        # merge all meshes
        return pv.merge(spheres)

    def get_all_site_meshes(self):
        meshes = [
            self.get_site_mesh(i)
            for i in range(len(self.structure))
            if i not in self.hidden_atoms
        ]
        return meshes

    def get_lattice_mesh(self):
        # get the lattice matrix
        a, b, c = self.structure.lattice.matrix
        # get the corners of the matrix
        corners = [np.array([0, 0, 0]), a, b, c, a + b, a + c, b + c, a + b + c]
        # get the indices indicating edges of the lattice
        edges = [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 4),
            (1, 5),
            (2, 4),
            (2, 6),
            (3, 5),
            (3, 6),
            (4, 7),
            (5, 7),
            (6, 7),
        ]
        # generate lines with pv
        lines = []
        for i, j in edges:
            line = pv.Line(corners[i], corners[j])
            lines.append(line)
        # combine and return
        return pv.merge(lines)

    def get_axes_actor(self):
        # TODO: Find a way to make this not point along cartesian axes
        axes = pv.AxesActor()
        return axes

    def get_plot(self):
        plotter = pv.Plotter()
        # add atoms
        atom_meshes = self.get_all_site_meshes()
        for atom_mesh, color in zip(atom_meshes, self.colors):
            plotter.add_mesh(
                atom_mesh,
                color=color,
                metallic=self.atom_metallicness,
                pbr=True,  # enable physical based rendering
            )
        # add lattice if desired
        if self.show_lattice:
            lattice_mesh = self.get_lattice_mesh()
            plotter.add_mesh(lattice_mesh, line_width=self.lattice_thickness, color="k")

        # add axes if desired
        if self.show_axes:
            axes = self.get_axes_actor()
            plotter.axes_actor = axes
            plotter.show_axes()

        return plotter
