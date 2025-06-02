# -*- coding: utf-8 -*-

"""
Defines a helper class for plotting Grids
"""
from itertools import product
from pathlib import Path

import numpy as np
import pyvista as pv
from numpy.typing import NDArray
from pymatgen.core.structure import Site, Structure
from vtk import vtkMatrix4x4, vtkTransform

from baderkit.plotting.defaults import ATOM_COLORS
from baderkit.utilities import Grid


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
        atom_metallicness: float = 0.0,
        show_axes: bool = True,
        **kwargs,
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
                ATOM_COLORS.get(s.specie.symbol, "#FFFFFF") for s in structure
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

    def get_structure_plot(self):
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

    def get_structure_plot_html(self, path: Path | None = None):
        plotter = self.get_structure_plot()
        if path is not None:
            plotter.export_html(path)
        return plotter.export_html()


class GridPlotter(StructurePlotter):
    def __init__(
        self,
        grid: Grid,
        show_caps: bool = True,
        iso_val: float = None,
        surface_opacity: float = 0.8,
        cap_opacity: float = 0.8,
        colormap: float = "viridis",  # Can be any colormap in matplotlib
        use_solid_surface_color: bool = False,
        use_solid_cap_color: bool = False,
        surface_color: tuple = (1.0, 1.0, 1.0),
        cap_color: tuple = (1.0, 1.0, 1.0),
        **kwargs,
    ):
        # apply StructurePlotter kwargs
        structure = grid.structure
        super().__init__(structure=structure, **kwargs)

        # Grid specific items
        self.show_caps = show_caps
        self.iso_val = iso_val
        self.surface_opacity = surface_opacity
        self.cap_opacity = cap_opacity
        self.colormap = colormap
        self.use_solid_surface_color = use_solid_surface_color
        self.use_solid_cap_color = use_solid_cap_color
        self.surface_color = surface_color
        self.cap_color = cap_color

        # determine default iso if not provided
        if iso_val is None:
            self.iso_val = np.mean(grid.total)

        # wrap values around to get one extra voxel on the far side of each axis.
        self.values = np.pad(
            grid.total, pad_width=((0, 1), (0, 1), (0, 1)), mode="wrap"
        )
        self.min_val = self.values.min()
        # make min val slightly above 0
        self.min_val += +0.0000001 * self.min_val
        self.max_val = self.values.max()
        # generate the structured grid
        indices = np.indices(self.values.shape).reshape(3, -1, order="F").T
        points = grid.get_cart_coords_from_vox(indices)
        structured_grid = pv.StructuredGrid()
        structured_grid.points = points
        structured_grid.dimensions = self.values.shape
        structured_grid["values"] = self.values.ravel(order="F")
        self.structured_grid = structured_grid

        # generate the surface
        self.surface = structured_grid.extract_surface()

    def get_surface_mesh_dict(self, iso_value: float) -> dict:
        mesh_dict = {}
        mesh_dict["iso"] = self.structured_grid.contour([iso_value])
        if self.show_caps:
            mesh_dict["cap"] = self.surface.contour_banded(
                2, rng=[iso_value, self.max_val], generate_contour_edges=False
            )
        return mesh_dict

    def get_cap_kwargs(self) -> dict:
        kwargs = {"opacity": self.cap_opacity, "pbr": True, "name": "cap"}
        if self.use_solid_cap_color:
            kwargs["color"] = self.cap_color
        else:
            kwargs["cmap"] = self.colormap
        return kwargs

    def get_surface_kwargs(self) -> dict:
        kwargs = {"opacity": self.surface_opacity, "pbr": True, "name": "iso"}
        if self.use_solid_surface_color:
            kwargs["color"] = self.surface_color
        else:
            kwargs["cmap"] = self.colormap
        return kwargs

    def get_grid_plot(self) -> pv.Plotter():
        # get initial plotter with structure
        plotter = self.get_structure_plot()

        # # Widgets do not work with html so for now I'm removing this
        # def update_isosurface(value):
        #     # Remove old actors
        #     plotter.remove_actor("iso")
        #     plotter.remove_actor("cap") if "cap" in plotter.actors else None
        #     # Generate new iso surface
        #     iso_dict = self.get_surface_mesh_dict(value)
        #     # Add new iso mesh
        #     plotter.add_mesh(iso_dict["iso"], **self.get_surface_kwargs())
        #     # Add cap mesh if present
        #     if "cap" in iso_dict:
        #         plotter.add_mesh(iso_dict["cap"], **self.get_cap_kwargs())

        # # Add the slider widget
        # plotter.add_slider_widget(
        #     callback=update_isosurface,
        #     rng=[self.min_val, self.max_val],
        #     value=self.iso_val,
        #     title="Isosurface Value",
        #     pointa=(0.9, 0.1),
        #     pointb=(0.9, 0.5),
        #     slider_width=0.03,
        #     tube_width=0.01,
        # )

        iso_dict = self.get_surface_mesh_dict(self.iso_val)
        # Add new iso mesh
        plotter.add_mesh(iso_dict["iso"], **self.get_surface_kwargs())
        # Add cap mesh if present
        if "cap" in iso_dict:
            plotter.add_mesh(iso_dict["cap"], **self.get_cap_kwargs())
        return plotter

    def get_grid_plot_html(self, path: Path | None = None):
        plotter = self.get_grid_plot()
        html = plotter.export_html(path)
        return html
