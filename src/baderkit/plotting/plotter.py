# -*- coding: utf-8 -*-

"""
Defines a helper class for plotting Grids
"""
from itertools import product
from io import StringIO

import numpy as np
import pyvista as pv
from numpy.typing import NDArray
import panel

from baderkit.plotting.defaults import ATOM_COLORS
from baderkit.utilities import Grid, Structure


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
        background: str = "#FFFFFF",
        view_indices: NDArray = [1,0,0],
        # parallel_projection: bool = True,
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
        self.background = background
        self.view_indices = view_indices
        # self.parallel_projection = parallel_projection
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

    def get_camera_position(self):
        h,k,l = self.view_indices
        # check for all 0s and adjust
        if all([x==0 for x in [h,k,l]]):
            h,k,l = 1,0,0
        # convert to cart coords
        view_direction = self.structure.get_cart_from_miller(h, k, l)
        # construct camera position
        # Define camera distance from the origin
        # TODO: Calculate this somehow
        camera_distance = 30.0
        
        # Set focal point as center of lattice
        matrix = self.structure.lattice.matrix
        far_corner = np.sum(matrix, axis=0)
        focal_point = far_corner/2
        
        camera_position = focal_point + camera_distance * view_direction
        
        # Use the z-axis as the view up, unless its parallel
        # TODO: Make sure this is rigorous
        z_axis = np.array([0, 0, 1])
        if np.allclose(np.abs(np.dot(view_direction, z_axis)), 1.0):
            view_up = np.array([0, 1, 0])  # fallback to y-axis if z is parallel
        else:
            view_up = z_axis
        # return camera position
        return [
            tuple(camera_position),  # where the camera is
            tuple(focal_point),      # where it's looking
            tuple(view_up)           # which direction is up
        ]
    
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

    def get_structure_plot(self, off_screen: bool = False):
        plotter = pv.Plotter(off_screen=off_screen)
        # set background
        plotter.set_background(self.background)
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
        
        # set camera direction
        plotter.camera_position = self.get_camera_position()
        
        # set camera perspective type
        # if self.parallel_projection:
        #     plotter.enable_parallel_projection()
        
        return plotter

    def get_structure_plot_html(self, width: int, height: int):
        plotter = self.get_structure_plot(off_screen=True)
        plotter.show(auto_close=False)
        vtk_pane = panel.pane.VTK(
            plotter.ren_win, 
            width=width, 
            height=height,
            
            )
        # Create HTML file
        with StringIO() as model_bytes:
            vtk_pane.save(
                model_bytes,
            )
            panel_html = model_bytes.getvalue()
        return panel_html


class GridPlotter(StructurePlotter):
    def __init__(
        self,
        grid: Grid,
        show_surface: bool = True,
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
        self.show_surface = show_surface
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
        if self.show_surface:
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

    def get_grid_plot(self, off_screen: bool = False) -> pv.Plotter():
        # get initial plotter with structure
        plotter = self.get_structure_plot(off_screen=off_screen)
        # get dict of surface meshes
        iso_dict = self.get_surface_mesh_dict(self.iso_val)
        # Add new iso mesh
        if "iso" in iso_dict.keys():
            plotter.add_mesh(iso_dict["iso"], **self.get_surface_kwargs())
        # Add cap mesh if present
        if "cap" in iso_dict:
            plotter.add_mesh(iso_dict["cap"], **self.get_cap_kwargs())
        return plotter

    def get_grid_plot_html(self, width: int, height: int):
        plotter = self.get_grid_plot(off_screen=True)
        plotter.show(auto_close=False)
        vtk_pane = panel.pane.VTK(plotter.ren_win, width=width, height=height)
        # breakpoint()
        # Create HTML file
        with StringIO() as model_bytes:
            vtk_pane.save(
                model_bytes,
            )
            panel_html = model_bytes.getvalue()
        return panel_html
