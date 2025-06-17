# -*- coding: utf-8 -*-

"""
Defines a helper class for plotting Grids
"""
import io
import multiprocessing as mp
import sys
from itertools import product
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from numpy.typing import NDArray

from baderkit.core import Bader, Grid, Structure
from baderkit.plotting.core.defaults import ATOM_COLORS

# BUG-FIX We use multiprocessing to export html because on linux/mac an error
# will throw if this is not done as a main process. We also force fork as our
# start method to avoid pickling issues.
if sys.platform != "win32":
    mp.set_start_method("fork", force=True)


def _export_html(queue: Queue, plotter: pv.Plotter):
    queue.put(plotter.export_html(filename=None))


class StructurePlotter:

    def __init__(
        self,
        structure: Structure,
        off_screen: bool = False,
    ):
        """
        A convenience class for creating plots of crystal structures using
        pyvista's package for VTK.

        Parameters
        ----------
        structure : Structure
            The pymatgen Structure object to plot.
        off_screen : bool, optional
            Whether or not the plotter should be in offline mode. The default is False.

        Returns
        -------
        None.

        """
        # sort and relabel structure for consistency
        structure = structure.copy()
        structure.sort()
        structure.relabel_sites()
        # create initial class variables
        self.structure = structure
        self._off_screen = off_screen
        self._visible_atoms = [i for i in range(len(self.structure))]
        self._show_lattice = True
        self._wrap_atoms = True
        self._lattice_thickness = 0.1
        self._atom_metallicness = 0.0
        self._background = "#FFFFFF"
        self._view_indices = [1, 0, 0]
        self._camera_rotation = (0.0,)
        self._show_axes = True
        self._parallel_projection = True
        self._radii = [s.specie.atomic_radius for s in structure]
        self._colors = [ATOM_COLORS.get(s.specie.symbol, "#FFFFFF") for s in structure]
        # generate initial plotter
        self.plotter = self._create_structure_plot(off_screen)
        self.view_indices = [1, 0, 0]
        self.up_indices = [0, 0, 1]

    ###########################################################################
    # Properties and Setters
    ###########################################################################
    @property
    def visible_atoms(self) -> list[int]:
        """

        Returns
        -------
        list[int]
            A list of atom indices to display in the plot.

        """
        return self._visible_atoms

    @visible_atoms.setter
    def visible_atoms(self, visible_atoms: list[int]):
        # update visibility of atoms
        for i, site in enumerate(self.structure):
            label = site.label
            actor = self.plotter.actors[f"{label}"]
            if i in visible_atoms:
                actor.visibility = True
            else:
                actor.visibility = False
        # set visible atoms
        self._visible_atoms = visible_atoms

    @property
    def show_lattice(self) -> bool:
        """

        Returns
        -------
        bool
            Whether or not to display the outline of the unit cell.

        """
        return self._show_lattice

    @show_lattice.setter
    def show_lattice(self, show_lattice: bool):
        actor = self.plotter.actors["lattice"]
        actor.visibility = show_lattice
        self._show_lattice = show_lattice

    # @property
    # def wrap_atoms(self):
    #     return self._wrap_atoms

    # TODO: Make two sets of atoms with and without wraps?
    # @wrap_atoms.setter
    # def wrap_atoms(self, wrap_atoms: bool):
    #     actor = self.plotter.

    @property
    def lattice_thickness(self) -> float:
        """

        Returns
        -------
        float
            The thickness of the lines outlining the unit cell.

        """
        return self._lattice_thickness

    @lattice_thickness.setter
    def lattice_thickness(self, lattice_thickness: float):
        actor = self.plotter.actors["lattice"]
        actor.prop.line_width = lattice_thickness
        self._lattice_thickness = lattice_thickness

    @property
    def atom_metallicness(self) -> float:
        """

        Returns
        -------
        float
            The amount of metallic character in the atom display.

        """
        return self._atom_metallicness

    @atom_metallicness.setter
    def atom_metallicness(self, atom_metallicness: float):
        # update all atoms
        for site in self.structure:
            label = site.label
            actor = self.plotter.actors[f"{label}"]
            actor.prop.metallic = atom_metallicness
        self._atom_metallicness = atom_metallicness

    @property
    def background(self) -> str:
        """

        Returns
        -------
        str
            The color of the plot background as a hex code.

        """
        return self._background

    @background.setter
    def background(self, background: str):
        self.plotter.set_background(background)
        self._background = background

    @property
    def show_axes(self) -> bool:
        """

        Returns
        -------
        bool
            Whether or not to show the axis widget. Note this currently only
            displays the cartesian axes.

        """
        return self._show_axes

    @show_axes.setter
    def show_axes(self, show_axes: bool):
        if show_axes:
            self.plotter.add_axes()
        else:
            self.plotter.hide_axes()
        self._show_axes = show_axes

    @property
    def parallel_projection(self) -> bool:
        """

        Returns
        -------
        bool
            If True, a parallel projection scheme will be used rather than
            perspective.

        """
        return self._parallel_projection

    @parallel_projection.setter
    def parallel_projection(self, parallel_projection: bool):
        if parallel_projection:
            self.plotter.renderer.enable_parallel_projection()
        else:
            self.plotter.renderer.disable_parallel_projection()
        self._parallel_projection = parallel_projection

    @property
    def radii(self) -> list[float]:
        """

        Returns
        -------
        list[float]
            The radius to display for each atom in the structure. The actual
            displayed radius will be 0.3*radius.

        """
        return self._radii

    @radii.setter
    def radii(self, radii: list[float]):
        # fix radii to be a list and make any negative values == 0.01
        radii = list(radii)
        for i, val in enumerate(radii):
            if val <= 0:
                radii[i] = 0.01
        # check which radii have changed and replace these atoms
        old_radii = self.radii
        # update radii
        self._radii = radii
        # for each site, check if the radius has changed and if it has remove it
        # then remake
        for i, (site, old_r, new_r, color) in enumerate(
            zip(self.structure, old_radii, radii, self.colors)
        ):
            if old_r == new_r:
                continue
            # otherwise remove the actor, regenerate, and replot
            self.plotter.remove_actor(f"{site.label}")
            atom_mesh = self.get_site_mesh(i)
            self.plotter.add_mesh(
                atom_mesh,
                color=color,
                metallic=self.atom_metallicness,
                pbr=True,  # enable physical based rendering
                name=f"{site.label}",
            )

    @property
    def colors(self) -> list[str]:
        """

        Returns
        -------
        list[str]
            The colors to use for each atom as hex codes.

        """
        return self._colors

    @colors.setter
    def colors(self, colors: list[str]):
        # for each site, check if the radius has changed and if it has remove it
        # then remake
        for site, old_color, new_color in zip(self.structure, self.colors, colors):
            if old_color == new_color:
                continue
            actor = self.plotter.actors[f"{site.label}"]
            actor.prop.color = new_color
        self._colors = colors

    @property
    def atom_df(self) -> pd.DataFrame:
        """

        Returns
        -------
        atom_df : TYPE
            A dataframe summarizing the properties of the atom meshes.

        """
        # construct a pandas dataframe for each atom
        visible = []
        for i in range(len(self.structure)):
            if i in self.visible_atoms:
                visible.append(True)
            else:
                visible.append(False)
        atom_df = pd.DataFrame(
            {
                "Label": self.structure.labels,
                "Visible": visible,
                "Color": self.colors,
                "Radius": self.radii,
            }
        )
        return atom_df

    @atom_df.setter
    def atom_df(self, atom_df: pd.DataFrame):
        visible = atom_df["Visible"]
        visible_atoms = []
        for i, val in enumerate(visible):
            if val == True:
                visible_atoms.append(i)
        # set each property from the dataframe
        self.visible_atoms = visible_atoms
        self.colors = atom_df["Color"]
        self.radii = atom_df["Radius"]

    @property
    def view_indices(self) -> NDArray[int]:
        """

        Returns
        -------
        NDArray[int]
            The miller indices of the plane that the camera is perpendicular to.

        """
        return self._view_indices

    @view_indices.setter
    def view_indices(self, view_indices: NDArray[int]):
        assert len(view_indices) == 3 and all(
            type(i) == int for i in view_indices
        ), "View indices must be an array or list of miller indices"
        h, k, l = view_indices
        camera_position = self.get_camera_position_from_miller(
            h, k, l, self.camera_rotation
        )
        self.camera_position = camera_position
        # reset the camera zoom so that it fits the screen
        self.plotter.reset_camera()
        self._view_indices = view_indices

    @property
    def camera_rotation(self) -> float:
        """

        Returns
        -------
        float
            The rotation of the camera from the default. The default is to set
            the camera so that the upwards view is as close to the z axis as
            possible, or the y axis if the view indices are perpendicular to z.

        """
        return self._camera_rotation

    @camera_rotation.setter
    def camera_rotation(self, camera_rotation: float):
        h, k, l = self.view_indices
        camera_position = self.get_camera_position_from_miller(h, k, l, camera_rotation)
        self.camera_position = camera_position
        # reset the camera zoom so that it fits the screen
        self.plotter.reset_camera()
        self._camera_rotation = camera_rotation

    @property
    def camera_position(self) -> list[tuple, tuple, tuple]:
        """

        Returns
        -------
        list[tuple, tuple, tuple]
            The set of tuples defining the camera position. In order, this is
            the camera's position, the focal point, and the view up vector.

        """
        pos = self.plotter.camera_position
        # convert to list for serializability
        return [list(pos[0]), list(pos[1]), list(pos[2])]

    @camera_position.setter
    def camera_position(self, camera_position: NDArray):
        camera_position = np.array(camera_position).astype(float)
        if camera_position.ndim == 1:
            h, k, l = camera_position
            camera_pos = self.get_camera_position_from_miller(h, k, l)
            self.plotter.camera_position = camera_pos
        else:
            # convert to tuples
            camera_position = [
                tuple(camera_position[0]),
                tuple(camera_position[1]),
                tuple(camera_position[2]),
            ]
            self.plotter.camera_position = camera_position

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

    def get_camera_position_from_miller(
        self,
        h: int,
        k: int,
        l: int,
        rotation: float = 0,
    ) -> list[tuple, tuple, tuple]:
        """
        Creates a camera position list from a list of miller indices.

        Parameters
        ----------
        h : int
            First miller index.
        k : int
            Second miller index.
        l : int
            Third miller index.
        rotation: float
            The rotation in degrees of the camera. The default of 0 will arrange
            the camera as close to Z=1 as possible, or in the case that it this
            is parallel, it will default to close to Y=1.

        Returns
        -------
        list[tuple, tuple, tuple]
            The set of tuples defining the camera position. In order, this is
            the camera's position, the focal point, and the view up vector.

        """
        # check for all 0s and adjust
        if all([x == 0 for x in [h, k, l]]):
            h, k, l = 1, 0, 0
        # convert to vector perpendicular to the miller plane
        view_direction = self.structure.get_cart_from_miller(h, k, l)
        # Calculate a distance to the camera that doesn't clip any bodies. It's
        # fine if this is very large as methods using this function should reset
        # the camera after. We use half the sum of all lattice sides plus the largest
        # atom radius as this should always be well outside the camera's range
        camera_distance = sum(self.structure.lattice.lengths) + max(self.radii)

        # Set focal point as center of lattice
        matrix = self.structure.lattice.matrix
        far_corner = np.sum(matrix, axis=0)
        focal_point = far_corner / 2
        # set the cameras position by adding the view direction to the focal point.
        # The position is scaled by multiplying by the desired distance
        camera_position = focal_point + view_direction * camera_distance

        # Find an orthogonal vector that has the maximum z value. This is done
        # using Gram-Schmidt orthogonalization.
        z_axis = np.array([0, 0, 1])
        view_up = z_axis - np.dot(z_axis, view_direction) * view_direction
        norm_proj_z = np.linalg.norm(view_up)
        if norm_proj_z < 1e-14:
            # fallback to y-axis if view direction is exactly perpendicular to
            # the z direction
            y_axis = np.array([0, 1, 0])
            view_up = y_axis - np.dot(y_axis, view_direction) * view_direction

        # Now we rotate the camera. We intentionally rotate counter clockwise to
        # make the structure appear to rotate clockwise.
        # convert degrees to radians
        angle_rad = np.deg2rad(rotation)
        view_up_rot = view_up * np.cos(angle_rad) + np.cross(
            view_direction, view_up
        ) * np.sin(angle_rad)
        # return camera position
        return [
            tuple(camera_position),  # where the camera is
            tuple(focal_point),  # where it's looking
            tuple(view_up_rot),  # which direction is up
        ]

    def get_site_mesh(self, site_idx: int) -> pv.PolyData:
        """
        Generates a mesh for the provided site index.

        Parameters
        ----------
        site_idx : int
            The index of the atom to create the mesh for.

        Returns
        -------
        pv.PolyData
            A pyvista mesh representing an atom.

        """
        site = self.structure[site_idx]
        radius = self.radii[site_idx]
        frac_coords = site.frac_coords
        # wrap atom if on edge
        if self._wrap_atoms:
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

    def get_all_site_meshes(self) -> list[pv.PolyData]:
        """
        Gets a list of pyvista meshes representing the atoms in the structure

        Returns
        -------
        meshes : pv.PolyData
            A list of pyvista meshes representing each atom.

        """
        meshes = [self.get_site_mesh(i) for i in range(len(self.structure))]
        return meshes

    def get_lattice_mesh(self) -> pv.PolyData:
        """
        Generates the mesh representing the outline of the unit cell.

        Returns
        -------
        pv.PolyData
            A pyvista mesh representing the outline of the unit cell.

        """
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

    def _create_structure_plot(self, off_screen: bool) -> pv.Plotter:
        """
        Generates a pyvista.Plotter object from the current class variables.
        This is called when the class is first instanced and generally shouldn't
        be called again.

        Parameters
        ----------
        off_screen : bool
            Whether or not the plotter should run in off_screen mode.

        Returns
        -------
        plotter : pv.Plotter
            A pyvista Plotter object representing the provided Structure object.

        """
        plotter = pv.Plotter(off_screen=off_screen)
        # set background
        plotter.set_background(self.background)
        # add atoms
        atom_meshes = self.get_all_site_meshes()
        for i, (site, atom_mesh, color) in enumerate(
            zip(self.structure, atom_meshes, self.colors)
        ):
            actor = plotter.add_mesh(
                atom_mesh,
                color=color,
                metallic=self.atom_metallicness,
                pbr=True,  # enable physical based rendering
                name=f"{site.label}",
            )
            if not i in self.visible_atoms:
                actor.visibility = False

        # add lattice if desired
        lattice_mesh = self.get_lattice_mesh()
        plotter.add_mesh(
            lattice_mesh,
            line_width=self.lattice_thickness,
            color="k",
            name="lattice",
        )

        # # set camera direction
        # plotter.camera_position = self.get_camera_position()

        # set camera perspective type
        if self.parallel_projection:
            plotter.renderer.enable_parallel_projection()

        return plotter

    def show(self):
        """
        Renders the plot to a window. After closing the window, a new instance
        must be created to plot again. Pressing q pauses the rendering allowing
        changes to be made without fully exiting.

        Returns
        -------
        None.

        """
        self.plotter.show(auto_close=False)

    def update(self):
        """
        Updates the pyvista plotter object when linked with a render window in
        Trame. Generally this is not needed outside of Trame applications.

        Returns
        -------
        None.

        """
        self.plotter.update()

    def rebuild(self) -> pv.Plotter:
        """
        Builds a new pyvista plotter object representing the current state of
        the Plotter class.

        Returns
        -------
        pv.Plotter
            A pyvista Plotter object representing the current state of the
            StructurePlotter class.

        """
        return self._create_structure_plot(self._off_screen)

    def get_plot_html(self) -> str:
        """
        Creates an html string representing the current state of the StructurePlotter
        class.

        Returns
        -------
        str
            The html string representing the current StructurePlotter class.

        """
        if sys.platform == "win32":
            # We can return the html directly without opening a subprocess. And
            # we need to because the "fork" start method doesn't work
            html_plotter = self.plotter.export_html(filename=None)
            return html_plotter.read()
        # BUG-FIX: On Linux and maybe MacOS, pyvista's export_html must be run
        # as a main process. To do this within our streamlit apps, we use python's
        # multiprocess to run the process as is done in [stpyvista](https://github.com/edsaac/stpyvista/blob/main/src/stpyvista/trame_backend.py)
        queue = Queue(maxsize=1)
        process = Process(target=_export_html, args=(queue, self.plotter))
        process.start()
        html_plotter = queue.get().read()
        process.join()
        return html_plotter

    # def get_plot_vtksz(self):
    #     # Create temp file path manually
    #     with tempfile.NamedTemporaryFile(suffix=".vtkjs", delete=False) as f:
    #         temp_path = f.name  # Just get the path, don't use the open file
    #     # Now write to it
    #     self.plotter.export_vtksz(temp_path)
    #     # Read the contents
    #     with open(temp_path, "rb") as f:
    #         content = f.read()
    #     # Clean up
    #     os.remove(temp_path)

    #     return content

    def get_plot_screenshot(
        self,
        filename: str | Path | io.BytesIO = None,
        transparent_background: bool = None,
        return_img: bool = True,
        window_size: tuple[int, int] = None,
        scale: int = None,
    ) -> NDArray[float]:
        """
        Creates a screenshot of the current state of the StructurePlotter class.
        This is a wraparound of pyvista's screenshot method

        Parameters
        ----------
        filename: str | Path | io.BytesIO
            Location to write image to. If None, no image is written.

        transparent_background: bool
            Whether to make the background transparent.
            The default is looked up on the plotter’s theme.

        return_img: bool
            If True, a numpy.ndarray of the image will be returned. Defaults to
            True.

        window_size: tuple[int, int]
            Set the plotter’s size to this (width, height) before taking the
            screenshot.

        scale: int
            Set the factor to scale the window size to make a higher resolution image. If None this will use the image_scale property on this plotter which defaults to one.

        Returns
        -------
        NDArray[float]
            Array containing pixel RGB and alpha. Sized:

            [Window height x Window width x 3] if transparent_background is set to False.

            [Window height x Window width x 4] if transparent_background is set to True.

        """
        plotter = self.rebuild()
        plotter.camera = self.plotter.camera.copy()
        screenshot = plotter.screenshot(
            filename=filename,
            transparent_background=transparent_background,
            return_img=return_img,
            window_size=window_size,
            scale=scale,
        )
        plotter.close()
        return screenshot


class GridPlotter(StructurePlotter):
    def __init__(
        self,
        grid: Grid,
        off_screen: bool = False,
        # downscale: int | None = 400,
    ):
        """
        A convenience class for creating plots of crystal structures and isosurfaces
        using pyvista's package for VTK.

        Parameters
        ----------
        grid : Grid
            The Grid object to use for isosurfaces. The structure will be pulled
            from this grid.
        off_screen : bool, optional
            Whether or not the plotter should be in offline mode. The default is False.

        Returns
        -------
        None.

        """
        # apply StructurePlotter kwargs
        structure = grid.structure
        super().__init__(structure=structure, off_screen=off_screen)

        # Grid specific items
        # if downscale is not None:
        #     if grid.voxel_resolution > downscale:
        #         # downscale the grid for speed
        #         logging.info("Grid is above desired resolution. Downscaling.")
        #         grid = grid.regrid(downscale)
        self.grid = grid
        self._show_surface = True
        self._show_caps = True
        self._surface_opacity = 0.8
        self._cap_opacity = 0.8
        self._colormap = "viridis"
        self._use_solid_surface_color = False
        self._use_solid_cap_color = False
        self._surface_color = "#BA8E23"
        self._cap_color = "#BA8E23"

        # wrap values around to get one extra voxel on the far side of each axis.
        values = np.pad(grid.total, pad_width=((0, 1), (0, 1), (0, 1)), mode="wrap")
        self.shape = values.shape
        self.values = values.ravel(order="F")
        self.min_val = self.values.min()
        # make min val slightly above 0
        self.min_val += +0.0000001 * self.min_val
        self.max_val = self.values.max()
        # determine default iso if not provided
        self._iso_val = self.min_val  # np.mean(grid.total)
        # generate the structured grid
        indices = np.indices(self.shape).reshape(3, -1, order="F").T
        self.points = grid.get_cart_coords_from_vox(indices)
        self.structured_grid = self._make_structured_grid(self.values)
        # generate the surface
        self.surface = self.structured_grid.extract_surface()
        # update plotter
        self.plotter = self._create_grid_plot(off_screen)

    def _make_structured_grid(self, values: NDArray[float]) -> pv.StructuredGrid:
        """
        Creates a pyvista StructuredGrid object for making isosurfaces. This
        should generally not be called directly.

        Parameters
        ----------
        values : NDArray[float]
            A 3xN array of values representing the data in the structured grid.
            These should be raveled/reshaped using Fortran's conventions (order='F'')

        Returns
        -------
        structured_grid : pv.StructuredGrid
            A pyvista StructuredGrid with values representing the grid data.

        """
        structured_grid = pv.StructuredGrid()
        structured_grid.points = self.points
        structured_grid.dimensions = self.shape
        structured_grid["values"] = values
        return structured_grid

    @property
    def show_surface(self) -> bool:
        """

        Returns
        -------
        bool
            whether or not to display the isosurface.

        """
        return self._show_surface

    @show_surface.setter
    def show_surface(self, show_surface: bool):
        if "iso" in self.plotter.actors.keys():
            actor = self.plotter.actors["iso"]
            actor.visibility = show_surface
        self._show_surface = show_surface

    @property
    def show_caps(self) -> bool:
        """

        Returns
        -------
        bool
            Whether or not to display caps on the isosurface.

        """
        return self._show_caps

    @show_caps.setter
    def show_caps(self, show_caps: bool):
        if "cap" in self.plotter.actors.keys():
            actor = self.plotter.actors["cap"]
            actor.visibility = show_caps
        self._show_caps = show_caps

    @property
    def surface_opacity(self) -> float:
        """

        Returns
        -------
        float
            Opacity of the isosurface.

        """
        return self._surface_opacity

    @surface_opacity.setter
    def surface_opacity(self, surface_opacity: float):
        if "iso" in self.plotter.actors.keys():
            actor = self.plotter.actors["iso"]
            actor.prop.opacity = surface_opacity
        self._surface_opacity = surface_opacity

    @property
    def cap_opacity(self) -> float:
        """

        Returns
        -------
        float
            Opacity of the caps.

        """
        return self._cap_opacity

    @cap_opacity.setter
    def cap_opacity(self, cap_opacity: float):
        if "cap" in self.plotter.actors.keys():
            actor = self.plotter.actors["cap"]
            actor.prop.opacity = cap_opacity
        self._cap_opacity = cap_opacity

    @property
    def colormap(self) -> str:
        """

        Returns
        -------
        str
            The colormap for the caps and isosurface. This is ignored when the
            surface or caps are set to use solid colors. Valid options are those
            available in matplotlib.

        """
        return self._colormap

    @colormap.setter
    def colormap(self, colormap: str):
        # update settings
        self._colormap = colormap
        if not self.use_solid_surface_color:
            self._add_iso_mesh()
        if not self.use_solid_cap_color:
            self._add_cap_mesh()

    @property
    def use_solid_surface_color(self) -> bool:
        """

        Returns
        -------
        bool
            whether or not to use a solid color for the isosurface.
        """
        return self._use_solid_surface_color

    # TODO: Figure out a way to set the cmap without remaking the surface?
    @use_solid_surface_color.setter
    def use_solid_surface_color(self, use_solid_surface_color: bool):
        # update property
        self._use_solid_surface_color = use_solid_surface_color
        # remove surface and add it back with new color/cmap
        self._add_iso_mesh()

    @property
    def use_solid_cap_color(self) -> bool:
        """

        Returns
        -------
        bool
            whether or not to use a solid color for the caps.
        """
        return self._use_solid_cap_color

    @use_solid_cap_color.setter
    def use_solid_cap_color(self, use_solid_cap_color: bool):
        # update property
        self._use_solid_cap_color = use_solid_cap_color
        # remove cap and add it back with new color/cmap
        self._add_cap_mesh()

    @property
    def surface_color(self) -> str:
        """

        Returns
        -------
        str
            The color to use for the surface as a hex string. This is ignored if
            the surface is not set to use solid colors.

        """
        return self._surface_color

    @surface_color.setter
    def surface_color(self, surface_color: str):
        self._surface_color = surface_color
        if self.use_solid_surface_color:
            self._add_iso_mesh()

    @property
    def cap_color(self):
        """

        Returns
        -------
        str
            The color to use for the caps as a hex string. This is ignored if
            the caps are not set to use solid colors.

        """
        return self._cap_color

    @cap_color.setter
    def cap_color(self, cap_color: str):
        self._cap_color = cap_color
        if self.use_solid_cap_color:
            self._add_cap_mesh()

    @property
    def iso_val(self) -> float:
        """

        Returns
        -------
        float
            The value to set the isosurface to.

        """
        return self._iso_val

    @iso_val.setter
    def iso_val(self, iso_val: float):
        # make sure iso value is within range
        iso_val = max(self.min_val, min(iso_val, self.max_val))
        self._update_surface_mesh(iso_val)
        self._add_iso_mesh()
        self._add_cap_mesh()

    def _update_surface_mesh(self, iso_value: float):
        """
        Updates the surface meshes to the provided iso_value

        Parameters
        ----------
        iso_value : float
            The value to update the surface meshes to

        Returns
        -------
        None.

        """
        self.iso = self.structured_grid.contour([iso_value])
        self.cap = self.surface.contour_banded(
            2, rng=[iso_value, self.max_val], generate_contour_edges=False
        )

    def _get_surface_kwargs(self) -> dict:
        """
        Generates the keyword arguments to use when adding the surface to
        the plotter. We need this because setting a solid color vs. a colormap
        requires different keywords

        Returns
        -------
        dict
            The keyword arguments for setting the surface mesh in the plotter.

        """
        kwargs = {
            "opacity": self.surface_opacity,
            "pbr": True,
            "name": "iso",
        }
        kwargs["color"] = self.surface_color
        if self.use_solid_surface_color:
            kwargs["color"] = self.surface_color
        else:
            kwargs["colormap"] = self.colormap
            kwargs["scalars"] = "values"
            kwargs["clim"] = [self.min_val, self.max_val]
            kwargs["show_scalar_bar"] = False
        return kwargs

    def _get_cap_kwargs(self) -> dict:
        """
        Generates the keyword arguments to use when adding the caps to
        the plotter. We need this because setting a solid color vs. a colormap
        requires different keywords

        Returns
        -------
        dict
            The keyword arguments for setting the caps mesh in the plotter.

        """
        kwargs = {
            "opacity": self.cap_opacity,
            "pbr": True,
            "name": "cap",
        }
        if self.use_solid_cap_color:
            kwargs["color"] = self.cap_color
        else:
            kwargs["cmap"] = self.colormap
            kwargs["scalars"] = "values"
            kwargs["clim"] = [self.min_val, self.max_val]
            kwargs["show_scalar_bar"] = False
        return kwargs

    def _add_iso_mesh(self):
        """
        Removes the current isosurface mesh than adds a new one.

        Returns
        -------
        None.

        """
        if self.show_surface:
            if "iso" in self.plotter.actors.keys():
                self.plotter.remove_actor("iso")
            if len(self.iso["values"]) > 0:
                self.plotter.add_mesh(self.iso, **self._get_surface_kwargs())

    def _add_cap_mesh(self) -> dict:
        """
        Removes the current cap mesh than adds a new one.

        Returns
        -------
        None.

        """
        if self.show_caps:
            if "cap" in self.plotter.actors.keys():
                self.plotter.remove_actor("cap")
            if len(self.iso["values"]) > 0:
                self.plotter.add_mesh(self.cap, **self._get_cap_kwargs())

    def _create_grid_plot(self, off_screen) -> pv.Plotter():
        """
        Generates a pyvista.Plotter object from the current class variables.
        This is called when the class is first instanced and generally shouldn't
        be called again.

        Parameters
        ----------
        off_screen : bool
            Whether or not the plotter should run in off_screen mode.

        Returns
        -------
        plotter : pv.Plotter
            A pyvista Plotter object representing the provided Structure object.

        """
        # get initial plotter with structure
        plotter = self._create_structure_plot(off_screen=off_screen)
        # generate initial surface meshes
        self._update_surface_mesh(self.iso_val)
        # Add iso mesh
        if len(self.iso["values"]) > 0:
            plotter.add_mesh(self.iso, **self._get_surface_kwargs())
        # Add cap mesh
        if len(self.cap["values"]) > 0:
            plotter.add_mesh(self.cap, **self._get_cap_kwargs())
        return plotter

    def rebuild(self):
        """
        Builds a new pyvista plotter object representing the current state of
        the Plotter class.

        Returns
        -------
        pv.Plotter
            A pyvista Plotter object representing the current state of the
            GridPlotter class.

        """
        return self._create_grid_plot(self._off_screen)


class BaderPlotter(GridPlotter):
    def __init__(
        self,
        bader: Bader,
        off_screen: bool = False,
    ):
        """
        A convenience class for creating plots of individual Bader basins
        using pyvista's package for VTK.

        Parameters
        ----------
        bader : Bader
            The Bader object to use for isolating basins and creating isosurfaces.
            The structure will be pulled from the charge grid.
        off_screen : bool, optional
            Whether or not the plotter should be in offline mode. The default is False.

        Returns
        -------
        None.

        """
        # apply StructurePlotter kwargs
        grid = bader.charge_grid
        super().__init__(grid=grid, off_screen=off_screen)
        self.bader = bader

        # pad the label arrays then flatten them
        padded_basins = np.pad(
            bader.basin_labels, pad_width=((0, 1), (0, 1), (0, 1)), mode="wrap"
        )
        padded_atoms = np.pad(
            bader.atom_labels, pad_width=((0, 1), (0, 1), (0, 1)), mode="wrap"
        )
        # padded_basins = bader.basin_labels
        # padded_atoms = bader.atom_labels
        self.flat_bader_basins = padded_basins.ravel(order="F")
        self.flat_atom_basins = padded_atoms.ravel(order="F")

        # get the initial empty list of visible atom labels and visible basin labels
        self._visible_bader_basins = set()
        self._visible_atom_basins = set()
        self.visible_bader_basins = []
        self.visible_atom_basins = []
        self._hidden_mask = np.zeros(len(self.flat_bader_basins), dtype=bool)

    @property
    def visible_bader_basins(self) -> list[int]:
        """

        Returns
        -------
        list[int]
            A list of bader basin indices that are currently visible.

        """
        return self._visible_bader_basins

    @visible_bader_basins.setter
    def visible_bader_basins(self, visible_bader_basins: set[int]):
        # make sure input is set
        visible_bader_basins = set(visible_bader_basins)
        # set visible basins
        self._visible_bader_basins = visible_bader_basins
        # update plotter
        self._update_plotter_mask()

    @property
    def visible_atom_basins(self) -> list[int]:
        """

        Returns
        -------
        list[int]
            A list of atom indices whose basins are currently visible.

        """
        return self._visible_atom_basins

    @visible_atom_basins.setter
    def visible_atom_basins(self, visible_atom_basins: set[int]):
        # make sure input is set
        visible_atom_basins = set(visible_atom_basins)
        # update visible basins set
        self._visible_atom_basins = visible_atom_basins
        # update plotter
        self._update_plotter_mask()

    def _update_plotter_mask(self):
        """
        Updates the mask indicating which areas of the grid should not be shown
        then sets the regions to -1.

        Returns
        -------
        None.

        """
        hidden_mask = ~(
            np.isin(self.flat_bader_basins, list(self._visible_bader_basins))
            | np.isin(self.flat_atom_basins, list(self._visible_atom_basins))
        )
        self._hidden_mask = hidden_mask
        # NOTE: using hide_cells works, but results in some funky artifacting.
        # Maybe there's a way to get it to work, but for now I'm replacing it
        # for visual quality
        # self.structured_grid.hide_cells(self.hidden_mask, inplace=True)
        # update structured_grid
        temp_values = self.values.copy()
        temp_values[hidden_mask] = -1
        self.structured_grid = self._make_structured_grid(temp_values)
        # update the surface
        self.surface = self.structured_grid.extract_surface()
        # update plotter
        self.iso_val = self.iso_val
