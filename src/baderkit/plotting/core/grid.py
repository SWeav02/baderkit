# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pyvista as pv
from numpy.typing import NDArray


from baderkit.core import Grid
from .structure import StructurePlotter

    
class GridPlotter(StructurePlotter):
    def __init__(
        self,
        grid: Grid,
        **structure_kwargs,
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

        Returns
        -------
        None.

        """
        # apply StructurePlotter kwargs
        super().__init__(structure=grid.structure, **structure_kwargs)

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
        self._min_val = self.values.min()
        # make min val slightly above 0
        self._min_val += 0.0000001 * self._min_val
        self._max_val = self.values.max()
        # determine default iso if not provided
        self._iso_val = self._min_val  # np.mean(grid.total)
        # generate the structured grid
        indices = np.indices(self.shape).reshape(3, -1, order="F").T
        self.points = grid.grid_to_cart(indices)
        self.structured_grid = self._make_structured_grid(self.values)
        # generate the surface
        self.surface = self.structured_grid.extract_surface()
        # update plotter
        self.plotter = self._create_grid_plot()

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
    def min_val(self) -> str:
        return self._min_val

    @min_val.setter
    def min_val(self, value: float):

        # make sure value is below max value
        value = min(self.max_val, value)

        # update settings
        self._min_val = value
        self._add_cap_mesh()

    @property
    def max_val(self) -> str:
        return self._max_val

    @max_val.setter
    def max_val(self, value: float):

        # make sure value is above min value
        value = max(self.min_val, value)

        # update settings
        self._max_val = value
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
        self._iso_val = iso_val
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

    def _create_grid_plot(self) -> pv.Plotter():
        """
        Generates a pyvista.Plotter object from the current class variables.
        This is called when the class is first instanced and generally shouldn't
        be called again.

        Returns
        -------
        plotter : pv.Plotter
            A pyvista Plotter object representing the provided Structure object.

        """
        if type(self.plotter) == StructurePlotter:
            plotter = self.plotter
        else:
            # get initial plotter with structure
            plotter = self._create_structure_plot()
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
        return self._create_grid_plot()

    def get_slice_plot(
        self,
        hkl: NDArray,
        d: float,
        include_atoms: bool = True,
        filepath: Path = None,
        **write_kwargs,
    ):
        """
        Generates a pyvista plot of a slice at the requested miller plane. If
        a filepath is provided, the plot is written and no plot object is returned.

        Parameters
        ----------
        hkl : NDArray
            The miller indices of the plane to use.
        d : float
            The multiplier for the d-spacing of the current plane
        include_atoms : bool, optional
            Whether or not atoms should be incuded. Only atoms whose sphere mesh
            is sliced by the plane are included. The default is True.
        filepath : Path, optional
            The filepath to write the plot to if desired. The default is None.
        **write_kwargs
            any additional keyword arguments to provide to the plot writer.

        Returns
        -------
        p : pv.plotter | None
            the pyvista plot of the slice or None if a filepath was provided.

        """
        h, k, l = hkl
        # get normal vector in cart coords
        normal = self.structure.get_cart_from_miller(h, k, l)
        n = self.structure.lattice.d_hkl(hkl)
        origin = normal * n * d
        slice_plane = self.structured_grid.slice(normal=normal, origin=origin)

        # create plotter
        p = pv.Plotter(off_screen=True)
        p.add_mesh(
            slice_plane,
            scalars="values",
            cmap=self.colormap,
            clim=(self.min_val, self.max_val),
            show_scalar_bar=False,
        )

        # if desired, add any atoms that sit on/near the plane
        if include_atoms:
            for i, (site, color) in enumerate(zip(self.structure, self.colors)):
                center = site.coords
                radius = self.radii[i]
                dist = np.dot(center - origin, normal)
                if not abs(dist) <= radius:
                    continue
                # otherwise remove the actor, regenerate, and replot
                atom_mesh = self.get_site_mesh(i)
                p.add_mesh(
                    atom_mesh,
                    color=color,
                    metallic=self.atom_metallicness,
                    pbr=True,  # enable physical based rendering
                    name=f"{site.label}",
                )
        # set camera to be perpendicular
        camera = self.get_camera_position_from_miller(h, k, l)
        p.camera_position = camera
        p.camera.parallel_projection = True
        # p.reset_camera() # zoom to fit
        self._set_camera_tight(p.camera)

        if filepath is not None:
            p.screenshot(filepath, **write_kwargs)
            p.close()
            return

        return p

