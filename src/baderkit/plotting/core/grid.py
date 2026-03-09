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
        show_surface = True,
        show_caps = True,
        surface_opacity = 0.8,
        cap_opacity = 0.8,
        colormap = "viridis",
        use_solid_surface_color = False,
        use_solid_cap_color = False,
        surface_color = "#BA8E23",
        cap_color = "#BA8E23",
        iso_value = None,
        **structure_kwargs,
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

        self.grid = grid
        self._show_surface = show_surface
        self._show_caps = show_caps
        self._surface_opacity = surface_opacity
        self._cap_opacity = cap_opacity
        self._colormap = colormap
        self._use_solid_surface_color = use_solid_surface_color
        self._use_solid_cap_color = use_solid_cap_color
        self._surface_color = pv.Color(surface_color)
        self._cap_color = pv.Color(cap_color)


        self._min_val = grid.total.min() + 0.0000001
        self._max_val = grid.total.max()
        if iso_value is None:
            self._iso_value = (self._min_val  + self._max_val) / 4
        else:
            self._iso_value = max(self.min_val, min(iso_value, self.max_val))
        
        # meshes
        self._surface_mesh = None
        self._cap_mesh = None
        self._slice_meshes = {}
        self._slice_planes = {}
        self._slice_hkls = {}
        
        # apply StructurePlotter kwargs
        super().__init__(structure=grid.structure, **structure_kwargs)
        

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
        if show_surface != self.show_surface:
            self._show_surface = show_surface
            self._update_surface_actor()

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
        if show_caps != self.show_caps:
            self._show_caps = show_caps
            self._update_cap_actor()

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
        if surface_opacity == self.surface_opacity:
            return
        actor = self.plotter.actors.get("iso", None)
        if actor is not None:
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
        if cap_opacity == self.cap_opacity:
            return
        actor = self.plotter.actors.get("cap", None)
        if actor is not None:
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
        if colormap == self.colormap:
            return
        
        # update settings
        self._colormap = colormap
        self._update_clims_cmaps()

    @property
    def min_val(self) -> str:
        return self._min_val

    @min_val.setter
    def min_val(self, value: float):
        if value == self.min_val:
            return

        # update settings
        self._min_val = min(self.max_val, value)
        self._update_clims_cmaps()

    @property
    def max_val(self) -> str:
        return self._max_val

    @max_val.setter
    def max_val(self, value: float):
        if value == self.max_val:
            return

        # update settings
        self._max_val = max(self.min_val, value)
        self._update_clims_cmaps()

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
        if use_solid_surface_color != self.use_solid_surface_color:   
            # update property
            self._use_solid_surface_color = use_solid_surface_color
            self._update_surface_actor()

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
        if use_solid_cap_color != self.use_solid_cap_color:   
            # update property
            self._use_solid_cap_color = use_solid_cap_color
            self._update_cap_actor()

    @property
    def surface_color(self) -> pv.Color:
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
        color = pv.Color(surface_color)
        if color == self.surface_color:
            return
        self._surface_color = color
        actor = self.plotter.actors.get("iso", None)
        if actor is not None:
            actor.prop.color = color

    @property
    def cap_color(self) -> pv.Color:
        """

        Returns
        -------
        str
            The color to use for the caps as a hex string. This is ignored if
            the caps are not set to use solid colors.

        """
        return self._cap_color

    @cap_color.setter
    def cap_color(self, cap_color: str | NDArray):
        color = pv.Color(cap_color)
        if color == self.cap_color:
            return
        self._cap_color = color
        actor = self.plotter.actors.get("cap", None)
        if actor is not None:
            actor.prop.color = color

    @property
    def iso_value(self) -> float:
        """

        Returns
        -------
        float
            The value to set the isosurface to.

        """
        return self._iso_value

    @iso_value.setter
    def iso_value(self, iso_value: float):
        if iso_value == self.iso_value:
            return
        # make sure iso value is within range
        iso_value = max(self.min_val, min(iso_value, self.max_val))
        # update
        self._iso_value = iso_value
        self._update_grid_meshes()
        self._update_surface_actor()
        self._update_cap_actor()

    
    def _create_plot(self) -> pv.Plotter():
        """
        Generates a pyvista.Plotter object from the current class variables.
        This is called when the class is first instanced and generally shouldn't
        be called again.

        Returns
        -------
        plotter : pv.Plotter
            A pyvista Plotter object representing the provided Structure object.

        """
        plotter = super()._create_plot()

        # generate initial surface meshes
        self._make_structured_grid()
        self._update_grid_meshes()
        self._update_surface_actor()
        self._update_cap_actor()
        for key, (hkl, d) in self._slice_hkls.items():
            self.add_slice(hkl, d, key)
        return plotter

    
    
    def _update_surface_actor(self):
        surface_kwargs = {
            "opacity": self.surface_opacity,
            "pbr": self.pbr,
            "name": "iso",
            "color": self.surface_color,
            }

        if not self.use_solid_surface_color:
            surface_kwargs.update({
                "scalars": "values",
                "clim": [self.min_val, self.max_val],
                "show_scalar_bar": False,
                "colormap": self.colormap
                })
        
        if self.show_surface and len(self._surface_mesh["values"]) > 0:
            self.plotter.add_mesh(self._surface_mesh, **surface_kwargs)
        else:
            actor = self.plotter.actors.get("iso", None)
            if actor is not None:
                self.plotter.remove_actor(actor)
        
            
    def _update_cap_actor(self):
        cap_kwargs = {
            "opacity": self.cap_opacity,
            "pbr": self.pbr,
            "name": "cap",
            "color": self.cap_color,
            }
        if not self.use_solid_cap_color:
            cap_kwargs.update({
                "scalars": "values",
                "clim": [self.min_val, self.max_val],
                "show_scalar_bar": False,
                "colormap": self.colormap
                })
        
        if self.show_caps and len(self._cap_mesh["values"]) > 0:
                self.plotter.add_mesh(self._cap_mesh, **cap_kwargs)
        else:
            actor = self.plotter.actors.get("cap", None)
            if actor is not None:
                self.plotter.remove_actor(actor)
            
    def _update_grid_meshes(self):
        """
        Updates the surface meshes to the provided iso_value


        Returns
        -------
        None.

        """
        
        self._surface_mesh = self._structured_grid.contour([self.iso_value]).triangulate()
        self._cap_mesh = self._surface.contour_banded(
            2, rng=[self.iso_value, self.max_val], generate_contour_edges=False
        ).triangulate()

    def _update_clims_cmaps(self):
        actors = ["iso", "cap"]
        actors.extend([f"slice_{i}" for i in self._slice_meshes.keys()])
        for actor_str in actors:
            actor = self.plotter.actors.get(actor_str, None)
            if actor is not None:
                actor.prop.clim = (self.min_val, self.max_val)
                actor.prop.cmap = self.colormap

    def _make_structured_grid(self) -> pv.StructuredGrid:
        """
        Creates a pyvista StructuredGrid object for making isosurfaces. This
        should generally only be called once
        
        Returns
        -------
        structured_grid : pv.StructuredGrid
            A pyvista StructuredGrid with values representing the grid data.

        """

        grid = self.grid
        values = np.pad(grid.total, pad_width=((0, 1), (0, 1), (0, 1)), mode="wrap")
        shape = values.shape
        indices = np.indices(shape).reshape(3, -1, order="F").T
        points = grid.grid_to_cart(indices)

        # create structured grid
        structured_grid = pv.StructuredGrid()
        structured_grid.points = points
        structured_grid.dimensions = shape
        structured_grid["values"] = values.ravel(order="F")
        
        # save and extract surface
        self._structured_grid = structured_grid
        self._surface = structured_grid.extract_surface()
        

    def add_slice(
        self,
        hkl: NDArray,
        d: float=1.0,
        key=None,
            ):
        """
        Adds a slice of the grid to the plot. If a key is provided, this updates
        the corresponding slice rather than adding a new one.

        Parameters
        ----------
        hkl : NDArray
            The miller indices of the plane
        d : float
            The multiplier for the d-spacing of the plane
        key : int, optional
            A integer key for an existing plane to update. The default is None.

        """
        
        if key is not None:
            name = f"slice_{key}"
            assert name in self._slice_meshes.keys(), "Key must correspond to an existing slice"
        else:
            if len(self._slice_meshes.keys()) > 0:
                idx = max(list(self._slice_meshes.keys())) + 1
            else:
                idx = 0
            name = f"slice_{idx}"
            
        h, k, l = hkl
        # get normal vector in cart coords
        normal = self.structure.get_cart_from_miller(h, k, l)
        n = self.structure.lattice.d_hkl(hkl)
        origin = normal * n * d
        slice_plane = self._structured_grid.slice(normal=normal, origin=origin)
            
        self._slice_meshes[name] = slice_plane
        self._slice_planes[name] = (origin, normal)
        self._slice_hkls[name] = (hkl, d)
        # get key if no
        # create plotter
        self.plotter.add_mesh(
            slice_plane,
            scalars="values",
            cmap=self.colormap,
            clim=(self.min_val, self.max_val),
            show_scalar_bar=False,
            name=name
        )
        
    def remove_slice(self, key):
        name = f"slice_{key}"
        if name in self._slice_meshes.keys():
            del(self._slice_meshes[name])
            del(self._slice_planes[name])
            del(self._slice_hkls[name])
        actor = self.plotter.actors.get(name, None)
        if actor is not None:
            self.plotter.remove_actor(actor)

    def plot_slice(
        self,
        key,
        include_atoms: bool = True,
        filename: Path = None,
        **write_kwargs,
    ):
        """
        Generates a pyvista plot of a slice at the requested miller plane. If
        a filename is provided, the plot is written and no plot object is returned.
    
        Parameters
        ----------
        key : int
            The key of the plane to plot
        include_atoms : bool, optional
            Whether or not atoms should be incuded. Only atoms whose sphere mesh
            is sliced by the plane are included. The default is True.
        filename : Path, optional
            The filename to write the plot to if desired. The default is None.
        **write_kwargs
            any additional keyword arguments to provide to the plot writer.
    
        Returns
        -------
        p : pv.plotter | None
            the pyvista plot of the slice or None if a filename was provided.
    
        """
        if key is not None:
            name = f"slice_{key}"
            assert name in self._slice_meshes.keys(), "Key must correspond to an existing slice"
        # create plotter
        mesh = self._slice_meshes[name]
        
        p = StructurePlotter(
            structure=self.structure,
            off_screen=True,
            show_axes=False,
            show_lattice=False,
            )
        p.plotter.add_mesh(
            mesh,
            scalars="values",
            cmap=self.colormap,
            clim=(self.min_val, self.max_val),
            show_scalar_bar=False,
        )

        origin, normal = self._slice_planes[name]
        # if desired, add any atoms that sit on/near the plane
        if include_atoms:
            # get wrapped atom points
            atom_poly = p._wrapped_atom_poly
            points = atom_poly.points
            include_coords = np.zeros(len(points), dtype=np.bool_)
            for wrap_idx, (atom_idx, center) in enumerate(zip(self._map_wrapped_to_atoms, points)):

                radius = self.atom_radii[atom_idx]*self.radii_scale
                dist = np.dot(center - origin, normal)
                if abs(dist) >= radius:
                    continue
                # otherwise add 
                include_coords[wrap_idx] = True

            # get atom colors
            atom_colors = self.atom_colors[self._map_wrapped_to_atoms]
            # get alpha values
            alpha = self.visible_atoms[self._map_wrapped_to_atoms]
            # set alpha to zero at unwanted atoms
            alpha[~include_coords] = 0.0
            # update poly data scalars
            atom_poly["atom_colors"] = np.column_stack((atom_colors, alpha))
            atom_poly["atom_radii"] = self.atom_radii[self._map_wrapped_to_atoms] * self.radii_scale
            
            # generate glyphs
            glyphs = atom_poly.glyph(geom=self._sphere_mesh, scale="atom_radii", orient=False)

            # add the atom glyphs to our plotter. This automatically overwrites any
            # previous meshes
            p.plotter.add_mesh(
                glyphs,
                scalars="atom_colors",
                rgb=True,
                name="atom_glyphs",
                pbr=self.pbr,
            )
        else:
            # otherwise, remove all atoms from the plot
            visible = p.visible_atoms
            visible[:] = 0.0
            p.visible_atoms=visible
        
        # set camera to be perpendicular
        p.set_camera_to_vector(origin=origin, normal=normal)
        p._set_camera_tight()
    
        if filename is not None:
            p.get_plot_screenshot(filename=filename, **write_kwargs)
        else:
            image= p.get_plot_screenshot(return_image=True, **write_kwargs)
            return image