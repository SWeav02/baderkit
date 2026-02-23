from itertools import product# -*- coding: utf-8 -*-

import numpy as np
import pyvista as pv
from numpy.typing import NDArray

from baderkit.core.critical_points import CriticalPoints
from .grid import GridPlotter
from baderkit.plotting.core.defaults import CRIT_COLORS, CONNECTION_COLORS
    
class CriticalPointsPlotter(GridPlotter):
    def __init__(
        self,
        critical_points: CriticalPoints,
        **grid_kwargs,
    ):
        """
        A convenience class for creating plots of critical points and their
        connections using pyvista's package for VTK.

        Parameters
        ----------
        critical_points : CriticalPoints
            The CriticalPoints object to use for generating critical points
            and connections

        Returns
        -------
        None.

        """
        # apply StructurePlotter kwargs
        grid = critical_points.reference_grid
        
        # set initial critical point objects
        self.critical_points = critical_points
        self.morse_graph = critical_points.morse_graph
        
        # set critical coord properties
        self._visible_critical_points = np.array([1.0 for i in range(4)], dtype=float)
        self._critical_point_radii = np.array([0.75 for s in range(4)])
        self._critical_point_colors = np.array([pv.Color(CRIT_COLORS.get(crit_type, "#FFFFFF")).float_rgb for crit_type in range(4)])
        
        # set edge properties
        self._visible_connections = np.array([1.0 for i in range(3)])
        self._connection_thickness = 4.0
        self._connection_colors = np.array([pv.Color(CONNECTION_COLORS.get(crit_type, "#FFFFFF")).float_rgb for crit_type in range(3)])
        
        
        # meshes
        self._wrapped_crit_poly = None
        self._crit_poly = None
        self._map_wrapped_to_crits = None
        self._map_to_crits = None
        self._connection_meshes = None
        self._wrapped_connection_meshes = None
        super().__init__(grid=grid, **grid_kwargs)
        
        # update to hide by default        
        self.show_caps = False
        self.show_surface = False

    @property
    def visible_critical_points(self) -> list[int]:
        """

        Returns
        -------
        list[int]
            A list of critical points to include in the plot

        """
        return self._visible_critical_points

    @visible_critical_points.setter
    def visible_critical_points(self, visible_critical_points: set[int]):
        # convert to array
        visible_critical_points = np.array(visible_critical_points, dtype=np.float64)
        
        # make sure we have the write shape
        assert self.visible_critical_points.shape == (4,), "Length must be 4"

        # set visible atoms
        self._visible_critical_points = visible_critical_points
        # update
        self._update_critical_meshes()
        
    @property
    def visible_connections(self) -> list[int]:
        """

        Returns
        -------
        list[int]
            A list of critical points to include in the plot

        """
        return self._visible_connections

    @visible_connections.setter
    def visible_connections(self, visible_connections: set[int]):
        # convert to array
        visible_connections = np.array(visible_connections, dtype=np.float64)
        
        # make sure we have the write shape
        assert self.visible_connections.shape == (3,), "Length must be 3"

        # set visible connections
        self._visible_connections = visible_connections
        # update
        self._update_connection_meshes()

    @property
    def critical_point_colors(self) -> list[str]:
        """

        Returns
        -------
        list[str]
            The colors to use for each atom as hex codes.

        """
        return self._critical_point_colors

    @critical_point_colors.setter
    def critical_point_colors(self, colors: list[str]):
        # for each site, check if the radius has changed and if it has remove it
        # then remake
        assert len(colors) == (4,), "Length must be 4"
        
        # convert provided colors to rgb
        crit_colors = np.array([pv.Color(i).float_rgb for i in colors], dtype=np.float64)
        self._critical_point_colors = crit_colors
        self._update_critical_meshes()
        
    @property
    def connection_colors(self) -> list[str]:
        """

        Returns
        -------
        list[str]
            The colors to use for each atom as hex codes.

        """
        return self._connection_colors

    @connection_colors.setter
    def connection_colors(self, colors: list[str]):
        # for each site, check if the radius has changed and if it has remove it
        # then remake
        assert len(colors) == (3,), "Length must be 3"
        
        # convert provided colors to rgb
        connection_colors = np.array([pv.Color(i).float_rgb for i in colors], dtype=np.float64)
        self._connection_colors = connection_colors
        self._update_connection_meshes()
        
    @property
    def critical_point_radii(self) -> list[float]:
        """

        Returns
        -------
        list[float]
            The radius to display for each atom in the structure. The actual
            displayed radius will be 0.3*radius.

        """
        return self._critical_point_radii

    @critical_point_radii.setter
    def critical_point_radii(self, critical_point_radii: NDArray[float]):
        # fix atom_radii to be a list and make any negative values == 0.01
        critical_point_radii = np.array(critical_point_radii, dtype=np.float64)
        assert self.critical_point_radii.shape == critical_point_radii.shape, "Length match the number of critical points"
        
        critical_point_radii[critical_point_radii<=0.01] = 0.01
        self._critical_point_radii = critical_point_radii
        self._update_crit_meshes()
            
    @property
    def connection_thickness(self) -> list[float]:
        """

        Returns
        -------
        list[float]
            The thickness to use for each connection

        """
        return self._connection_thickness

    @connection_thickness.setter
    def connection_thickness(self, thickness: float):
        self._connection_thickness = thickness
        # TODO: this can probably be updated with out removing/adding actors
        self._update_connection_meshes()
            
    @property
    def wrap_atoms(self):
        return self._wrap_atoms
            
    @wrap_atoms.setter
    def wrap_atoms(self, wrap_atoms: bool):
        assert type(wrap_atoms) == bool, "wrap_atoms must be a bool"
        
        # update
        self._wrap_atoms = wrap_atoms
        self._update_site_meshes()
        self._update_critical_meshes()   
    

    @staticmethod
    def _split_and_wrap_line_frac(p0, p1, eps=1e-12):
        p0 = np.asarray(p0, float)
        p1 = np.asarray(p1, float)
    
        v = p1 - p0
    
        ts = [0.0, 1.0]
        for i in range(3):
            if abs(v[i]) < eps:
                continue
            for m in range(int(np.floor(min(p0[i], p1[i]))),
                           int(np.ceil (max(p0[i], p1[i])))):
                t = (m - p0[i]) / v[i]
                if eps < t < 1 - eps:
                    ts.append(t)
    
        ts = np.unique(ts)
        ts.sort()
    
        segs = []
        for t0, t1 in zip(ts[:-1], ts[1:]):
            A = p0 + t0 * v
            B = p0 + t1 * v
    
            # choose a *single* periodic image using the midpoint
            mid = 0.5 * (A + B)
            shift = -np.floor(mid)

            segs.append((A + shift, B + shift))
            
        return segs


    def _get_connection_mesh(self, p0, p1, tol=0.01) -> pv.PolyData:
        """
        Generates a line mesh from a starting point to an ending point, wrapping
        at periodic boundaries

        """
        unwrapped_line_actors = []
        wrapped_line_actors = []
        # get line segments
        segments = self._split_and_wrap_line_frac(p0, p1)
        for f0, f1 in segments:
            # convert to cartesian coords
            c0 = self.grid.frac_to_cart(f0)
            c1 = self.grid.frac_to_cart(f1)
            # add line segment
            unwrapped_line_actors.append(
                pv.Line(c0, c1)
                )
            wrapped_line_actors.append(
                pv.Line(c0, c1)
                )
            # if both points are very close to the same connections, we also get the
            # transformed versions
        
            transforms0 = [
                [0, 1] if abs(f) < tol else [0, -1] if abs(f - 1) < tol else [0]
                for f in f0
            ]
            transforms1 = [
                [0, 1] if abs(f) < tol else [0, -1] if abs(f - 1) < tol else [0]
                for f in f1
            ]

            shifts0 = set(product(*transforms0))
            shifts1 = set(product(*transforms1))
            shifts = [np.array(i) for i in shifts0 if i in shifts1]
            for shift in shifts:
                x0 = f0+shift
                x1 = f1+shift
                # convert to cartesian coords
                c0 = self.grid.frac_to_cart(x0)
                c1 = self.grid.frac_to_cart(x1)
                wrapped_line_actors.append(
                    pv.Line(c0, c1)
                    )
        
        return pv.merge(unwrapped_line_actors), pv.merge(wrapped_line_actors)

    def _create_connection_meshes(self):
        unwrapped_meshes = [[] for i in range(3)]
        wrapped_meshes = [[] for i in range(3)]

        for i, j, image in self.morse_graph.edges(data="image"):
            crit_type0=self.morse_graph.nodes[i]["type_idx"]
            crit_type1=self.morse_graph.nodes[j]["type_idx"]
            # only allow connections from lower to higher critical point types
            if crit_type0 >= crit_type1:
                continue
            
            # get fractional coords of each point
            p0 = self.morse_graph.nodes[i]["frac_coords"]
            p1 = self.morse_graph.nodes[j]["frac_coords"]
            # shift p1 to image
            p1 = p1 + image

            # get lines with and without wrapping
            unwrapped, wrapped = self._get_connection_mesh(p0,p1)
            unwrapped_meshes[crit_type0].append(unwrapped)
            wrapped_meshes[crit_type0].append(wrapped)

        self._connection_meshes = unwrapped_meshes
        self._wrapped_connection_meshes = wrapped_meshes
        
    def _update_connection_meshes(self):
        if self._wrapped_connection_meshes is None or self._connection_meshes is None:
            self._create_connection_meshes()
            
        # get appropriate poly data
        if self.wrap_atoms:
            connections = self._wrapped_connection_meshes
        else:
            connections = self._connection_meshes
        
        # get connection colors
        connection_colors = self.connection_colors
        # get visible connections
        visible_connections = self.visible_connections
        
        for idx, (color, visible) in enumerate(zip(connection_colors, visible_connections)):
            if not visible:
                # remove related actors
                actor = self.plotter.actors.get(f"connections_{idx}", None)
                if actor is not None:
                    self.plotter.remove_actor(actor)
                continue
            # update actor
            mesh = pv.merge(connections[idx])

            self.plotter.add_mesh(
                mesh,
                color=color,
                name=f"connections_{idx}",
                render_lines_as_tubes=True,
                line_width=self.connection_thickness,
            )

    def _create_critical_polydata(self):
        wrapped_crit_coords = []
        crit_coords = []
        corresponding_sites = []
        wrapped_corresponding_sites = []
        
        for i in self.morse_graph.nodes:
            crit_type = self.morse_graph.nodes[i]["type_idx"]
            frac_coords = self.morse_graph.nodes[i]["frac_coords"]
            crit_coords.append(frac_coords @ self.structure.lattice.matrix)
            corresponding_sites.append(crit_type)
            # get wrapped coords as well
            wrapped_coords, shifts = self._wrap_near_edge(frac_coords)
            cart_coords = wrapped_coords @ self.structure.lattice.matrix
            # add to lists
            wrapped_crit_coords.extend(cart_coords)
            for j, shift in enumerate(shifts):
                wrapped_corresponding_sites.append(crit_type)

        # save the map from wrapped points to their original site
        self._map_wrapped_to_crits = np.array(wrapped_corresponding_sites, dtype=int)
        self._map_to_crits = np.array(corresponding_sites, dtype=int)
        
        # create poly data of points for both the atoms with and without wrapping
        self._crit_poly = pv.PolyData(crit_coords)
        self._wrapped_crit_poly = pv.PolyData(wrapped_crit_coords)

    def _update_critical_meshes(self):
        if self._wrapped_crit_poly is None or self._crit_poly is None:
            self._create_critical_polydata()
            
        # get appropriate poly data
        if self.wrap_atoms:
            crit_map = self._map_wrapped_to_crits
            # get poly data including wrapped atoms
            crits = self._wrapped_crit_poly
        else:
            crit_map = self._map_to_crits
            # get poly data including wrapped atoms
            crits = self._crit_poly
        
        # get crit colors
        crit_colors = self.critical_point_colors[crit_map]
        # get alpha values
        alpha = self.visible_critical_points[crit_map]
        
        # update poly data scalars
        crits["crit_colors"] = np.column_stack((crit_colors, alpha))
        crits["crit_radii"] = self.critical_point_radii[crit_map] * self.radii_scale
        
        # generate glyphs
        glyphs = crits.glyph(geom=self._sphere_mesh, scale="crit_radii", orient=False)

        # add the crit glyphs to our plotter. This automatically overwrites any
        # previous meshes
        self.plotter.add_mesh(
            glyphs,
            scalars="crit_colors",
            rgb=True,
            name="crit_glyphs",
            pbr=True,
        )

    def _create_plot(self) -> pv.Plotter:
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
        
        # add site meshes
        self._update_critical_meshes()
        self._update_connection_meshes()

        return plotter