# -*- coding: utf-8 -*-
from itertools import product

import numpy as np
import pyvista as pv
from numpy.typing import NDArray
from scipy.spatial import ConvexHull

from baderkit.core.critical_points import CriticalPoints
from .grid import GridPlotter
from baderkit.plotting.core.defaults import CRIT_COLORS, CONNECTION_COLORS
    
# TODO:
    # get integral paths rather than just connections

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
        
        # critical point settings
        self._show_minima = True
        self._show_saddle1 = True
        self._show_saddle2 = True
        self._show_maxima = True
        
        self._minima_color = pv.Color(CRIT_COLORS.get(0, "#FFFFFF"))
        self._saddle1_color = pv.Color(CRIT_COLORS.get(1, "#FFFFFF"))
        self._saddle2_color = pv.Color(CRIT_COLORS.get(2, "#FFFFFF"))
        self._maxima_color = pv.Color(CRIT_COLORS.get(3, "#FFFFFF"))
        
        self._critical_radii = 0.25
        self._ring_cage_opacity = 0.5

        # edge settings
        self._show_saddle1_minima = False
        self._show_saddle2_maxima = True
        self._show_saddle1_saddle2 = False
        
        self._saddle1_minima_color = pv.Color(CONNECTION_COLORS.get(0,  "#FFFFFF"))
        self._saddle2_maxima_color = pv.Color(CONNECTION_COLORS.get(1,  "#FFFFFF"))
        self._saddle1_saddle2_color = pv.Color(CONNECTION_COLORS.get(2,  "#FFFFFF"))
        
        self._connection_thickness = 3.0
        
        # crit meshes
        self._minima_point_polydata = None
        self._minima_ring_meshes = None
        self._minima_cage_meshes = None
        
        self._saddle1_point_polydata = None
        self._saddle1_ring_meshes = None
        self._saddle1_cage_meshes = None
        
        self._saddle2_point_polydata = None
        self._saddle2_ring_meshes = None
        self._saddle2_cage_meshes = None
        
        self._maxima_point_polydata = None
        self._maxima_ring_meshes = None
        self._maxima_cage_meshes = None
        
        # connection meshes
        self._saddle1_minima_meshes = None
        self._saddle2_maxima_meshes = None
        self._saddle1_saddle2_meshes = None
        
        super().__init__(grid=grid, **grid_kwargs)
        
        # update to hide by default        
        self.show_caps = False
        self.show_surface = False
        
    ###########################################################################
    # Crit Property Setting
    ###########################################################################
    @property
    def show_minima(self) -> bool:
        """

        Returns
        -------
        bool
            whether or not to show minima points

        """
        return self._show_minima

    @show_minima.setter
    def show_minima(self, show_minima: bool):
        # set visible atoms
        self._show_minima = show_minima
        # update
        self._update_critical_meshes()
        
    @property
    def show_saddle1(self) -> bool:
        """

        Returns
        -------
        bool
            whether or not to show saddle1 points

        """
        return self._show_saddle1

    @show_saddle1.setter
    def show_saddle1(self, show_saddle1: bool):
        # set visible atoms
        self._show_saddle1 = show_saddle1
        # update
        self._update_critical_meshes()
        
    @property
    def show_saddle2(self) -> bool:
        """

        Returns
        -------
        bool
            whether or not to show saddle2 points

        """
        return self._show_saddle2

    @show_saddle2.setter
    def show_saddle2(self, show_saddle2: bool):
        # set visible atoms
        self._show_saddle2 = show_saddle2
        # update
        self._update_critical_meshes()
        
    @property
    def show_maxima(self) -> bool:
        """

        Returns
        -------
        bool
            whether or not to show maxima points

        """
        return self._show_maxima

    @show_maxima.setter
    def show_maxima(self, show_maxima: bool):
        # set visible atoms
        self._show_maxima = show_maxima
        # update
        self._update_critical_meshes()
        
    @property
    def minima_color(self) -> pv.Color:
        """

        Returns
        -------
        pv.Color
            The color to display the minima as

        """
        return self._minima_color

    @minima_color.setter
    def minima_color(self, minima_color: pv.Color | str):
        # convert to color
        color = pv.Color(minima_color)
        # set visible atoms
        self._minima_color = color
        # update
        self._update_critical_meshes()
        
    @property
    def saddle1_color(self) -> pv.Color:
        """

        Returns
        -------
        pv.Color
            The color to display the saddle1 as

        """
        return self._saddle1_color

    @saddle1_color.setter
    def saddle1_color(self, saddle1_color: pv.Color | str):
        # convert to color
        color = pv.Color(saddle1_color)
        # set visible atoms
        self._saddle1_color = color
        # update
        self._update_critical_meshes()
        
    @property
    def saddle2_color(self) -> pv.Color:
        """

        Returns
        -------
        pv.Color
            The color to display the saddle2 as

        """
        return self._saddle2_color

    @saddle2_color.setter
    def saddle2_color(self, saddle2_color: pv.Color | str):
        # convert to color
        color = pv.Color(saddle2_color)
        # set visible atoms
        self._saddle2_color = color
        # update
        self._update_critical_meshes()
        
    @property
    def maxima_color(self) -> pv.Color:
        """

        Returns
        -------
        pv.Color
            The color to display the maxima as

        """
        return self._maxima_color

    @maxima_color.setter
    def maxima_color(self, maxima_color: pv.Color | str):
        # convert to color
        color = pv.Color(maxima_color)
        # set visible atoms
        self._maxima_color = color
        # update
        self._update_critical_meshes()
        

    @property
    def critical_radii(self) -> float:
        """

        Returns
        -------
            The radius to display for each critical point in the structure. 
            The actual displayed radius will be 0.3*radius.

        """
        return self._critical_radii

    @critical_radii.setter
    def critical_radii(self, critical_radii: float):
        
        critical_radii = max(critical_radii, 0.1)
        
        self._critical_radii = critical_radii
        self._update_critical_meshes()
        
    @property
    def ring_cage_opacity(self) -> float:
        """

        Returns
        -------
            The opacity of ring and cage meshes

        """
        return self._ring_cage_opacity
        
    @ring_cage_opacity.setter
    def ring_cage_opacity(self, ring_cage_opacity: float):
        
        self._ring_cage_opacity = ring_cage_opacity
        self._update_critical_meshes()
        
    ###########################################################################
    # Connection Mesh Creation
    ###########################################################################
    
    @property
    def show_saddle1_minima(self) -> bool:
        """

        Returns
        -------
        bool
            whether or not to show connections between type1 saddles and minima

        """
        return self._show_saddle1_minima

    @show_saddle1_minima.setter
    def show_saddle1_minima(self, show_saddle1_minima: bool):
        # set visible atoms
        self._show_saddle1_minima = show_saddle1_minima
        # update
        self._update_connection_meshes()
        
    @property
    def show_saddle2_maxima(self) -> bool:
        """

        Returns
        -------
        bool
            whether or not to show connections between type2 saddles and maxima

        """
        return self._show_saddle2_maxima

    @show_saddle2_maxima.setter
    def show_saddle2_maxima(self, show_saddle2_maxima: bool):
        # set visible atoms
        self._show_saddle2_maxima = show_saddle2_maxima
        # update
        self._update_connection_meshes()
        
    @property
    def show_saddle1_saddle2(self) -> bool:
        """

        Returns
        -------
        bool
            whether or not to show connections between type1 and type2 saddles

        """
        return self._show_saddle1_saddle2

    @show_saddle1_saddle2.setter
    def show_saddle1_saddle2(self, show_saddle1_saddle2: bool):
        # set visible atoms
        self._show_saddle1_saddle2 = show_saddle1_saddle2
        # update
        self._update_connection_meshes()
        
    @property
    def saddle1_minima_color(self) -> pv.Color:
        """

        Returns
        -------
        pv.Color
            The color to display saddle1 to minima connections as

        """
        return self._saddle1_minima_color

    @saddle1_minima_color.setter
    def saddle1_minima_color(self, saddle1_minima_color: pv.Color | str):
        # convert to color
        color = pv.Color(saddle1_minima_color)
        # set visible atoms
        self._saddle1_minima_color = color
        # update
        self._update_connection_meshes()
        
    @property
    def saddle2_maxima_color(self) -> pv.Color:
        """

        Returns
        -------
        pv.Color
            The color to display saddle2 to maxima connections as

        """
        return self._saddle2_maxima_color

    @saddle2_maxima_color.setter
    def saddle2_maxima_color(self, saddle2_maxima_color: pv.Color | str):
        # convert to color
        color = pv.Color(saddle2_maxima_color)
        # set visible atoms
        self._saddle2_maxima_color = color
        # update
        self._update_connection_meshes()
        
    @property
    def saddle1_saddle2_color(self) -> pv.Color:
        """

        Returns
        -------
        pv.Color
            The color to display saddle1 to saddle2 connections as

        """
        return self._saddle1_saddle2_color

    @saddle1_saddle2_color.setter
    def saddle1_saddle2_color(self, saddle1_saddle2_color: pv.Color | str):
        # convert to color
        color = pv.Color(saddle1_saddle2_color)
        # set visible atoms
        self._saddle1_saddle2_color = color
        # update
        self._update_connection_meshes()
        
    @property
    def connection_thickness(self) -> float:
        """

        Returns
        -------
        float
            The thickness to use for each connection

        """
        return self._connection_thickness

    @connection_thickness.setter
    def connection_thickness(self, thickness: float):
        self._connection_thickness = thickness
        # TODO: this can probably be updated with out removing/adding actors
        self._update_connection_meshes()
    
    ###########################################################################
    # Critical Point Mesh Creation
    ###########################################################################
    
    @staticmethod
    def _create_polyhull(coords):
        x=coords[:,0]
        y=coords[:,1]
        z=coords[:,2]
        hull = ConvexHull(np.column_stack((x, y, z)))
        faces = np.column_stack(
            (3*np.ones((len(hull.simplices), 1), dtype=int), 
             hull.simplices)
            ).flatten()
        poly = pv.PolyData(hull.points, faces)
        return poly

    @staticmethod
    def _sort_ring_points(coords: np.ndarray):
        center = coords.mean(axis=0)
        X = coords - center
    
        _, _, vh = np.linalg.svd(X, full_matrices=False)
        e1, e2 = vh[0], vh[1]
    
        x = X @ e1
        y = X @ e2
        angles = np.arctan2(y, x)
    
        order = np.argsort(angles)
        return coords[order], angles[order], center, e1, e2
    
    def _smooth_ring_polar(self, coords, window=5, n_iter=2):
        coords, angles, center, e1, e2 = self._sort_ring_points(coords)
    
        # polar radius
        X = coords - center
        r = np.sqrt((X @ e1)**2 + (X @ e2)**2)
    
        # periodic smoothing
        for _ in range(n_iter):
            r = np.convolve(
                np.r_[r[-window:], r, r[:window]],
                np.ones(2*window+1)/(2*window+1),
                mode="valid"
            )
    
        # reconstruct in plane
        x = r * np.cos(angles)
        y = r * np.sin(angles)
    
        smoothed = center + np.outer(x, e1) + np.outer(y, e2)
        return smoothed
    
    @staticmethod
    def _resample_closed_curve(coords, n_points=None):
        if n_points is None:
            n_points = len(coords)
    
        pts = np.vstack([coords, coords[0]])
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.r_[0, np.cumsum(d)]
        s /= s[-1]
    
        t = np.linspace(0, 1, n_points + 1)[:-1]
    
        out = np.empty((n_points, 3))
        for i in range(3):
            out[:, i] = np.interp(t, s, pts[:, i])
        return out
    
    def _create_ring_polyline(self, coords, n_sides=12):
        coords = self._smooth_ring_polar(
            coords,
            window=4,
            n_iter=2
        )
    
        coords = self._resample_closed_curve(coords, len(coords))
    
        n = len(coords)
        points = np.vstack([coords, coords[0]])
    
        lines = np.column_stack((
            np.full(n, 2),
            np.arange(n),
            np.arange(1, n + 1)
        )).astype(np.int64).ravel()
    
        poly = pv.PolyData(points)
        poly.lines = lines
    
        return poly.tube(
            radius=self.critical_radii * 0.1,
            n_sides=n_sides,
            capping=False
        )
    
    def _create_crit_meshes_by_type(self, crit_type):
        point_coords = []
        ring_meshes = []
        cage_meshes = []
        
        for i in self.morse_graph.nodes:
            crit_type1 = self.morse_graph.nodes[i]["type_idx"]
            # skip points that are not part of the requested critical type
            if crit_type != crit_type1:
                continue
            frac_coords = self.morse_graph.nodes[i]["frac_coords"]
            group_type = self.morse_graph.nodes[i]["group_type"]
            group = self.morse_graph.nodes[i]["voxel_group"]

            # get voxel group in fractional coordinates and wrap to be as close
            # as possible
            group_frac = self.grid.grid_to_frac(group)
            ref = frac_coords
            group_frac = group_frac - np.round(group_frac - ref)
            
            if group_type == 0:
                # get shifted coords near edges of cell
                wrapped_coords, shifts = self._wrap_near_edge(frac_coords)
                point_coords.extend(wrapped_coords @ self.structure.lattice.matrix)
            elif group_type == 1:
                # sort and coords
                group_frac = self._smooth_ring_polar(group_frac)
                # resample
                group_frac = self._resample_closed_curve(group_frac)
                # get all shifts with part of the polyhedra in the cell
                shifts = self._wrap_group_near_edge_shifts(group_frac)
                for shift in shifts:
                    # shift voxel group
                    group_frac1 = group_frac + shift
                    # convert to cartesian coordinates
                    group_cart = group_frac1 @ self.structure.lattice.matrix
                    # create mesh
                    mesh = self._create_ring_polyline(group_cart)
                    cage_meshes.append(mesh)
            elif group_type == 2:
                # get all shifts with part of the polyhedra in the cell
                shifts = self._wrap_group_near_edge_shifts(group_frac)
                for shift in shifts:
                    # shift voxel group
                    group_frac1 = group_frac + shift
                    # convert to cartesian coordinates
                    group_cart = group_frac1 @ self.structure.lattice.matrix
                    # create mesh
                    mesh = self._create_polyhull(group_cart)
                    cage_meshes.append(mesh)
        # create polydata from points if any
        if point_coords:
            point_polydata = pv.PolyData(point_coords)
        else:
            point_polydata = False
        # merge meshes if they exist
        if ring_meshes:
            ring_meshes = pv.merge(ring_meshes)
        else:
            ring_meshes = False
            
        if cage_meshes:
            cage_meshes = pv.merge(cage_meshes)
        else:
            cage_meshes = False
        
        return point_polydata, ring_meshes, cage_meshes
    
    def _create_all_crit_meshes(self):
        # set all mesh types
        for i, name in enumerate(("minima", "saddle1", "saddle2", "maxima")):
            point_polydata, ring_meshes, cage_meshes = self._create_crit_meshes_by_type(i)
            setattr(self, f"_{name}_point_polydata", point_polydata)
            setattr(self, f"_{name}_ring_meshes", ring_meshes)
            setattr(self, f"_{name}_cage_meshes", cage_meshes)
    
    def _update_critical_meshes(self):
        if self._minima_point_polydata is None:
            self._create_all_crit_meshes()
        
        # add each set of critical points
        for i, name in enumerate(("minima", "saddle1", "saddle2", "maxima")):
            visible = getattr(self, f"show_{name}")
            color = getattr(self, f"{name}_color")
            
            # remove connections that aren't visible
            if not visible:
                actor0 = self.plotter.actors.get(f"{name}_glyph", None)
                actor1 = self.plotter.actors.get(f"{name}_poly", None)
                if actor0 is not None:
                    self.plotter.remove_actor(actor0)
                if actor1 is not None:
                    self.plotter.remove_actor(actor1)
                continue
            
            point_polydata = getattr(self, f"_{name}_point_polydata")
            ring_meshes = getattr(self, f"_{name}_ring_meshes")
            cage_meshes = getattr(self, f"_{name}_cage_meshes")
            
            
            if point_polydata:
                # make point glyphs
                radii = self.critical_radii * self.radii_scale
                
                # generate glyphs
                glyphs = point_polydata.glyph(
                    geom=self._sphere_mesh, 
                    factor=radii, 
                    orient=False)
                # add to plot
                self.plotter.add_mesh(
                    glyphs,
                    name=f"{name}_glyph",
                    color=color,
                    pbr=self.pbr,
                )

            all_meshes = []
            # append ring/cage meshes if they exist
            if ring_meshes:
                all_meshes.append(ring_meshes)
            if cage_meshes:
                all_meshes.append(cage_meshes)
                
            # if we have any meshes, merge them
            if all_meshes:
                # merge all meshes to reduce total number of actors
                total_mesh = pv.merge(all_meshes)
                
                # add to plotter
                self.plotter.add_mesh(
                    total_mesh,
                    name=f"{name}_crit",
                    color=color,
                    opacity=self.ring_cage_opacity,
                    pbr=self.pbr,
                )
            
    ###########################################################################
    # Connection Mesh Creation
    ###########################################################################
    

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
        wrapped_line_actors = []
        # get line segments
        segments = self._split_and_wrap_line_frac(p0, p1)
        for f0, f1 in segments:
            # convert to cartesian coords
            c0 = self.grid.frac_to_cart(f0)
            c1 = self.grid.frac_to_cart(f1)
            # add line segment
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
        
        if wrapped_line_actors:
            return pv.merge(wrapped_line_actors)
        else:
            return False

    def _create_connection_meshes(self):
        for connection, (crit1, crit2) in zip(
                ("saddle1_minima", "saddle2_maxima", "saddle1_saddle2"),
                ((1,0), (2,3), (1,2)),
                ):
            meshes = []
            for (i,j,_), data_dict in self.morse_graph.edges.items():
                crit_type0=self.morse_graph.nodes[i]["type_idx"]
                crit_type1=self.morse_graph.nodes[j]["type_idx"]
                if crit_type0 != crit1 or crit_type1 != crit2:
                    continue
                
                p0 = data_dict["p0"]
                p1 = data_dict["p1"]
                # get lines with and without wrapping
                wrapped = self._get_connection_mesh(p0,p1)
                if wrapped:
                    meshes.append(wrapped)
                
            # merge meshes and set
            if meshes:
                meshes = pv.merge(meshes)
            else:
                meshes = False
            setattr(self, f"_{connection}_meshes", meshes)

        
    def _update_connection_meshes(self):
        if self._saddle1_minima_meshes is None:
            self._create_connection_meshes()
            
        for connection in ("saddle1_minima", "saddle2_maxima", "saddle1_saddle2"):
            visible = getattr(self, f"show_{connection}")
            color = getattr(self, f"{connection}_color")

            # remove connections that aren't visible
            if not visible:
                actor = self.plotter.actors.get(f"{connection}", None)
                if actor is not None:
                    self.plotter.remove_actor(actor)
                continue
            
            # add mesh
            mesh = getattr(self, f"_{connection}_meshes")

            if mesh:
                self.plotter.add_mesh(
                    mesh,
                    color=color,
                    name=f"{connection}",
                    line_width=self.connection_thickness,
                    render_lines_as_tubes=True,
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
