# -*- coding: utf-8 -*-

import numpy as np
import pyvista as pv

from baderkit.core.critical_points import CriticalPoints
from .grid import GridPlotter
from baderkit.plotting.core.defaults import CRIT_COLORS
    
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
        super().__init__(grid=grid, **grid_kwargs)
        self.critical_points = critical_points
        self.critical_graph = critical_points.morse_graph
        
        self._visible_critical_points = [i for i in range(len(self.critical_graph.nodes))]
        self._critical_point_radii = [0.25 for s in self.critical_graph.nodes]
        self._critical_point_colors = [CRIT_COLORS.get(self.critical_graph.nodes[cp]["type"], "#FFFFFF") for cp in self.critical_graph.nodes]
        
        self.plotter = self._create_critical_plot()
        
        self.show_caps = False
        self.show_surface = False
        self.visible_atoms = []
    # visible critical points
    # visible connections
    # critical radii
    # connection thickness
    # critical colors
    # connection colors
    

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
        # update visibility of critical points
        for i in range(len(self.critical_graph)):
            actor = self.plotter.actors[f"crit_{i}"]
            if i in visible_critical_points:
                actor.visibility = True
            else:
                actor.visibility = False
        # set visible points
        self._visible_critical_points = visible_critical_points

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
        for crit_idx, old_color, new_color in zip(self.critical_points, self.critical_point_colors, colors):
            if old_color == new_color:
                continue
            actor = self.plotter.actors[f"crit_{crit_idx}"]
            actor.prop.color = new_color
        self._critical_point_colors = colors
        
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
    def critical_point_radii(self, critical_point_radii: list[float]):
        # fix critical_point_radii to be a list and make any negative values == 0.01
        critical_point_radii = list(critical_point_radii)
        for i, val in enumerate(critical_point_radii):
            if val <= 0:
                critical_point_radii[i] = 0.01
        # check which critical_point_radii have changed and replace these atoms
        old_critical_point_radii = self.critical_point_radii
        # update critical_point_radii
        self._critical_point_radii = critical_point_radii
        # for each site, check if the radius has changed and if it has remove it
        # then remake
        for i, (crit_idx, old_r, new_r, color) in enumerate(
            zip(
                range(len(self.critical_graph)), 
                old_critical_point_radii, 
                critical_point_radii, 
                self.critical_point_colors,
                )
        ):
            if old_r == new_r:
                continue
            # otherwise remove the actor, regenerate, and replot
            self.plotter.remove_actor(f"crit_{crit_idx}")
            atom_mesh = self.get_crit_mesh(i)
            self.plotter.add_mesh(
                atom_mesh,
                color=color,
                metallic=self.atom_metallicness,
                pbr=True,  # enable physical based rendering
                name=f"crit_{crit_idx}",
            )
        
    def _create_critical_plot(self) -> pv.Plotter():
        
        if type(self.plotter) == GridPlotter:
            plotter = self.plotter
        else:
            # get initial plotter with structure
            plotter = self._create_grid_plot()

        # add critical points
        crit_meshes = self.get_all_crit_meshes()
        for crit_idx, (crit_mesh, color) in enumerate(
            zip(crit_meshes, self.critical_point_colors)
        ):
            actor = plotter.add_mesh(
                crit_mesh,
                color=color,
                metallic=self.atom_metallicness,
                pbr=True,  # enable physical based rendering
                name=f"crit_{crit_idx}",
            )
            if not crit_idx in self.visible_critical_points:
                actor.visibility = False
                
        # add edges
        edge_meshes, crit_types, edge_starts, edge_ends = self.get_all_edge_meshes()
        for edge_idx, (edge_mesh, crit_type, crit0, crit1) in enumerate(
            zip(edge_meshes, crit_types, edge_starts, edge_ends)
        ):
            actor = plotter.add_mesh(
                edge_mesh,
                name=f"edge_{edge_idx}",
            )

        return plotter
    
    def rebuild(self):
        """
        Builds a new pyvista plotter object representing the current state of
        the Plotter class.

        Returns
        -------
        pv.Plotter
            A pyvista Plotter object representing the current state of the
            CriticalPointPlotter class.

        """
        return self._create_critical_plot()
    
    def get_crit_mesh(self, crit_idx: int) -> pv.PolyData:
        """
        Generates a mesh for the provided critical point index.

        Parameters
        ----------
        crit_idx : int
            The index of the critical point to create the mesh for.

        Returns
        -------
        pv.PolyData
            A pyvista mesh representing an atom.

        """
        critical_node = self.critical_graph.nodes[crit_idx]
        radius = self.critical_point_radii[crit_idx]
        frac_coords = critical_node["frac_coords"]
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
                    # theta_resolution=30,
                    # phi_resolution=30,
                )
            )
        # merge all meshes
        return pv.merge(spheres)
    
    def get_all_crit_meshes(self) -> list[pv.PolyData]:
        """
        Gets a list of pyvista meshes representing the atoms in the structure

        Returns
        -------
        meshes : pv.PolyData
            A list of pyvista meshes representing each atom.

        """
        meshes = [self.get_crit_mesh(i) for i in range(len(self.critical_graph))]
        return meshes

    @staticmethod
    def _split_and_wrap_line_frac(p0, p1, image, eps=1e-12):
        """
        Split a periodic line segment into unit-cell-wrapped pieces.
    
        Parameters
        ----------
        p0 : (3,) float
            Start fractional coordinate
        p1 : (3,) float
            End fractional coordinate (in base cell)
        image : (3,) int
            Periodic image vector that p1 connects to
        eps : float
            Numerical tolerance
    
        Returns
        -------
        segments : list of (p0, p1)
            Each p0, p1 are wrapped fractional coords in [0,1)
        """
    
        p1u = p1 + image
        delta = p1u - p0
    
        ts = [0.0, 1.0]
    
        for d in range(3):
            if abs(delta[d]) < eps:
                continue
    
            kmin = int(np.floor(min(p0[d], p1u[d]))) - 1
            kmax = int(np.ceil (max(p0[d], p1u[d]))) + 1
    
            for k in range(kmin, kmax + 1):
                t = (k - p0[d]) / delta[d]
                if eps < t < 1.0 - eps:
                    ts.append(t)
    
        ts = np.array(ts)
        ts.sort()
    
        # Deduplicate
        ts_unique = [ts[0]]
        for t in ts[1:]:
            if abs(t - ts_unique[-1]) > eps:
                ts_unique.append(t)
    
        segments = []
    
        for i in range(len(ts_unique) - 1):
            t0 = ts_unique[i]
            t1 = ts_unique[i + 1]
    
            p0 = p0 + t0 * delta
            p1 = p0 + t1 * delta
    
            # Wrap into central cell
            p0 = p0 % 1.0
            p1 = p1 % 1.0
    
            segments.append((p0, p1))
        # breakpoint()
        return segments


    def get_edge_mesh(self, p0, p1, image) -> pv.PolyData:
        """
        Generates a line mesh from a starting point to an ending point, wrapping
        at periodic boundaries

        """
        segments = self._split_and_wrap_line_frac(p0, p1, image)
        line_actors = []
        for f0, f1 in segments:
            # convert to cartesian coords
            c0 = self.grid.frac_to_cart(f0)
            c1 = self.grid.frac_to_cart(f1)
            # add line segment
            line_actors.append(
                pv.Line(c0, c1)
                )
        
        return pv.merge(line_actors)

    def get_all_edge_meshes(self):
        meshes = []
        edge_types = []
        edge_starts = []
        edge_ends = []
        for i in range(len(self.critical_graph)):
            edge_type=self.critical_graph.nodes[i]["type"]
            if edge_type != "maxima":
                continue
            p0 = self.critical_graph.nodes[i]["frac_coords"]
            for j in self.critical_graph.successors(i):
                # skip points above the parent
                if j < i:
                    continue
                p1 = self.critical_graph.nodes[j]["frac_coords"]
                image = self.critical_graph.edges[i,j,0]["image"]
                # get line
                meshes.append(self.get_edge_mesh(p0,p1,image))
                edge_types.append(self.critical_graph.nodes[i]["type"])
                edge_starts.append(i)
                edge_ends.append(j)
        return meshes, edge_types, edge_starts, edge_ends
