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
                    theta_resolution=30,
                    phi_resolution=30,
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

