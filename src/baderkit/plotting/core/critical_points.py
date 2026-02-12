from itertools import product# -*- coding: utf-8 -*-

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
        self._critical_point_radii = [0.75 for s in self.critical_graph.nodes]
        self._critical_point_colors = [CRIT_COLORS.get(self.critical_graph.nodes[cp]["type"], "#FFFFFF") for cp in self.critical_graph.nodes]
        
        self._visible_edges = [i for i in range(len(self.critical_graph.edges))]
        self._edge_thicknesses = [4.0 for s in self.critical_graph.edges]
        self._edge_colors = ["black" for edge in self.critical_graph.edges]
        
        
        self.plotter = self._create_critical_plot()
        
        self.show_caps = False
        self.show_surface = False
        self.visible_atoms = []
    

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
            actor = self.plotter.actors[f"crit_{i}{self._actor_suffix}"]
            if i in visible_critical_points:
                actor.visibility = True
            else:
                actor.visibility = False
        # set visible points
        self._visible_critical_points = visible_critical_points
        
    @property
    def visible_edges(self) -> list[int]:
        """

        Returns
        -------
        list[int]
            A list of critical points to include in the plot

        """
        return self._visible_edges

    @visible_edges.setter
    def visible_edges(self, visible_edges: set[int]):
        # update visibility of critical points
        for i in range(len(self.critical_graph)):
            actor = self.plotter.actors[f"edge_{i}{self._actor_suffix}"]
            if i in visible_edges:
                actor.visibility = True
            else:
                actor.visibility = False
        # set visible points
        self._visible_edges = visible_edges

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
            for suffix in ["", "_wrapped"]:
                actor = self.plotter.actors[f"crit_{crit_idx}{suffix}"]
                actor.prop.color = new_color
        self._critical_point_colors = colors
        
    @property
    def edge_colors(self) -> list[str]:
        """

        Returns
        -------
        list[str]
            The colors to use for each atom as hex codes.

        """
        return self._edge_colors

    @edge_colors.setter
    def edge_colors(self, colors: list[str]):
        # for each site, check if the radius has changed and if it has remove it
        # then remake
        for edge_idx, (old_color, new_color) in enumerate(zip(self.edge_colors, colors)):
            if old_color == new_color:
                continue
            for suffix in ["", "_wrapped"]:
                actor = self.plotter.actors[f"edge_{edge_idx}{suffix}"]
                actor.prop.color = new_color
        self._edge_colors = colors
        
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
            crit_mesh, wrapped_crit_mesh = self.get_crit_mesh(i)
            for suffix, mesh, hide in zip(
                    ["", "_wrapped"], 
                    (crit_mesh, wrapped_crit_mesh),
                    (self.wrap_atoms, ~self.wrap_atoms)
                    ):
                self.plotter.remove_actor(f"crit_{crit_idx}{suffix}")
                actor = self.plotter.add_mesh(
                    mesh,
                    color=color,
                    metallic=self.atom_metallicness,
                    pbr=True,  # enable physical based rendering
                    name=f"crit_{crit_idx}{suffix}",
                )
                if hide or i not in self.visible_critical_points:
                    actor.visibility = False
            
    @property
    def edge_thicknesses(self) -> list[float]:
        """

        Returns
        -------
        list[float]
            The thickness to use for each edge

        """
        return self._edge_thicknesses

    @edge_thicknesses.setter
    def edge_thicknesses(self, thicknesses: list[float]):
        for edge_idx, (old_thickness, new_thickness) in enumerate(zip(self.edge_thicknesses, thicknesses)):
            if old_thickness == new_thickness:
                continue
            for suffix in ["", "_wrapped"]:
                actor = self.plotter.actors[f"edge_{edge_idx}{suffix}"]
                actor.prop.line_width = new_thickness
        self._edge_thicknesses = thicknesses
            
    @property
    def wrap_atoms(self):
        return self._wrap_atoms
            
    @wrap_atoms.setter
    def wrap_atoms(self, wrap_atoms: bool):
        self._edit_wrapped_atoms(wrap_atoms)
        
    def _edit_wrapped_atoms(self, wrap_atoms):
        # call equivalent method for atoms then update critical points
        super()._edit_wrapped_atoms(wrap_atoms)
        if wrap_atoms:
            suffix = "_wrapped"
            false_suffix = ""
        else:
            suffix = ""
            false_suffix = "_wrapped"
        for crit_idx in self.critical_graph.nodes:
            actor = self.plotter.actors[f"crit_{crit_idx}{suffix}"]
            actor1 = self.plotter.actors[f"crit_{crit_idx}{false_suffix}"]
            if crit_idx in self.visible_critical_points:
                actor.visibility = True
            else:
                actor.visibility = False
            actor1.visibility = False
        
    def _create_critical_plot(self) -> pv.Plotter():
        
        if type(self.plotter) == GridPlotter:
            plotter = self.plotter
        else:
            # get initial plotter with structure
            plotter = self._create_grid_plot()

        # add critical points
        crit_meshes, wrapped_crit_meshes = self.get_all_crit_meshes()
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
            if self.wrap_atoms or not crit_idx in self.visible_critical_points:
                actor.visibility = False
        # add atoms with wrapping
        for crit_idx, (atom_mesh, color) in enumerate(
            zip(wrapped_crit_meshes, self.critical_point_colors)
        ):
            actor = plotter.add_mesh(
                atom_mesh,
                color=color,
                metallic=self.atom_metallicness,
                pbr=True,  # enable physical based rendering
                name=f"crit_{crit_idx}_wrapped",
            )
            if not self.wrap_atoms or not crit_idx in self.visible_critical_points:
                actor.visibility = False
                
        # add edges
        unwrapped_meshes, wrapped_meshes, crit_types, edge_starts, edge_ends = self.get_all_edge_meshes()
        for edge_idx, (
                unwrapped_mesh, 
                wrapped_mesh, 
                crit_type, 
                crit0, 
                crit1,
                thickness,
                color
                ) in enumerate(
            zip(
                unwrapped_meshes, 
                wrapped_meshes, 
                crit_types, 
                edge_starts, 
                edge_ends,
                self.edge_thicknesses,
                self.edge_colors
                )
        ):
            # add edges without wrapping
            actor = plotter.add_mesh(
                unwrapped_mesh,
                name=f"edge_{edge_idx}",
                line_width=thickness,
                color=color
            )
            if self.wrap_atoms or not edge_idx in self.visible_edges:
                actor.visibility = False
            # add edges with wrapping
            actor = plotter.add_mesh(
                wrapped_mesh,
                name=f"edge_{edge_idx}_wrapped",
                line_width=thickness,
                color=color
                )
            if not self.wrap_atoms or not edge_idx in self.visible_edges:
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
            A pyvista mesh representing a critical point.

        """
        critical_node = self.critical_graph.nodes[crit_idx]
        radius = self.critical_point_radii[crit_idx]
        frac_coords = critical_node["frac_coords"]
        # get wrapped atoms
        wrapped_coords = self.get_edge_atom_fracs(frac_coords)

        # convert to cart coords
        cart_coords = frac_coords @ self.structure.lattice.matrix
        wrapped_cart_coords = wrapped_coords @ self.structure.lattice.matrix
        
        # get mesh for single atom
        unwrapped_mesh = pv.Sphere(
                radius=radius * 0.3,
                center=cart_coords,
                # theta_resolution=30,
                # phi_resolution=30,
            )

        # generate meshes for wrapped atoms
        spheres = []
        for cart_coord in wrapped_cart_coords:
            spheres.append(
                pv.Sphere(
                    radius=radius * 0.3,
                    center=cart_coord,
                    # theta_resolution=30,
                    # phi_resolution=30,
                )
            )
        # return unwrapped and wrapped meshes
        return unwrapped_mesh, pv.merge(spheres)

    def get_all_crit_meshes(self) -> list[pv.PolyData]:
        """
        Gets a list of pyvista meshes representing the atoms in the structure

        Returns
        -------
        meshes : pv.PolyData
            A list of pyvista meshes representing each atom.

        """
        unwrapped_meshes = []
        wrapped_meshes = []
        for i in range(len(self.critical_graph)):
            mesh, wrapped_mesh = self.get_crit_mesh(i)
            unwrapped_meshes.append(mesh)
            wrapped_meshes.append(wrapped_mesh)

        return unwrapped_meshes, wrapped_meshes

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


    def get_edge_mesh(self, p0, p1, tol=0.01) -> pv.PolyData:
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
            # if both points are very close to the same edges, we also get the
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

    def get_all_edge_meshes(self):
        unwrapped_meshes = []
        wrapped_meshes = []
        edge_types = []
        edge_starts = []
        edge_ends = []
        for i, j, image in self.critical_graph.edges(data="image"):
            edge_type=self.critical_graph.nodes[i]["type"]
            if edge_type not in ["maxima", "minima"]:
                continue
            p0 = self.critical_graph.nodes[i]["frac_coords"]
            p1 = self.critical_graph.nodes[j]["frac_coords"]
            # shift p1 to image
            p1 = p1 + image

            # get lines with and without wrapping
            unwrapped, wrapped = self.get_edge_mesh(p0,p1)
            unwrapped_meshes.append(unwrapped)
            wrapped_meshes.append(wrapped)
            edge_types.append(self.critical_graph.nodes[i]["type"])
            edge_starts.append(i)
            edge_ends.append(j)
            
        return unwrapped_meshes, wrapped_meshes, edge_types, edge_starts, edge_ends
