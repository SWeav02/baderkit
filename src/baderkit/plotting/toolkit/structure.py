# -*- coding: utf-8 -*-

import logging

import numpy as np
import pandas as pd
import pyvista as pv
from numpy.typing import NDArray
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius

from baderkit.toolkit.structure import Structure

from ..base import VtkPlotter
from ..base.defaults import ATOM_COLORS


class StructurePlotter(VtkPlotter):
    """
    A convenience class for creating plots of crystal structures using
    pyvista's package for VTK.

    Parameters
    ----------
    structure : Structure
        The pymatgen Structure object to plot.

    """

    def __init__(
        self,
        structure: Structure,
        wrap_atoms=True,
        atom_metallicness=0.0,
        atom_roughness=0.25,
        radii_scale=0.3,
        sphere_resolution: int = 40,
        atom_radii=None,
        atom_colors=None,
        show_bonds: bool = False,
        bond_radius: float = 0.08,
        bond_cutoff_scale: float = 1.2,
        bond_color: str | None = None,
        bond_resolution: int = 20,
        bond_pairs: list[tuple[int, int]] | None = None,
        **kwargs,
    ):

        # create initial class variables
        self._atom_opacities = np.array([1 for i in range(len(structure))], dtype=float)
        self._wrap_atoms = wrap_atoms
        self._atom_metallicness = atom_metallicness
        self._atom_roughness = atom_roughness
        self._radii_scale = radii_scale
        self._sphere_resolution = max(6, int(sphere_resolution))
        self._show_bonds = show_bonds
        self._bond_radius = float(bond_radius)
        self._bond_cutoff_scale = float(bond_cutoff_scale)
        self._bond_color = pv.Color(bond_color) if bond_color is not None else None
        self._bond_resolution = max(6, int(bond_resolution))
        self._bond_pairs = bond_pairs
        radii = []
        for s in structure:
            try:
                radius = s.specie.atomic_radius
            except:
                radius = 1.0
            radii.append(radius)
        self._atom_radii = np.array(radii)
        self._atom_colors = np.array(
            [ATOM_COLORS.get(s.specie.symbol, (1.00, 1.00, 1.00)) for s in structure]
        )

        # atom poly data
        self._map_wrapped_to_atoms = None
        self._atom_poly = None
        self._wrapped_atom_poly = None
        self._sphere_mesh = pv.Sphere(
            radius=1.0,
            theta_resolution=self._sphere_resolution,
            phi_resolution=self._sphere_resolution,
        )
        # generate initial plotter
        super().__init__(structure=structure, **kwargs)
        if atom_radii is not None:
            try:
                self.atom_radii = atom_radii
            except:
                logging.info("Improper atom radii provided. Defaults will be used")
        if atom_colors is not None:
            try:
                self.atom_colors = atom_colors
            except:
                logging.info("Improper atom colors provided. Defaults will be used")

    ###########################################################################
    # Properties and Setters
    ###########################################################################
    @property
    def atom_opacities(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            Whether or not each atom is visible. This is actually the opacity
            if desired.

        """
        return self._atom_opacities

    @atom_opacities.setter
    def atom_opacities(self, atom_opacities: NDArray[float]):

        # convert to array
        atom_opacities = np.array(atom_opacities, dtype=np.float64)

        # make sure we have the write shape
        assert (
            self.atom_opacities.shape == atom_opacities.shape
        ), "Length match the number of atoms"

        # set visible atoms
        self._atom_opacities = atom_opacities
        # update
        self._update_site_meshes()

    @property
    def wrap_atoms(self):
        return self._wrap_atoms

    @wrap_atoms.setter
    def wrap_atoms(self, wrap_atoms: bool):
        assert type(wrap_atoms) == bool, "wrap_atoms must be a bool"

        # update
        self._wrap_atoms = wrap_atoms
        self._update_site_meshes()

    @property
    def atom_radii(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The radius to display for each atom in the structure. The actual
            displayed radius will be radii_scale*radius.

        """
        return self._atom_radii

    @atom_radii.setter
    def atom_radii(self, atom_radii: NDArray[float]):
        # fix atom_radii to be a list and make any negative values == 0.01
        atom_radii = np.array(atom_radii, dtype=np.float64)
        assert (
            self.atom_radii.shape == atom_radii.shape
        ), "Length match the number of atoms"

        atom_radii[atom_radii <= 0.01] = 0.01
        self._atom_radii = atom_radii
        self._update_site_meshes()

    @property
    def radii_scale(self) -> float:
        """

        Returns
        -------
        float
            A constant to multiply atom radii by

        """
        return self._radii_scale

    @radii_scale.setter
    def radii_scale(self, radii_scale: float):
        # ensure scale is not at or below zero
        radii_scale = float(radii_scale)
        radii_scale = max(radii_scale, 0.1)
        # update
        self._radii_scale = radii_scale
        self._update_site_meshes()

    @property
    def atom_colors(self) -> list[str]:
        """

        Returns
        -------
        list[str]
            The atom_colors to use for each atom as hex codes.

        """
        return self._atom_colors

    @atom_colors.setter
    def atom_colors(self, atom_colors: NDArray[float | str]):
        # for each site, check if the radius has changed and if it has remove it
        # then remake
        assert len(self.atom_colors) == len(
            atom_colors
        ), "Length match the number of atoms"

        # convert provided colors to rgb
        atom_colors = np.array(
            [pv.Color(i).float_rgb for i in atom_colors], dtype=np.float64
        )
        self._atom_colors = atom_colors
        self._update_site_meshes()

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
        # update all atoms and bonds
        for actor_name in ("atom_glyphs", "bonds"):
            actor = self.plotter.actors.get(actor_name, None)
            if actor is not None:
                actor.prop.metallic = atom_metallicness
        self._atom_metallicness = atom_metallicness

    @property
    def atom_roughness(self) -> float:
        """

        Returns
        -------
        float
            The amount of roughness in the atom display.

        """
        return self._atom_roughness

    @atom_roughness.setter
    def atom_roughness(self, atom_roughness: float):
        # update all atoms and bonds
        for actor_name in ("atom_glyphs", "bonds"):
            actor = self.plotter.actors.get(actor_name, None)
            if actor is not None:
                actor.prop.roughness = atom_roughness
        self._atom_roughness = atom_roughness

    @property
    def sphere_resolution(self) -> int:
        """Theta/phi angular resolution of the atom sphere mesh."""
        return self._sphere_resolution

    @sphere_resolution.setter
    def sphere_resolution(self, resolution: int):
        resolution = max(6, int(resolution))
        if resolution != self._sphere_resolution:
            self._sphere_resolution = resolution
            self._sphere_mesh = pv.Sphere(
                radius=1.0,
                theta_resolution=resolution,
                phi_resolution=resolution,
            )
            self._update_site_meshes()

    @property
    def show_bonds(self) -> bool:
        """Whether to display covalent bond bars between atoms."""
        return self._show_bonds

    @show_bonds.setter
    def show_bonds(self, show_bonds: bool):
        if show_bonds != self._show_bonds:
            self._show_bonds = show_bonds
            self._update_bond_meshes()

    @property
    def bond_radius(self) -> float:
        """Radius of the bond cylinders in Angstroms."""
        return self._bond_radius

    @bond_radius.setter
    def bond_radius(self, bond_radius: float):
        bond_radius = max(0.01, float(bond_radius))
        if bond_radius != self._bond_radius:
            self._bond_radius = bond_radius
            if self._show_bonds:
                self._update_bond_meshes()

    @property
    def bond_cutoff_scale(self) -> float:
        """Scale factor multiplied by the sum of covalent radii to detect bonds."""
        return self._bond_cutoff_scale

    @bond_cutoff_scale.setter
    def bond_cutoff_scale(self, scale: float):
        scale = max(0.1, float(scale))
        if scale != self._bond_cutoff_scale:
            self._bond_cutoff_scale = scale
            if self._show_bonds:
                self._update_bond_meshes()

    @property
    def bond_color(self) -> pv.Color | None:
        """Uniform color for bond bars, or None for two-tone atom-colored bonds."""
        return self._bond_color

    @bond_color.setter
    def bond_color(self, color: str | None):
        parsed = pv.Color(color) if color is not None else None
        if parsed != self._bond_color:
            self._bond_color = parsed
            if self._show_bonds:
                self._update_bond_meshes()

    @property
    def bond_resolution(self) -> int:
        """Radial facet resolution for bond cylinders."""
        return self._bond_resolution

    @bond_resolution.setter
    def bond_resolution(self, resolution: int):
        resolution = max(6, int(resolution))
        if resolution != self._bond_resolution:
            self._bond_resolution = resolution
            if self._show_bonds:
                self._update_bond_meshes()

    @property
    def bond_pairs(self) -> list[tuple[int, int]] | None:
        """Optional explicit list of bonded site-index pairs."""
        return self._bond_pairs

    @bond_pairs.setter
    def bond_pairs(self, pairs: list[tuple[int, int]] | None):
        self._bond_pairs = pairs
        if self._show_bonds:
            self._update_bond_meshes()

    @property
    def atom_df(self) -> pd.DataFrame:
        """

        Returns
        -------
        atom_df : TYPE
            A dataframe summarizing the properties of the atom meshes.

        """
        # construct a pandas dataframe for each atom
        visible = self.atom_opacities > 0
        atom_df = pd.DataFrame(
            {
                "Label": self.structure.labels,
                "Visible": visible,
                "Color": self.atom_colors,
                "Radius": self.atom_radii,
            }
        )
        return atom_df

    @atom_df.setter
    def atom_df(self, atom_df: pd.DataFrame):
        # set each property from the dataframe
        self.atom_opacities = atom_df["Visible"]
        self.atom_colors = atom_df["Color"]
        self.atom_radii = atom_df["Radius"]

    def _create_atom_polydata(self):
        wrapped_atom_coords = []
        corresponding_sites = []

        for i in range(len(self.structure)):
            frac_coords = self.structure.frac_coords[i]
            # get wrapped atoms as well
            wrapped_coords, shifts = self._wrap_near_edge(frac_coords)
            cart_coords = wrapped_coords @ self.structure.lattice.matrix
            # add to lists
            wrapped_atom_coords.extend(cart_coords)
            for j, shift in enumerate(shifts):
                corresponding_sites.append(i)

        # save the map from wrapped points to their original site
        self._map_wrapped_to_atoms = np.array(corresponding_sites, dtype=int)
        # create poly data of points for both the atoms with and without wrapping
        self._atom_poly = pv.PolyData(self.structure.cart_coords)
        self._wrapped_atom_poly = pv.PolyData(wrapped_atom_coords)

    def _update_site_meshes(self):
        if self._wrapped_atom_poly is None or self._atom_poly is None:
            self._create_atom_polydata()

        # get appropriate poly data
        if self.wrap_atoms:
            # get poly data including wrapped atoms
            atoms = self._wrapped_atom_poly
            # get atom colors
            atom_colors = self.atom_colors[self._map_wrapped_to_atoms]
            # get alpha values
            alpha = self.atom_opacities[self._map_wrapped_to_atoms]
            # get radii
            radii = self.atom_radii[self._map_wrapped_to_atoms] * self.radii_scale
        else:
            # get poly data without wrapped atoms
            atoms = self._atom_poly
            # get atom colors
            atom_colors = self.atom_colors
            # get alpha values
            alpha = self.atom_opacities
            # get radii
            radii = self.atom_radii * self.radii_scale

        sphere = self._sphere_mesh
        # add alpha to colors
        colors = np.column_stack((atom_colors, alpha))

        # update poly data scalars
        atoms["atom_colors"] = colors
        atoms["atom_radii"] = radii

        # generate glyphs
        glyphs = atoms.glyph(geom=sphere, scale="atom_radii", orient=False)

        # add the atom glyphs to our plotter. This automatically overwrites any
        # previous meshes
        self.plotter.add_mesh(
            glyphs,
            scalars="atom_colors",
            rgb=True,
            name="atom_glyphs",
            pbr=self.pbr,
            smooth_shading=True,
            metallic=self.atom_metallicness,
            roughness=self.atom_roughness,
        )
        if self.show_bonds:
            self._update_bond_meshes()

    def _get_covalent_radii(self) -> NDArray[float]:
        radii = []
        for s in self.structure:
            symbol = s.specie.symbol
            r = CovalentRadius.radius.get(symbol, None)
            if r is None:
                try:
                    r = float(s.specie.atomic_radius)
                except Exception:
                    r = 1.0
            radii.append(float(r))
        return np.array(radii, dtype=np.float64)

    def _find_bond_segments(self) -> list[tuple[NDArray, NDArray, NDArray, NDArray]]:
        if self._wrapped_atom_poly is None or self._atom_poly is None:
            self._create_atom_polydata()

        if self.wrap_atoms:
            coords = np.asarray(self._wrapped_atom_poly.points, dtype=np.float64)
            site_ids = self._map_wrapped_to_atoms
        else:
            coords = np.asarray(self._atom_poly.points, dtype=np.float64)
            site_ids = np.arange(len(self.structure), dtype=int)

        opacities = self.atom_opacities[site_ids]
        visible_idx = np.where(opacities > 0)[0]
        if len(visible_idx) < 2:
            return []

        vis_coords = coords[visible_idx]
        vis_sites = site_ids[visible_idx]
        vis_colors = self.atom_colors[vis_sites]

        diff = vis_coords[:, None, :] - vis_coords[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)

        segments = []
        if self.bond_pairs is not None:
            pair_set = {(min(a, b), max(a, b)) for a, b in self.bond_pairs}
            n = len(visible_idx)
            for i in range(n):
                for j in range(i + 1, n):
                    key = (min(int(vis_sites[i]), int(vis_sites[j])), max(int(vis_sites[i]), int(vis_sites[j])))
                    if key in pair_set and dists[i, j] > 1e-6:
                        segments.append(
                            (vis_coords[i], vis_coords[j], vis_colors[i], vis_colors[j])
                        )
        else:
            cov_r = self._get_covalent_radii()[vis_sites]
            cutoffs = self.bond_cutoff_scale * (cov_r[:, None] + cov_r[None, :])
            ii, jj = np.where(
                np.triu((dists > 1e-6) & (dists <= cutoffs), k=1)
            )
            for i, j in zip(ii, jj):
                segments.append(
                    (vis_coords[i], vis_coords[j], vis_colors[i], vis_colors[j])
                )

        return segments

    def _update_bond_meshes(self):
        if not self.show_bonds:
            actor = self.plotter.actors.get("bonds", None)
            if actor is not None:
                self.plotter.remove_actor(actor)
            return

        segments = self._find_bond_segments()
        if not segments:
            actor = self.plotter.actors.get("bonds", None)
            if actor is not None:
                self.plotter.remove_actor(actor)
            return

        cylinders = []
        use_two_tone = self.bond_color is None
        for p1, p2, c1, c2 in segments:
            vec = p2 - p1
            length = float(np.linalg.norm(vec))
            if length < 1e-6:
                continue
            mid = 0.5 * (p1 + p2)
            if use_two_tone:
                cyl1 = pv.Cylinder(
                    center=0.5 * (p1 + mid),
                    direction=vec,
                    radius=self.bond_radius,
                    height=0.5 * length,
                    resolution=self.bond_resolution,
                )
                cyl1["bond_colors"] = np.tile(c1, (cyl1.n_points, 1))
                cyl2 = pv.Cylinder(
                    center=0.5 * (mid + p2),
                    direction=vec,
                    radius=self.bond_radius,
                    height=0.5 * length,
                    resolution=self.bond_resolution,
                )
                cyl2["bond_colors"] = np.tile(c2, (cyl2.n_points, 1))
                cylinders.extend((cyl1, cyl2))
            else:
                cyl = pv.Cylinder(
                    center=mid,
                    direction=vec,
                    radius=self.bond_radius,
                    height=length,
                    resolution=self.bond_resolution,
                )
                cylinders.append(cyl)

        bond_mesh = pv.merge(cylinders)
        mesh_kwargs = {
            "name": "bonds",
            "pbr": self.pbr,
            "smooth_shading": True,
            "metallic": self.atom_metallicness,
            "roughness": self.atom_roughness,
        }
        if use_two_tone:
            mesh_kwargs.update({"scalars": "bond_colors", "rgb": True})
        else:
            mesh_kwargs["color"] = self.bond_color

        self.plotter.add_mesh(bond_mesh, **mesh_kwargs)

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

        # add site meshes (and bonds if enabled)
        self._update_site_meshes()
        self._update_bond_meshes()

        return plotter
