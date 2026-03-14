# -*- coding: utf-8 -*-

import logging

import numpy as np
import pandas as pd
import pyvista as pv
from numpy.typing import NDArray

from baderkit.core import Structure
from baderkit.plotting.core.defaults import ATOM_COLORS

from .base import VtkPlotter


class StructurePlotter(VtkPlotter):

    def __init__(
        self,
        structure: Structure,
        wrap_atoms=True,
        atom_metallicness=0.0,
        radii_scale=0.3,
        atom_radii=None,
        atom_colors=None,
        **kwargs,
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
        qt_plotter : bool, optional
            Whether or not the plotter will use pyvistaqt for qt applications
        qt_frame
            If using pyvistaqt, the QFrame to link the plotter to.

        """

        # create initial class variables
        self._visible_atoms = np.array([1 for i in range(len(structure))], dtype=float)
        self._wrap_atoms = wrap_atoms
        self._atom_metallicness = atom_metallicness
        self._radii_scale = radii_scale
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
            radius=1.0, theta_resolution=15, phi_resolution=15
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
    def visible_atoms(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            Whether or not each atom is visible. This is actually the opacity
            if desired.

        """
        return self._visible_atoms

    @visible_atoms.setter
    def visible_atoms(self, visible_atoms: NDArray[float]):

        # convert to array
        visible_atoms = np.array(visible_atoms, dtype=np.float64)

        # make sure we have the write shape
        assert (
            self.visible_atoms.shape == visible_atoms.shape
        ), "Length match the number of atoms"

        # set visible atoms
        self._visible_atoms = visible_atoms
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
        # update all atoms
        actor = self.plotter.actors["atom_glyphs"]
        actor.prop.metallic = atom_metallicness
        self._atom_metallicness = atom_metallicness

    @property
    def atom_df(self) -> pd.DataFrame:
        """

        Returns
        -------
        atom_df : TYPE
            A dataframe summarizing the properties of the atom meshes.

        """
        # construct a pandas dataframe for each atom
        visible = self.visible_atoms > 0
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
        self.visible_atoms = atom_df["Visible"]
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
            alpha = self.visible_atoms[self._map_wrapped_to_atoms]
            # get radii
            radii = self.atom_radii[self._map_wrapped_to_atoms] * self.radii_scale
        else:
            # get poly data without wrapped atoms
            atoms = self._atom_poly
            # get atom colors
            atom_colors = self.atom_colors
            # get alpha values
            alpha = self.visible_atoms
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
        self._update_site_meshes()

        return plotter
