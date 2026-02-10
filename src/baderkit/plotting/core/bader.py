# -*- coding: utf-8 -*-

import numpy as np
from baderkit.core import Bader
from .grid import GridPlotter
    
class BaderPlotter(GridPlotter):
    def __init__(
        self,
        bader: Bader,
        grid_name: str = "reference_grid",
        **grid_kwargs,
    ):
        """
        A convenience class for creating plots of individual Bader basins
        using pyvista's package for VTK.

        Parameters
        ----------
        bader : Bader
            The Bader object to use for isolating basins and creating isosurfaces.
            The structure will be pulled from the charge grid.
        grid_name : str, optional
            The name of the grid property with the desired data to plot. Options
            are 'charge_grid', 'total_charge_grid', or 'reference_grid'. The
            default is 'reference_grid'

        Returns
        -------
        None.

        """
        # apply StructurePlotter kwargs
        grid = getattr(bader, grid_name)
        super().__init__(grid=grid, **grid_kwargs)
        self.bader = bader

        # pad the label arrays then flatten them
        padded_basins = np.pad(
            bader.maxima_basin_labels, pad_width=((0, 1), (0, 1), (0, 1)), mode="wrap"
        )
        padded_atoms = np.pad(
            bader.atom_labels, pad_width=((0, 1), (0, 1), (0, 1)), mode="wrap"
        )
        # padded_basins = bader.basin_labels
        # padded_atoms = bader.atom_labels
        self.flat_bader_basins = padded_basins.ravel(order="F")
        self.flat_atom_basins = padded_atoms.ravel(order="F")

        # get the initial empty list of visible atom labels and visible basin labels
        self._visible_bader_basins = set(
            [i for i, ai in enumerate(bader.basin_atoms) if ai == 0]
        )
        self._visible_atom_basins = set()
        self.visible_bader_basins = [
            i for i, ai in enumerate(bader.basin_atoms) if ai == 0
        ]
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
        self.iso_val = self._iso_val
