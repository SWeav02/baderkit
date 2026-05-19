# -*- coding: utf-8 -*-

from abc import ABC

import numpy as np

from baderkit.toolkit.structure import Structure
from ..toolkit import GridPlotter

class BaseAnalysis(GridPlotter, ABC):
    
    _label_grids = [] # e.g. maxima_basin_labels
    _alt_label_names = {} # e.g. "maxima_basin_labels" : "bader_basins
    
    def __init__(
        self,
        base_analysis,
        structure: Structure = None,
        grid_name: str = "reference_grid",
        **grid_kwargs,
    ):
        """
        This class contains convenience functions used by most complex
        analysis plotters in BaderKit. It should not be used directly.

        """
        # apply StructurePlotter kwargs
        grid = getattr(base_analysis, grid_name)
        if structure:
            grid = grid.copy()
            grid.structure = structure.copy()
        super().__init__(grid=grid, **grid_kwargs)
        
        self._grid_name = grid_name
        
        # set base analysis class
        setattr(self, base_analysis.__class__.__name__.lower(), base_analysis)
        
        # pad label grids
        for label_grid_str in self._label_grids:
            label_grid = getattr(base_analysis, label_grid_str, None)
            if label_grid is None:
                continue
            # pad the label arrays then flatten them
            padded_label_grid = np.pad(
                label_grid,
                pad_width=((0, 1), (0, 1), (0, 1)),
                mode="wrap",
            )
            setattr(self, f"_{label_grid_str}", padded_label_grid.ravel(order="F"))
            # create settings property
            alt_name = self._alt_label_names.get(label_grid_str, label_grid_str)
            self._make_visible_property(alt_name)

        # set visible basins
        self._update_plotter_mask()

    @property
    def grid_name(self) -> str:
        """

        Returns
        -------
        str
            The name of the grid to plot

        """
        return self._grid_name

    @grid_name.setter
    def grid_name(self, grid_name: str):
        assert grid_name in [
            "reference_grid",
            "charge_grid",
            "total_charge_grid",
        ]
        # set visible basins
        self._grid_name = grid_name
        self.grid = getattr(self.bader, grid_name)
        # update plotter
        self._update_plotter_mask()

    ###########################################################################
    # Utilities
    ###########################################################################
            
    def _make_visible_property(
        self,
        prop_name: str,
    ):
        item_description = " ".join(prop_name.split("_"))
    
        @property
        def prop(self) -> set[int]:
            f"""
            Returns
            -------
            set[int]
                The set of {item_description} indices currently visible.
            """
            return getattr(self, f"_visible_{prop_name}")
    
        @prop.setter
        def prop(self, values: set[int]):
            values = set(values)
            setattr(self, f"_visible_{prop_name}", values)
            self._update_plotter_mask()

        setattr(self, f"_visible_{prop_name}", set())
        setattr(type(self), f"visible_{prop_name}", prop)

    def _update_plotter_mask(self):
        """
        Updates the mask indicating which areas of the grid should not be shown
        then sets the regions to -1.

        Returns
        -------
        None.

        """
        # get the visible points
        visible_masks = []

        for label_name in self._label_grids:
            visible_values = getattr(
                self,
                f"_visible_{self._alt_label_names.get(label_name, label_name)}"
            )
        
            mask = np.isin(
                getattr(self,  f"_{label_name}"),
                list(visible_values),
            )
        
            visible_masks.append(mask)
        
        hidden_mask = ~np.logical_or.reduce(visible_masks)
        self._hidden_mask = hidden_mask
        # NOTE: using hide_cells works, but results in some funky artifacting.
        # Maybe there's a way to get it to work, but for now I'm replacing it
        # for visual quality
        # self.structured_grid.hide_cells(self.hidden_mask, inplace=True)
        # update structured_grid
        self.structured_grid = self._make_structured_grid()
        # update plotter
        self._update_grid_meshes()
        self._update_surface_actor()
        self._update_cap_actor()
