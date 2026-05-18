# -*- coding: utf-8 -*-

from baderkit.elf_analysis import Badelf

from ..base.base_analysis import BaseAnalysis



class BadelfPlotter(BaseAnalysis):
    
    _label_grids = [
        "atom_labels",
        ]
    _alt_label_names = {
        "atom_labels" : "atom_basins",
        }
    
    def __init__(
        self,
        badelf: Badelf,
        grid_name: str = "reference_grid",
        **kwargs,
    ):
        """
        A convenience class for creating plots of individual Bader basins
        using pyvista's package for VTK.

        Parameters
        ----------
        badelf : Badelf
            The Badelf object to use for isolating basins and creating isosurfaces.
            The structure will be pulled from the charge grid.
        grid_name : str, optional
            The name of the grid property with the desired data to plot. Options
            are 'charge_grid', 'total_charge_grid', or 'reference_grid'. The
            default is 'reference_grid'

        Returns
        -------
        None.

        """
        super().__init__(base_analysis=badelf, grid_name=grid_name, **kwargs)