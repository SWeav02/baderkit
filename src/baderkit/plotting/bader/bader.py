# -*- coding: utf-8 -*-

from baderkit.bader.bader import Bader

from ..base.base_analysis import BaseAnalysis


class BaderPlotter(BaseAnalysis):
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

    _label_grids = [
        "maxima_basin_labels",
        "atom_labels",
    ]
    _alt_label_names = {
        "maxima_basin_labels": "bader_basins",
        "atom_labels": "atom_basins",
    }

    def __init__(
        self,
        bader: Bader,
        grid_name: str = "reference_grid",
        **kwargs,
    ):

        super().__init__(base_analysis=bader, grid_name=grid_name, **kwargs)
