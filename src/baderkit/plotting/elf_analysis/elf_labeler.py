# -*- coding: utf-8 -*-

from baderkit.elf_analysis import ElfLabeler

from ..base.base_analysis import BaseAnalysis



class ElfLabelerPlotter(BaseAnalysis):
    """
    A convenience class for creating plots of chemical feature basins
    using pyvista's package for VTK.

    Parameters
    ----------
    elf_labeler : ElfLabeler
        The ElfLabeler object to use for isolating basins and creating isosurfaces.
        The structure will be pulled from the charge grid.
    grid_name : str, optional
        The name of the grid property with the desired data to plot. Options
        are 'charge_grid', 'total_charge_grid', or 'reference_grid'. The
        default is 'reference_grid'

    """
    
    _label_grids = [
        "type_basin_labels",
        ]
    _alt_label_names = {
        "type_basin_labels" : "chemical_features",
        }
    
    def __init__(
        self,
        elf_labeler: ElfLabeler,
        grid_name: str = "reference_grid",
        **kwargs,
    ):

        # get nna structure
        structure = elf_labeler.nna_structure
        
        super().__init__(
            base_analysis=elf_labeler, 
            structure=structure, 
            grid_name=grid_name, 
            **kwargs,
            )
        
    @property
    def chemical_features(self) -> list[str]:
        """

        Returns
        -------
        list[str]
            The types of chemical features found in the system.

        """
        return self.elflabeler.types_in_system