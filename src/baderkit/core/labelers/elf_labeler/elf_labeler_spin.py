# -*- coding: utf-8 -*-

"""
Extends the ElfLabeler to spin polarized calculations
"""

from pathlib import Path
import logging
from typing import TypeVar

import numpy as np
import plotly.graph_objects as go

from baderkit.core import Grid, Structure
from baderkit.core.labelers.bifurcation_graph import BifurcationGraph

from .elf_labeler import ElfLabeler

Self = TypeVar("Self", bound="ElfLabeler")

class SpinElfLabeler:
    def __init__(
            self,
            charge_grid: Grid,
            reference_grid: Grid,
            **kwargs,
            ):
        # First make sure the grids are actually spin polarized
        assert reference_grid.is_spin_polarized and charge_grid.is_spin_polarized, "ELF must be spin polarized. Use a spin polarized calculation or switch to the ElfLabeler class."
        # store the original grid
        self.original_reference_grid = reference_grid
        self.original_charge_grid = charge_grid
        # split the grids to spin up and spin down
        self.reference_grid_up, self.reference_grid_down = reference_grid.split_to_spin()
        self.charge_grid_up, self.charge_grid_down = charge_grid.split_to_spin()
        # check if spin up and spin down are the same
        if np.allclose(
            self.reference_grid_up.total, self.reference_grid_down.total, rtol=0, atol=1e-4
        ):
            logging.info("Spin grids are found to be equal. Only spin-up system will be used.")
            self._equal_spin = True
        else:
            self._equal_spin = False
        # create spin up and spin down elf analyzer instances
        self.elf_analyzer_up = ElfLabeler(
            reference_grid=self.reference_grid_up,
            charge_grid=self.charge_grid_up,
            **kwargs,
            )
        if not self._equal_spin:
            self.elf_analyzer_down = ElfLabeler(
                reference_grid=self.reference_grid_down, 
                charge_grid=self.charge_grid_down, 
                **kwargs,
                )
        else:
            self.elf_analyzer_down = self.elf_analyzer_up
        
    
    ###########################################################################
    # Properties for spin up and spin down systems
    ###########################################################################
    
    @property
    def structure(self) -> Structure:
        """
        Shortcut to grid's structure object
        """
        structure = self.original_reference_grid.structure.copy()
        structure.add_oxidation_state_by_guess()
        return structure
    
    @property
    def feature_structure_up(self) -> Structure:
        return self.elf_analyzer_up.feature_structure
    
    @property
    def feature_structure_down(self) -> Structure:
        return self.elf_analyzer_down.feature_structure
    
    @property
    def bifurcation_graph_up(self) -> BifurcationGraph:
        return self.elf_analyzer_up.bifurcation_graph
    
    @property
    def bifurcation_graph_down(self) -> BifurcationGraph:
        return self.elf_analyzer_down.bifurcation_graph
    
    @property
    def bifurcation_plot_up(self) -> go.Figure:
        return self.elf_analyzer_up.bifurcation_plot
    
    @property
    def bifurcation_plot_down(self) -> go.Figure:
        return self.elf_analyzer_down.bifurcation_plot
    
    def write_plots(self, filename: str | Path):
        filename = Path(filename)
    
        if filename.suffix:
            filename_up = filename.with_name(f"{filename.stem}_up{filename.suffix}")
            filename_down = filename.with_name(f"{filename.stem}_down{filename.suffix}")
        else:
            filename_up = filename.with_name(f"{filename.name}_up")
            filename_down = filename.with_name(f"{filename.name}_down")
    
        self.elf_analyzer_up.write_bifurcation_plot(filename_up)
        self.elf_analyzer_down.write_bifurcation_plot(filename_down)

    
    ###########################################################################
    # From methods
    ###########################################################################
    @classmethod
    def from_vasp(
        cls,
        charge_filename: Path | str = "CHGCAR",
        reference_filename: Path | str = "ELFCAR",
        **kwargs,
    ) -> Self:
        """
        Creates a SpinElfAnalysis class object from VASP files.

        Parameters
        ----------
        charge_filename : Path | str, optional
            The path to the CHGCAR like file that will be used for summing charge.
            The default is "CHGCAR".
        reference_filename : Path | str
            The path to ELFCAR like file that will be used for partitioning.
            If None, the charge file will be used for partitioning.
        total_only: bool
            If true, only the first set of data in the file will be read. This
            increases speed and reduced memory usage as the other data is typically
            not used.
            Defaults to True.
        **kwargs : dict
            Keyword arguments to pass to the Bader class.

        Returns
        -------
        Self
            A SpinElfAnalysis class object.

        """
        charge_grid = Grid.from_vasp(charge_filename, total_only=False)
        if reference_filename is None:
            reference_grid = None
        else:
            reference_grid = Grid.from_vasp(reference_filename, total_only=False)

        return cls(charge_grid=charge_grid, reference_grid=reference_grid, **kwargs)
    
    # TODO: Currently this class is only useful for VASP because .cube files
    # typically only contain a single grid. Is there a reason to create a convenience
    # function for cube files?