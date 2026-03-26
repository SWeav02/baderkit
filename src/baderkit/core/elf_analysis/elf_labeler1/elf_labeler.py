# -*- coding: utf-8 -*-

from pathlib import Path
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from baderkit.core.base.base_analysis import BaseAnalysis
from baderkit.core.bader.bader import Bader
from baderkit.core.toolkit import Grid
from baderkit.core.elf_analysis.overlap import BasinOverlap

from .enum_and_styling import FeatureType

Self = TypeVar("Self", bound="ElfLabeler")

class ElfLabeler(BaseAnalysis):
    """
    A convenience class for calculating the overlap between basins calculated
    in the charge density and a localization density such as ELF.

    """
    
    _reset_props = [
        "basin_types",
        "attractor_shapes",
        "along_bond",
        "heavily_polarized",
        ]
    
    _summary_props = [
        "basin_types",
        "attractor_shapes",
        "heavily_polarized",
        ]

    def __init__(
        self,
        charge_grid: Grid,
        reference_grid: Grid,
        total_charge_grid: Grid | None = None,
        polarization_cutoff: float = 0.5,
        **kwargs,
    ):
        # create bader objects
        self._overlap: BasinOverlap = BasinOverlap(
            charge_grid=charge_grid,
            total_charge_grid=total_charge_grid,
            reference_grid=reference_grid,
            **kwargs,
        )
        
        self._elf_bader = self.overlap.local_bader

        super().__init__(
            charge_grid=charge_grid,
            total_charge_grid=total_charge_grid,
            reference_grid=reference_grid,
            **kwargs,
        )
        
        self._polarization_cutoff = polarization_cutoff
        
    ###########################################################################
    # Settings
    ###########################################################################
    
    @property
    def polarization_cutoff(self) -> float:
        return self._polarization_cutoff

    @polarization_cutoff.setter
    def polarization_cutoff(self, value: float):
        self._polarization_cutoff = value
        self._reset_properties(
            include_properties=[
                "heavily_polarized",
                "basin_types",
                ],
            )
    
    ###########################################################################
    # Properties
    ###########################################################################
    @property
    def overlap(self) -> BasinOverlap:
        return self._overlap
    
    @property
    def elf_bader(self) -> Bader:
        return self._elf_bader
     
    @property
    def maxima_frac(self) -> NDArray[np.float64]:
        return self.elf_bader.maxima_frac
    
    @property
    def attractor_shapes(self) -> NDArray[str]:
        return self.overlap.attractor_shapes
    
    @property
    def basin_types(self) -> list[str]:
        if self._basin_types is None:
            self._label_basins()
        return [i.name for i in self._basin_types]
    
    @property
    def heavily_polarized(self):
        if self._heavily_polarized is None:
            self._heavily_polarized = self.overlap.polarization_indexes > self.polarization_cutoff
        return self._heavily_polarized
    
    def _label_basins(self):
        """
        Label scheme:
            shared:
                point/ring:
                    atom center -> ionic shell
                    along bond:
                        heavily shared -> covalent bond
                        barely shared -> ionic bond
                    not along bond:
                        heavily shared:
                            small -> metallic bond
                            medium -> multi-center bond?
                            large -> electride?
                        barely shared:
                            dominant atom has other bonds -> lone-pair
                            dominant atom has no bonds -> ionic shell
                cage -> ionic shell
            
            unshared:
                point:
                    atom center -> core
                    elsewhere -> lone-pair
                ring -> core?
                cage -> core
        """

        # create a list to store types
        types = []
        for feature_idx in range(len(self.maxima_frac)):
            shape = self.overlap.attractor_shapes[feature_idx]
            # check for core
            is_core = self.overlap.core_basins[feature_idx] != -1
            if is_core:
                types.append(FeatureType.core)
                continue
            # check for lone-pair
            if not self.overlap.shared_basins[feature_idx]:
                types.append(FeatureType.lone_pair)
                continue
            
            along_bond = self.overlap.along_bond[feature_idx]
            heavily_polarized = self.heavily_polarized[feature_idx]
            
            # check for ionic/covalent bond
            if not shape == "cage" and along_bond:
                if heavily_polarized:
                    types.append(FeatureType.ionic)
                else:
                    types.append(FeatureType.covalent)
            
            # check for ionic shells and nnas
            else:
                if heavily_polarized:
                    types.append(FeatureType.ionic_shell)
                else:
                    # breakpoint()
                    types.append(FeatureType.nna)
        self._basin_types = types

    @classmethod
    def from_vasp(
        cls,
        charge_filename: Path | str = "CHGCAR",
        reference_filename: Path | str = "ELFCAR",
        **kwargs,
    ) -> Self:
        """
        Creates a Bader class object from VASP files.

        Parameters
        ----------
        charge_filename : Path | str, optional
            The path to the CHGCAR like file that will be used for integrating charge.
            The default is "CHGCAR".
        reference_filename : Path |  str
            The path to ELFCAR like file that will be used for partitioning.
        total_charge_filename : Grid | None, optional
            The path to the CHGCAR like file used for determining vacuum regions
            in the system. For pseudopotential codes this represents the total
            electron density and should be provided whenever possible.
            If None, defaults to the charge_grid.
        total_only: bool
            If true, only the first set of data in each file will be read. This
            increases speed and reduced memory usage as the other data is typically
            not used.
            Defaults to True.
        **kwargs : dict
            Keyword arguments to pass to the class.

        Returns
        -------
        Self
            A BaseAnalysis class object.

        """
        return super().from_vasp(
            charge_filename=charge_filename,
            reference_filename=reference_filename,
            **kwargs
            )