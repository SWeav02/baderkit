# -*- coding: utf-8 -*-

import numpy as np
from numpy.typing import NDArray

from baderkit.core.base.base_analysis import BaseAnalysis
from baderkit.core.toolkit import Grid
from baderkit.core.elf_analysis.overlap import BasinOverlap


class ElfLabeler(BaseAnalysis):
    """
    A convenience class for calculating the overlap between basins calculated
    in the charge density and a localization density such as ELF.

    """
    
    _reset_props = [
        "basin_types"
        ]

    def __init__(
        self,
        charge_grid: Grid,
        reference_grid: Grid,
        total_charge_grid: Grid | None = None,
        **kwargs,
    ):
        # create bader objects
        self.overlap = BasinOverlap(
            charge_grid=charge_grid,
            total_charge_grid=total_charge_grid,
            reference_grid=total_charge_grid,
            **kwargs,
        )
        
        self.elf_bader = self.overlap.local_bader

        super().__init__(
            charge_grid=charge_grid,
            total_charge_grid=total_charge_grid,
            reference_grid=reference_grid,
            **kwargs,
        )
        
    @property
    def maxima_frac(self) -> NDArray[np.float64]:
        return self.elf_bader.maxima_frac
    
    @property
    def basin_types(self) -> list[str]:
        if self._basin_types is None:
            self._label_basins()
        return self._basin_types
    
    def _label_basins(self):
        # Label scheme:
            # shared:
                # point/ring:
                    # atom center -> ionic shell
                    # along bond:
                        # heavily shared -> covalent bond
                        # barely shared -> ionic bond
                    # not along bond:
                        # heavily shared:
                            # small -> metallic bond
                            # medium -> multi-center bond?
                            # large -> electride?
                        # barely shared:
                            # dominant atom has other bonds -> lone-pair
                            # dominant atom has no bonds -> ionic shell
                # cage -> ionic shell
            
            # unshared:
                # point:
                    # atom center -> core
                    # elsewhere -> lone-pair
                # ring -> core?
                # cage -> core
        pass
    
    #properties:
        # labeled basins
        # radii