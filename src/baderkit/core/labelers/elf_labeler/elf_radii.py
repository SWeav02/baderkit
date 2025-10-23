# -*- coding: utf-8 -*-
"""
This is a reimplementation of the ionic radius finder I created for BadELF in
[Simmate](https://github.com/jacksund/simmate/blob/main/src/simmate/apps/badelf/core/partitioning.py)
"""

from functools import cached_property
import numpy as np
from numpy.typing import NDArray
from baderkit.core import Structure, Grid

from baderkit.core.utilities.coord_env import get_atom_nearest_neighbors
from .elf_radii_numba import get_elf_radii
    


class ElfRadiiTools:
    def __init__(
            self, 
            grid: Grid,
            feature_labels: NDArray,
            feature_structure: Structure,
            covalent_symbol: str = "Z",
            ):
        
        self.grid = grid
        self.cubic_coeffs = grid.cubic_spline_coeffs
        self.feature_structure = feature_structure
        self.structure = grid.structure
        self.feature_labels = feature_labels
        self.covalent_symbol = covalent_symbol
        
        # NOTE: atom labels should correspond to labeled structure
    
    @cached_property
    def nearest_neighbors(self):
        # calculate the nearest neighbors
        neighbor_indices, neighbor_dists, neighbor_images = get_atom_nearest_neighbors(
            atom_frac_coords=self.structure.frac_coords,
            atom_cart_coords=self.structure.cart_coords,
            frac2cart=self.structure.lattice.matrix
            )
        
        return neighbor_indices, neighbor_dists, neighbor_images
    
    @cached_property
    def atomic_radii(self):
        
        # get nearest neighbor info
        neighbor_indices, neighbor_dists, neighbor_images = self.nearest_neighbors
        return get_elf_radii(
            equivalent_atoms=self.grid.equivalent_atoms,
            data=self.cubic_coeffs,
            feature_labels=self.feature_labels,
            atom_frac_coords=self.structure.frac_coords,
            neighbor_indices=neighbor_indices,
            neighbor_dists=neighbor_dists,
            neighbor_images=neighbor_images,
            covalent_labels=np.array(
                self.feature_structure.indices_from_symbol(self.covalent_symbol), 
                dtype=np.float64
                )
            )
    
    