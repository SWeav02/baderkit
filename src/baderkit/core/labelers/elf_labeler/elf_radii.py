# -*- coding: utf-8 -*-
"""
This is a reimplementation of the ionic radius finder I created for BadELF in
[Simmate](https://github.com/jacksund/simmate/blob/main/src/simmate/apps/badelf/core/partitioning.py)
"""

from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from baderkit.core import Grid, Structure
from baderkit.core.utilities.coord_env import get_atom_nearest_neighbors

from .elf_radii_numba import get_all_atom_elf_radii, get_elf_radii


class ElfRadiiTools:
    def __init__(
        self,
        grid: Grid,
        feature_labels: NDArray,
        feature_structure: Structure,
        covalent_symbols: list[str] = ["Z", "M"],
        override_structure: Structure | None = None,
    ):

        self.grid = grid
        self.cubic_coeffs = grid.cubic_spline_coeffs
        self.feature_structure = feature_structure
        self.feature_labels = feature_labels
        self.covalent_symbols = covalent_symbols

        if override_structure is None:
            self.structure = grid.structure
        else:
            self.structure = override_structure

        # NOTE: atom labels should correspond to labeled structure

    # @cached_property
    # def nearest_neighbors(self):
    #     # calculate the nearest neighbors
    #     neighbor_indices, neighbor_dists, neighbor_images = get_atom_nearest_neighbors(
    #         atom_frac_coords=self.structure.frac_coords,
    #         atom_cart_coords=self.structure.cart_coords,
    #         frac2cart=self.structure.lattice.matrix,
    #     )

    #     return neighbor_indices, neighbor_dists, neighbor_images

    # @cached_property
    # def _elf_radii_and_type(self):
    #     # get nearest neighbor info
    #     neighbor_indices, neighbor_dists, neighbor_images = self.nearest_neighbors
    #     return get_elf_radii(
    #         equivalent_atoms=self.structure.equivalent_atoms,
    #         data=self.cubic_coeffs,
    #         feature_labels=self.feature_labels,
    #         atom_frac_coords=self.structure.frac_coords,
    #         neighbor_indices=neighbor_indices,
    #         neighbor_dists=neighbor_dists,
    #         neighbor_images=neighbor_images,
    #         covalent_labels=np.array(
    #             self.feature_structure.indices_from_symbol(self.covalent_symbol),
    #             dtype=np.float64,
    #         ),
    #     )

    # @cached_property
    # def atom_elf_radii(self):
    #     return self._elf_radii_and_type[0]

    # @cached_property
    # def atom_elf_radii_types(self):
    #     return self._elf_radii_and_type[1]

    def get_all_neigh_elf_radii_and_type(
        self,
        site_indices,
        neigh_indices,
        neigh_frac_coords,
        neigh_dists,
    ):
        # first, get the unique bonds. We do this here because numba does not
        # currently allow np.unique axis argument
        equivalent_atoms = self.structure.equivalent_atoms
        equiv_sites = equivalent_atoms[site_indices]
        equiv_neighs = equivalent_atoms[neigh_indices]
        pair_array = np.column_stack((equiv_sites, equiv_neighs, neigh_dists))
        unique_bonds, index, inverse = np.unique(
            pair_array, return_inverse=True, return_index=True, axis=0
        )
        equivalent_bonds = index[inverse]
        
        # create a map of covalent labels
        covalent_labels = np.zeros(len(self.feature_structure), dtype=np.bool_)
        for symbol in self.covalent_symbols:
            covalent_labels[np.array(self.feature_structure.indices_from_symbol(symbol), dtype=np.int64)] = True

        return get_all_atom_elf_radii(
            site_indices=site_indices,
            neigh_indices=neigh_indices,
            site_frac_coords=self.structure.frac_coords,
            neigh_frac_coords=neigh_frac_coords,
            neigh_dists=neigh_dists,
            equivalent_bonds=equivalent_bonds,
            data=self.cubic_coeffs,
            feature_labels=self.feature_labels,
            covalent_labels=covalent_labels,
        )
