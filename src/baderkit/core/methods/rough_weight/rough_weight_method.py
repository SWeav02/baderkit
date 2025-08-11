# -*- coding: utf-8 -*-

import logging

import numpy as np

from baderkit.core.methods.base import MethodBase

from .rough_weight_numba import (  # reduce_weight_maxima,
    get_rough_weight_labels
)

class RoughWeightMethod(MethodBase):

    def run(self):
        """
        Assigns basin weights to each voxel and assigns charge using
        the weight method:
            M. Yu and D. R. Trinkle,
            Accurate and efficient algorithm for Bader charge integration,
            J. Chem. Phys. 134, 064111 (2011).

        Returns
        -------
        None.

        """
        reference_grid = self.reference_grid.copy()
        # get the voronoi neighbors, their distances, and the area of the corresponding
        # facets. This is used to calculate the volume flux from each voxel
        neighbor_transforms, neighbor_dists, facet_areas, _ = (
            reference_grid.voxel_voronoi_facets
        )
        neighbor_weights = facet_areas/neighbor_dists
        logging.info("Sorting reference data")
        data = reference_grid.total
        shape = reference_grid.shape
        # get voxel indices
        flat_voxel_coords = np.indices(shape, dtype=np.int64).reshape(3, -1).T
        # sort data from high to low
        sorted_data_indices = np.flip(np.argsort(data.ravel(), kind="stable"))
        sorted_voxel_coords = flat_voxel_coords[sorted_data_indices]
        # remove vacuum points from our list of voxel indices
        sorted_voxel_coords = sorted_voxel_coords[
            : len(sorted_voxel_coords) - self.num_vacuum
        ]
        # Get the flux of volume from each voxel to its neighbor.
        logging.info("Calculating weight assignments")
        all_neighbor_transforms, all_neighbor_dists = reference_grid.voxel_26_neighbors
        labels, self._maxima_mask = get_rough_weight_labels(
            data=data,
            sorted_voxel_coords=sorted_voxel_coords.copy(),
            neighbor_transforms=neighbor_transforms,
            neighbor_weights=neighbor_weights,
            all_neighbor_transforms=all_neighbor_transforms,
            all_neighbor_dists=all_neighbor_dists,
        )
        # reduce maxima/labels and save frac coords.
        # NOTE: reduction algorithm returns with unlabeled values as -1
        labels, self._maxima_frac = self.reduce_label_maxima(labels)
        # assign all results
        results = {
            "basin_labels": labels,
        }
        # assign charges/volumes, etc.
        results.update(self.get_basin_charges_and_volumes(labels))
        results.update(self.get_extras())
        return results
