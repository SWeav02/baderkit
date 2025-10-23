# -*- coding: utf-8 -*-

import logging

import numpy as np

from baderkit.core.methods.base import MethodBase

from .weight_numba import (  # reduce_charge_volume,; get_labels,
    get_weight_assignments,
    sort_maxima_frac,
)


class WeightMethod(MethodBase):

    def _run_bader(self, labels):
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
        reference_grid = self.reference_grid
        charge_grid = self.charge_grid
        reference_data = reference_grid.total
        charge_data = charge_grid.total
        shape = reference_grid.shape

        # We need our maxima to follow the same ordering as the other methods.
        # We want to get this order down before we run our method
        self._maxima_frac, self._maxima_vox, maxima_indices = sort_maxima_frac(
            self.maxima_frac, self.maxima_vox, shape
        )

        logging.info("Sorting Reference Data")
        # sort data from lowest to highest
        sorted_indices = np.argsort(reference_data.ravel(), kind="stable")

        # remove vacuum from sorted indices
        sorted_indices = sorted_indices[self.num_vacuum :]
        # flip to move from high to low
        sorted_indices = np.flip(sorted_indices)
        # get the voronoi neighbors, their distances, and the area of the corresponding
        # facets. This is used to calculate the volume flux from each voxel
        neighbor_transforms, neighbor_dists, facet_areas, _ = (
            reference_grid.point_neighbor_voronoi_transforms
        )
        # # get a single alpha corresponding to the area/dist
        neighbor_alpha = facet_areas / neighbor_dists

        # Get the flux of volume from each voxel to its neighbor.
        logging.info("Assigning Charges and Volumes")
        all_neighbor_transforms, all_neighbor_dists = (
            reference_grid.point_neighbor_transforms
        )
        labels, charges, volumes = get_weight_assignments(
            reference_data,
            labels,
            charge_data,
            sorted_indices,
            neighbor_transforms,
            neighbor_alpha,
            all_neighbor_transforms,
            all_neighbor_dists,
            self.maxima_mask,
            maxima_indices,
        )
        # reconstruct a 3D array with our labels
        labels = labels.reshape(shape)

        # adjust charges from vasp convention
        charges /= shape.prod()
        # adjust volumes from voxel count
        volumes *= reference_grid.point_volume
        # assign all values
        results = {
            "basin_labels": labels,
            "basin_charges": charges,
            "basin_volumes": volumes,
            "vacuum_charge": self.charge_grid.total[self.vacuum_mask].sum()
            / shape.prod(),
            "vacuum_volume": (self.num_vacuum / reference_grid.ngridpts)
            * reference_grid.structure.volume,
        }
        return results
