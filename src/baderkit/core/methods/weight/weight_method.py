# -*- coding: utf-8 -*-

import logging

import numpy as np

from baderkit.core.methods.base import MethodBase
from baderkit.core.methods.shared_numba import combine_maxima_frac

from .weight_numba import (  # reduce_charge_volume,
    get_labels,
    get_weight_assignments,
    sort_maxima_vox,
)


class WeightMethod(MethodBase):

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
        reference_grid = self.reference_grid
        charge_grid = self.charge_grid
        reference_data = reference_grid.total
        charge_data = charge_grid.total
        shape = reference_grid.shape

        logging.info("Sorting Reference Data")
        # sort data from lowest to highest
        sorted_indices = np.argsort(reference_data.ravel(), kind="stable")

        # remove vacuum from sorted indices
        sorted_indices = sorted_indices[self.num_vacuum :]
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
        labels, charges, volumes, maxima_vox, sorted_maxima_mask = (
            get_weight_assignments(
                reference_data,
                charge_data,
                sorted_indices,
                neighbor_transforms,
                neighbor_alpha,
                all_neighbor_transforms,
                all_neighbor_dists,
            )
        )

        logging.info("Assigning Labels")
        # flip charges to go from high to low
        charges = np.flip(charges)
        volumes = np.flip(volumes)
        # get actual labels
        labels = get_labels(
            labels,
            np.flip(sorted_indices),
            np.flip(sorted_maxima_mask),
        )
        # reconstruct a 3D array with our labels
        labels = labels.reshape(shape)
        # our maxima are already fully reduced by the current method, but we need
        # the combined frac coords. Get those here
        maxima_frac = reference_grid.grid_to_frac(maxima_vox)
        self._maxima_frac = combine_maxima_frac(
            labels,
            maxima_vox,
            maxima_frac,
        )
        self._maxima_vox = sort_maxima_vox(
            maxima_vox,
            shape[0],
            shape[1],
            shape[2],
        )
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
        results.update(self.get_extras())
        return results
