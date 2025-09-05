# -*- coding: utf-8 -*-

import logging

import numpy as np

from baderkit.core.methods.base import MethodBase

from .weight_numba import (
    reduce_charge_volume,
    get_labels,
    # get_neighbor_flux,
    get_weight_assignments,
    relabel_reduced_maxima,
)

# TODO: Use list of list storage for initial flux calcs. For points that would
# be given no flux, check if they're true maxima and make all flux point to highest
# neighbor. That should prevent fake maxima so we don't need to peform maxima reduction


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
        
        logging.info("Sorting reference data")
        # sort data from lowest to highest
        sorted_indices = np.argsort(reference_data.ravel(), kind="stable")

        # remove vacuum from sorted indices
        sorted_indices = sorted_indices[self.num_vacuum:]
        # get the voronoi neighbors, their distances, and the area of the corresponding
        # facets. This is used to calculate the volume flux from each voxel
        neighbor_transforms, neighbor_dists, facet_areas, _ = (
            reference_grid.point_neighbor_voronoi_transforms
        )
        # # get a single alpha corresponding to the area/dist
        neighbor_alpha = facet_areas / neighbor_dists

        # Get the flux of volume from each voxel to its neighbor.
        logging.info("Calculating voxel flux contributions")
        all_neighbor_transforms, all_neighbor_dists = (
            reference_grid.point_neighbor_transforms
        )
        labels, charges, volumes, self._maxima_mask = get_weight_assignments(
            reference_data,
            charge_data.ravel(),
            sorted_indices,
            neighbor_transforms,
            neighbor_alpha,
            all_neighbor_transforms,
            all_neighbor_dists,
        )
        # flip charges to go from high to low
        charges = np.flip(charges)
        volumes = np.flip(volumes)
        # get actual labels
        labels = labels.ravel()
        labels = get_labels(
            labels,
            np.flip(sorted_indices),
        )
        # We now have our roots. Relabel so that they go from 0 to the length of our
        # roots
        unique_roots, labels = np.unique(labels, return_inverse=True)
        # If we had at least one vacuum point, we need to subtract our labels by
        # 1 to recover the vacuum label.
        if -1 in unique_roots:
            labels -= 1
        # reconstruct a 3D array with our labels
        labels = labels.reshape(shape)
        
        # reduce maxima/labels and save frac coords.
        # NOTE: reduction algorithm returns with unlabeled values as -1
        labels, self._maxima_frac, label_map = self.reduce_label_maxima(labels, True)
        basin_num = len(self._maxima_frac)
        charges, volumes = reduce_charge_volume(
            label_map,
            charges,
            volumes,
            basin_num,
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
