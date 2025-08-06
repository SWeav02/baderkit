# -*- coding: utf-8 -*-

import logging

import numpy as np

from baderkit.core.methods.base import MethodBase
from baderkit.core.methods.shared_numba import get_edges

from .neargrid_numba import (
    get_neargrid_labels,
    get_ongrid_and_rgrads,
    refine_neargrid,
)


class NeargridMethod(MethodBase):

    def run(self):
        """
        Assigns voxels to basins and calculates charge using the near-grid
        method:
            W. Tang, E. Sanville, and G. Henkelman
            A grid-based Bader analysis algorithm without lattice bias
            J. Phys.: Condens. Matter 21, 084204 (2009)

        Returns
        -------
        None.

        """
        grid = self.reference_grid.copy()
        # get neigbhor transforms
        neighbor_transforms, neighbor_dists = grid.voxel_26_neighbors
        matrix = grid.matrix
        # convert to lattice vectors as columns
        dir2car = matrix.T
        # get lattice to cartesian matrix
        lat2car = dir2car / grid.shape[np.newaxis, :]
        # get inverse for cartesian to lattice matrix
        car2lat = np.linalg.inv(lat2car)
        logging.info("Calculating gradients")
        highest_neighbors, all_drs, self._maxima_mask = get_ongrid_and_rgrads(
            data=grid.total,
            car2lat=car2lat,
            neighbor_dists=neighbor_dists,
            neighbor_transforms=neighbor_transforms,
            vacuum_mask=self.vacuum_mask,
        )
        logging.info("Calculating initial labels")
        # get initial labels
        labels = get_neargrid_labels(
            data=grid.total,
            highest_neighbors=highest_neighbors,
            all_drs=all_drs,
            maxima_mask=self.maxima_mask,
            vacuum_mask=self.vacuum_mask,
            neighbor_dists=neighbor_dists,
            neighbor_transforms=neighbor_transforms,
        )
        # we now have an array with labels ranging from 0 up (if theres vacuum)
        # or 1 up (if no vacuum). We want to reduce the number of maxima if there
        # are any that border each other. Our reduction algorithm requires unlabeled
        # or vacuum points to be -1 and 0 and up for basins
        labels -= 1
        # reduce labels
        labels, self._maxima_frac = self.reduce_label_maxima(labels)
        # Increase values so vacuum points are labeled by 1 and basins are 2 and up.
        # the reduction algorithm returns vacuum as -1 and basins start at 0
        labels += 2
        # get maxima positions, not including vacuum
        maxima_vox = self.maxima_vox
        # We want to combine any adjacent maxima. This both reduces the number
        # of basins and reduces the nubmer of edges. This often heavily decreases
        # the number of refinements that need to be performed

        reassignments = 1
        # get our edges, not including edges on the vacuum.
        # NOTE: Should the vacuum edges be refined as well in case some voxels
        # are added to it?
        refinement_mask = get_edges(
            labeled_array=labels,
            neighbor_transforms=neighbor_transforms,
            vacuum_mask=self.vacuum_mask,
        )
        # initialize a mask where voxels are already checked to prevent
        # reassignment. We include vacuum voxels from the start
        checked_mask = self.vacuum_mask.copy()
        # add maxima to mask so they don't get checked
        for i, j, k in maxima_vox:
            refinement_mask[i, j, k] = False
            checked_mask[i, j, k] = True

        while reassignments > 0:
            # get refinement indices
            refinement_indices = np.argwhere(refinement_mask)
            if len(refinement_indices) == 0:
                # there's nothing to refine so we break
                break
            print(f"Refining {len(refinement_indices)} points")
            # reassign edges
            labels, reassignments, refinement_mask, checked_mask = refine_neargrid(
                data=grid.total,
                labels=labels,
                refinement_indices=refinement_indices,
                refinement_mask=refinement_mask,
                checked_mask=checked_mask,
                maxima_mask=self.maxima_mask,
                highest_neighbors=highest_neighbors,
                all_drs=all_drs,
                neighbor_dists=neighbor_dists,
                neighbor_transforms=neighbor_transforms,
                vacuum_mask=self.vacuum_mask,
            )

            print(f"{reassignments} values changed")
            # if our refinement method is single, we cancel the loop here
            if self.refinement_method == "single":
                break
        # Our labels currently span 1 and up, with 1 corresponding to vacuum. We
        # subtract by 2 to return to -1 as vacuum and labels spanning 0 up
        labels -= 2
        # get all results
        results = {
            "basin_labels": labels,
        }
        # assign charges/volumes, etc.
        results.update(self.get_basin_charges_and_volumes(labels))
        results.update(self.get_extras())
        return results
