# -*- coding: utf-8 -*-

import logging

import numpy as np

from baderkit.core.methods.base import MethodBase
from baderkit.core.methods.shared_numba import get_edges

from .neargrid_numba import (
    get_gradient_pointers,
    refine_fast_neargrid,
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
        logging.info("Calculating gradients")
        # calculate gradients and pointers to best neighbors
        labels, gradients, self._maxima_mask = get_gradient_pointers(
            data=grid.total,
            dir2lat=self.dir2lat,
            neighbor_dists=neighbor_dists,
            neighbor_transforms=neighbor_transforms,
            vacuum_mask=self.vacuum_mask,
            initial_labels=grid.all_voxel_indices,
        )
        # Convert to 1D. We use the same name for minimal memory
        labels = labels.ravel()
        # Find roots
        # NOTE: Vacuum points are indicated by a value of -1 and we want to
        # ignore these
        logging.info("Finding roots")
        labels = self.get_roots(labels)
        # We now have our roots. Relabel so that they go from 0 to the length of our
        # roots
        unique_roots, labels = np.unique(labels, return_inverse=True)
        # shift back to vacuum at -1
        if -1 in unique_roots:
            labels -= 1
        # reconstruct a 3D array with our labels
        labels = labels.reshape(grid.shape)
        # reduce maxima/basins
        labels, self._maxima_frac = self.reduce_label_maxima(labels)
        # shift to vacuum at 0
        labels += 1

        # Now we refine the edges with the neargrid method
        reassignments = 1
        # get our edges, not including edges on the vacuum.
        refinement_mask = get_edges(
            labeled_array=labels,
            neighbor_transforms=neighbor_transforms,
            vacuum_mask=self.vacuum_mask,
        )
        # remove maxima from refinement
        refinement_mask[self.maxima_mask] = False
        # note these labels should not be reassigned again in future cycles
        labels[refinement_mask] = -labels[refinement_mask]

        while reassignments > 0:
            # get refinement indices
            refinement_indices = np.argwhere(refinement_mask)
            if len(refinement_indices) == 0:
                # there's nothing to refine so we break
                break
            print(f"Refining {len(refinement_indices)} points")
            # reassign edges
            labels, reassignments, refinement_mask = refine_fast_neargrid(
                data=grid.total,
                labels=labels,
                refinement_indices=refinement_indices,
                refinement_mask=refinement_mask,
                maxima_mask=self.maxima_mask,
                gradients=gradients,
                neighbor_dists=neighbor_dists,
                neighbor_transforms=neighbor_transforms,
            )

            print(f"{reassignments} values changed")
        # switch negative labels back to positive and subtract by 1 to get to
        # correct indices
        labels = np.abs(labels) - 1
        # get all results
        results = {
            "basin_labels": labels,
        }
        # assign charges/volumes, etc.
        results.update(self.get_basin_charges_and_volumes(labels))
        results.update(self.get_extras())
        return results
