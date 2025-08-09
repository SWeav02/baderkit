# -*- coding: utf-8 -*-

import logging

import numpy as np

from baderkit.core.methods.base import MethodBase
from baderkit.core.methods.shared_numba import get_edges

from .fast_neargrid_numba import (
    get_ongrid_rgrads_pointers,
    refine_fast_neargrid,
)


class FastNeargridMethod(MethodBase):

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
        pointers_3d, highest_neighbors, all_drs, self._maxima_mask = get_ongrid_rgrads_pointers(
            data=grid.total,
            car2lat=car2lat,
            neighbor_dists=neighbor_dists,
            neighbor_transforms=neighbor_transforms,
            vacuum_mask=self.vacuum_mask,
            initial_labels=grid.all_voxel_indices,
        )
        pointers = pointers_3d.ravel()
        # NOTE: Vacuum points are indicated by a value of -1 and we want to
        # ignore these
        logging.info("Finding roots")
        # mask for non-vacuum indices (not -1)
        if self.num_vacuum:
            valid = pointers != -1
        else:
            valid = None
        pointers = self.get_roots(pointers, valid)
        # We now have our roots. Relabel so that they go from 0 to the length of our
        # roots
        unique_roots, labels_flat = np.unique(pointers, return_inverse=True)
        # If we had at least one vacuum point, we need to subtract our labels by
        # 1 to recover the vacuum label.
        if -1 in unique_roots:
            labels_flat -= 1
        # reconstruct a 3D array with our labels
        labels = labels_flat.reshape(grid.shape)
        # reduce maxima/basins
        labels, self._maxima_frac = self.reduce_label_maxima(labels)
        # Now we refine the edges with the neargrid method
        

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
        for i, j, k in self.maxima_vox:
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
            labels, reassignments, refinement_mask, checked_mask = refine_fast_neargrid(
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
            )

            print(f"{reassignments} values changed")
        # get all results
        results = {
            "basin_labels": labels,
        }
        # assign charges/volumes, etc.
        results.update(self.get_basin_charges_and_volumes(labels))
        results.update(self.get_extras())
        return results
