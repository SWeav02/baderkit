# -*- coding: utf-8 -*-

import logging

import numpy as np

from baderkit.core.methods.base import MethodBase

from .ongrid_numba import get_steepest_pointers


class OngridMethod(MethodBase):

    def run(self):
        """
        Assigns voxels to basins and calculates charge using the on-grid
        method:
            G. Henkelman, A. Arnaldsson, and H. Jónsson
            A fast and robust algorithm for Bader decomposition of charge density,
            Comput. Mater. Sci. 36, 354-360 (2006)

        Returns
        -------
        None.

        """
        grid = self.reference_grid
        data = grid.total
        shape = data.shape
        # get shifts to move from a voxel to the 26 surrounding voxels
        neighbor_transforms, neighbor_dists = grid.voxel_26_neighbors
        # For each voxel, get the label of the surrounding voxel that has the highest
        # density
        logging.info("Calculating steepest neighbors")
        pointers_3d, self._maxima_mask = get_steepest_pointers(
            data=data,
            initial_labels=grid.all_voxel_indices,
            neighbor_transforms=neighbor_transforms,
            neighbor_dists=neighbor_dists,
            vacuum_mask=self.vacuum_mask,
        )
        # ravel the best labels to get a 1D array pointing from each voxel to its steepest
        # neighbor
        pointers = pointers_3d.ravel()
        # Our pointers object is a 1D array pointing each voxel to its parent voxel. We
        # essentially have a classic forest of trees problem where each maxima is
        # a root and we want to point all of our voxels to their respective root.
        # We being a while loop. In each loop, we remap our pointers to point at
        # the index that its parent was pointing at.
        # NOTE: Vacuum points are indicated by a value of -1 and we want to
        # ignore these
        logging.info("Finding roots")
        pointers = self.get_roots(pointers)
        # We now have our roots. Relabel so that they go from 0 to the length of our
        # roots
        unique_roots, labels_flat = np.unique(pointers, return_inverse=True)
        # If we had at least one vacuum point, we need to subtract our labels by
        # 1 to recover the vacuum label.
        if -1 in unique_roots:
            labels_flat -= 1
        # reconstruct a 3D array with our labels
        labels = labels_flat.reshape(shape)
        # reduce maxima/basins
        labels, self._maxima_frac = self.reduce_label_maxima(labels)
        # assign all results
        results = {
            "basin_labels": labels,
        }
        # assign charges/volumes, etc.
        results.update(self.get_basin_charges_and_volumes(labels))
        results.update(self.get_extras())
        return results
