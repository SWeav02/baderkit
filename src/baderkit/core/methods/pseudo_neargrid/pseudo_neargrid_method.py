# -*- coding: utf-8 -*-

import logging

import numpy as np

from baderkit.core.methods.base import MethodBase

from .pseudo_neargrid_numba import get_pseudo_neargrid_labels


class PseudoNeargridMethod(MethodBase):

    def run(self):
        """
        Assigns voxels to basins and calculates charge using the pseudo-neargrid
        method.

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
        # sort points from lowest to highest
        shape = grid.shape
        # flatten data and get initial 1D and 3D voxel indices
        flat_data = grid.total.ravel()
        flat_voxel_coords = np.indices(shape).reshape(3, -1).T
        # sort data from high to low
        sorted_data_indices = np.argsort(flat_data, kind="stable")
        sorted_voxel_coords = flat_voxel_coords[sorted_data_indices]
        # remove vacuum voxels from the sorted voxel coords
        sorted_voxel_coords = sorted_voxel_coords[self.num_vacuum :]
        # get pointers
        pointers, self._maxima_mask = get_pseudo_neargrid_labels(
            data=grid.total,
            car2lat=car2lat,
            neighbor_transforms=neighbor_transforms,
            neighbor_dists=neighbor_dists,
            sorted_voxel_coords=sorted_voxel_coords,
            vacuum_mask=self.vacuum_mask,
            initial_labels=grid.all_voxel_indices,
        )
        # Get roots of pointers
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
        labels = labels_flat.reshape(shape)
        # reduce labels for adjacent maxima and store
        labels, self._maxima_frac = self.reduce_label_maxima(
            labels,
        )
        # assign all results
        results = {
            "basin_labels": labels,
        }
        # assign charges/volumes, etc.
        results.update(self.get_basin_charges_and_volumes(labels))
        results.update(self.get_extras())
        return results
