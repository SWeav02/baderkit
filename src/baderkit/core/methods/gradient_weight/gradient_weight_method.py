# -*- coding: utf-8 -*-

import logging

import numpy as np

from baderkit.core.methods.base import MethodBase

from .gradient_weight_numba import get_gradient_pointers


class GradientWeightMethod(MethodBase):
    """
    Assigns each grid point to its highest neighbor determined by calculating
    a weighted pseudo gradient.
    """

    def run(self):
        reference_grid = self.reference_grid.copy()
        # # get the voronoi neighbors, their distances, and the area of the corresponding
        # # facets.
        # neighbor_transforms, neighbor_dists, facet_areas, _ = (
        #     reference_grid.voxel_voronoi_facets
        # )
        neighbor_transforms, neighbor_dists = reference_grid.voxel_26_neighbors
        # Get the transforms in cartesian coordinates, normalize, and weight by
        # area and distance
        neighbor_carts = reference_grid.get_cart_coords_from_vox(neighbor_transforms)
        # neighbor_carts = (
        #     neighbor_carts.T
        #     * facet_areas
        #     / (neighbor_dists * np.linalg.norm(neighbor_carts, axis=1))
        # ).T
        neighbor_carts = (
            neighbor_carts.T / (neighbor_dists * np.linalg.norm(neighbor_carts, axis=1))
        ).T
        # Get all neighbors
        # all_neighbor_transforms, all_neighbor_dists = reference_grid.voxel_26_neighbors
        # Now we get pointers
        logging.info("Calculating steepest neighbors")
        pointers_3d, self._maxima_mask = get_gradient_pointers(
            initial_labels=reference_grid.all_voxel_indices,
            data=reference_grid.total,
            neighbor_transforms=neighbor_transforms,
            neighbor_dists=neighbor_dists,
            weighted_cart=neighbor_carts,
            # all_neighbor_transforms=all_neighbor_transforms,
            # all_neighbor_dists=all_neighbor_dists,
            vacuum_mask=self.vacuum_mask,
            car2lat=self.car2lat,
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
        labels = labels_flat.reshape(reference_grid.shape)
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
