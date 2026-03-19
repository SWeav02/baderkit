# -*- coding: utf-8 -*-

import logging

import numpy as np

from baderkit.core.bader.methods.base import MethodBase
from baderkit.core.utilities.basic import coords_to_flat, get_lowest_uint

from .weight_numba import (  # reduce_charge_volume,; get_labels,
    get_weight_assignments,
    sort_extrema_frac,
)


class WeightMethod(MethodBase):

    def _run_bader(self):
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
        labels = self.labels
        images = self.images
        reference_data = reference_grid.total
        charge_data = charge_grid.total
        shape = reference_grid.shape
        nx, ny, nz = shape
        ny_nz = ny * nz

        # get flat indices of extrema
        extrema_indices = np.empty(len(self.extrema_vox), dtype=np.uint32)
        for idx, (i, j, k) in enumerate(self.extrema_vox):
            extrema_indices[idx] = coords_to_flat(i, j, k, ny_nz, nz)
        # sort low to high so that searchsorted works
        extrema_indices = np.sort(extrema_indices)

        # mark vacuum points
        labels[self.vacuum_mask.ravel()] = np.iinfo(labels.dtype).max - 1

        logging.info("Sorting Reference Data")
        # sort data from lowest to highest, ignoring vacuum points
        sorted_indices = np.argsort(reference_data[~self.vacuum_mask], kind="stable")

        # map these truncated sorted indices back to the original flat indices.
        # We also flip the indices to go from high to low here when looking for
        # maxima.
        non_vacuum_indices = np.where((~self.vacuum_mask).ravel())[0]
        if not self.use_minima:
            sorted_indices = non_vacuum_indices[np.flip(sorted_indices)]
        else:
            sorted_indices = non_vacuum_indices[sorted_indices]

        # get the voronoi neighbors, their distances, and the area of the corresponding
        # facets. This is used to calculate the volume flux from each voxel
        neighbor_transforms, neighbor_dists, facet_areas, _ = (
            reference_grid.point_neighbor_voronoi_transforms
        )

        # get a single alpha corresponding to the area/dist
        neighbor_alpha = facet_areas / neighbor_dists

        # Get the flux of volume from each voxel to its neighbor.
        logging.info("Assigning Charges and Volumes")
        all_neighbor_transforms, all_neighbor_dists = (
            reference_grid.point_neighbor_transforms
        )

        labels, images, charges, volumes = get_weight_assignments(
            reference_data,
            labels,
            images,
            charge_data,
            sorted_indices,
            neighbor_transforms,
            neighbor_alpha,
            all_neighbor_transforms,
            all_neighbor_dists,
            self.extrema_mask,
            extrema_indices,
            use_minima=self.use_minima,
        )

        # reconstruct a 3D array with our labels
        labels = labels.reshape(shape)
        # update unassigned labels
        unassigned_mask = labels == np.iinfo(labels.dtype).max - 1
        vacuum_label = len(self.extrema_vox)
        dtype = get_lowest_uint(vacuum_label)
        labels[unassigned_mask] = vacuum_label


        # adjust charges from vasp convention
        charges /= shape.prod()
        # adjust volumes from voxel count
        volumes *= reference_grid.point_volume
        # condense images
        images = self.condense_images(images)
        images = images.reshape(shape)

        # set results
        self._labels = labels.astype(dtype)
        self._images = images
        self._charges = charges
        self._volumes = volumes
        self._vacuum_charge = (
            self.charge_grid.total[unassigned_mask].sum() / shape.prod()
        )
        self._vacuum_volume = (
            (self.num_vacuum / reference_grid.ngridpts)
            * reference_grid.structure.volume,
        )