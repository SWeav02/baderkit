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
        # # get sorted indices from highest to lowest
        # sorted_indices = np.flip(np.argsort(reference_data.ravel(), kind="stable"))
        sorted_indices = np.argsort(reference_data.ravel(), kind="stable")
        # get a map from index to sorted index
        # indices_to_sorted = np.empty_like(sorted_indices)
        # indices_to_sorted[sorted_indices] = np.arange(reference_grid.ngridpts)

        # remove vacuum from sorted indices
        sorted_indices = sorted_indices[:reference_grid.ngridpts-self.num_vacuum]
        # get the voronoi neighbors, their distances, and the area of the corresponding
        # facets. This is used to calculate the volume flux from each voxel
        neighbor_transforms, neighbor_dists, facet_areas, _ = (
            reference_grid.point_neighbor_voronoi_transforms
        )
        # # get a single alpha corresponding to the area/dist
        neighbor_alpha = facet_areas / neighbor_dists
        # logging.info("Sorting reference data")
        # data = reference_grid.total
        # shape = reference_grid.shape
        # # flatten data and get array of coordinates
        # sorted_data = data.ravel()
        # sorted_charge = self.charge_grid.total.ravel()
        # sorted_coords = np.indices(shape, dtype=np.int64).reshape(3, -1).T
        # # get sorted indices from lowest to highest and remove vacuum
        # sorted_indices = np.argsort(sorted_data, kind="stable")[self.num_vacuum :]
        # # get sorted data
        # sorted_data = sorted_data[sorted_indices]
        # sorted_charge = sorted_charge[sorted_indices]
        # sorted_coords = sorted_coords[sorted_indices]
        # # get pointers from 3D indices to sorted 1D
        # sorted_pointers = np.empty(shape, dtype=np.int64)
        # sorted_pointers[
        #     sorted_coords[:, 0], sorted_coords[:, 1], sorted_coords[:, 2]
        # ] = np.arange(len(sorted_coords), dtype=np.int64)

        # Get the flux of volume from each voxel to its neighbor.
        logging.info("Calculating voxel flux contributions")
        all_neighbor_transforms, all_neighbor_dists = (
            reference_grid.point_neighbor_transforms
        )
        labels, charges, volumes, self._maxima_mask = get_weight_assignments(
            reference_data,
            charge_data.copy(),
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
