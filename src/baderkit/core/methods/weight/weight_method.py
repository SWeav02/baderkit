# -*- coding: utf-8 -*-

import logging

import numpy as np

from baderkit.core.methods.base import MethodBase

from .weight_numba import (
    get_multi_weight_voxels,
    get_neighbor_flux,
    get_single_weight_voxels,
    reduce_weight_maxima,
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
        reference_grid = self.reference_grid.copy()
        # get the voronoi neighbors, their distances, and the area of the corresponding
        # facets. This is used to calculate the volume flux from each voxel
        neighbor_transforms, neighbor_dists, facet_areas, _ = (
            reference_grid.voxel_voronoi_facets
        )
        logging.info("Sorting reference data")
        data = reference_grid.total
        shape = reference_grid.shape
        # flatten data and get initial 1D and 3D voxel indices
        flat_data = data.ravel()
        flat_voxel_indices = np.arange(np.prod(shape))
        flat_voxel_coords = np.indices(shape, dtype=np.int64).reshape(3, -1).T
        # sort data from high to low
        sorted_data_indices = np.flip(np.argsort(flat_data, kind="stable"))
        # create an array that maps original voxel indices to their range in terms
        # of data
        flat_sorted_voxel_indices = np.empty_like(flat_voxel_indices)
        flat_sorted_voxel_indices[sorted_data_indices] = flat_voxel_indices
        # Get a 3D grid representing this data and the corresponding 3D indices
        sorted_voxel_indices = flat_sorted_voxel_indices.reshape(shape)
        sorted_voxel_coords = flat_voxel_coords[sorted_data_indices]
        # remove vacuum points from our list of voxel indices
        sorted_voxel_coords = sorted_voxel_coords[
            : len(sorted_voxel_coords) - self.num_vacuum
        ]
        # Get the flux of volume from each voxel to its neighbor.
        logging.info("Calculating voxel flux contributions")
        flux_array, neigh_indices_array, weight_maxima_mask = get_neighbor_flux(
            data=data,
            sorted_voxel_coords=sorted_voxel_coords.copy(),
            voxel_indices=sorted_voxel_indices,
            neighbor_transforms=neighbor_transforms,
            neighbor_dists=neighbor_dists,
            facet_areas=facet_areas,
        )
        # get the voxel coords of the maxima found throught the weight method
        weight_maxima_vox = sorted_voxel_coords[weight_maxima_mask]
        # Calculate the weights for each voxel to each basin
        logging.info("Calculating weights, charges, and volumes")
        # get charge and volume info
        charge_data = self.charge_grid.total
        flat_charge_data = charge_data.ravel()
        sorted_flat_charge_data = flat_charge_data[sorted_data_indices]
        # remove vacuum from charge data
        sorted_flat_charge_data = sorted_flat_charge_data[: len(sorted_voxel_coords)]
        voxel_volume = reference_grid.voxel_volume

        # There are a few ways that we might end up with extra, non-physical
        # maxima. The default weight method doesn't use all 26 neighbors when
        # defining maxima, which can result in some strange assignments. Additionally,
        # a maximum might be between two symmetrical points, resulting in both
        # points being labeled as maxima. I believe it is more reasonable to
        # reduce these maxima, and this also saves time/memory for the rest of
        # the process.

        # first we reduce to maxima that are higher than the 26 nearest neighbors
        logging.info("Reducing maxima")
        all_neighbor_transforms, all_neighbor_dists = reference_grid.voxel_26_neighbors
        maxima_connections = reduce_weight_maxima(
            weight_maxima_vox,
            data,
            all_neighbor_transforms,
            all_neighbor_dists,
        )
        # NOTE: The maxima are already sorted from highest to lowest
        # We now have a 1D array pointing each maximum to the index of the
        # actual maximum it connects to. We want to reset these so that they
        # run from 0 upward with -1 being unlabeled
        unique_maxima, labels_flat = np.unique(maxima_connections, return_inverse=True)
        # create a labels array and label maxima
        labels = np.full(data.shape, -1, dtype=np.int64)
        labels[
            weight_maxima_vox[:, 0],
            weight_maxima_vox[:, 1],
            weight_maxima_vox[:, 2],
        ] = labels_flat
        # Get the true maxima vox coords
        maxima_vox_coords = weight_maxima_vox[
            maxima_connections == np.arange(len(maxima_connections))
        ]
        # Now reduce maxima that are adjacent.
        self._maxima_mask = np.zeros(data.shape, dtype=np.bool_)
        self._maxima_mask[
            maxima_vox_coords[:, 0],
            maxima_vox_coords[:, 1],
            maxima_vox_coords[:, 2],
        ] = True
        # reduce labels and save frac coords.
        # NOTE: reduction algorithm returns with unlabeled values as -1
        labels, self._maxima_frac = self.reduce_label_maxima(labels)
        maxima_num = len(self.maxima_frac)
        # get labels for voxels with one weight
        labels, unassigned_mask, charges, volumes = get_single_weight_voxels(
            neigh_indices_array=neigh_indices_array,
            sorted_voxel_coords=sorted_voxel_coords,
            data=data,
            maxima_num=maxima_num,
            sorted_flat_charge_data=sorted_flat_charge_data,
            voxel_volume=voxel_volume,
            labels=labels,
        )
        # Now we have the labels for the voxels that have exactly one weight.
        # We want to get the weights for those that are split. To do this, we
        # need an array with a (N, maxima_num) shape, where N is the number of
        # unassigned voxels. Then we also need an array pointing each unassigned
        # voxel to its point in this array
        unass_to_vox_pointer = np.where(unassigned_mask)[0]
        unassigned_num = len(unass_to_vox_pointer)

        vox_to_unass_pointer = np.full(len(neigh_indices_array), -1, dtype=np.int64)
        vox_to_unass_pointer[unassigned_mask] = np.arange(unassigned_num)

        # get labels, charges, and volumes
        labels, charges, volumes = get_multi_weight_voxels(
            flux_array=flux_array,
            neigh_indices_array=neigh_indices_array,
            labels=labels,
            unass_to_vox_pointer=unass_to_vox_pointer,
            vox_to_unass_pointer=vox_to_unass_pointer,
            sorted_voxel_coords=sorted_voxel_coords,
            charge_array=charges,
            volume_array=volumes,
            sorted_flat_charge_data=sorted_flat_charge_data,
            voxel_volume=voxel_volume,
            maxima_num=maxima_num,
        )

        charges /= shape.prod()
        # assign all values
        results = {
            "basin_labels": labels,
            "basin_charges": charges,
            "basin_volumes": volumes,
            "vacuum_charge": charge_data[self.vacuum_mask].sum() / shape.prod(),
            "vacuum_volume": (self.num_vacuum / reference_grid.voxel_num)
            * reference_grid.structure.volume,
        }
        results.update(self.get_extras())
        return results
