# -*- coding: utf-8 -*-

import logging

import numpy as np

from baderkit.core.methods.base import MethodBase

from .weight_numba import (  # reduce_weight_maxima,
    get_multi_weight_voxels,
    get_neighbor_flux,
    get_single_weight_voxels,
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
        all_neighbor_transforms, all_neighbor_dists = reference_grid.voxel_26_neighbors
        flux_array, neigh_indices_array, weight_maxima_mask = get_neighbor_flux(
            data=data,
            sorted_voxel_coords=sorted_voxel_coords.copy(),
            voxel_indices=sorted_voxel_indices,
            neighbor_transforms=neighbor_transforms,
            neighbor_dists=neighbor_dists,
            facet_areas=facet_areas,
            all_neighbor_transforms=all_neighbor_transforms,
            all_neighbor_dists=all_neighbor_dists,
        )
        # NOTE: The maxima found through this method are now the same as those
        # in other methods, without the need to reduce them in an additional step
        # get the voxel coords of the maxima found throught the weight method
        weight_maxima_vox = sorted_voxel_coords[weight_maxima_mask]
        # create the maxima mask
        self._maxima_mask = np.zeros(data.shape, dtype=np.bool_)
        self._maxima_mask[
            weight_maxima_vox[:, 0],
            weight_maxima_vox[:, 1],
            weight_maxima_vox[:, 2],
        ] = True

        # get charge and volume info
        charge_data = self.charge_grid.total
        flat_charge_data = charge_data.ravel()
        sorted_flat_charge_data = flat_charge_data[sorted_data_indices]
        # remove vacuum from charge data
        sorted_flat_charge_data = sorted_flat_charge_data[: len(sorted_voxel_coords)]
        voxel_volume = reference_grid.voxel_volume
        # create a labels array and label maxima
        labels = np.full(data.shape, -1, dtype=np.int64)
        labels[self._maxima_mask] = np.arange(len(weight_maxima_vox))
        # reduce maxima/labels and save frac coords.
        # NOTE: reduction algorithm returns with unlabeled values as -1
        labels, self._maxima_frac = self.reduce_label_maxima(labels)
        maxima_num = len(self.maxima_frac)

        # Calculate the weights for each voxel to each basin
        logging.info("Calculating weights, charges, and volumes")
        # first get labels for voxels with one weight
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
