# -*- coding: utf-8 -*-

import logging

import numpy as np

from baderkit.core.methods.base import MethodBase

from .weight_numba import (  # reduce_charge_volume,
    get_labels,
    get_neighbor_flux,
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
        reference_grid = self.reference_grid.copy()
        # get the voronoi neighbors, their distances, and the area of the corresponding
        # facets. This is used to calculate the volume flux from each voxel
        neighbor_transforms, neighbor_dists, facet_areas, _ = (
            reference_grid.point_neighbor_voronoi_transforms
        )
        # get a single alpha corresponding to the area/dist
        neighbor_alpha = facet_areas / neighbor_dists
        logging.info("Sorting reference data")
        data = reference_grid.total
        shape = reference_grid.shape
        # flatten data and get array of coordinates
        sorted_data = data.ravel()
        sorted_charge = self.charge_grid.total.ravel()
        sorted_coords = np.indices(shape, dtype=np.int64).reshape(3, -1).T
        # get sorted indices from lowest to highest and remove vacuum
        sorted_indices = np.argsort(sorted_data, kind="stable")[self.num_vacuum :]
        # get sorted data
        sorted_data = sorted_data[sorted_indices]
        sorted_charge = sorted_charge[sorted_indices]
        sorted_coords = sorted_coords[sorted_indices]
        # get pointers from 3D indices to sorted 1D
        sorted_pointers = np.empty(shape, dtype=np.int64)
        sorted_pointers[
            sorted_coords[:, 0], sorted_coords[:, 1], sorted_coords[:, 2]
        ] = np.arange(len(sorted_coords), dtype=np.int64)

        # Get the flux of volume from each voxel to its neighbor.
        logging.info("Calculating voxel flux contributions")
        all_neighbor_transforms, all_neighbor_dists = (
            reference_grid.point_neighbor_transforms
        )
        neigh_fluxes, neigh_pointers, weight_maxima_mask = get_neighbor_flux(
            data=data,
            sorted_coords=sorted_coords.copy(),
            sorted_pointers=sorted_pointers,
            neighbor_transforms=neighbor_transforms,
            neighbor_alpha=neighbor_alpha,
            all_neighbor_transforms=all_neighbor_transforms,
            all_neighbor_dists=all_neighbor_dists,
        )
        # We want to reduce the number of basins prior to the next step
        weight_maxima_vox = sorted_coords[weight_maxima_mask]
        # create the maxima mask
        self._maxima_mask = np.zeros(data.shape, dtype=np.bool_)
        self._maxima_mask[
            weight_maxima_vox[:, 0],
            weight_maxima_vox[:, 1],
            weight_maxima_vox[:, 2],
        ] = True
        # get labels
        labels = np.full(data.shape, -1, dtype=np.int64)
        labels[self._maxima_mask] = np.arange(len(weight_maxima_vox))
        # reduce maxima/labels and save frac coords.
        # NOTE: reduction algorithm returns with unlabeled values as -1
        labels, self._maxima_frac, label_map = self.reduce_label_maxima(labels, True)
        maxima_num = len(self.maxima_frac)
        # Our label grid goes from 0 (or -1 with vac) upward. We need to relabel
        # the maxima with an actual maximum index so that our pointers work.
        labels, label_map = relabel_reduced_maxima(
            labels,
            maxima_num,
            weight_maxima_vox,
            reference_grid.flat_grid_indices,
        )

        logging.info("Assigning charges and volumes")
        labels, charges, volumes = get_weight_assignments(
            data,
            labels,
            label_map,
            sorted_coords,
            sorted_charge,
            reference_grid.flat_grid_indices,
            neigh_fluxes,
            neigh_pointers,
            weight_maxima_mask,
            maxima_num,
        )
        # NOTE: The maxima found through this method are now the same as those
        # in other methods, without the need to reduce them in an additional step

        logging.info("Finding labels")
        # our current labels are pointers like the ongrid/neargrid methods. We
        # need to get the roots
        labels = labels.ravel()
        # get roots
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
        # rearrange the charges/volumes to match the labels
        # get the voxel coords of the maxima found throught the weight method
        maxima_vox = (
            np.round(reference_grid.frac_to_grid(self._maxima_frac))
            / reference_grid.shape
        ).astype(int)
        maxima_labels = labels[maxima_vox[:, 0], maxima_vox[:, 1], maxima_vox[:, 2]]
        # reorganize charges/volumes to match properly ordered maxima labels
        sorted_maxima = np.argsort(maxima_labels, kind="stable")
        charges = charges[sorted_maxima]
        volumes = volumes[sorted_maxima]

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
