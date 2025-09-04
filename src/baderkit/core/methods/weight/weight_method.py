# -*- coding: utf-8 -*-

import logging

import numpy as np

from baderkit.core.methods.base import MethodBase

from .weight_numba import (
    get_neighbor_flux,
    get_weight_assignments,
    reduce_charge_volume,
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
        # get sorted indices from lowest to highest and remove vacuum
        sorted_indices = np.flip(np.argsort(reference_data.ravel(), kind="stable")[self.num_vacuum:])
        sorted_charge = charge_data.ravel()[sorted_indices]
        # get the voronoi neighbors, their distances, and the area of the corresponding
        # facets. This is used to calculate the volume flux from each voxel
        neighbor_transforms, neighbor_dists, facet_areas, _ = (
            reference_grid.point_neighbor_voronoi_transforms
        )
        # get a single alpha corresponding to the area/dist
        neighbor_alpha = facet_areas / neighbor_dists
        
        # calculate flux flowing from each lower neighbor to each point
        logging.info("Calculating voxel flux contributions")
        neigh_fluxes, neigh_pointers, labels, maxima_mask = get_neighbor_flux(
            data=reference_data,
            neighbor_transforms=neighbor_transforms,
            neighbor_alpha=neighbor_alpha,
            sorted_indices=sorted_indices,
        )
        # reorder flux/pointers
        neigh_fluxes = neigh_fluxes[sorted_indices]
        neigh_pointers = neigh_pointers[sorted_indices]
        maxima_mask = maxima_mask[sorted_indices]
        # breakpoint()
        # breakpoint()
        # calculate charges/volumes
        logging.info("Calculating charges and volumes")
        # get full 26 neighbors for ongrid steps
        all_neighbor_transforms, all_neighbor_dists = (
            reference_grid.point_neighbor_transforms
        )
        charges, volumes, labels, maxima_labels = get_weight_assignments(
            reference_data,
            labels,
            sorted_charge,
            sorted_indices,
            neigh_fluxes,
            neigh_pointers,
            maxima_mask,
            all_neighbor_transforms,
            all_neighbor_dists,
        )
        
        # get labels
        logging.info("Assigning labels")
        labels = labels.ravel()
        labels = self.get_roots(labels)
        # We now have our roots. Relabel so that they go from 0 to the length of our
        # roots
        unique_roots, labels = np.unique(labels, return_inverse=True)
        # shift back to vacuum at -1
        if -1 in unique_roots:
            labels -= 1
        # reconstruct a 3D array with our labels
        labels = labels.reshape(reference_grid.shape)
        # get maxima mask
        maxima_mask = np.zeros(charge_grid.ngridpts, dtype=np.bool_)
        maxima_mask[unique_roots] = True
        self._maxima_mask = maxima_mask.reshape(shape)
        
        # reduce maxima/basins
        labels, self._maxima_frac, label_map = self.reduce_label_maxima(labels, True)
        # combine charges/volumes
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
