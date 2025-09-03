# -*- coding: utf-8 -*-

import logging
import time

import numpy as np

from baderkit.core.methods.base import MethodBase

from .weight_numba import (
    get_neighbor_flux,
    get_weight_assignments,
    reduce_charge_volume,
    get_labels,
    get_labels_fine,
    # remove_edge_labels,
    # relabel_edges,
)

# TODO: Use list of list storage for initial flux calcs. For points that would
# be given no flux, check if they're true maxima and make all flux point to highest
# neighbor. That should prevent fake maxima so we don't need to peform maxima reduction


class WeightMethod(MethodBase):
    _refine_edges = True

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
        # get sorted indices from lowest to highest
        sorted_indices = np.argsort(sorted_data, kind="stable")
        # sort data and coords
        sorted_data = sorted_data[sorted_indices]
        sorted_charge = sorted_charge[sorted_indices]
        sorted_coords = sorted_coords[sorted_indices]
        # remove vaccum coords
        sorted_data = sorted_data[self.num_vacuum:]
        sorted_charge = sorted_charge[self.num_vacuum:]
        sorted_coords = sorted_coords[self.num_vacuum:]
        # get pointers from 3D indices to sorted 1D
        sorted_pointers = np.empty(shape, dtype=np.int64)
        sorted_pointers[sorted_coords[:,0],sorted_coords[:,1],sorted_coords[:,2]] = np.arange(len(sorted_coords), dtype=np.int64)
        
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
        logging.info("Calculating weights, charges, and volumes")
        
        labels, charges, volumes = get_weight_assignments(
            data,
            sorted_coords,
            sorted_charge,
            reference_grid.flat_grid_indices,
            neigh_fluxes,
            neigh_pointers,
            weight_maxima_mask,
            )
        # NOTE: The maxima found through this method are now the same as those
        # in other methods, without the need to reduce them in an additional step
        # get the voxel coords of the maxima found throught the weight method
        maxima_vox = sorted_coords[weight_maxima_mask]
        
        logging.info("Finding roots")
        if self._refine_edges:
            maxima_labels = reference_grid.flat_grid_indices[maxima_vox[:,0], maxima_vox[:,1], maxima_vox[:,2]]
            labels = get_labels_fine(
                labels,
                maxima_labels,
                reference_grid.flat_grid_indices,
                sorted_indices,
                neigh_pointers,
                # neigh_fluxes,
                volumes,
                sorted_coords,
                    )
        else:
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
        maxima_labels = labels[maxima_vox[:,0],maxima_vox[:,1],maxima_vox[:,2]]
        # reorganize charges/volumes to match properly ordered maxima labels
        sorted_maxima = np.argsort(maxima_labels, kind="stable")
        charges = charges[sorted_maxima]
        volumes = volumes[sorted_maxima]
        
        # create the maxima mask
        self._maxima_mask = np.zeros(data.shape, dtype=np.bool_)
        self._maxima_mask[
            maxima_vox[:, 0],
            maxima_vox[:, 1],
            maxima_vox[:, 2],
        ] = True
        
        # reduce maxima/basins
        labels, self._maxima_frac, label_map = self.reduce_label_maxima(labels, True)
        
        # reduce charges/volumes
        charges, volumes = reduce_charge_volume(label_map, charges, volumes, len(self._maxima_frac))
        
        # TODO: Make label edge refinement optional for workflows that don't need
        # it.
        # Labels that are exactly split to multiple basins may not be reasonable.
        # we reassign them here
        # if self._refine_edges:
        #     # first, sort from highest to lowest values
        #     to_refine = np.flip(to_refine)
        #     # now relabel the edges
        #     labels = relabel_edges(
        #         to_refine,
        #         sorted_coords,
        #         sorted_charge,
        #         labels,
        #         fluxes,
        #         neigh_pointers,
        #         len(self._maxima_frac),
        #             )
        
        # adjust charges from vasp convention
        charges /= shape.prod()
        # adjust volumes from voxel count
        volumes *= reference_grid.point_volume
        # assign all values
        results = {
            "basin_labels": labels,
            "basin_charges": charges,
            "basin_volumes": volumes,
            "vacuum_charge": self.charge_grid.total[self.vacuum_mask].sum() / shape.prod(),
            "vacuum_volume": (self.num_vacuum / reference_grid.ngridpts)
            * reference_grid.structure.volume,
        }
        results.update(self.get_extras())
        return results
