# -*- coding: utf-8 -*-

import logging

import numpy as np

from baderkit.core.methods.base import MethodBase
from baderkit.core.methods.neargrid.neargrid_numba import (
    get_gradient_pointers_overdetermined,
    get_gradient_pointers_simple,
    refine_fast_neargrid,
)
from baderkit.core.methods.shared_numba import combine_maxima_frac, get_edges

from .neargrid_weight_numba import (
    get_edge_charges_volumes,
    get_interior_basin_charges_and_volumes,
)


class NeargridWeightMethod(MethodBase):

    _use_overdetermined = False

    def run(self):
        """
        Assigns voxels to basins and calculates charge using the near-grid
        method:
            W. Tang, E. Sanville, and G. Henkelman
            A grid-based Bader analysis algorithm without lattice bias
            J. Phys.: Condens. Matter 21, 084204 (2009)

        Returns
        -------
        None.

        """
        reference_grid = self.reference_grid
        charge_grid = self.charge_grid
        reference_data = reference_grid.total
        charge_data = charge_grid.total
        shape = reference_grid.shape
        # get neigbhor transforms
        neighbor_transforms, neighbor_dists = reference_grid.point_neighbor_transforms
        logging.info("Calculating Gradients")
        if not self._use_overdetermined:
            # calculate gradients and pointers to best neighbors
            labels, gradients, self._maxima_mask = get_gradient_pointers_simple(
                data=reference_data,
                dir2lat=self.dir2lat,
                neighbor_dists=neighbor_dists,
                neighbor_transforms=neighbor_transforms,
                vacuum_mask=self.vacuum_mask,
            )
        else:
            # NOTE: This is an alternatvie method using an overdetermined system
            # of all 26 neighbors to calculate the gradient. I didn't see any
            # improvement for NaCl or H2O, but both were cubic systems.
            # get cartesian transforms and normalize
            cart_transforms = reference_grid.grid_to_cart(neighbor_transforms)
            norm_cart_transforms = (
                cart_transforms.T / np.linalg.norm(cart_transforms, axis=1)
            ).T
            # get the pseudo inverse
            inv_norm_cart_trans = np.linalg.pinv(norm_cart_transforms[:13])
            # calculate gradients and pointers to best neighbors
            labels, gradients, self._maxima_mask = get_gradient_pointers_overdetermined(
                data=reference_data,
                car2lat=self.car2lat,
                inv_norm_cart_trans=inv_norm_cart_trans,
                neighbor_dists=neighbor_dists,
                neighbor_transforms=neighbor_transforms,
                vacuum_mask=self.vacuum_mask,
            )
        # Find roots
        # NOTE: Vacuum points are indicated by a value of -1 and we want to
        # ignore these
        logging.info("Finding Roots")
        labels = self.get_roots(labels)
        # We now have our roots. Relabel so that they go from 0 to the length of our
        # roots
        unique_roots, labels = np.unique(labels, return_inverse=True)
        # shift back to vacuum at -1
        if -1 in unique_roots:
            labels -= 1
        # reconstruct a 3D array with our labels
        labels = labels.reshape(shape)
        # get frac coords
        maxima_frac = reference_grid.grid_to_frac(self.maxima_vox)
        self._maxima_frac = combine_maxima_frac(
            labels,
            self.maxima_vox,
            maxima_frac,
        )
        logging.info("Starting Edge Refinement")
        # shift to vacuum at 0
        labels += 1

        # Now we refine the edges with the neargrid method
        # Get our edges, not including edges on the vacuum.
        refinement_mask = get_edges(
            labeled_array=labels,
            neighbor_transforms=neighbor_transforms,
            vacuum_mask=self.vacuum_mask,
        )
        # remove maxima from refinement
        refinement_mask[self.maxima_mask] = False
        # note these labels should not be reassigned again in future cycles
        labels[refinement_mask] = -labels[refinement_mask]
        labels = refine_fast_neargrid(
            data=reference_data,
            labels=labels,
            refinement_mask=refinement_mask,
            maxima_mask=self.maxima_mask,
            gradients=gradients,
            neighbor_dists=neighbor_dists,
            neighbor_transforms=neighbor_transforms,
        )

        # switch negative labels back to positive and subtract by 1 to get to
        # correct indices
        labels = np.abs(labels) - 1

        # get final edges
        edge_mask = get_edges(
            labels,
            neighbor_transforms,
            self.vacuum_mask,
        )
        logging.info("Assigning Interior Charge and Volume")
        # get interior charge/volume
        charges, volumes, vacuum_charge, vacuum_volume = (
            get_interior_basin_charges_and_volumes(
                data=charge_data,
                labels=labels,
                cell_volume=reference_grid.structure.volume,
                maxima_num=len(self.maxima_frac),
                edge_mask=edge_mask,
            )
        )

        logging.info("Assigning Edge Charges and Volumes")
        # Now we want to split the charge of the edges using the weight method.
        # get voronoi neighbors/weights
        voronoi_neighbor_transforms, voronoi_neighbor_dists, facet_areas, _ = (
            reference_grid.point_neighbor_voronoi_transforms
        )
        neighbor_alpha = facet_areas / voronoi_neighbor_dists

        # Get the data at the edges
        edge_indices = np.argwhere(edge_mask)
        edge_data = charge_data[edge_mask]
        # sort the data
        sorted_indices = np.argsort(edge_data, kind="stable")
        # get edge charge/volume
        charges, volumes = get_edge_charges_volumes(
            reference_data=reference_data,
            charge_data=charge_data,
            edge_indices=edge_indices,
            sorted_indices=sorted_indices,
            labels=labels,
            charges=charges,
            volumes=volumes,
            neighbor_transforms=voronoi_neighbor_transforms,
            neighbor_alpha=neighbor_alpha,
            all_neighbor_transforms=neighbor_transforms,
            all_neighbor_dists=neighbor_dists,
        )

        volumes = volumes * reference_grid.structure.volume / reference_grid.ngridpts
        charges = charges / reference_grid.ngridpts

        # get all results
        results = {
            "basin_labels": labels,
            "basin_charges": charges,
            "basin_volumes": volumes,
            "vacuum_charge": vacuum_charge,
            "vacuum_volume": vacuum_volume,
        }
        results.update(self.get_extras())
        return results
