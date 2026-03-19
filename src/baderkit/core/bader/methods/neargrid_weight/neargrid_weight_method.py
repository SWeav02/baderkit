# -*- coding: utf-8 -*-

import logging

import numpy as np

from baderkit.core.bader.methods.base import MethodBase
from baderkit.core.bader.methods.neargrid.neargrid_numba import (
    get_gradient_pointers_overdetermined,
    get_gradient_pointers_simple,
    refine_fast_neargrid,
)
from baderkit.core.utilities.basic import get_lowest_int, get_lowest_uint
from baderkit.core.utilities.basins import get_edges_w_flat_images, get_edges_w_images

from .neargrid_weight_numba import (
    get_edge_charges_volumes,
    get_interior_basin_charges_and_volumes,
)


class NeargridWeightMethod(MethodBase):

    _use_overdetermined = False

    def _run_bader(self):
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
        labels = self.labels
        images = self.images
        reference_data = reference_grid.total
        charge_data = charge_grid.total
        shape = reference_grid.shape
        # get neigbhor transforms
        neighbor_transforms, neighbor_dists = reference_grid.point_neighbor_transforms
        logging.info("Calculating Gradients")
        if not self._use_overdetermined:
            # calculate gradients and pointers to best neighbors
            labels, images, gradients = get_gradient_pointers_simple(
                data=reference_data,
                labels=labels,
                images=images,
                dir2lat=self.dir2lat,
                neighbor_dists=neighbor_dists,
                neighbor_transforms=neighbor_transforms,
                vacuum_mask=self.vacuum_mask,
                extrema_mask=self.extrema_mask,
                use_minima=self.use_minima,
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
            labels, images, gradients, self._extrema_mask = (
                get_gradient_pointers_overdetermined(
                    data=reference_data,
                    labels=labels,
                    images=images,
                    car2lat=self.car2lat,
                    inv_norm_cart_trans=inv_norm_cart_trans,
                    neighbor_dists=neighbor_dists,
                    neighbor_transforms=neighbor_transforms,
                    vacuum_mask=self.vacuum_mask,
                    extrema_mask=self.extrema_mask,
                    use_minima=self.use_minima,
                )
            )

        # Find roots
        logging.info("Finding Roots")
        labels, images = self.get_roots(labels, images)

        # reconstruct a 3D array with our labels. make sure our data type can
        # include negative values so that we can mark points needing refinement
        dtype = get_lowest_int(len(self.extrema_vox) + 1)
        labels = labels.reshape(shape).astype(dtype)

        logging.info("Starting Edge Refinement")

        # shift indices to start at 1
        labels += 1

        # Now we refine the edges with the neargrid method
        # Get our edges, not including edges on the vacuum.
        vacuum_label = len(self.extrema_vox) + 1
        refinement_mask = get_edges_w_flat_images(
            labeled_array=labels,
            images=images,
            neighbor_transforms=neighbor_transforms,
            vacuum_label=vacuum_label,
        )
        vacuum_mask = labels == vacuum_label
        # remove extrema from refinement
        refinement_mask[self.extrema_mask] = False
        # note these labels and the vacuum should not be reassigned again in future cycles
        labels[refinement_mask & vacuum_mask] = -labels[
            refinement_mask & vacuum_mask
        ]
        labels, images = refine_fast_neargrid(
            data=reference_data,
            labels=labels,
            images=images,
            refinement_mask=refinement_mask,
            extrema_mask=self.extrema_mask,
            gradients=gradients,
            neighbor_dists=neighbor_dists,
            neighbor_transforms=neighbor_transforms,
            vacuum_label=-vacuum_label,
            use_minima=self.use_minima,
        )
        # switch negative labels back to positive and subtract by 1 to get to
        # correct indices
        labels = np.abs(labels) - 1
        dtype = get_lowest_uint(len(self.extrema_vox))
        labels = labels.reshape(shape).astype(dtype)
        vacuum_label -= 1
        # condense images
        images = self.condense_images(images)
        images = images.reshape(shape)

        # get final edges
        edge_mask = get_edges_w_images(
            labeled_array=labels,
            images=images,
            neighbor_transforms=neighbor_transforms,
            vacuum_label=vacuum_label,
        )

        logging.info("Assigning Interior Charge and Volume")
        # get interior charge/volume
        charges, volumes, vacuum_charge, vacuum_volume = (
            get_interior_basin_charges_and_volumes(
                data=charge_data,
                labels=labels,
                cell_volume=reference_grid.structure.volume,
                extrema_num=len(self.extrema_vox),
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
        edge_data = reference_data[edge_mask]
        # sort the data
        sorted_indices = np.argsort(edge_data, kind="stable")
        if self.use_minima:
            sorted_indices = np.flip(sorted_indices)
        # get edge charge/volume
        charges, volumes = get_edge_charges_volumes(
            reference_data=reference_data,
            charge_data=charge_data,
            edge_indices=edge_indices,
            sorted_indices=sorted_indices,
            labels=labels,
            charges=charges,
            volumes=volumes,
            vacuum_label=vacuum_label,
            neighbor_transforms=voronoi_neighbor_transforms,
            neighbor_alpha=neighbor_alpha,
            all_neighbor_transforms=neighbor_transforms,
            all_neighbor_dists=neighbor_dists,
            use_minima=self.use_minima,
        )

        volumes = volumes * reference_grid.structure.volume / reference_grid.ngridpts
        charges = charges / reference_grid.ngridpts

        # set results
        self._labels = labels
        self._images = images
        self._charges = charges
        self._volumes = volumes
        self._vacuum_charge = vacuum_charge
        self._vacuum_volume = vacuum_volume