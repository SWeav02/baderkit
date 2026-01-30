# -*- coding: utf-8 -*-

import logging

import numpy as np

from baderkit.core.bader.methods.base import MethodBase

from baderkit.core.utilities.basic import get_lowest_uint

from .ongrid_numba import get_steepest_pointers


class OngridMethod(MethodBase):

    def _run_bader(self, labels, shifts):
        """
        Assigns voxels to basins and calculates charge using the on-grid
        method:
            G. Henkelman, A. Arnaldsson, and H. Jónsson
            A fast and robust algorithm for Bader decomposition of charge density,
            Comput. Mater. Sci. 36, 354-360 (2006)

        Returns
        -------
        None.

        """
        grid = self.reference_grid
        data = grid.total
        shape = data.shape
        # get shifts to move from a voxel to the 26 surrounding voxels
        neighbor_transforms, neighbor_dists = grid.point_neighbor_transforms
        # For each voxel, get the label of the surrounding voxel that has the highest
        # density
        logging.info("Calculating Steepest Neighbors")
        labels, shifts = get_steepest_pointers(
            data=data,
            labels=labels,
            shifts=shifts,
            neighbor_transforms=neighbor_transforms,
            neighbor_dists=neighbor_dists,
            vacuum_mask=self.vacuum_mask,
            maxima_mask=self.maxima_mask,
        )
        
        # Our pointers object is a 1D array pointing each voxel to its parent voxel. We
        # essentially have a classic forest of trees problem where each maxima is
        # a root and we want to point all of our voxels to their respective root.
        # We being a while loop. In each loop, we remap our pointers to point at
        # the index that its parent was pointing at.
        # NOTE: Vacuum points are indicated by a value of -1 and we want to
        # ignore these
        logging.info("Finding Roots")
        labels, shifts = self.get_roots(labels, shifts)
        shifts = self.condense_shifts(shifts)
            
        # reconstruct a 3D array with our labels and images
        labels = labels.reshape(shape)
        shifts = shifts.reshape(shape)

        # assign all results
        results = {
            "basin_labels": labels,
            "basin_shifts": shifts,
        }
        # assign charges/volumes, etc.
        logging.info("Assigning Charges and Volumes")
        results.update(self.get_basin_charges_and_volumes(labels))
        return results
