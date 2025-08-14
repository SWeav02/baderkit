# # -*- coding: utf-8 -*-

"""
This method is the original version trying to emulate the Henkelman group's code
as close as possible. It has been replaced with a faster method that is quite
differet, but is kept here in case we ever need to revert back or include it somehow.
"""

# import logging

# import numpy as np

# from baderkit.core.methods.base import MethodBase
# from baderkit.core.methods.shared_numba import get_edges

# from .neargrid_numba import (
#     get_neargrid_labels,
#     get_ongrid_and_rgrads,
#     # refine_neargrid,
# )
# from baderkit.core.methods.neargrid.neargrid_numba import refine_fast_neargrid


# class NeargridOldMethod(MethodBase):

#     def run(self):
#         """
#         Assigns voxels to basins and calculates charge using the near-grid
#         method:
#             W. Tang, E. Sanville, and G. Henkelman
#             A grid-based Bader analysis algorithm without lattice bias
#             J. Phys.: Condens. Matter 21, 084204 (2009)

#         Returns
#         -------
#         None.

#         """
#         grid = self.reference_grid.copy()
#         # get neigbhor transforms
#         neighbor_transforms, neighbor_dists = grid.voxel_26_neighbors
#         logging.info("Calculating gradients")
#         highest_neighbors, all_drs, gradients, self._maxima_mask = get_ongrid_and_rgrads(
#             data=grid.total,
#             dir2lat=self.dir2lat,
#             neighbor_dists=neighbor_dists,
#             neighbor_transforms=neighbor_transforms,
#             vacuum_mask=self.vacuum_mask,
#         )
#         logging.info("Calculating initial labels")
#         # get initial labels
#         labels = get_neargrid_labels(
#             data=grid.total,
#             highest_neighbors=highest_neighbors,
#             all_drs=all_drs,
#             maxima_mask=self.maxima_mask,
#             vacuum_mask=self.vacuum_mask,
#             neighbor_dists=neighbor_dists,
#             neighbor_transforms=neighbor_transforms,
#         )
#         labels -= 1
#         # reduce labels
#         labels, self._maxima_frac = self.reduce_label_maxima(labels)
#         # we now have an array with labels ranging from 0 up (if theres vacuum)
#         # or 1 up (if no vacuum). We want to reduce the number of maxima if there
#         # are any that border each other. Our reduction algorithm requires unlabeled
#         # or vacuum points to be -1 and 0 and up for basins
#         # shift to vacuum at 0
#         labels += 1

#         # Now we refine the edges with the neargrid method
#         # Get our edges, not including edges on the vacuum.
#         refinement_mask = get_edges(
#             labeled_array=labels,
#             neighbor_transforms=neighbor_transforms,
#             vacuum_mask=self.vacuum_mask,
#         )
#         # remove maxima from refinement
#         refinement_mask[self.maxima_mask] = False
#         # note these labels should not be reassigned again in future cycles
#         labels[refinement_mask] = -labels[refinement_mask]
#         labels = refine_fast_neargrid(
#             data=grid.total,
#             labels=labels,
#             refinement_mask=refinement_mask,
#             maxima_mask=self.maxima_mask,
#             gradients=gradients,
#             neighbor_dists=neighbor_dists,
#             neighbor_transforms=neighbor_transforms,
#         )

#         # switch negative labels back to positive and subtract by 1 to get to
#         # correct indices
#         labels = np.abs(labels) - 1
#         # get all results
#         results = {
#             "basin_labels": labels,
#         }
#         # assign charges/volumes, etc.
#         results.update(self.get_basin_charges_and_volumes(labels))
#         results.update(self.get_extras())
#         return results
