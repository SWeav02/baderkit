# -*- coding: utf-8 -*-

import copy
import itertools
import logging
import time
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from baderkit.core.toolkit import Grid
from baderkit.core.utilities.basic import get_lowest_uint
from baderkit.core.utilities.interpolation import (
    refine_critical_points,
    refine_extrema,
)
from baderkit.core.utilities.persistence import (
    get_canonical_saddle_connections,
    get_single_point_saddles,
    group_by_persistence,
)

from .shared_numba import (  # combine_neigh_extrema,; get_neighboring_basin_connections_w_images,; get_edges_w_images,
    get_basin_charges_and_volumes,
    get_thin_basin_edges,
    get_basin_min_and_max,
    get_extrema,
    initialize_labels_from_extrema,
    update_final_images,
    update_labels_and_images,
)

# This allows for Self typing and is compatible with python 3.10
Self = TypeVar("Self", bound="MethodBase")

# TODO:
# 2. allow for setting periodic boundaries
# 3. get basins counting crossing boundaries as separate
# 3. continue with critical point finder
# 4. update bifurcation method


class MethodBase:
    """
    A base class that all Bader methods inherit from. Designed to handle the
    basin, charge, and volume assignments which are unique to each method.

    Methods are dynamically imported by the Bader class so that we don't need to
    list out the methods in multiple places.
    The method must follow a specific naming convention and be placed in a module
    with a specific name.

    For example, a method with the name example-name
        class name:  ExampleNameMethod
        module name: example_name

    Parameters
    ----------
    charge_grid : Grid
        A Grid object with the charge density that will be integrated.
    reference_grid : Grid
        A grid object whose values will be used to construct the basins.
    vacuum_mask : NDArray[bool]
        A 3D Numpy array with the same shape as the grids that is True where
        points belong to the vacuum
    num_vacuum : int,
        The number of vacuum points in the system.
    persistence_tol: float, optional
        The persistence score tolerance. Pairs of extrema with scores below
        this tolerance will be combined. The score is calculated as:
            dist * (lower_max - saddle_value) / higher_max
        where 'dist' is the cartesian distance between the extrema, lower_max
        is the value at the maximum with a lower value, saddle_value is the
        highest value at which there is a connection between the two extremas
        descending manifold (basin), and higher_max is the value at the maximum
        with a higher value.


    """

    def __init__(
        self,
        charge_grid: Grid,
        reference_grid: Grid,
        vacuum_mask: NDArray[bool],
        num_vacuum: int,
        persistence_tol: float = 0.01,
        use_minima: bool = False,
    ):
        # define variables needed by all methods
        self.charge_grid = charge_grid
        self.reference_grid = reference_grid
        self.vacuum_mask = vacuum_mask
        self.num_vacuum = num_vacuum
        self.use_minima = use_minima

        # scale persistence tolerance to voxel size
        self.persistence_tol = persistence_tol

        # These variables are also often needed but are calculated during the run
        self._extrema_mask = None
        self._extrema_vox = None
        self._extrema_groups = None
        self._extrema_frac = None
        self._car2lat = None
        self._dir2lat = None

    def run(self) -> dict:
        """
        Runs the main bader method and returns a dictionary with values for:
            - extrema_frac
            - extrema_vox
            - extrema_groups
            - extrema_basin_labels
            - extrema_basin_images
            - extrema_ref_values
            - basin_charges
            - basin_volumes
            - vacuum_charges
            - vacuum_volumes
        """
        # first we initialize our label array with all values set to placeholders
        dtype = get_lowest_uint(self.reference_grid.ngridpts)
        vacuum_label = np.iinfo(dtype).max
        labels = np.full(
            self.reference_grid.ngridpts, vacuum_label, dtype=dtype
        )
        # get neighbor transforms
        neighbor_transforms, neighbor_dists = (
            self.reference_grid.point_neighbor_transforms
        )

        # get ongrid extrema
        extrema_vox = self.extrema_vox
        # refine to closest nearby extrema. This helps with ensuring proper
        # initialization
        extrema_frac = extrema_vox / self.reference_grid.shape
        if self.use_minima:
            target_index = 0
        else:
            target_index = 3
        extrema_frac, _ = refine_critical_points(
            points=extrema_frac,
            data=self.reference_grid.total,
            target_index=target_index,
        )

        # initialize our labels, combining false extrema
        logging.info("Initializing Labels")
        t0 = time.time()
        (
            labels,
            images,
            self._extrema_vox,
            self._extrema_frac,
            self._extrema_groups,
        ) = initialize_labels_from_extrema(
            labels=labels,
            data=self.reference_grid.total,
            extrema_frac=extrema_frac,
            extrema_mask=self.extrema_mask,
            neighbor_transforms=neighbor_transforms,
            neighbor_dists=neighbor_dists,
            persistence_tol=self.persistence_tol,
            use_minima=self.use_minima,
            matrix=self.reference_grid.matrix,
            method="linear",
        )

        t1 = time.time()
        logging.info("Initialization Complete")
        logging.info(f"Time: {round(t1-t0,2)}")

        # now run bader
        results = self._run_bader(labels, images)
        labels = results["extrema_basin_labels"]
        images = results["extrema_basin_images"]

        # sort extrema coordinates ensuring they go from lowest to highest label
        extrema_labels = labels[
            self.extrema_vox[:, 0],
            self.extrema_vox[:, 1],
            self.extrema_vox[:, 2],
        ]
        extrema_sorted = np.argsort(extrema_labels)
        self._extrema_vox = self.extrema_vox[extrema_sorted]
        self._extrema_groups = [self.extrema_groups[i] for i in extrema_sorted]

        # Now we want to combine any remaining noisy extrema based on their
        # persistence.
        logging.info("Combining Low-Persistence Basins")
        
        # get saddle locations and values
        saddle_connections, saddle_coords, saddle_values = (
            self.get_saddle_connections(labels, images)
        )
        breakpoint()

        # get saddle coords in cartesian coordinates
        saddle_cart = self.reference_grid.grid_to_cart(saddle_coords)
        # get extrema unions based on persistence and update lowest persistence
        # values for extrema
        (
            extrema_roots,
            root_transforms,
        ) = group_by_persistence(
            data=self.reference_grid.total,
            critical_vox=self.extrema_vox,
            basin_connections=saddle_connections,
            saddle_values=saddle_values,
            saddle_cart=saddle_cart,
            persistence_tol=self.persistence_tol,
            use_minima=self.use_minima,
            matrix=self.reference_grid.matrix,
        )

        # update extrema children, labels, charges, and volumes
        charges = results["basin_charges"]
        volumes = results["basin_volumes"]
        # get unique roots. The indices that result are the labels we want to
        # assign our points to
        final_extrema, inverse, indices = np.unique(
            extrema_roots, return_inverse=True, return_index=True
        )

        # next we want to sort the final indices from low to high
        extrema_vox = self.extrema_vox[final_extrema]
        extrema_vals = self.reference_grid.total[
            extrema_vox[:, 0], extrema_vox[:, 1], extrema_vox[:, 2]
        ]
        ordered_indices = np.argsort(extrema_vals)[::-1]  # descending
        # reorder final_extrema, persistence_cutoffs
        final_extrema = final_extrema[ordered_indices]
        # get the final extrema groups and voxels
        extrema_groups = [self.extrema_groups[i] for i in final_extrema]
        extrema_vox = self.extrema_vox[final_extrema]

        # persistence_cutoffs = persistence_cutoffs[final_extrema]
        final_charges = charges[final_extrema]
        final_volumes = volumes[final_extrema]

        # For each old maximum, get the new index we want to map them onto.
        new_roots = np.empty_like(inverse, dtype=int)
        for new_idx, old_idx in enumerate(ordered_indices):
            new_roots[old_idx] = new_idx

        final_roots = np.empty_like(indices, dtype=int)
        for idx, old_idx in enumerate(indices):
            final_roots[idx] = new_roots[old_idx]

        # combine children, charges, volumes
        for max_idx, (root, old_root) in enumerate(
            zip(final_roots, extrema_roots)
        ):
            if max_idx != old_root:
                extrema_groups[root] = np.append(
                    extrema_groups[root], self.extrema_groups[max_idx], axis=0
                )
                final_charges[root] += charges[max_idx]
                final_volumes[root] += volumes[max_idx]

        # relabel grid
        labels, images = update_labels_and_images(
            labels=labels,
            images=images,
            label_map=final_roots,
            image_map=root_transforms,
            vacuum_mask=self.vacuum_mask,
        )
        # save final results
        self._extrema_vox = extrema_vox
        self._extrema_groups = extrema_groups
        # persistence_cutoffs = persistence_cutoffs[final_extrema]

        if self.use_minima:
            logging.info("Refining Minima")
        else:
            logging.info("Refining Maxima")

        # todo: interpolate this instead?
        extrema_values = self.reference_grid.total[
            self.extrema_vox[:, 0],
            self.extrema_vox[:, 1],
            self.extrema_vox[:, 2],
        ]
        # refine extrema using a newton refinement
        extrema_vox = refine_extrema(
            extrema_coords=self.extrema_vox,
            data=self.reference_grid.total,
            labels=labels,
            lattice=self.reference_grid.matrix,
            use_minima=self.use_minima,
        )

        # get shifts
        extrema_vox_rounded = np.round(extrema_vox)
        shifts = extrema_vox_rounded // self.reference_grid.shape

        extrema_frac = self.reference_grid.grid_to_frac(extrema_vox) - shifts

        self._extrema_vox = extrema_vox_rounded % self.reference_grid.shape
        self._extrema_frac = extrema_frac

        # update images one last time for any shifts
        nonzero_shifts = np.max(np.abs(shifts), axis=1) > 0
        if np.any(nonzero_shifts):
            images = update_final_images(
                labels=labels,
                images=images,
                image_map=-shifts,
                important_mask=nonzero_shifts,
                vacuum_mask=self.vacuum_mask,
            )

        results.update(
            {
                "extrema_basin_labels": labels,
                "extrema_basin_images": images,
                "extrema_vox": self.extrema_vox.astype(np.int16),
                "basin_charges": final_charges,
                "basin_volumes": final_volumes,
                "extrema_voxel_groups": self.extrema_groups,
                "extrema_frac": self.extrema_frac,
                "extrema_ref_values": extrema_values,
                # "extrema_persistence_values": persistence_cutoffs,
            }
        )
        return results

    def _run_bader(self, labels, images) -> dict:
        """
        This is the main function that each method must have. It must return a
        dictionary with values for:
            - extrema_frac
            - extrema_basin_images
            - basin_charges
            - basin_volumes
            - vacuum_charges
            - vacuum_volumes

        """
        raise NotImplementedError(
            "No run method has been implemented for this Bader Method."
        )

    ###########################################################################
    # Properties used by most or all methods
    ###########################################################################

    @property
    def extrema_mask(self) -> NDArray[bool]:
        """

        Returns
        -------
        NDArray[bool]
            A mask representing the voxels that are local extrema.

        """
        if self._extrema_mask is None:
            data = self.reference_grid.total
            neighbor_transforms, _ = (
                self.reference_grid.point_neighbor_transforms
            )
            vacuum_mask = self.vacuum_mask
            self._extrema_mask = get_extrema(
                data=data,
                neighbor_transforms=neighbor_transforms,
                vacuum_mask=vacuum_mask,
                use_minima=self.use_minima,
            )
        return self._extrema_mask

    @property
    def extrema_vox(self) -> NDArray[int]:
        """

        Returns
        -------
        NDArray[int]
            An Nx3 array representing the voxel indices of each local maximum.

        """
        if self._extrema_vox is None:
            self._extrema_vox = np.argwhere(self.extrema_mask)
        return self._extrema_vox

    @property
    def extrema_groups(self) -> NDArray[int]:
        """

        Returns
        -------
        NDArray[int]
            An Nx3 array representing the voxel indices of each local maximum.

        """
        assert (
            self._extrema_groups is not None
        ), "Maxima children must be set by run method"
        return self._extrema_groups

    @property
    def extrema_frac(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            An Nx3 array representing the fractional coordinates of each local
            maximum. These are set after extrema/basin reduction so there may be
            fewer than the number of extrema_vox.

        """
        assert (
            self._extrema_frac is not None
        ), "Maxima frac must be set by run method"
        return self._extrema_frac

    @property
    def car2lat(self) -> NDArray[float]:
        if self._car2lat is None:
            grid = self.reference_grid.copy()
            matrix = grid.matrix
            # convert to lattice vectors as columns
            dir2car = matrix.T
            # get lattice to cartesian matrix
            lat2car = dir2car / grid.shape[np.newaxis, :]
            # get inverse for cartesian to lattice matrix
            self._car2lat = np.linalg.inv(lat2car)
        return self._car2lat

    @property
    def dir2lat(self) -> NDArray[float]:
        if self._dir2lat is None:
            self._dir2lat = self.car2lat.dot(self.car2lat.T)
        return self._dir2lat

    ###########################################################################
    # Functions used by most or all methods
    ###########################################################################

    def get_basin_charges_and_volumes(
        self,
        labels: NDArray[int],
    ):
        """
        Calculates the charges and volumes for the basins and vacuum from the
        provided label array. This is used by most methods except for `weight`.

        Parameters
        ----------
        labels : NDArray[int]
            A 3D array of the same shape as the reference grid with entries
            representing the basin the voxel belongs to.

        Returns
        -------
        dict
            A dictionary with information on charges, volumes, and siginificant
            basins.

        """
        logging.info("Calculating basin charges and volumes")
        grid = self.charge_grid
        # NOTE: I used to use numpy directly, but for systems with many basins
        # it was much slower than doing a loop with numba.
        charges, volumes, vacuum_charge, vacuum_volume = (
            get_basin_charges_and_volumes(
                data=grid.total,
                labels=labels,
                cell_volume=grid.structure.volume,
                extrema_num=len(self.extrema_vox),
            )
        )
        return {
            "basin_charges": charges,
            "basin_volumes": volumes,
            "vacuum_charge": vacuum_charge,
            "vacuum_volume": vacuum_volume,
        }

    def get_roots(
        self, pointers: NDArray[int], images: NDArray[int]
    ) -> NDArray[int]:
        """
        Finds the roots of a 1D array of pointers where each index points to its
        parent and sums images across periodic boundaries

        Parameters
        ----------
        pointers : NDArray[int]
            A 1D array where each entry points to that entries parent.
        images : NDArray[int]
            A Nx3 array where each entry indicates a shift across a periodic
            boundary
        Returns
        -------
        pointers : NDArray[int]
            A 1D array where each entry points to that entries root parent.

        """
        # temporarily set all vacuum points to point to the first vacuum
        # index. This way, when looking for minima, points assigned to the
        # vacuum will be updated properly
        max_val = np.iinfo(pointers.dtype).max
        vacuum_mask = pointers == max_val
        vacuum_points = np.where(vacuum_mask)[0]
        if len(vacuum_points) > 0:
            vacuum_point = np.where(vacuum_mask)[0][0]
            pointers[vacuum_mask] = vacuum_point
        n = 0
        while n < 100:
            n += 1
            # create a copy to avoid modifying in-place before comparison
            new_parents = pointers.copy()

            images += images[pointers]
            # for non-vacuum entries, reassign each index to the value at the
            # index it is pointing to
            new_parents = pointers[pointers]

            # check if we have the same value as before
            if np.all(new_parents == pointers):
                break

            pointers = new_parents

        # relabel our vacuum back to the max
        if len(vacuum_points) > 0:
            vacuum_mask = pointers == vacuum_point
            pointers[vacuum_mask] = max_val

        # We now have our roots. Relabel so that they go from 0 to the length of our
        # roots with the highest value being the vacuum
        unique_roots = np.unique(pointers)
        dtype = get_lowest_uint(len(unique_roots))
        pointers = np.searchsorted(unique_roots, pointers).astype(
            dtype, copy=False
        )

        return pointers, images

    def condense_images(self, images: NDArray[int]):
        """
        Converts Nx3 images into N images where each possible shift is represented
        as a single uint8

        Parameters
        ----------
        images : NDArray[int]
            A Nx3 array where each entry indicates a shift across a periodic
            boundary

        Returns
        -------
        A 1D array of length N representing each cells shift

        """
        shift_map = np.zeros((3, 3, 3), dtype=np.uint8)
        shift_trans = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
        for shift_idx, (i, j, k) in enumerate(shift_trans):
            shift_map[i, j, k] = shift_idx
        images = shift_map[images[:, 0], images[:, 1], images[:, 2]]
        return images

    def get_saddle_connections(
        self,
        labels,
        images,
    ):

        neighbor_transforms, neighbor_dists, _, _ = (
            self.reference_grid.point_neighbor_voronoi_transforms
        )
        # get edges
        edge_mask = get_thin_basin_edges(
            data=self.reference_grid.total,
            labels=labels,
            images=images,
            neighbor_transforms=neighbor_transforms,
            vacuum_mask=self.vacuum_mask,
            use_minima=self.use_minima,
        )

        saddle_coords, saddle_connections = (
            get_canonical_saddle_connections(
                labels=labels,
                images=images,
                data=self.reference_grid.total,
                neighbor_transforms=neighbor_transforms,
                edge_mask=edge_mask,
                matrix=self.reference_grid.matrix,
                use_minima=self.use_minima,
            )
        )
        breakpoint()
        # next:
            # refine saddle points and values
            # get persistence

        return saddle_connections, saddle_coords

    def copy(self) -> Self:
        """

        Returns
        -------
        Self
            A deep copy of this Method object.

        """
        return copy.deepcopy(self)
