# -*- coding: utf-8 -*-

import itertools
import logging
import time
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from baderkit.core.toolkit import Grid
from baderkit.core.utilities.basic import get_lowest_uint
from baderkit.core.utilities.basins import (  # combine_neigh_extrema,; get_neighboring_basin_connections_w_images,; get_edges_w_images,
    get_basin_charges_and_volumes,
    get_extrema,
    get_thin_basin_edges,
    update_labels_and_images,
)
from baderkit.core.utilities.critical_points import (
    get_saddles_from_basins,
    get_single_point_saddles,
    refine_critical_points,
    remove_adjacent_saddles,
    remove_false_saddles,
)
from baderkit.core.utilities.persistence import (
    group_by_persistence,
    init_by_approx_persistence,
)
from baderkit.core.utilities.transforms import ALL_NEIGHBOR_TRANSFORMS

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
        persistence_tol: float,
        use_minima: bool,
    ):
        # define variables needed by all methods
        self.charge_grid = charge_grid
        self.reference_grid = reference_grid
        self.vacuum_mask = vacuum_mask
        self.num_vacuum = num_vacuum
        self.use_minima = use_minima

        # scale persistence tolerance to voxel size
        self.persistence_tol = persistence_tol

        # create attributes that will be passed at the end of the
        # run method
        self._charges = None
        self._volumes = None
        self._vacuum_charge = None
        self._vacuum_volume = None
        self._labels = None
        self._images = None
        self._saddle_vox = None
        self._saddle_frac = None
        self._saddle_connections = None
        self._extrema_vox = None
        self._extrema_groups = None
        self._extrema_frac = None

        # These variables are also often needed but are calculated during the run
        self._extrema_mask = None
        self._original_vox = None
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
            - basin_charges
            - basin_volumes
            - vacuum_charges
            - vacuum_volumes
        """
        if self.use_minima:
            extrema = "Minima"
        else:
            extrema = "Maxima"

        # first we initialize our label array with all values set to placeholders
        dtype = get_lowest_uint(self.reference_grid.ngridpts)
        vacuum_label = np.iinfo(dtype).max
        labels = np.full(self.reference_grid.ngridpts, vacuum_label, dtype=dtype)

        # get ongrid maxima
        logging.info(f"Finding on-grid {extrema}")
        t0 = time.time()
        extrema_mask = self.extrema_mask
        extrema_vox = np.argwhere(extrema_mask).astype(np.int16)
        self._original_vox = extrema_vox
        t1 = time.time()
        logging.info(f"{len(extrema_vox)} on-grid {extrema} Found")
        logging.info(f"Time: {round(t1-t0,2)}")

        # initialize our labels, combining false extrema
        logging.info("Initializing Labels")
        t0 = time.time()
        (
            self._labels,
            self._images,
            self._extrema_vox,
            self._extrema_frac,
        ) = init_by_approx_persistence(
            labels=labels,
            data=self.reference_grid.total,
            extrema_mask=extrema_mask,
            extrema_vox=extrema_vox,
            persistence_tol=self.persistence_tol,
            use_minima=self.use_minima,
            matrix=self.reference_grid.matrix,
            max_cart_offset=0.25,
        )
        t1 = time.time()

        logging.info(
            f"Initialization Complete. Reduced to {len(self.extrema_vox)} {extrema}."
        )
        logging.info(f"Time: {round(t1-t0,2)}")

        # now run bader
        logging.info("Running Basin Assignment Algorithm")
        t0 = time.time()
        self._run_bader()
        t1 = time.time()
        logging.info("Basin Assignment Complete")
        logging.info(f"Time: {round(t1-t0,2)}")

        # Now we want to combine any remaining noisy extrema based on their
        # persistence.
        logging.info("Locating potential saddle points")
        t0 = time.time()
        all_connections, all_coords, all_vals = self._get_possible_saddle_connections()
        best_connections, best_coords, best_vals, best_cart = (
            self._compute_best_saddles(all_connections, all_coords, all_vals)
        )
        t1 = time.time()
        logging.info("Saddle Location Complete")
        logging.info(f"Time: {round(t1-t0,2)}")

        logging.info("Combining Low-Persistence Basins")
        t0 = time.time()
        extrema_roots, root_transforms = group_by_persistence(
            data=self.reference_grid.total,
            extrema_frac=self.extrema_frac,
            extrema_vox=self.extrema_vox,
            basin_connections=best_connections,
            saddle_values=best_vals,
            saddle_carts=best_cart,
            persistence_tol=self.persistence_tol,
            use_minima=self.use_minima,
            matrix=self.reference_grid.matrix,
        )
        # update labels/images
        self._build_final_extrema(extrema_roots, root_transforms)

        # update saddles
        self._update_saddles(all_coords)

        saddle_type = 1 if self.use_minima else 2
        t1 = time.time()
        logging.info(f"Reduced to {len(self.extrema_vox)} {extrema}")
        logging.info(f"Time: {round(t1-t0,2)}")

        results = {
            "extrema_basin_labels": self.labels,
            "extrema_basin_images": self.images,
            "basin_charges": self.charges,
            "basin_volumes": self.volumes,
            "ongrid_extrema_groups": self.extrema_groups,
            "extrema_vox": self.extrema_vox,
            "extrema_frac": self.extrema_frac,
            f"saddle{saddle_type}_vox": self.saddle_vox,
            f"saddle{saddle_type}_frac": self.saddle_frac,
            f"saddle{saddle_type}_connections": self.saddle_connections,
        }
        return results

    def _run_bader(self) -> dict:
        """
        This is the main function that each method must have. It must set the
        following class attributes
            - labels
            - images
            - charges
            - volumes
            - vacuum_charges
            - vacuum_volumes

        """
        raise NotImplementedError(
            "No run method has been implemented for this Bader Method."
        )

    ###########################################################################
    # Properties returned by all methodes
    ###########################################################################

    @property
    def labels(self) -> NDArray[int]:
        """

        Returns
        -------
        NDArray[int]
            The attractor each point in the grid belongs to. This is updated
            throughout the method.

        """
        assert self._labels is not None, "Labels must be set by init method"
        return self._labels

    @property
    def images(self) -> NDArray[int]:
        """

        Returns
        -------
        NDArray[int]
            The image of the attractor each point in the grid belongs to. This is updated
            throughout the method.

        """
        assert self._images is not None, "Images must be set by init method"
        return self._images

    @property
    def charges(self) -> NDArray[int]:
        """

        Returns
        -------
        NDArray[int]
            The charge assigned to each attractor

        """
        assert self._charges is not None, "charges must be set by the method"
        return self._charges

    @property
    def volumes(self) -> NDArray[int]:
        """

        Returns
        -------
        NDArray[int]
            The volume assigned to each attractor

        """
        assert self._volumes is not None, "volumes must be set by the method"
        return self._volumes

    @property
    def vacuum_charge(self) -> NDArray[int]:
        """

        Returns
        -------
        NDArray[int]
            The charge assigned to the vacuum

        """
        assert (
            self._vacuum_charge is not None
        ), "vacuum charge must be set by the method"
        return self._vacuum_charge

    @property
    def vacuum_volume(self) -> NDArray[int]:
        """

        Returns
        -------
        NDArray[int]
            The volume assigned to the vacuum

        """
        assert (
            self._vacuum_volume is not None
        ), "vacuum volume must be set by the method"
        return self._vacuum_volume

    @property
    def extrema_mask(self) -> NDArray[int]:
        """

        Returns
        -------
        NDArray[int]
            A mask that is true where ongrid maxima exist

        """
        if self._extrema_mask is None:
            self._extrema_mask = get_extrema(
                data=self.reference_grid.total,
                neighbor_transforms=ALL_NEIGHBOR_TRANSFORMS,
                vacuum_mask=self.vacuum_mask,
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
        assert self._extrema_vox is not None, "Maxima vox must be set by init method"
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
        ), "Maxima children must be set by init method"
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
        assert self._extrema_frac is not None, "Maxima frac must be set by init method"
        return self._extrema_frac

    @property
    def saddle_vox(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            An Nx3 array representing the coordinates of each saddle
            in the system

        """
        assert self._saddle_vox is not None, "Saddle coords must be set by run method"
        return self._saddle_vox

    @property
    def saddle_frac(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            An Nx3 array representing the coordinates of each saddle
            in the system

        """
        assert self._saddle_vox is not None, "Saddle coords must be set by run method"
        return self._saddle_frac

    @property
    def saddle_connections(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            An Nx3 array representing the attractors connected through each
            saddle and the image of the second attractor

        """
        assert (
            self._saddle_connections is not None
        ), "Saddle connections must be set by run method"
        return self._saddle_connections

    ###########################################################################
    # Properties for convenience
    ###########################################################################

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
        charges, volumes, vacuum_charge, vacuum_volume = get_basin_charges_and_volumes(
            data=grid.total,
            labels=labels,
            cell_volume=grid.structure.volume,
            extrema_num=len(self.extrema_vox),
        )
        self._charges = charges
        self._volumes = volumes
        self._vacuum_charge = vacuum_charge
        self._vacuum_volume = vacuum_volume

    def get_roots(self, pointers: NDArray[int], images: NDArray[int]) -> NDArray[int]:
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
        pointers = np.searchsorted(unique_roots, pointers).astype(dtype, copy=False)

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

    def _get_possible_saddle_connections(self):
        labels = self.labels
        images = self.images
        neighbor_transforms, neighbor_dists = (
            self.reference_grid.point_neighbor_transforms
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

        # get possible saddles
        saddle_vox, saddle_connections = get_saddles_from_basins(
            labels=labels,
            images=images,
            data=self.reference_grid.total,
            edge_mask=edge_mask,
            use_minima=self.use_minima,
        )

        # get values at each saddle
        saddle_values = self.reference_grid.total[
            saddle_vox[:, 0],
            saddle_vox[:, 1],
            saddle_vox[:, 2],
        ]

        return saddle_connections, saddle_vox, saddle_values

    def _compute_best_saddles(self, connections, coords, values):
        temp = np.sort(connections[:, :2], axis=1)

        _, idx, inv = np.unique(
            temp,
            axis=0,
            return_index=True,
            return_inverse=True,
        )

        unique_connections = connections[idx]

        best_vals, best_indices = get_single_point_saddles(
            connection_values=values,
            connection_indices=inv,
            initial_indices=idx,
            num_connections=len(unique_connections),
        )

        best_coords = coords[best_indices]
        best_connections = connections[best_indices]

        best_cart = self.reference_grid.grid_to_cart(best_coords)

        return best_connections, best_coords, best_vals, best_cart

    def _generate_voxel_groups(self):
        # get labels for all ongrid extrema
        all_vox = self._original_vox
        all_labels = self.labels[
            all_vox[:, 0],
            all_vox[:, 1],
            all_vox[:, 2],
        ]

        # sort from low to high
        order = np.argsort(all_labels)
        vox_sorted = all_vox[order]
        labels_sorted = all_labels[order]

        # count number of points in each group then split
        counts = np.bincount(labels_sorted)
        offsets = np.cumsum(counts)
        self._extrema_groups = np.split(vox_sorted, offsets[:-1])

    def _build_final_extrema(self, extrema_roots, root_transforms):

        # get the final extrema
        final_extrema, inverse, indices = np.unique(
            extrema_roots,
            return_inverse=True,
            return_index=True,
        )

        # sort from high to low
        extrema_vox = self.extrema_vox[final_extrema]

        data = self.reference_grid.total
        extrema_vals = data[tuple(extrema_vox.T)]

        order = np.argsort(extrema_vals)[::-1]

        final_extrema = final_extrema[order]
        self._extrema_vox = self.extrema_vox[final_extrema]
        self._extrema_frac = self.extrema_frac[final_extrema]

        # create map from old indices to final
        root_map = np.argsort(order)[indices]

        # update charges and volumes
        self._charges = np.bincount(
            root_map, weights=self.charges, minlength=len(final_extrema)
        )
        self._volumes = np.bincount(
            root_map, weights=self.volumes, minlength=len(final_extrema)
        )

        # update final labels and images
        self._labels, self._images = update_labels_and_images(
            labels=self.labels,
            images=self.images,
            label_map=root_map,
            image_map=root_transforms,
            vacuum_mask=self.vacuum_mask,
        )

        # get final ongrid extrema groups
        self._generate_voxel_groups()

    def _update_saddles(self, all_coords):
        # remove any possible saddles with large gradients
        saddle_vox, saddle_connections = remove_false_saddles(
            all_coords,
            labels=self.labels,
            images=self.images,
            data=self.reference_grid.total,
            matrix=self.reference_grid.matrix,
            use_minima=self.use_minima,
        )

        # refine saddle positions
        idx = 1 if self.use_minima else 2
        refined_vox, successes = refine_critical_points(
            critical_coords=saddle_vox,
            data=self.reference_grid.total,
            matrix=self.reference_grid.matrix,
            target_index=idx,
        )

        successes = np.where(successes)[0]
        refined_vox = refined_vox[successes]
        saddle_vox = saddle_vox[successes]
        saddle_connections = saddle_connections[successes]

        # get important saddles
        important = remove_adjacent_saddles(
            refined_vox,
            self.reference_grid.shape,
        )

        refined_vox = refined_vox[important]
        saddle_frac = refined_vox / self.reference_grid.shape
        self._saddle_vox = saddle_vox[important]
        self._saddle_frac = saddle_frac
        self._saddle_connections = saddle_connections[important]
