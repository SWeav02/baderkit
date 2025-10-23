# -*- coding: utf-8 -*-

import copy
import logging
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from baderkit.core.toolkit import Grid
from baderkit.core.toolkit.grid_numba import refine_maxima

from .shared_numba import (  # combine_neigh_maxima,
    get_basin_charges_and_volumes,
    get_maxima,
    initialize_labels_from_maxima,
)

# This allows for Self typing and is compatible with python 3.10
Self = TypeVar("Self", bound="MethodBase")


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

    """

    def __init__(
        self,
        charge_grid: Grid,
        reference_grid: Grid,
        vacuum_mask: NDArray[bool],
        num_vacuum: int,
    ):
        """

        Parameters
        ----------
        charge_grid : Grid
            A Grid object with the charge density that will be integrated.
        reference_grid : Grid
            A grid object whose values will be used to construct the basins.
        vacuum_tol: float, optional
            The value below which a point will be considered part of the vacuum.
            The default is 0.001.
        normalize_vacuum: bool, optional
            Whether or not the reference data needs to be converted to real space
            units for vacuum tolerance comparison. This should be set to True if
            the data follows VASP's CHGCAR standards, but False if the data should
            be compared as is (e.g. in ELFCARs)

        Returns
        -------
        None.

        """
        # define variables needed by all methods
        self.charge_grid = charge_grid
        self.reference_grid = reference_grid
        self.vacuum_mask = vacuum_mask
        self.num_vacuum = num_vacuum

        # These variables are also often needed but are calculated during the run
        self._maxima_mask = None
        self._maxima_vox = None
        self._maxima_frac = None
        self._car2lat = None
        self._dir2lat = None

    def run(self) -> dict:
        """
        Runs the main bader method and returns a dictionary with values for:
            - basin_maxima_frac
            - basin_maxima_vox
            - basin_maxima_ref_values
            - basin_charges
            - basin_volumes
            - vacuum_charges
            - vacuum_volumes
            - significant_basins
        """
        # all methods require finding maxima. For consistency and to combine
        # adjacent maxima, I do this the same way for all methods. Because this
        # step is generally ~400x faster than the rest of the method, I think it's
        # ok to not try and do it during the actual method

        # get our initial maxima
        maxima_vox = self.maxima_vox
        # get neighbor transforms
        neighbor_transforms, _ = self.reference_grid.point_neighbor_transforms
        # now merge our maxima and initialize our labels
        labels, self._maxima_frac, self._maxima_vox = initialize_labels_from_maxima(
            data=self.reference_grid.total,
            spline_coeffs=self.reference_grid.cubic_spline_coeffs,
            maxima_vox=maxima_vox,
        )

        # now run bader
        results = self._run_bader(labels)

        # refine maxima using a quadratic fit
        refined_maxima_frac, maxima_values = refine_maxima(
            maxima_coords=self.maxima_frac,
            data=self.reference_grid.total,
            lattice=self.reference_grid.matrix,
        )

        self._maxima_frac = refined_maxima_frac

        results.update(
            {
                "basin_maxima_vox": self.maxima_vox,
                "basin_maxima_frac": self.maxima_frac,
                "basin_maxima_ref_values": maxima_values,
            }
        )
        return results

    def _run_bader(self, labels) -> dict:
        """
        This is the main function that each method must have. It must return a
        dictionary with values for:
            - basin_maxima_frac
            - basin_charges
            - basin_volumes
            - vacuum_charges
            - vacuum_volumes
            - significant_basins

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        dict
            DESCRIPTION.

        """
        raise NotImplementedError(
            "No run method has been implemented for this Bader Method."
        )

    ###########################################################################
    # Properties used by most or all methods
    ###########################################################################

    @property
    def maxima_mask(self) -> NDArray[bool]:
        """

        Returns
        -------
        NDArray[bool]
            A mask representing the voxels that are local maxima.

        """
        if self._maxima_mask is None:
            data = self.reference_grid.total
            neighbor_transforms, _ = self.reference_grid.point_neighbor_transforms
            vacuum_mask = self.vacuum_mask
            self._maxima_mask = get_maxima(
                data=data,
                neighbor_transforms=neighbor_transforms,
                vacuum_mask=vacuum_mask,
                use_minima=False,
            )
        return self._maxima_mask

    @property
    def maxima_vox(self) -> NDArray[int]:
        """

        Returns
        -------
        NDArray[int]
            An Nx3 array representing the voxel indices of each local maximum.

        """
        if self._maxima_vox is None:
            self._maxima_vox = np.argwhere(self.maxima_mask)
        return self._maxima_vox

    @property
    def maxima_frac(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            An Nx3 array representing the fractional coordinates of each local
            maximum. These are set after maxima/basin reduction so there may be
            fewer than the number of maxima_vox.

        """
        assert self._maxima_frac is not None, "Maxima frac must be set by run method"
        return self._maxima_frac

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
            maxima_num=len(self.maxima_frac),
        )
        return {
            "basin_charges": charges,
            "basin_volumes": volumes,
            "vacuum_charge": vacuum_charge,
            "vacuum_volume": vacuum_volume,
        }

    def get_roots(self, pointers: NDArray[int]) -> NDArray[int]:
        """
        Finds the roots of a 1D array of pointers where each index points to its
        parent.

        Parameters
        ----------
        pointers : NDArray[int]
            A 1D array where each entry points to that entries parent.

        Returns
        -------
        pointers : NDArray[int]
            A 1D array where each entry points to that entries root parent.

        """
        # mask for non-vacuum indices (not -1)
        if self.num_vacuum:
            valid = pointers != -1
        else:
            valid = None
        if valid is not None:
            while True:
                # create a copy to avoid modifying in-place before comparison
                new_parents = pointers.copy()

                # for non-vacuum entries, reassign each index to the value at the
                # index it is pointing to
                new_parents[valid] = pointers[pointers[valid]]

                # check if we have the same value as before
                if np.all(new_parents == pointers):
                    break

                # update only non-vacuum entries
                pointers[valid] = new_parents[valid]
        else:
            while True:
                # create a copy to avoid modifying in-place before comparison
                new_parents = pointers.copy()

                # for non-vacuum entries, reassign each index to the value at the
                # index it is pointing to
                new_parents = pointers[pointers]

                # check if we have the same value as before
                if np.all(new_parents == pointers):
                    break

                pointers = new_parents
        return pointers

    def copy(self) -> Self:
        """

        Returns
        -------
        Self
            A deep copy of this Method object.

        """
        return copy.deepcopy(self)
