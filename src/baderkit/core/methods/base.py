# -*- coding: utf-8 -*-

import copy
import logging
from pathlib import Path
from typing import Literal, TypeVar

import numpy as np
from numpy.typing import NDArray

from baderkit.core.grid import Grid

from .shared_numba import combine_neigh_maxima, get_basin_charges_and_volumes

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
        vacuum_tol: float = 1.0e-3,
        normalize_vacuum: bool = True,
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
        self.vacuum_tol = vacuum_tol
        self.normalize_vacuum = normalize_vacuum

        # These variables are also often needed but are calculated during the run
        self._vacuum_mask = None
        self._num_vacuum = None
        self._maxima_mask = None
        self._maxima_vox = None
        self._maxima_frac = None

    def run(self) -> dict:
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
    def vacuum_mask(self) -> NDArray[bool]:
        """

        Returns
        -------
        NDArray[bool]
            A mask representing the voxels that belong to the vacuum.

        """
        if self._vacuum_mask is None:
            if self.normalize_vacuum:
                self._vacuum_mask = self.reference_grid.total < (
                    self.vacuum_tol / self.reference_grid.structure.volume
                )
            else:
                self._vacuum_mask = self.reference_grid.total < self.vacuum_tol
        return self._vacuum_mask

    @property
    def num_vacuum(self) -> int:
        """

        Returns
        -------
        int
            The number of vacuum points in the array

        """
        if self._num_vacuum is None:
            self._num_vacuum = np.count_nonzero(self.vacuum_mask)
        return self._num_vacuum

    @property
    def maxima_mask(self) -> NDArray[bool]:
        """

        Returns
        -------
        NDArray[bool]
            A mask representing the voxels that are local maxima.

        """
        assert self._maxima_mask is not None, "Maxima mask must be set by run method"
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

    ###########################################################################
    # Functions used by most or all methods
    ###########################################################################

    def get_extras(self):
        """

        Returns
        -------
        dict
            Collects the important class variables.

        """
        # TODO: This has to be called in every method currently. This should be
        # moved to a method in this abstract class to avoid repeat code/forgetting
        return {
            "vacuum_mask": self.vacuum_mask,
            "num_vacuum": self.num_vacuum,
            "basin_maxima_vox": self.maxima_vox,
            "basin_maxima_frac": self.maxima_frac,
        }

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

    @staticmethod
    def get_roots(
        pointers: NDArray[int], valid: NDArray[bool] | None = None
    ) -> NDArray[int]:
        """
        Finds the roots of a 1D array of pointers where each index points to its
        parent.

        Parameters
        ----------
        pointers : NDArray[int]
            A 1D array where each entry points to that entries parent.
        valid : NDArray[bool] | None, optional
            A mask the same shape as the pointers array. False values will be
            ignored. The default is None.

        Returns
        -------
        pointers : NDArray[int]
            A 1D array where each entry points to that entries root parent.

        """
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

    def reduce_label_maxima(
        self,
        labels: NDArray[int],
    ) -> (NDArray[int], NDArray[float]):
        """
        Combines maxima/basins that are adjacent to one another.

        Parameters
        ----------
        labels : NDArray[int]
            A 3D array representing current basin assignments.

        Returns
        -------
        labels : NDArray[int]
            A 3D array representing the new basin assignments
        frac_coords : NDArray[float]
            The averaged fractional coordinates for each set of adjacent maxima.

        """
        # TODO: stop reassignments if no reduction is needed
        maxima_vox = self.maxima_vox
        # order maxima voxels from lowest to highest labels
        # 1. get corresponding basin labels for maxima
        maxima_labels = labels[maxima_vox[:, 0], maxima_vox[:, 1], maxima_vox[:, 2]]
        # 2. sort from lowest to highest
        maxima_sorted_indices = np.argsort(maxima_labels)
        maxima_vox = maxima_vox[maxima_sorted_indices]
        # reduce maxima
        neighbor_transforms, _ = self.reference_grid.voxel_26_neighbors
        new_labels, frac_coords = combine_neigh_maxima(
            labels,
            neighbor_transforms,
            maxima_vox,
            maxima_vox / self.reference_grid.shape,
            self.maxima_mask,
        )
        # get the highest label
        max_label = maxima_labels.max()
        # if there are any unlabeled points in our label array, they will be
        # marked as -1. np.choose requires values to be 0, 1, 2, ...
        if -1 in labels:
            # shift to start at 0
            labels += 1
            max_label += 1
            # add -1 to the start of our new labels to reassign back to -1
            new_labels = np.insert(new_labels, 0, -1)
        # update_labels
        # Create a mapping from old to new labels. The initial labels should be
        # 0 and up
        mapping = np.arange(max_label + 1)
        mapping[mapping] = new_labels
        # Apply mapping
        labels = mapping[labels]
        return labels, frac_coords

    def copy(self) -> Self:
        """

        Returns
        -------
        Self
            A deep copy of this Method object.

        """
        return copy.deepcopy(self)
