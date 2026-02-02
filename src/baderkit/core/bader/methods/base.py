# -*- coding: utf-8 -*-

import copy
import logging
from typing import TypeVar
import itertools

import numpy as np
from numpy.typing import NDArray

from baderkit.core.toolkit import Grid
from baderkit.core.utilities.interpolation import refine_maxima

from baderkit.core.utilities.basic import get_lowest_uint
from .shared_numba import (  # combine_neigh_maxima,
    get_basin_charges_and_volumes,
    get_maxima,
    initialize_labels_from_maxima,
    get_neighboring_basin_connections,
    get_edges,
    group_by_persistence,
)

# This allows for Self typing and is compatible with python 3.10
Self = TypeVar("Self", bound="MethodBase")

# TODO:
    # 1. Add periodic awareness to weight method
    # 2. allow for non-periodic boundaries?
    # 3. continue with critical point finder

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
        The persistence score tolerance. Pairs of maxima with scores below
        this tolerance will be combined. The score is calculated as:
            dist * (lower_max - saddle_value) / higher_max
        where 'dist' is the cartesian distance between the maxima, lower_max
        is the value at the maximum with a lower value, saddle_value is the
        highest value at which there is a connection between the two maximas
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
    ):
        # define variables needed by all methods
        self.charge_grid = charge_grid
        self.reference_grid = reference_grid
        self.vacuum_mask = vacuum_mask
        self.num_vacuum = num_vacuum
        self.persistence_tol = persistence_tol

        # These variables are also often needed but are calculated during the run
        self._maxima_mask = None
        self._maxima_vox = None
        self._maxima_children = None
        self._maxima_frac = None
        self._car2lat = None
        self._dir2lat = None

    def run(self) -> dict:
        """
        Runs the main bader method and returns a dictionary with values for:
            - maxima_frac
            - maxima_vox
            - maxima_children
            - maxima_basin_labels
            - maxima_basin_images
            - maxima_ref_values
            - basin_charges
            - basin_volumes
            - vacuum_charges
            - vacuum_volumes
        """
        # first we initialize our label array with all values set to placeholders
        dtype = get_lowest_uint(self.reference_grid.ngridpts)
        vacuum_label = np.iinfo(dtype).max
        labels = np.full(self.reference_grid.ngridpts, vacuum_label, dtype=dtype)
        # get neighbor transforms
        neighbor_transforms, neighbor_dists = (
            self.reference_grid.point_neighbor_transforms
        )

        # all methods require finding maxima. For consistency and to combine
        # adjacent maxima, I do this the same way for all methods. Because this
        # step is generally ~400x faster than the rest of the method, I think it's
        # ok to not try and do it during the actual method

        # get our initial maxima
        labels, images, self._maxima_vox, self._maxima_children = initialize_labels_from_maxima(
            labels=labels,
            data=self.reference_grid.total,
            maxima_mask=self.maxima_mask,
            neighbor_transforms=neighbor_transforms,
            neighbor_dists=neighbor_dists,
            lattice=self.reference_grid.structure.lattice.matrix,
            persistence_tol=self.persistence_tol,
        )

        # now run bader
        results = self._run_bader(labels, images)
        labels = results["maxima_basin_labels"]
        
        # Now we want to combine any remaining noisy maxima based on their
        # rigorous discrete persistence.
        logging.info("Combining Low-Persistence Basins")
        
        # get edges
        edge_mask = get_edges(
            labeled_array=labels,
            neighbor_transforms=neighbor_transforms,
            vacuum_mask=self.vacuum_mask,
            )
        
        # get the values maxima connect at
        basin_connections, connection_values = get_neighboring_basin_connections(
            labeled_array=labels,
            data=self.reference_grid.total,
            neighbor_transforms=neighbor_transforms,
            vacuum_mask=self.vacuum_mask,
            edge_mask=edge_mask,
            label_num=len(self.maxima_vox),
            )

        # get maxima unions based on persistence
        maxima_roots = group_by_persistence(
            data=self.reference_grid.total,
            critical_vox=self.maxima_vox, 
            connections=basin_connections, 
            connection_values=connection_values, 
            lattice=self.reference_grid.structure.lattice.matrix,
            persistence_tol=self.persistence_tol,
            )
        
        # update maxima children, labels, charges, and volumes
        charges = results["basin_charges"]
        volumes = results["basin_volumes"]
        final_maxima, new_roots = np.unique(maxima_roots, return_inverse=True)

        for max_idx, root in enumerate(maxima_roots):
            if max_idx != root:
                # combine children
                self.maxima_children[root]=np.append(self.maxima_children[root], self.maxima_children[max_idx], axis=0)
                # add charge/volume
                charges[root] += charges[max_idx]
                volumes[root] += volumes[max_idx]

        labels[~self.vacuum_mask] = new_roots[labels[~self.vacuum_mask]]
        self._maxima_vox = self.maxima_vox[final_maxima]
        self._maxima_children = [self.maxima_children[i] for i in final_maxima]
        final_charges = charges[final_maxima]
        final_volumes = volumes[final_maxima]
        
        logging.info("Refining Maxima")
        # refine maxima using a quadratic fit
        self._maxima_vox, refined_maxima_frac, maxima_values = refine_maxima(
            maxima_coords=self.maxima_vox,
            maxima_children=self.maxima_children,
            data=self.reference_grid.total,
            labels=labels,
            lattice=self.reference_grid.matrix,
        )

        self._maxima_frac = refined_maxima_frac

        results.update(
            {
                "maxima_vox": self.maxima_vox,
                "basin_charges": final_charges,
                "basin_volumes": final_volumes,
                "basin_maxima_children": self.maxima_children,
                "maxima_frac": self.maxima_frac,
                "maxima_ref_values": maxima_values,
            }
        )
        return results

    def _run_bader(self, labels, images) -> dict:
        """
        This is the main function that each method must have. It must return a
        dictionary with values for:
            - maxima_frac
            - maxima_basin_images
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
    def maxima_children(self) -> NDArray[int]:
        """

        Returns
        -------
        NDArray[int]
            An Nx3 array representing the voxel indices of each local maximum.

        """
        assert self._maxima_children is not None, "Maxima children must be set by run method"
        return self._maxima_children

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
            maxima_num=len(self.maxima_vox),
        )
        return {
            "basin_charges": charges,
            "basin_volumes": volumes,
            "vacuum_charge": vacuum_charge,
            "vacuum_volume": vacuum_volume,
        }

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
        # mask for non-vacuum indices (not max possible value)
        if self.num_vacuum:
            valid = pointers != np.iinfo(pointers.dtype).max
        else:
            valid = None
        if valid is not None:
            while True:
                # create a copy to avoid modifying in-place before comparison
                new_parents = pointers.copy()

                # for non-vacuum entries, reassign each index to the value at the
                # index it is pointing to
                images[valid] += images[pointers[valid]]
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

                images += images[pointers]
                # for non-vacuum entries, reassign each index to the value at the
                # index it is pointing to
                new_parents = pointers[pointers]

                # check if we have the same value as before
                if np.all(new_parents == pointers):
                    break

                pointers = new_parents
        # We now have our roots. Relabel so that they go from 0 to the length of our
        # roots
        # vacuum_val = np.iinfo(pointers.dtype).max
        unique_roots = np.unique(pointers)
        dtype = get_lowest_uint(len(unique_roots))
        pointers = np.searchsorted(unique_roots, pointers).astype(dtype, copy=False)
        
        # # if we have any vacuum, relabel to the highest available value
        # if vacuum_val in unique_roots:
        #     pointers[pointers==pointers.max()] = np.iinfo.dtype.max
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
        shift_map = np.zeros((3,3,3), dtype=np.uint8)
        shift_trans = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
        for shift_idx, (i,j,k) in enumerate(shift_trans):
            shift_map[i,j,k] = shift_idx
            
        images = shift_map[images[:,0],images[:,1],images[:,2]]
        return images
        

    def copy(self) -> Self:
        """

        Returns
        -------
        Self
            A deep copy of this Method object.

        """
        return copy.deepcopy(self)
