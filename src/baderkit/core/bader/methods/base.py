# -*- coding: utf-8 -*-

import copy
import logging
from typing import TypeVar
import itertools

import numpy as np
from numpy.typing import NDArray

from baderkit.core.toolkit import Grid
from baderkit.core.utilities.interpolation import refine_extrema

from baderkit.core.utilities.basic import get_lowest_uint
from .shared_numba import (  # combine_neigh_extrema,
    get_basin_charges_and_volumes,
    get_extrema,
    initialize_labels_from_extrema,
    # get_neighboring_basin_connections_w_images,
    # get_edges_w_images,
    group_by_persistence,
    get_basin_edges,
    get_canonical_saddle_connections,
    get_single_point_saddles
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
        self.persistence_tol = persistence_tol
        self.use_minima = use_minima

        # These variables are also often needed but are calculated during the run
        self._extrema_mask = None
        self._extrema_vox = None
        self._extrema_children = None
        self._extrema_frac = None
        self._car2lat = None
        self._dir2lat = None

    def run(self) -> dict:
        """
        Runs the main bader method and returns a dictionary with values for:
            - extrema_frac
            - extrema_vox
            - extrema_children
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
        labels = np.full(self.reference_grid.ngridpts, vacuum_label, dtype=dtype)
        # get neighbor transforms
        neighbor_transforms, neighbor_dists = (
            self.reference_grid.point_neighbor_transforms
        )
        
        # NOTE: We do not want to use any vacuum for finding minima. If we did, many
        # minima would be found along the eges of the vacuum, which is not the
        # desired behavior. Instead we want minima in the vacuum and any voxels
        # assigned to them to all be considered part of the vacuum and not valid
        # manifolds. To deal with this, we save the vacuum here, and replace it
        # with an empty mask

        # all methods require finding extrema. For consistency and to combine
        # adjacent extrema, I do this the same way for all methods. Because this
        # step is generally ~400x faster than the rest of the method, I think it's
        # ok to not try and do it during the actual method

        # get our initial extrema
        if self.use_minima:
            opp_extreme_val = self.reference_grid.total.max()
        else:
            opp_extreme_val = self.reference_grid.total.min()
        labels, images, self._extrema_vox, self._extrema_children, persistence_cutoffs = initialize_labels_from_extrema(
            labels=labels,
            data=self.reference_grid.total,
            extrema_mask=self.extrema_mask,
            neighbor_transforms=neighbor_transforms,
            neighbor_dists=neighbor_dists,
            lattice=self.reference_grid.structure.lattice.matrix,
            opp_extreme_val=opp_extreme_val,
            persistence_tol=self.persistence_tol,
            use_minima=self.use_minima,
        )
        # breakpoint()

        # now run bader
        results = self._run_bader(labels, images)
        labels = results["extrema_basin_labels"]
        images = results["extrema_basin_images"]

        # Now we want to combine any remaining noisy extrema based on their
        # rigorous discrete persistence.
        logging.info("Combining Low-Persistence Basins")
        
        saddle_connections, saddle_coords, saddle_values = self.get_saddle_connections(labels, images)

        # get extrema unions based on persistence and update lowest persistence
        # values for extrema
        extrema_roots, persistence_cutoffs = group_by_persistence(
            data=self.reference_grid.total,
            critical_vox=self.extrema_vox, 
            basin_connections=saddle_connections, 
            saddle_values=saddle_values, 
            lattice=self.reference_grid.structure.lattice.matrix,
            persistence_tol=self.persistence_tol,
            persistence_cutoffs=persistence_cutoffs,
            use_minima=self.use_minima,
            )

        # update extrema children, labels, charges, and volumes
        charges = results["basin_charges"]
        volumes = results["basin_volumes"]
        final_extrema, indices, new_roots  = np.unique(extrema_roots, return_inverse=True, return_index=True)
        extrema_vox = self.extrema_vox[indices]
        
        # reorder from highest to lowest
        extrema_vals = self.reference_grid.total[extrema_vox[:,0],extrema_vox[:,1],extrema_vox[:,2]]
        ordered_indices = np.flip(np.argsort(extrema_vals))
        
        final_extrema = final_extrema[ordered_indices]
        new_roots = ordered_indices[new_roots]
        persistence_cutoffs = persistence_cutoffs[final_extrema]
        

        for max_idx, root in enumerate(extrema_roots):
            if max_idx != root:
                # combine children
                self.extrema_children[root]=np.append(self.extrema_children[root], self.extrema_children[max_idx], axis=0)
                # add charge/volume
                charges[root] += charges[max_idx]
                volumes[root] += volumes[max_idx]
        # relabel combined basins
        labels[~self.vacuum_mask] = new_roots[labels[~self.vacuum_mask]]
        self._extrema_vox = self.extrema_vox[final_extrema]
        self._extrema_children = [np.array(self.extrema_children[i],dtype=np.uint16) for i in final_extrema]
        final_charges = charges[final_extrema]
        final_volumes = volumes[final_extrema]
        
        if self.use_minima:
            logging.info("Refining Minima")  
        else:
            logging.info("Refining Maxima")
        # refine extrema using a quadratic fit
        self._extrema_vox, refined_extrema_frac, extrema_values = refine_extrema(
            extrema_coords=self.extrema_vox,
            extrema_children=self.extrema_children,
            data=self.reference_grid.total,
            labels=labels,
            lattice=self.reference_grid.matrix,
            use_minima=self.use_minima
        )

        self._extrema_frac = refined_extrema_frac
        
        # convert extrema vox to 

        results.update(
            {
                "extrema_basin_labels": labels,
                "extrema_basin_images": images,
                "extrema_vox": self.extrema_vox.astype(np.uint16),
                "basin_charges": final_charges,
                "basin_volumes": final_volumes,
                "extrema_voxel_groups": self.extrema_children,
                "extrema_frac": self.extrema_frac,
                "extrema_ref_values": extrema_values,
                "extrema_persistence_values": persistence_cutoffs,
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
            neighbor_transforms, _ = self.reference_grid.point_neighbor_transforms
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
    def extrema_children(self) -> NDArray[int]:
        """

        Returns
        -------
        NDArray[int]
            An Nx3 array representing the voxel indices of each local maximum.

        """
        assert self._extrema_children is not None, "Maxima children must be set by run method"
        return self._extrema_children

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
        assert self._extrema_frac is not None, "Maxima frac must be set by run method"
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
        charges, volumes, vacuum_charge, vacuum_volume = get_basin_charges_and_volumes(
            data=grid.total,
            labels=labels,
            cell_volume=grid.structure.volume,
            extrema_num=len(self.extrema_vox),
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
        # temporarily set all vacuum points to point to the first vacuum
        # index. This way, when looking for minima, points assigned to the
        # vacuum will be updated properly
        max_val = np.iinfo(pointers.dtype).max
        vacuum_mask = pointers == max_val
        vacuum_points = np.where(vacuum_mask)[0]
        if len(vacuum_points) > 0:
            vacuum_point = np.where(vacuum_mask)[0][0]
            pointers[vacuum_mask] = vacuum_point
        n=0
        while n<100:
            n+=1
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
        shift_map = np.zeros((3,3,3), dtype=np.uint8)
        shift_trans = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
        for shift_idx, (i,j,k) in enumerate(shift_trans):
            shift_map[i,j,k] = shift_idx
        images = shift_map[images[:,0],images[:,1],images[:,2]]
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
        edge_mask = get_basin_edges(
            labels=labels,
            images=images,
            neighbor_transforms=neighbor_transforms,
            vacuum_mask=self.vacuum_mask,
            )
        
        saddle_coords, saddle_connections, connection_vals = get_canonical_saddle_connections(
            labels=labels,
            images=images,
            data=self.reference_grid.total,
            neighbor_transforms=neighbor_transforms,
            edge_mask=edge_mask,
            use_minima=self.use_minima,
        )
        
        unique_connections, inverse = np.unique(saddle_connections[:,:3],axis=0, return_inverse=True)

        # get the best saddle points
        saddle_indices, saddle_values = get_single_point_saddles(
            data=self.reference_grid.total,
            connection_values=connection_vals,
            saddle_coords=saddle_coords,
            connection_indices=inverse,
            num_connections=len(unique_connections),
            use_minima=self.use_minima,
        )
        saddle_coords = saddle_coords[saddle_indices]
        saddle_connections = saddle_connections[saddle_indices]
        
        return saddle_connections, saddle_coords, saddle_values
        

    def copy(self) -> Self:
        """

        Returns
        -------
        Self
            A deep copy of this Method object.

        """
        return copy.deepcopy(self)
