# -*- coding: utf-8 -*-

import copy
import inspect
import logging
from itertools import product
from pathlib import Path
from typing import Literal, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from rich.progress import track

from baderkit.core.grid import Grid
from baderkit.core.numba_functions import (
    get_basin_charges_and_volumes,
    get_edges,
    get_multi_weight_voxels,
    get_neargrid_labels,
    get_neighbor_flux,
    get_ongrid_and_rgrads,
    get_reverse_neargrid_labels,
    get_single_weight_voxels,
    get_steepest_pointers,
    get_vacuum_mask,
    reduce_maxima,
    refine_neargrid,
)
from baderkit.core.structure import Structure

# This allows for Self typing and is compatible with python 3.10
Self = TypeVar("Self", bound="Bader")


class Bader:
    """
    Class for running Bader analysis on a regular grid. For information on each
    method, see our [docs](https://sweav02.github.io/baderkit/)
    """

    def __init__(
        self,
        charge_grid: Grid,
        reference_grid: Grid,
        method: Literal[
            "ongrid",
            "neargrid",
            "reverse-neargrid",
            "weight",
            "hybrid-weight",
        ] = "reverse-neargrid",
        refinement_method: Literal["recursive", "single"] = "recursive",
        directory: Path = Path("."),
        vacuum_tol: float = 1.0e-3,
        normalize_vacuum: bool = True,
        bader_tol: float = 1.0e-3,
    ):
        """

        Parameters
        ----------
        charge_grid : Grid
            A Grid object with the charge density that will be integrated.
        reference_grid : Grid
            A grid object whose values will be used to construct the basins.
        method : Literal["ongrid", "neargrid", "reverse-neargrid", "weight", "hybrid-weight"], optional
            The algorithm to use for generating bader basins. If None, defaults
            to weight.
        refinement_method : Literal["recursive", "single"], optional
            For methods that refine the basin edges (neargrid), whether to
            refine the edges until none change or to refine a single time. If
            None, defaults to recursive.
        directory : Path, optional
            The directory that files will be written to by default.
            The default is Path("."), or the current active directory.
        vacuum_tol: float, optional
            The value below which a point will be considered part of the vacuum.
            The default is 0.001.
        normalize_vacuum: bool, optional
            Whether or not the reference data needs to be converted to real space
            units for vacuum tolerance comparison. This should be set to True if
            the data follows VASP's CHGCAR standards, but False if the data should
            be compared as is (e.g. in ELFCARs)
        bader_tol: float, optional
            The value below which a basin will not be considered significant. This
            is used to avoid writing out data that is likely not valuable.
            The default is 0.001.

        Returns
        -------
        None.

        """
        self.charge_grid = charge_grid
        self.reference_grid = reference_grid
        self.method = method
        self.directory = directory
        self.refinement_method = refinement_method
        self.vacuum_tol = vacuum_tol
        self.normalize_vacuum = normalize_vacuum
        self.bader_tol = bader_tol

        # define hidden class variables. This allows us to cache properties and
        # still be able to recalculate them if needed, though that should only
        # be done by advanced users
        # Assigned by run_bader
        self._basin_labels = None
        self._basin_maxima_frac = None
        self._basin_charges = None
        self._basin_volumes = None
        self._basin_surface_distances = None
        self._basin_edges = None
        self._vacuum_charge = None
        self._vacuum_volume = None
        self._significant_basins = None
        # others assigned by calling the property directly, but usually during
        # run_bader
        self._vacuum_mask = None
        self._num_vacuum = None
        self._structure = None
        # Assigned by run_atom_assignment
        self._basin_atoms = None
        self._basin_atom_dists = None
        self._atom_labels = None
        self._atom_charges = None
        self._atom_volumes = None
        self._atom_surface_distances = None

    @property
    def basin_labels(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            A 3D array of the same shape as the reference grid with entries
            representing the basin the voxel belongs to. Note that for some
            methods (e.g. weight) the voxels have weights for each basin.
            These will be stored in the basin_weights property.

        """
        if self._basin_labels is None:
            self.run_bader()
        return self._basin_labels

    @property
    def basin_maxima_frac(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The fractional coordinates of each attractor.

        """
        if self._basin_maxima_frac is None:
            self.run_bader()
        return self._basin_maxima_frac

    @property
    def basin_charges(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The charges assigned to each attractor.

        """
        if self._basin_charges is None:
            self.run_bader()
        return self._basin_charges

    @property
    def basin_volumes(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The volume assigned to each attractor.

        """
        if self._basin_volumes is None:
            self.run_bader()
        return self._basin_volumes

    @property
    def basin_surface_distances(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The distance from each basin maxima to the nearest point on
            the basins surface

        """
        if self._basin_surface_distances is None:
            self._get_basin_surface_distances()
        return self._basin_surface_distances

    @property
    def basin_atoms(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The atom index of each basin is assigned to.

        """
        if self._basin_atoms is None:
            self.run_atom_assignment()
        return self._basin_atoms

    @property
    def basin_atom_dists(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The distance from each attractor to the nearest atom

        """
        if self._basin_atom_dists is None:
            self.run_atom_assignment()
        return self._basin_atom_dists

    @property
    def significant_basins(self) -> NDArray[bool]:
        if self._significant_basins is None:
            self.run_bader()
        return self._significant_basins

    @property
    def atom_labels(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            A 3D array of the same shape as the reference grid with entries
            representing the atoms the voxel belongs to.

            Note that for some methods (e.g. weight) some voxels have fractional
            assignments for each basin and this will not represent exactly how
            charges are assigned.

        """
        if self._atom_labels is None:
            self.run_atom_assignment()
        return self._atom_labels

    @property
    def atom_charges(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The charge assigned to each atom

        """
        if self._atom_charges is None:
            self.run_atom_assignment()
        return self._atom_charges

    @property
    def atom_volumes(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The volume assigned to each atom

        """
        if self._atom_volumes is None:
            self.run_atom_assignment()
        return self._atom_volumes

    @property
    def atom_surface_distances(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The distance from each atom to the nearest point on the atoms surface.

        """
        if self._atom_surface_distances is None:
            self._get_atom_surface_distances()
        return self._atom_surface_distances

    @property
    def structure(self) -> Structure:
        """

        Returns
        -------
        Structure
            The pymatgen structure basins are assigned to.

        """
        if self._structure is None:
            self._structure = self.reference_grid.structure.copy()
            self._structure.relabel_sites(ignore_uniq=True)
        return self._structure

    @property
    def basin_edges(self) -> NDArray[np.bool_]:
        """

        Returns
        -------
        NDArray[np.bool_]
            A mask with the same shape as the input grids that is True at points
            on basin edges.

        """
        if self._basin_edges is None:
            self._basin_edges = self.get_basin_edges(self.basin_labels)
        return self._basin_edges

    @property
    def vacuum_charge(self) -> float:
        """

        Returns
        -------
        float
            The charge assigned to the vacuum.

        """
        if self._vacuum_charge is None:
            self.run_bader()
        return self._vacuum_charge

    @property
    def vacuum_volume(self) -> float:
        """

        Returns
        -------
        float
            The total volume assigned to the vacuum.

        """
        if self._vacuum_volume is None:
            self.run_bader()
        return self._vacuum_volume

    @property
    def vacuum_mask(self) -> NDArray[bool]:
        """

        Returns
        -------
        NDArray[bool]
            A mask representing the voxels that belong to the vacuum.

        """
        if self._vacuum_mask is None:
            # logging.info("Finding Vacuum Points")
            # Find the vacuum voxels
            self._vacuum_mask = get_vacuum_mask(
                data=self.reference_grid.total,
                cell_volume=self.structure.volume,
                vacuum_threshold=self.vacuum_tol,
                normalize_vac=self.normalize_vacuum,
            )
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

    @staticmethod
    def methods() -> list[str]:
        """

        Returns
        -------
        list[str]
            A list of the available methods.

        """

        return [
            "ongrid",
            "neargrid",
            "reverse-neargrid",
            "weight",
            "hybrid-weight",
        ]

    @staticmethod
    def get_basin_edges(
        basin_labels: NDArray[float],
        vacuum_mask: NDArray[bool] = None,
        neighbor_transforms: NDArray = None,
    ) -> NDArray[np.bool_]:
        """
        Gets a mask representing the edges of a labeled array.

        Parameters
        ----------
        basin_labels : NDArray[float]
            A 3D numpy array of integers assigning points to basins.
        neighbor_transforms : NDArray, optional
            The transformations from each voxel to its neighbors. Providing None
            will result in the 26 nearest neighbors being used.
        vacuum_mask: NDArray[np.bool_]
            A 3D array representing the location of the vacuum. If None, the
            vacuum will be treated like a basin.

        Returns
        -------
        NDArray[np.bool_]
            A mask with the same shape as the input grid that is True at points
            on basin edges.


        """

        # If no specific neighbors are provided, we default to all 26 neighbors
        if neighbor_transforms is None:
            neighbor_transforms = list(product([-1, 0, 1], repeat=3))
            neighbor_transforms.remove((0, 0, 0))  # Remove the (0, 0, 0) self-shift
            neighbor_transforms = np.array(neighbor_transforms)
        if vacuum_mask is None:
            vacuum_mask = np.zeros(basin_labels.shape, dtype=np.bool_)
        return get_edges(
            basin_labels,
            neighbor_transforms=neighbor_transforms,
            vacuum_mask=vacuum_mask,
        )

    @staticmethod
    def get_basin_charges_and_volumes(
        basin_labels: NDArray[int],
        grid: Grid,
        maxima_num: int,
    ):
        """
        Calculate the volume and charge for each basin in the input label array

        Parameters
        ----------
        basin_labels : NDArray[int]
            A 3D numpy array with the same shape as the grid indicating the basin
            or atom each point belongs to.
        grid : Grid
            The charge grid to integrate over.

        Returns
        -------
        (NDArray[float], NDArray[float])
            A tuple of 1D array where the first is the assigned to each labeled
            basin and the second is the corresponding assigned volume.

        """
        logging.info("Calculating basin charges and volumes")
        # NOTE: I used to use numpy directly, but for systems with many basins
        # it was much slower than doing a loop with numba.
        charges, volumes, vacuum_charge, vacuum_volume = get_basin_charges_and_volumes(
            data=grid.total,
            labels=basin_labels,
            cell_volume=grid.structure.volume,
            maxima_num=maxima_num,
        )
        return charges, volumes, vacuum_charge, vacuum_volume

    def run_bader(self) -> None:
        """
        Runs the entire bader process and saves results to class variables.

        Raises
        ------
        ValueError
            The class method variable must be 'ongrid', 'neargrid',
            'reverse-neargrid', 'weight' or 'hybrid-weight'.

        Returns
        -------
        None.

        """
        if self.method == "ongrid":
            self._run_bader_ongrid()

        elif self.method == "neargrid":
            self._run_bader_neargrid()

        elif self.method == "reverse-neargrid":
            self._run_bader_reverse_near_grid()

        elif self.method == "weight":
            self._run_bader_weight()

        elif self.method == "hybrid-weight":
            self._run_bader_weight(hybrid=True)

        else:
            raise ValueError(
                f"{self.method} is not a valid algorithm."
                "Acceptable values are 'ongrid', 'neargrid', 'reverse-neargrid', 'weight', and 'hybrid-weight'"
            )

    def _set_basin_properties_from_labels(
        self,
        maxima_vox: NDArray[int],
    ) -> None:
        """
        Calculates various properties from a label array including basin
        charge/volume, vacuum charge/volume, significant basins, and maxima
        position

        Parameters
        ----------
        maxima_vox : NDArray[bool]
            An array of voxel coordinates representing the positions of the
            maxima.

        Returns
        -------
        None

        """
        labels = self.basin_labels
        # get corresponding basin labels for maxima
        maxima_labels = labels[maxima_vox[:, 0], maxima_vox[:, 1], maxima_vox[:, 2]]
        # sort from lowest to highest
        maxima_sorted_indices = np.argsort(maxima_labels)
        maxima_vox = maxima_vox[maxima_sorted_indices]
        # calculate frac coords and save
        maxima_frac_coords = self.reference_grid.get_frac_coords_from_vox(maxima_vox)
        self._basin_maxima_frac = maxima_frac_coords
        # get charge and volume for each label and vacuum
        (
            basin_charges,
            basin_volumes,
            vacuum_charge,
            vacuum_volume,
        ) = self.get_basin_charges_and_volumes(
            basin_labels=labels,
            grid=self.charge_grid,
            maxima_num=len(maxima_labels),
        )
        # get significant basins
        significant_basins = basin_charges > self.bader_tol
        # save charges/volumes.
        self._significant_basins = significant_basins
        self._basin_charges, self._basin_volumes = basin_charges, basin_volumes
        self._vacuum_charge, self._vacuum_volume = vacuum_charge, vacuum_volume
        # get maxima coords
        maxima_frac = self.reference_grid.get_frac_coords_from_vox(maxima_vox)
        self._basin_maxima_frac = maxima_frac

    def _get_bader_on_grid(self):
        """
        Calculates the ongrid labels for each voxel. This does not assign any
        class properties and should usually be used as part of other functions

        Returns
        -------
        labels : NDArray[np.int64]
            A 3D grid of labels representing current voxel assignments.
        maxima_mask : NDArray[np.int64]
            A 3D grid representing the location of maxima

        """
        grid = self.reference_grid
        data = grid.total
        shape = data.shape
        # get shifts to move from a voxel to the 26 surrounding voxels
        neighbor_transforms, neighbor_dists = grid.voxel_26_neighbors
        # For each voxel, get the label of the surrounding voxel that has the highest
        # density
        logging.info("Calculating steepest neighbors")
        best_label = get_steepest_pointers(
            data=data,
            initial_labels=grid.all_voxel_indices,
            neighbor_transforms=neighbor_transforms,
            neighbor_dists=neighbor_dists,
            vacuum_mask=self.vacuum_mask,
        )
        # ravel the best labels to get a 1D array pointing from each voxel to its steepest
        # neighbor
        pointers = best_label.ravel()
        # Our pointers object is a 1D array pointing each voxel to its parent voxel. We
        # essentially have a classic forest of trees problem where each maxima is
        # a root and we want to point all of our voxels to their respective root.
        # We being a while loop. In each loop, we remap our pointers to point at
        # the index that its parent was pointing at.
        # NOTE: Vacuum points are indicated by a value of -1 and we want to
        # ignore these
        logging.info("Finding roots")
        # mask for non-vacuum indices (not -1)
        valid = pointers != -1
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

        # We now have our roots. Relabel so that they go from 0 to the length of our
        # roots
        unique_roots, labels_flat = np.unique(pointers, return_inverse=True)
        # If we had at least one vacuum point, we need to subtract our labels by
        # 1 to recover the vacuum label.
        if -1 in unique_roots:
            labels_flat -= 1
        # reconstruct a 3D array with our labels
        labels = labels_flat.reshape(shape)
        # find the position of the maxima
        maxima_mask = best_label == grid.all_voxel_indices
        return labels, maxima_mask

    def _run_bader_ongrid(self):
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
        labels, maxima_mask = self._get_bader_on_grid()
        maxima_vox = np.argwhere(maxima_mask)
        # store our labels
        self._basin_labels = labels
        # assign charges/volumes, etc.
        self._set_basin_properties_from_labels(maxima_vox)

    def _run_bader_neargrid(self):
        """
        Assigns voxels to basins and calculates charge using the near-grid
        method:
            W. Tang, E. Sanville, and G. Henkelman
            A grid-based Bader analysis algorithm without lattice bias
            J. Phys.: Condens. Matter 21, 084204 (2009)

        Parameters
        ----------
        hybrid : bool, optional
            If True, the first round of assignments will be done using the ongrid
            method and refinements will use the neargrid. The default is False.

        Returns
        -------
        None.

        """
        grid = self.reference_grid.copy()
        # get neigbhor transforms
        neighbor_transforms, neighbor_dists = grid.voxel_26_neighbors
        matrix = grid.matrix
        # convert to lattice vectors as columns
        dir2car = matrix.T
        # get lattice to cartesian matrix
        lat2car = dir2car / grid.shape[np.newaxis, :]
        # get inverse for cartesian to lattice matrix
        car2lat = np.linalg.inv(lat2car)
        logging.info("Calculating gradients")
        best_neighbors, all_drs, maxima_mask = get_ongrid_and_rgrads(
            data=grid.total,
            car2lat=car2lat,
            neighbor_dists=neighbor_dists,
            neighbor_transforms=neighbor_transforms,
            vacuum_mask=self.vacuum_mask,
        )
        logging.info("Calculating initial labels")
        # get initial labels
        labels = get_neargrid_labels(
            data=grid.total,
            best_neighbors=best_neighbors,
            all_drs=all_drs,
            maxima_mask=maxima_mask,
            vacuum_mask=self.vacuum_mask,
            neighbor_dists=neighbor_dists,
            neighbor_transforms=neighbor_transforms,
        )
        # Increase values by 1 so that vacuum is labeled as 1 not 0
        labels += 1
        # get maxima positions, not including vacuum
        maxima_vox = np.argwhere(maxima_mask & ~self.vacuum_mask)
        reassignments = 1
        # get our edges, not including edges on the vacuum.
        # NOTE: Should the vacuum edges be refined as well in case some voxels
        # are added to it?
        refinement_mask = get_edges(
            labeled_array=labels,
            neighbor_transforms=neighbor_transforms,
            vacuum_mask=self.vacuum_mask,
        )
        # initialize a mask where voxels are already checked to prevent
        # reassignment. We include vacuum voxels from the start
        checked_mask = self.vacuum_mask.copy()
        # add maxima to mask so they don't get checked
        for i, j, k in maxima_vox:
            refinement_mask[i, j, k] = False
            checked_mask[i, j, k] = True

        while reassignments > 0:
            # get refinement indices
            refinement_indices = np.argwhere(refinement_mask)
            if len(refinement_indices) == 0:
                # there's nothing to refine so we break
                break
            print(f"Refining {len(refinement_indices)} points")
            # reassign edges
            labels, reassignments, refinement_mask, checked_mask = refine_neargrid(
                data=grid.total,
                labels=labels,
                refinement_indices=refinement_indices,
                refinement_mask=refinement_mask,
                checked_mask=checked_mask,
                maxima_mask=maxima_mask,
                best_neighbors=best_neighbors,
                all_drs=all_drs,
                neighbor_dists=neighbor_dists,
                neighbor_transforms=neighbor_transforms,
                vacuum_mask=self.vacuum_mask,
            )

            print(f"{reassignments} values changed")
            # if our refinement method is single, we cancel the loop here
            if self.refinement_method == "single":
                break
        # Our labels currently span 1 and up, with 1 corresponding to vacuum. We
        # subtract by 2 to return to -1 as vacuum and labels spanning 0 up
        labels -= 2
        # assign labels
        self._basin_labels = labels
        # assign charges/volumes, etc.
        self._set_basin_properties_from_labels(maxima_vox)

    def _run_bader_reverse_near_grid(self):
        """
        Assigns voxels to basins and calculates charge using a variation of
        the neargrid method. The reference grid is first sorted from highest to
        lowest, and the gradient is followed in a descending rather than ascending
        manor. The core concepts are still based on the original neargrid method:
            W. Tang, E. Sanville, and G. Henkelman
            A grid-based Bader analysis algorithm without lattice bias
            J. Phys.: Condens. Matter 21, 084204 (2009)

        Returns
        -------
        None.

        """
        grid = self.reference_grid.copy()
        # get neigbhor transforms
        neighbor_transforms, neighbor_dists = grid.voxel_26_neighbors
        matrix = grid.matrix
        # convert to lattice vectors as columns
        dir2car = matrix.T
        # get lattice to cartesian matrix
        lat2car = dir2car / grid.shape[np.newaxis, :]
        # get inverse for cartesian to lattice matrix
        car2lat = np.linalg.inv(lat2car)
        logging.info("Calculating gradients")
        best_neighbors, all_drs, maxima_mask = get_ongrid_and_rgrads(
            data=grid.total,
            car2lat=car2lat,
            neighbor_dists=neighbor_dists,
            neighbor_transforms=neighbor_transforms,
            vacuum_mask=self.vacuum_mask,
        )
        logging.info("Sorting reference data")
        shape = grid.shape
        # flatten data and get initial 1D and 3D voxel indices
        flat_data = grid.total.ravel()
        flat_voxel_coords = np.indices(shape).reshape(3, -1).T
        # sort data from high to low
        sorted_data_indices = np.flip(np.argsort(flat_data, kind="stable"))
        sorted_voxel_coords = flat_voxel_coords[sorted_data_indices]
        logging.info("Assigning labels")
        # get assignments and maxima mask
        labels = get_reverse_neargrid_labels(
            data=grid.total,
            ordered_voxel_coords=sorted_voxel_coords,
            best_neighbors=best_neighbors,
            all_drs=all_drs,
            neighbor_transforms=neighbor_transforms,
            neighbor_dists=neighbor_dists,
            maxima_mask=maxima_mask,
            num_vacuum=self.num_vacuum,
        )
        # adjust labels to 0 index convention. We don't adjust values below 0 as
        # these are part of the vacuum
        labels[labels >= 0] -= 1
        # assign labels
        self._basin_labels = labels
        # get maxima voxels
        maxima_vox = np.argwhere(maxima_mask & ~self.vacuum_mask)
        # assign charges/volumes, etc.
        self._set_basin_properties_from_labels(maxima_vox)

    def _run_bader_weight(self, hybrid: bool = False):
        """
        Assigns basin weights to each voxel and assigns charge using
        the weight method:
            M. Yu and D. R. Trinkle,
            Accurate and efficient algorithm for Bader charge integration,
            J. Chem. Phys. 134, 064111 (2011).

        Parameters
        ----------
        hybrid : bool, optional
            If True, the maxima will be reduced to voxels that have higher values
            than the 26 neighbors surrounding them. The default is False.

        Returns
        -------
        None.

        """
        reference_grid = self.reference_grid.copy()

        # get the voronoi neighbors, their distances, and the area of the corresponding
        # facets. This is used to calculate the volume flux from each voxel
        neighbor_transforms, neighbor_dists, facet_areas, _ = (
            reference_grid.voxel_voronoi_facets
        )
        logging.info("Sorting reference data")
        data = reference_grid.total
        shape = reference_grid.shape
        # flatten data and get initial 1D and 3D voxel indices
        flat_data = data.ravel()
        flat_voxel_indices = np.arange(np.prod(shape))
        flat_voxel_coords = np.indices(shape, dtype=np.int64).reshape(3, -1).T
        # sort data from high to low
        sorted_data_indices = np.flip(np.argsort(flat_data, kind="stable"))
        # create an array that maps original voxel indices to their range in terms
        # of data
        flat_sorted_voxel_indices = np.empty_like(flat_voxel_indices)
        flat_sorted_voxel_indices[sorted_data_indices] = flat_voxel_indices
        # Get a 3D grid representing this data and the corresponding 3D indices
        sorted_voxel_indices = flat_sorted_voxel_indices.reshape(shape)
        sorted_voxel_coords = flat_voxel_coords[sorted_data_indices]
        # remove vacuum points from our list of voxel indices
        sorted_voxel_coords = sorted_voxel_coords[
            : len(sorted_voxel_coords) - self.num_vacuum
        ]
        # Get the flux of volume from each voxel to its neighbor
        logging.info("Calculating voxel flux contributions")
        flux_array, neigh_indices_array, maxima_mask = get_neighbor_flux(
            data=data,
            sorted_voxel_coords=sorted_voxel_coords.copy(),
            voxel_indices=sorted_voxel_indices,
            neighbor_transforms=neighbor_transforms,
            neighbor_dists=neighbor_dists,
            facet_areas=facet_areas,
        )
        # get the frac coords of the maxima
        maxima_vox_coords = sorted_voxel_coords[maxima_mask]
        # maxima_frac_coords = reference_grid.get_frac_coords_from_vox(maxima_vox_coords)
        maxima_num = len(maxima_vox_coords)
        # Calculate the weights for each voxel to each basin
        logging.info("Calculating weights, charges, and volumes")
        # get charge and volume info
        charge_data = self.charge_grid.total
        flat_charge_data = charge_data.ravel()
        sorted_flat_charge_data = flat_charge_data[sorted_data_indices]
        # remove vacuum from charge data
        sorted_flat_charge_data = sorted_flat_charge_data[: len(sorted_voxel_coords)]
        voxel_volume = reference_grid.voxel_volume

        # If we are using the hybrid method, we first assign maxima based on
        # their 26 neighbors rather than the reduced voxel ones
        if hybrid:
            logging.info("Reducing maxima")
            all_neighbor_transforms, all_neighbor_dists = (
                reference_grid.voxel_26_neighbors
            )
            maxima_connections = reduce_maxima(
                maxima_vox_coords,
                data,
                all_neighbor_transforms,
                all_neighbor_dists,
            )
            # NOTE: The maxima are already sorted from highest to lowest
            # We now have a 1D array pointing each maximum to the index of the
            # actual maximum it connects to. We want to reset these so that they
            # run from 0 upward
            unique_maxima, labels_flat = np.unique(
                maxima_connections, return_inverse=True
            )

            # create a labels array and label maxima
            labels = np.full(data.shape, -1, dtype=np.int64)
            labels[
                maxima_vox_coords[:, 0],
                maxima_vox_coords[:, 1],
                maxima_vox_coords[:, 2],
            ] = labels_flat
            # update maxima_num
            maxima_num = len(np.unique(maxima_connections))
            # update maxima vox coords
            maxima_vox_coords = maxima_vox_coords[
                maxima_connections == np.arange(len(maxima_connections))
            ]

        else:
            labels = None

        # label maxima frac coords
        maxima_frac_coords = reference_grid.get_frac_coords_from_vox(maxima_vox_coords)
        self._basin_maxima_frac = maxima_frac_coords

        # get labels for voxels with one weight
        labels, unassigned_mask, charges, volumes = get_single_weight_voxels(
            neigh_indices_array=neigh_indices_array,
            sorted_voxel_coords=sorted_voxel_coords,
            data=data,
            maxima_num=maxima_num,
            sorted_flat_charge_data=sorted_flat_charge_data,
            voxel_volume=voxel_volume,
            labels=labels,
        )
        # Now we have the labels for the voxels that have exactly one weight.
        # We want to get the weights for those that are split. To do this, we
        # need an array with a (N, maxima_num) shape, where N is the number of
        # unassigned voxels. Then we also need an array pointing each unassigned
        # voxel to its point in this array
        unass_to_vox_pointer = np.where(unassigned_mask)[0]
        unassigned_num = len(unass_to_vox_pointer)

        # TODO: Check if the weights array ever actually needs to be the full maxima num wide
        # get unassigned voxel index pointer
        vox_to_unass_pointer = np.full(len(neigh_indices_array), -1, dtype=np.int64)
        vox_to_unass_pointer[unassigned_mask] = np.arange(unassigned_num)

        # get labels, charges, and volumes
        labels, charges, volumes = get_multi_weight_voxels(
            flux_array=flux_array,
            neigh_indices_array=neigh_indices_array,
            labels=labels,
            unass_to_vox_pointer=unass_to_vox_pointer,
            vox_to_unass_pointer=vox_to_unass_pointer,
            sorted_voxel_coords=sorted_voxel_coords,
            charge_array=charges,
            volume_array=volumes,
            sorted_flat_charge_data=sorted_flat_charge_data,
            voxel_volume=voxel_volume,
            maxima_num=maxima_num,
        )

        charges /= shape.prod()
        self._basin_labels = labels
        self._basin_charges = charges
        self._basin_volumes = volumes
        # calculate vacuum charge
        self._vacuum_charge = (charge_data.sum() / shape.prod()) - charges.sum()
        self._vacuum_volume = self.structure.volume - volumes.sum()
        # get significant basins
        significant_basins = charges > self.bader_tol
        self._significant_basins = significant_basins

    def run_atom_assignment(self, structure: Structure = None):
        """
        Assigns bader basins to the atoms in the provided structure.
        """
        # Default structure
        structure = structure or self.structure
        self._structure = structure

        # Shorthand access
        basins = self.basin_maxima_frac  # (N_basins, 3)
        atoms = structure.frac_coords  # (N_atoms, 3)
        L = structure.lattice.matrix  # (3, 3)
        N_basins, N_atoms = len(basins), len(atoms)

        logging.info("Assigning atom properties")

        # Vectorized deltas, minimum‑image wrapping
        diffs = atoms[None, :, :] - basins[:, None, :]
        diffs += np.where(diffs <= -0.5, 1, 0)
        diffs -= np.where(diffs >= 0.5, 1, 0)

        # Cartesian diffs & distances
        cart = np.einsum("bij,jk->bik", diffs, L)
        dists = np.linalg.norm(cart, axis=2)

        # Basin→atom assignment & distances
        basin_atoms = np.argmin(dists, axis=1)  # (N_basins,)
        basin_atom_dists = dists[np.arange(N_basins), basin_atoms]  # (N_basins,)

        # Atom labels per grid point
        # NOTE: append -1 so that vacuum gets assigned to -1 in the atom_labels
        # array
        basin_atoms = np.insert(basin_atoms, len(basin_atoms), -1)
        atom_labels = basin_atoms[self.basin_labels]

        # Sum up charges/volumes per atom in one shot. slice with -1 is necessary
        # to prevent no negative value error
        atom_charges = np.bincount(
            basin_atoms[:-1], weights=self.basin_charges, minlength=N_atoms
        )
        atom_volumes = np.bincount(
            basin_atoms[:-1], weights=self.basin_volumes, minlength=N_atoms
        )

        # Store everything
        self._basin_atoms = basin_atoms[:-1]
        self._basin_atom_dists = basin_atom_dists
        self._atom_labels = atom_labels
        self._atom_charges = atom_charges
        self._atom_volumes = atom_volumes

    def _get_atom_surface_distances(self):
        """
        Calculates the distance from each atom to the nearest surface. This is
        automatically called during the atom assignment and generally should
        not be called manually.

        Returns
        -------
        None.

        """
        atom_labeled_voxels = self.atom_labels
        atom_radii = []
        edge_mask = self.get_basin_edges(atom_labeled_voxels)
        for atom_index in track(
            range(len(self.structure)), description="Calculating atom radii"
        ):
            # get the voxels corresponding to the interior edge of this basin
            atom_edge_mask = (atom_labeled_voxels == atom_index) & edge_mask
            edge_vox_coords = np.argwhere(atom_edge_mask)
            # convert to frac coords
            edge_frac_coords = self.reference_grid.get_frac_coords_from_vox(
                edge_vox_coords
            )
            atom_frac_coord = self.structure.frac_coords[atom_index]
            # Get the difference in coords between atom and edges
            coord_diff = atom_frac_coord - edge_frac_coords
            # Wrap any coords that are more than 0.5 or less than -0.5
            coord_diff -= np.round(coord_diff)
            # Convert to cartesian coordinates
            cart_coords = self.reference_grid.get_cart_coords_from_frac(coord_diff)
            # Calculate distance of each
            norm = np.linalg.norm(cart_coords, axis=1)
            if len(norm) == 0:
                logging.warning(f"No volume assigned to atom at site {atom_index}.")
                atom_radii.append(0)
            else:
                atom_radii.append(norm.min())
        atom_radii = np.array(atom_radii)
        self._atom_surface_distances = atom_radii

    def _get_basin_surface_distances(self):
        """
        Calculates the distance from each basin maxima to the nearest surface.
        This is automatically called during the atom assignment and generally
        should not be called manually.

        Returns
        -------
        None.

        """
        basin_labeled_voxels = self.basin_labels
        basin_radii = []
        edge_mask = self.basin_edges
        for basin in track(
            range(len(self.basin_maxima_frac)), description="Calculating feature radii"
        ):
            # We only calculate the edges for significant basins
            if not self.significant_basins[basin]:
                continue
            basin_edge_mask = (basin_labeled_voxels == basin) & edge_mask
            edge_vox_coords = np.argwhere(basin_edge_mask)
            edge_frac_coords = self.reference_grid.get_frac_coords_from_vox(
                edge_vox_coords
            )
            basin_frac_coord = self.basin_maxima_frac[basin]

            coord_diff = basin_frac_coord - edge_frac_coords
            coord_diff -= np.round(coord_diff)
            cart_coords = self.reference_grid.get_cart_coords_from_frac(coord_diff)
            norm = np.linalg.norm(cart_coords, axis=1)
            basin_radii.append(norm.min())
        basin_radii = np.array(basin_radii)
        self._basin_surface_distances = basin_radii

    @classmethod
    def from_vasp(
        cls,
        charge_filename: Path | str = "CHGCAR",
        reference_filename: Path | None | str = None,
        **kwargs,
    ) -> Self:
        """
        Creates a Bader class object from VASP files.

        Parameters
        ----------
        charge_filename : Path | str, optional
            The path to the CHGCAR like file that will be used for summing charge.
            The default is "CHGCAR".
        reference_filename : Path | None | str, optional
            The path to CHGCAR like file that will be used for partitioning.
            If None, the charge file will be used for partitioning.
        **kwargs : dict
            Keyword arguments to pass to the Bader class.

        Returns
        -------
        Self
            A Bader class object.

        """
        charge_grid = Grid.from_vasp(charge_filename)
        if reference_filename is None:
            reference_grid = charge_grid.copy()
        else:
            reference_grid = Grid.from_vasp(reference_filename)
        return cls(charge_grid=charge_grid, reference_grid=reference_grid, **kwargs)

    @classmethod
    def from_cube(
        cls,
        charge_filename: Path | str,
        reference_filename: Path | None | str = None,
        **kwargs,
    ) -> Self:
        """
        Creates a Bader class object from .cube files.

        Parameters
        ----------
        charge_filename : Path | str, optional
            The path to the .cube file that will be used for summing charge.
        reference_filename : Path | None | str, optional
            The path to .cube file that will be used for partitioning.
            If None, the charge file will be used for partitioning.
        **kwargs : dict
            Keyword arguments to pass to the Bader class.

        Returns
        -------
        Self
            A Bader class object.

        """
        charge_grid = Grid.from_cube(charge_filename)
        if reference_filename is None:
            reference_grid = charge_grid.copy()
        else:
            reference_grid = Grid.from_cube(reference_filename)
        return cls(charge_grid=charge_grid, reference_grid=reference_grid, **kwargs)

    @classmethod
    def from_dynamic(
        cls,
        charge_filename: Path | str,
        reference_filename: Path | None | str = None,
        format: Literal["vasp", "cube", None] = None,
        **kwargs,
    ) -> Self:
        """
        Creates a Bader class object from VASP or .cube files. If no format is
        provided the method will automatically try and determine the file type
        from the name

        Parameters
        ----------
        charge_filename : Path | str
            The path to the file containing the charge density that will be
            integrated.
        reference_filename : Path | None | str, optional
            The path to the file that will be used for partitioning.
            If None, the charge file will be used for partitioning.
        format : Literal["vasp", "cube", None], optional
            The format of the grids to read in. If None, the formats will be
            guessed from the file names.
        **kwargs : dict
            Keyword arguments to pass to the Bader class.

        Returns
        -------
        Self
            A Bader class object.

        """

        charge_grid = Grid.from_dynamic(charge_filename, format=format)
        if reference_filename is None:
            reference_grid = charge_grid.copy()
        else:
            reference_grid = Grid.from_dynamic(reference_filename, format=format)
        return cls(charge_grid=charge_grid, reference_grid=reference_grid, **kwargs)

    def copy(self) -> Self:
        """

        Returns
        -------
        Self
            A deep copy of this Bader object.

        """
        return copy.deepcopy(self)

    @property
    def results_summary(self) -> dict:
        """

        Returns
        -------
        results_dict : dict
            A dictionary summary of all results

        """
        results_dict = {
            "method": self.method,
            "basin_maxima_frac": self.basin_maxima_frac,
            "basin_charges": self.basin_charges,
            "basin_volumes": self.basin_volumes,
            "basin_surface_distances": self.basin_surface_distances,
            "basin_atoms": self.basin_atoms,
            "basin_atom_dists": self.basin_atom_dists,
            "atom_charges": self.atom_charges,
            "atom_volumes": self.atom_volumes,
            "atom_surface_distances": self.atom_surface_distances,
            "structure": self.structure,
            "vacuum_charge": self.vacuum_charge,
            "vacuum_volume": self.vacuum_volume,
            "significant_basins": self.significant_basins,
        }
        return results_dict

    def write_basin_volumes(
        self,
        basin_indices: NDArray,
        directory: str | Path = None,
        file_prefix: str = "CHGCAR",
        data_type: Literal["charge", "reference"] = "charge",
    ):
        """
        Writes bader basins to vasp-like files. Points belonging to the basin
        will have values from the charge or reference grid, and all other points
        will be 0. Filenames are written as {file_prefix}_b{i} where i is the
        basin index.

        Parameters
        ----------
        basin_indices : NDArray
            The list of basin indices to write
        directory: str | Path
            The directory to write the files in. If None, the directory currently
            assigned to the Bader object will work.
        file_prefix : str, optional
            The string to append to each file name. The default is "CHGCAR".
        data_type : Literal["charge", "reference"], optional
            Which file to write from. The default is "charge".

        Returns
        -------
        None.

        """
        if data_type == "charge":
            grid = self.charge_grid.copy()
        elif data_type == "reference":
            grid = self.reference_grid.copy()

        data_array = grid.total
        if directory is None:
            directory = self.directory
        for basin in basin_indices:
            mask = self.basin_labels == basin
            data_array_copy = data_array.copy()
            data_array_copy[~mask] = 0
            data = {"total": data_array_copy}
            grid = Grid(structure=self.structure, data=data)
            grid.write_file(directory / f"{file_prefix}_b{basin}")

    def write_all_basin_volumes(
        self,
        directory: str | Path = None,
        file_prefix: str = "CHGCAR",
        data_type: Literal["charge", "reference"] = "charge",
    ):
        """
        Writes all bader basins to vasp-like files. Points belonging to the basin
        will have values from the charge or reference grid, and all other points
        will be 0. Filenames are written as {file_prefix}_b{i} where i is the
        basin index.

        Parameters
        ----------
        directory: str | Path
            The directory to write the files in. If None, the directory currently
            assigned to the Bader object will work.
        file_prefix : str, optional
            The string to append to each file name. The default is "CHGCAR".
        data_type : Literal["charge", "reference"], optional
            Which file to write from. The default is "charge".

        Returns
        -------
        None.

        """
        basin_indices = np.where(self.significant_basins)[0]
        self.write_basin_volumes(
            basin_indices=basin_indices,
            directory=directory,
            file_prefix=file_prefix,
            data_type=data_type,
        )

    def write_basin_volumes_sum(
        self,
        basin_indices: NDArray,
        directory: str | Path = None,
        file_prefix: str = "CHGCAR",
        data_type: Literal["charge", "reference"] = "charge",
    ):
        """
        Writes the union of the provided bader basins to vasp-like files.
        Points belonging to the basins will have values from the charge or
        reference grid, and all other points will be 0. Filenames are written
        as {file_prefix}_bsum.

        Parameters
        ----------
        basin_indices : NDArray
            The list of basin indices to sum and write
        directory: str | Path
            The directory to write the files in. If None, the directory currently
            assigned to the Bader object will work.
        file_prefix : str, optional
            The string to append to each file name. The default is "CHGCAR".
        data_type : Literal["charge", "reference"], optional
            Which file to write from. The default is "charge".

        Returns
        -------
        None.

        """
        if data_type == "charge":
            grid = self.charge_grid.copy()
        elif data_type == "reference":
            grid = self.reference_grid.copy()

        data_array = grid.total
        if directory is None:
            directory = self.directory
        mask = np.isin(self.basin_labels, basin_indices)
        data_array_copy = data_array.copy()
        data_array_copy[~mask] = 0
        data = {"total": data_array_copy}
        grid = Grid(structure=self.structure, data=data)
        grid.write_file(directory / f"{file_prefix}_bsum")

    def write_atom_volumes(
        self,
        atom_indices: NDArray,
        directory: str | Path = None,
        file_prefix: str = "CHGCAR",
        data_type: Literal["charge", "reference"] = "charge",
    ):
        """
        Writes atomic basins to vasp-like files. Points belonging to the atom
        will have values from the charge or reference grid, and all other points
        will be 0. Filenames are written as {file_prefix}_a{i} where i is the
        atom index.

        Parameters
        ----------
        atom_indices : NDArray
            The list of atom indices to write
        directory: str | Path
            The directory to write the files in. If None, the directory currently
            assigned to the Bader object will work.
        file_prefix : str, optional
            The string to append to each file name. The default is "CHGCAR".
        data_type : Literal["charge", "reference"], optional
            Which file to write from. The default is "charge".

        Returns
        -------
        None.

        """
        if data_type == "charge":
            grid = self.charge_grid.copy()
        elif data_type == "reference":
            grid = self.reference_grid.copy()

        data_array = grid.total
        if directory is None:
            directory = self.directory
        for atom_index in atom_indices:
            mask = self.atom_labels == atom_index
            data_array_copy = data_array.copy()
            data_array_copy[~mask] = 0
            data = {"total": data_array_copy}
            grid = Grid(structure=self.structure, data=data)
            grid.write_file(directory / f"{file_prefix}_a{atom_index}")

    def write_all_atom_volumes(
        self,
        directory: str | Path = None,
        file_prefix: str = "CHGCAR",
        data_type: Literal["charge", "reference"] = "charge",
    ):
        """
        Writes all atomic basins to vasp-like files. Points belonging to the atom
        will have values from the charge or reference grid, and all other points
        will be 0. Filenames are written as {file_prefix}_a{i} where i is the
        atom index.

        Parameters
        ----------
        directory: str | Path
            The directory to write the files in. If None, the directory currently
            assigned to the Bader object will work.
        file_prefix : str, optional
            The string to append to each file name. The default is "CHGCAR".
        data_type : Literal["charge", "reference"], optional
            Which file to write from. The default is "charge".

        Returns
        -------
        None.

        """
        atom_indices = np.array(range(len(self.structure)))
        self.write_atom_volumes(
            atom_indices=atom_indices,
            directory=directory,
            file_prefix=file_prefix,
            data_type=data_type,
        )

    def write_atom_volumes_sum(
        self,
        atom_indices: NDArray,
        directory: str | Path = None,
        file_prefix: str = "CHGCAR",
        data_type: Literal["charge", "reference"] = "charge",
    ):
        """
        Writes the union of the provided atom basins to vasp-like files.
        Points belonging to the atoms will have values from the charge or
        reference grid, and all other points will be 0. Filenames are written
        as {file_prefix}_asum.

        Parameters
        ----------
        atom_indices : NDArray
            The list of atom indices to sum and write
        directory: str | Path
            The directory to write the files in. If None, the directory currently
            assigned to the Bader object will work.
        file_prefix : str, optional
            The string to append to each file name. The default is "CHGCAR".
        data_type : Literal["charge", "reference"], optional
            Which file to write from. The default is "charge".

        Returns
        -------
        None.

        """
        if data_type == "charge":
            grid = self.charge_grid.copy()
        elif data_type == "reference":
            grid = self.reference_grid.copy()

        data_array = grid.total
        if directory is None:
            directory = self.directory
        mask = np.isin(self.atom_labels, atom_indices)
        data_array_copy = data_array.copy()
        data_array_copy[~mask] = 0
        data = {"total": data_array_copy}
        grid = Grid(structure=self.structure, data=data)
        grid.write_file(directory / f"{file_prefix}_asum")

    def get_atom_results_dataframe(self) -> pd.DataFrame:
        """
        Collects a summary of results for the atoms in a pandas DataFrame.

        Returns
        -------
        atoms_df : pd.DataFrame
            A table summarizing the atomic basins.

        """
        # Get atom results summary
        atom_frac_coords = self.structure.frac_coords
        atoms_df = pd.DataFrame(
            {
                "label": self.structure.labels,
                "x": atom_frac_coords[:, 0],
                "y": atom_frac_coords[:, 1],
                "z": atom_frac_coords[:, 2],
                "charge": self.atom_charges,
                "volume": self.atom_volumes,
                "surface_dist": self.atom_surface_distances,
            }
        )
        return atoms_df

    def get_basin_results_dataframe(self):
        """
        Collects a summary of results for the basins in a pandas DataFrame.

        Returns
        -------
        basin_df : pd.DataFrame
            A table summarizing the basins.

        """
        subset = self.significant_basins
        basin_frac_coords = self.basin_maxima_frac[subset]
        basin_df = pd.DataFrame(
            {
                "atoms": self.basin_atoms[subset],
                "x": basin_frac_coords[:, 0],
                "y": basin_frac_coords[:, 1],
                "z": basin_frac_coords[:, 2],
                "charge": self.basin_charges[subset],
                "volume": self.basin_volumes[subset],
                "surface_dist": self.basin_surface_distances,
            }
        )
        return basin_df

    def write_results_summary(
        self,
        directory: Path | str | None = None,
    ):
        """
        Writes a summary of atom and basin results to .tsv files.

        Parameters
        ----------
        directory : Path | str | None, optional
            The directory to write to. If None, the current directory assigned
            to the bader class will be used.

        Returns
        -------
        None.

        """
        if directory is None:
            directory = self.directory

        # Get atom results summary
        atoms_df = self.get_atom_results_dataframe()
        formatted_atoms_df = atoms_df.copy()
        numeric_cols = formatted_atoms_df.select_dtypes(include="number").columns
        formatted_atoms_df[numeric_cols] = formatted_atoms_df[numeric_cols].map(
            lambda x: f"{x:.6f}"
        )

        # Get basin results summary
        basin_df = self.get_basin_results_dataframe()
        formatted_basin_df = basin_df.copy()
        numeric_cols = formatted_basin_df.select_dtypes(include="number").columns
        formatted_basin_df[numeric_cols] = formatted_basin_df[numeric_cols].map(
            lambda x: f"{x:.6f}"
        )

        # Determine max width per column including header
        atom_col_widths = {
            col: max(len(col), formatted_atoms_df[col].map(len).max())
            for col in atoms_df.columns
        }
        basin_col_widths = {
            col: max(len(col), formatted_basin_df[col].map(len).max())
            for col in basin_df.columns
        }

        # Write to file with aligned columns using tab as separator
        for df, col_widths, name in zip(
            [formatted_atoms_df, formatted_basin_df],
            [atom_col_widths, basin_col_widths],
            ["bader_atom_summary.tsv", "bader_basin_summary.tsv"],
        ):
            with open(directory / name, "w") as f:
                # Write header
                header = "\t".join(f"{col:<{col_widths[col]}}" for col in df.columns)
                f.write(header + "\n")

                # Write rows
                for _, row in df.iterrows():
                    line = "\t".join(
                        f"{val:<{col_widths[col]}}" for col, val in row.items()
                    )
                    f.write(line + "\n")
