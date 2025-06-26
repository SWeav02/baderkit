# -*- coding: utf-8 -*-

import copy
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
    get_edges,
    get_multi_weight_voxels,
    get_neargrid_labels,
    get_neighbor_flux,
    get_single_weight_voxels,
    get_steepest_pointers,
    propagate_edges,
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
            "ongrid", "neargrid", "hybrid-neargrid", "weight", "hybrid-weight"
        ] = None,
        refinement_method: Literal["recursive", "single"] = None,
        directory: Path = Path("."),
    ):
        """

        Parameters
        ----------
        charge_grid : Grid
            A Grid object with the charge density that will be integrated.
        reference_grid : Grid
            A grid object whose values will be used to construct the basins.
        method : Literal["ongrid", "neargrid", "hybrid-neargrid", "weight", "hybrid-weight"], optional
            The algorithm to use for generating bader basins. If None, defaults
            to weight.
        refinement_method : Literal["recursive", "single"], optional
            For methods that refine the basin edges (neargrid), whether to
            refine the edges until none change or to refine a single time. If
            None, defaults to recursive.
        directory : Path, optional
            The directory that files will be written to by default.
            The default is Path("."), or the current active directory.

        Returns
        -------
        None.

        """
        self.charge_grid = charge_grid
        self.reference_grid = reference_grid
        if method is not None:
            self.method = method
        else:
            self.method = "weight"
        self.directory = directory
        if refinement_method is not None:
            self.refinement_method = refinement_method
        else:
            self.refinement_method = "recursive"

        # define hidden class variables. This allows us to cache properties and
        # still be able to recalculate them
        # Assigned by run_bader
        self._basin_labels = None
        self._basin_maxima_frac = None
        self._basin_charges = None
        self._basin_volumes = None
        self._basin_surface_distances = None
        self._basin_edges = None
        # Assigned by run_atom_assignment
        self._basin_atoms = None
        self._basin_atom_dists = None
        self._atom_labels = None
        self._atom_charges = None
        self._atom_volumes = None
        self._atom_surface_distances = None
        self._structure = None

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

    @staticmethod
    def methods() -> list[str]:
        """

        Returns
        -------
        list[str]
            A list of the available methods.

        """

        return ["ongrid", "neargrid", "hybrid-neargrid", "weight", "hybrid-weight"]

    @staticmethod
    def get_basin_edges(
        basin_labels: NDArray[float], neighbor_transforms: NDArray = None
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
        return get_edges(basin_labels, neighbor_transforms=neighbor_transforms)

    @staticmethod
    def get_basin_charges_and_volumes(
        basin_labels: NDArray[int],
        grid: Grid,
    ) -> tuple[NDArray[float], NDArray[float]]:
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
        voxel_volume = grid.voxel_volume
        charge_array = grid.total
        charges = []
        volumes = []
        maxima_num = len(np.unique(basin_labels))
        for j in range(maxima_num):
            mask = basin_labels == j
            charge_values = charge_array[mask]
            charges.append(charge_values.sum())
            volumes.append(voxel_volume * len(charge_values))
        charges = np.array(charges)
        volumes = np.array(volumes)
        # adjust charges
        charges /= grid.shape.prod()
        return charges, volumes

    def run_bader(self) -> None:
        """
        Runs the entire bader process and saves results to class variables.

        Raises
        ------
        ValueError
            The class method variable must be 'ongrid', 'neargrid', 'weight' or
            'hybrid-weight'.

        Returns
        -------
        None.

        """
        if self.method == "ongrid":
            self._run_bader_on_grid()

        elif self.method == "neargrid":
            self._run_bader_near_grid()

        elif self.method == "hybrid-neargrid":
            self._run_bader_near_grid(hybrid=True)

        elif self.method == "weight":
            self._run_bader_weight()

        elif self.method == "hybrid-weight":
            self._run_bader_weight(hybrid=True)

        else:
            raise ValueError(
                f"{self.method} is not a valid algorithm."
                "Acceptable values are 'ongrid' and 'weight'"
            )

    def _run_bader_on_grid(self):
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

        # get an array where each entry is that voxels unique label
        initial_labels = np.arange(np.prod(shape)).reshape(shape)

        # get shifts to move from a voxel to the 26 surrounding voxels
        neighbor_transforms, neighbor_dists = grid.voxel_26_neighbors

        # For each voxel, get the label of the surrounding voxel that has the highest
        # elf
        logging.info("Calculating steepest neighbors")
        best_label = get_steepest_pointers(
            data=data,
            initial_labels=initial_labels,
            neighbor_transforms=neighbor_transforms,
            neighbor_dists=neighbor_dists,
        )

        # ravel the best labels to get a 1D array pointing from each voxel to its steepest
        # neighbor
        pointers = best_label.ravel()
        # Our pointers object is a 1D array pointing each voxel to its parent voxel. We
        # essentially have a classic forrest of trees problem where each maxima is
        # a root and we want to point all of our voxels to their respective root.
        # We being a while loop. In each loop, we remap our pointers to point at
        # the index that its parent was pointing at.
        logging.info("Finding roots")
        while True:
            # reassign each index to the value at the index it is pointing to
            new_parents = pointers[pointers]
            # check if we have the same value as before
            if np.all(new_parents == pointers):
                break
            # if not, relabel our pointers
            pointers = new_parents
        # We now have our roots. Relabel so that they go from 0 to the length of our
        # roots
        unique_roots, labels_flat = np.unique(pointers, return_inverse=True)
        # reconstruct a 3D array with our labels
        labels = labels_flat.reshape(shape)
        # store our labels
        self._basin_labels = labels

        # get maxima voxels
        maxima_mask = best_label == initial_labels
        maxima_vox = np.argwhere(maxima_mask)
        # get corresponding basin labels
        maxima_labels = labels[maxima_vox[:, 0], maxima_vox[:, 1], maxima_vox[:, 2]]
        if not np.all(np.equal(maxima_labels, np.sort(maxima_labels))):
            breakpoint()

        # get maxima coords
        maxima_frac = grid.get_frac_coords_from_vox(maxima_vox)
        self._basin_maxima_frac = maxima_frac

        # get charge and volume for each label
        basin_charges, basin_volumes = self.get_basin_charges_and_volumes(
            basin_labels=labels,
            grid=self.charge_grid,
        )

        self._basin_charges, self._basin_volumes = basin_charges, basin_volumes

    def _run_bader_near_grid(self, hybrid: bool = False):
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
        data = grid.total
        shape = data.shape
        neighbor_transforms, neighbor_dists = grid.voxel_26_neighbors
        matrix = grid.matrix
        # convert to lattice vectors as columns
        dir2car = matrix.T
        # get lattice to cartesian matrix
        lat2car = dir2car / grid.shape[np.newaxis, :]
        # get inverse for cartesian to lattice matrix
        car2lat = np.linalg.inv(lat2car)
        if not hybrid:
            # we want to make our initial assignments using the neargrid method
            labels, maxima_mask = get_neargrid_labels(
                data=data,
                car2lat=car2lat,
                neighbor_transforms=neighbor_transforms,
                neighbor_dists=neighbor_dists,
            )
        else:
            # we want to make our initial assignments using the ongrid method
            # get an array where each entry is that voxels unique label
            initial_labels = np.arange(np.prod(shape)).reshape(shape)
            # get shifts to move from a voxel to the 26 surrounding voxels
            neighbor_transforms, neighbor_dists = grid.voxel_26_neighbors
            # For each voxel, get the label of the surrounding voxel that has the highest
            # elf
            logging.info("Calculating steepest neighbors")
            best_label = get_steepest_pointers(
                data=data,
                initial_labels=initial_labels,
                neighbor_transforms=neighbor_transforms,
                neighbor_dists=neighbor_dists,
            )
            # create a maxima mask
            maxima_mask = best_label == initial_labels
            # ravel the best labels to get a 1D array pointing from each voxel to its steepest
            # neighbor
            pointers = best_label.ravel()
            # Our pointers object is a 1D array pointing each voxel to its parent voxel. We
            # essentially have a classic forrest of trees problem where each maxima is
            # a root and we want to point all of our voxels to their respective root.
            # We being a while loop. In each loop, we remap our pointers to point at
            # the index that its parent was pointing at.
            logging.info("Finding roots")
            while True:
                # reassign each index to the value at the index it is pointing to
                new_parents = pointers[pointers]
                # check if we have the same value as before
                if np.all(new_parents == pointers):
                    break
                # if not, relabel our pointers
                pointers = new_parents
            # We now have our roots. Relabel so that they go from 0 to the length of our
            # roots
            unique_roots, labels_flat = np.unique(pointers, return_inverse=True)
            # reconstruct a 3D array with our labels
            labels = labels_flat.reshape(shape)
            # convert to index 1
            labels += 1

        reassignments = 1
        maxima_vox = np.argwhere(maxima_mask)
        # get our edges
        refinement_mask = get_edges(
            labeled_array=labels, neighbor_transforms=neighbor_transforms
        )
        checked_mask = np.zeros(refinement_mask.shape, dtype=np.bool_)
        while reassignments > 0:
            # remove maxima from edge mask
            for i, j, k in maxima_vox:
                refinement_mask[i, j, k] = False
            # get edge indices
            refinement_indices = np.argwhere(refinement_mask)
            print(f"Refining {len(refinement_indices)} points")
            # reassign edges
            labels, reassignments, refinement_mask, checked_mask = refine_neargrid(
                data=data,
                labels=labels,
                refinement_indices=refinement_indices,
                refinement_mask=refinement_mask,
                checked_mask=checked_mask,
                maxima_mask=maxima_mask,
                car2lat=car2lat,
                neighbor_transforms=neighbor_transforms,
                neighbor_dists=neighbor_dists,
            )
            print(f"{reassignments} values changed")
            # if our refinement method is single, we cancel the loop here
            if self.refinement_method == "single":
                break

        # get corresponding basin labels
        maxima_labels = labels[maxima_vox[:, 0], maxima_vox[:, 1], maxima_vox[:, 2]]
        # sort from lowest to highest
        maxima_sorted_indices = np.argsort(maxima_labels)
        maxima_vox = maxima_vox[maxima_sorted_indices]
        maxima_frac_coords = grid.get_frac_coords_from_vox(maxima_vox)
        self._basin_maxima_frac = maxima_frac_coords
        # adjust labels to 0 index convention
        labels -= 1
        # assign labels
        self._basin_labels = labels
        # get charge and volume for each label
        basin_charges, basin_volumes = self.get_basin_charges_and_volumes(
            basin_labels=labels,
            grid=self.charge_grid,
        )
        self._basin_charges, self._basin_volumes = basin_charges, basin_volumes

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
        shape = data.shape
        # flatten data and get initial 1D and 3D voxel indices
        flat_data = data.ravel()
        flat_voxel_indices = np.arange(np.prod(shape))
        flat_voxel_coords = np.indices(shape).reshape(3, -1).T
        # sort data from high to low
        sorted_data_indices = np.flip(np.argsort(flat_data, kind="stable"))
        # create an array that maps original voxel indices to their range in terms
        # of data
        flat_sorted_voxel_indices = np.empty_like(flat_voxel_indices)
        flat_sorted_voxel_indices[sorted_data_indices] = flat_voxel_indices
        # Get a 3D grid representing this data and the corresponding 3D indices
        sorted_voxel_indices = flat_sorted_voxel_indices.reshape(shape)
        sorted_voxel_coords = flat_voxel_coords[sorted_data_indices]
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
        voxel_volume = reference_grid.voxel_volume

        # If we are using the hybrid method, we first assign maxima based on
        # their 26 neighbors rather than the reduced voxel ones
        if hybrid:
            logging.info("Reducing maxima")
            # get an array where each entry is that voxels unique label
            initial_labels = np.arange(np.prod(shape)).reshape(shape)
            # get shifts to move from a voxel to the 26 surrounding voxels
            all_neighbor_transforms = np.array(
                [s for s in product([-1, 0, 1], repeat=3) if s != (0, 0, 0)]
            )
            # get distance from each voxel to its neighbor in cartesian coordinates. This
            # allows us to normalize the gradients
            cartesian_shifts = reference_grid.get_cart_coords_from_vox(
                all_neighbor_transforms
            )
            cartesian_dists = np.linalg.norm(cartesian_shifts, axis=1)
            best_label = get_steepest_pointers(
                data=data,
                initial_labels=initial_labels,
                neighbor_transforms=all_neighbor_transforms,
                neighbor_dists=cartesian_dists,
            )
            # ravel the best labels to get a 1D array pointing from each voxel to its steepest
            # neighbor
            pointers = best_label.ravel()
            # Our pointers object is a 1D array pointing each voxel to its parent voxel. We
            # essentially have a classic forrest of trees problem where each maxima is
            # a root and we want to point all of our voxels to their respective root.
            # We being a while loop. In each loop, we remap our pointers to point at
            # the index that its parent was pointing at.
            while True:
                # reassign each index to the value at the index it is pointing to
                new_parents = pointers[pointers]
                # check if we have the same value as before
                if np.all(new_parents == pointers):
                    break
                # if not, relabel our pointers
                pointers = new_parents
            # before reorganizing, update the voxel coords
            new_maxima_mask = pointers.reshape(data.shape) == initial_labels
            maxima_vox_coords = np.argwhere(new_maxima_mask)
            # reorganize by maxima
            pointers = pointers[sorted_data_indices]
            maxima_labels = pointers[maxima_mask]
            maxima_coords = sorted_voxel_coords[maxima_mask]
            # get the unique maxima and the corresponding label for each
            unique_maxima, labels_flat = np.unique(maxima_labels, return_inverse=True)
            # create an labels array and label maxima
            labels = np.full(data.shape, -1, dtype=np.int64)
            labels[maxima_coords[:, 0], maxima_coords[:, 1], maxima_coords[:, 2]] = (
                labels_flat
            )
            # update maxima_num
            maxima_num = len(unique_maxima)

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
        # need an array with a N, maxima_num shape, where N is the number of
        # unassigned voxels. Then we also need an array pointing each unassigned
        # voxel to its point in this array
        unass_to_vox_pointer = np.where(unassigned_mask)[0]
        unassigned_num = len(unass_to_vox_pointer)

        # TODO: Check if the weights array ever actually needs to be the full maxima num wide
        # get unassigned voxel index pointer
        vox_to_unass_pointer = np.full(len(flat_charge_data), -1, dtype=np.int64)
        vox_to_unass_pointer[unassigned_mask] = np.arange(unassigned_num)

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

        charges /= reference_grid.shape.prod()
        self._basin_labels = labels
        self._basin_charges = charges
        self._basin_volumes = volumes

    def run_atom_assignment(self, structure: Structure = None):
        """
        Assigns bader basins to the atoms in the provided structure. If
        no structure is provided, defaults to the reference grid structure.

        This method generally shouldn't be run manually, but is useful
        for reassigning basins when working with dummy atoms (e.g. in electrides)

        Parameters
        ----------
        structure : Structure, optional
            The pymatgen structure to assign basins to. If None, the structure
            of the reference grid will be used.

        Returns
        -------
        None.

        """
        if structure is None:
            structure = self.structure
        self._structure = structure
        # Get the frac coords for each basin and atom. These must be in the
        # same order as the corresponding basin labels
        basin_frac_coords = self.basin_maxima_frac
        atom_frac_coords = structure.frac_coords
        logging.info("Assigning atom properties")
        # create arrays for atom properties
        basin_atoms = np.empty(len(basin_frac_coords), dtype=int)
        basin_atom_dists = np.empty(len(basin_frac_coords))
        atom_labels = np.zeros(self.basin_labels.shape, dtype=np.int64)
        atom_charges = np.zeros(len(atom_frac_coords))
        atom_volumes = np.zeros(len(atom_frac_coords))

        for i, frac_coord in enumerate(basin_frac_coords):
            # get the difference between this basin and all of the atoms
            diffs = atom_frac_coords - frac_coord
            # wrap anything below -0.5 or above 0.5
            diffs[diffs < -0.5] += 1
            diffs[diffs > 0.5] -= 1
            # convert to cartesian coords and calculate distance
            cart_diffs = diffs @ structure.lattice.matrix
            dists = np.linalg.norm(cart_diffs, axis=1)
            # get the lowest distance and corresponding atom
            min_dist = dists.min()
            assignment = np.argwhere(dists == min_dist)[0][0]
            # assign this atom label to this basin and update properties
            basin_atoms[i] = assignment
            basin_atom_dists[i] = min_dist
            atom_labels[self.basin_labels == i] = assignment
            try:
                atom_charges[assignment] += self.basin_charges[i]
            except:
                breakpoint()
            atom_volumes[assignment] += self.basin_volumes[i]

        # update class variables
        self._basin_atoms = basin_atoms
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
        basin_indices = np.array(range(len(self.basin_atoms)))
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
        basin_frac_coords = self.basin_maxima_frac
        basin_df = pd.DataFrame(
            {
                "atoms": self.basin_atoms,
                "x": basin_frac_coords[:, 0],
                "y": basin_frac_coords[:, 1],
                "z": basin_frac_coords[:, 2],
                "charge": self.basin_charges,
                "volume": self.basin_volumes,
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
