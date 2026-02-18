# -*- coding: utf-8 -*-

import importlib
import logging
import time
import warnings
from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pymatgen.io.vasp import Potcar

from baderkit.core.base.base_analysis import BaseAnalysis
from baderkit.core.toolkit import Structure

from .methods import Method
from .methods.shared_numba import (
    get_edges,
    get_min_avg_surface_dists,
    get_neighboring_basin_surface_area,
    get_persistence_groups,
)

# This allows for Self typing and is compatible with python 3.10
Self = TypeVar("Self", bound="Bader")

# TODO:
# - Add periodicity options (i.e. on vs. off etc.)
# - Improve docstrings, especially for write methods, so that they show kwargs


class Bader(BaseAnalysis):
    """
    Class for running Bader analysis on a regular grid. For information on each
    method, see our [docs](https://sweav02.github.io/baderkit/)

    Parameters
    ----------
    charge_grid : Grid
        The Grid object with the charge density that will be integrated.
    total_charge_grid : Grid | None, optional
        The Grid object used for determining vacuum regions in the system. For
        pseudopotential codes this represents the total electron density and should
        be provided whenever possible. If None, defaults to the charge_grid.
    reference_grid : Grid | None, optional
        The Grid object whose values will be used to construct the basins. This
        should typically only be set when partitioning functions other than the
        charge density (e.g. ELI-D, ELF, etc.).If None, defaults to the
        total_charge_grid.
    method : str | Method, optional
        The algorithm to use for generating bader basins.
    vacuum_tol : float | bool, optional
        If a float is provided, this is the value below which a point will
        be considered part of the vacuum. If a bool is provided, no vacuum
        will be used on False, and the default tolerance (0.001) will be used on True.
    nna_cutoff : float | bool, optional
        If a float is provided, any basins found at a distance in Angstroms greater
        than this cutoff will be considered non-nuclear attractors. If any are
        found, dummy atoms will be appended to the structure and regarded as
        separate species. If a bool is provided, NNAs will be assigned to the
        nearest atom on False or a default value (1 Ang) will be used on True.
    maxima_persistence_tol : float, optional
        It is common for false maxima to be found using only nearest neighbor
        points. To deal with this we combine pairs of basins that have low
        topological persistence.
        
        The persistence score is calculated as:
            
            score = (lower_max - connection_value) / connection_value
    maxima_persistence_tol : float, optional
        It is common for false minima to be found using only nearest neighbor
        points. To deal with this we combine pairs of basins that have low
        topological persistence. We use a separate tolerance as minima typically
        have lower persistence values.
        
        The persistence score is calculated as:
            
            score = (connection_value - higher_min) / connection_value


    """

    _reset_props = [
        # assigned by _run_bader
        "maxima_basin_labels",
        "maxima_basin_images",
        "maxima_frac",
        "maxima_vox",
        "maxima_voxel_groups",
        "maxima_charge_values",
        "maxima_ref_values",
        "maxima_persistence_values",
        "basin_charges",
        "basin_volumes",
        # assigned by _run_minima_bader
        "minima_basin_labels",
        "minima_basin_images",
        "minima_frac",
        "minima_vox",
        "minima_voxel_groups",
        "minima_charge_values",
        "minima_ref_values",
        "minima_persistence_values",
        # Assigned by calling the property
        "basin_min_surface_distances",
        "basin_avg_surface_distances",
        "basin_edges",
        "atom_edges",
        "basin_surface_areas",
        "basin_contact_surface_areas",
        "atom_surface_areas",
        "atom_contact_surface_areas",
        # Assigned by run_atom_assignment
        "basin_atoms",
        "basin_atom_dists",
        "atom_labels",
        "atom_charges",
        "atom_volumes",
        "atom_min_surface_distances",
        "atom_avg_surface_distances",
    ]

    def __init__(
        self,
        method: str | Method = Method.weight,
        nna_cutoff: float | bool = False,
        maxima_persistence_tol: float = 0.05,
        minima_persistence_tol: float = 0.005,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # ensure the method is valid
        valid_methods = [m.value for m in Method]
        if isinstance(method, Method):
            self._method = method
        elif method in valid_methods:
            self._method = Method(method)
        else:
            raise ValueError(
                f"Invalid method '{method}'. Available options are: {valid_methods}"
            )
        
        if nna_cutoff is True:
            nna_cutoff = 1.0
        self._nna_cutoff = nna_cutoff
        self._maxima_persistence_tol = maxima_persistence_tol
        self._minima_persistence_tol = minima_persistence_tol

        # whether or not to use overdetermined gradients in neargrid methods.
        self._use_overdetermined = False

    ###########################################################################
    # Set Properties
    ###########################################################################

    @property
    def method(self) -> str:
        """

        Returns
        -------
        str
            The algorithm to use for generating bader basins. If None, defaults
            to neargrid.

        """
        return self._method

    @method.setter
    def method(self, value: str | Method):
        # Support both Method instances and their string values
        valid_values = [m.value for m in Method]
        if isinstance(value, Method):
            self._method = value
        elif value in valid_values:
            self._method = Method(value)
        else:
            raise ValueError(
                f"Invalid method '{value}'. Available options are: {valid_values}"
            )
        self._reset_properties(exclude_properties=["vacuum_mask", "num_vacuum", "vacuum_charge", "vacuum_volume",])
        
    @property
    def nna_cutoff(self) -> float:
        """

        Returns
        -------
        float
            The distance cutoff in angstroms above which a basin will be considered
            a non-nuclear attractor.
            
            If a float is provided, any basins found at a distance in Angstroms greater
            than this cutoff will be considered non-nuclear attractors. If any are
            found, dummy atoms will be appended to the structure and regarded as
            separate species. If a bool is provided, NNAs will be assigned to the
            nearest atom on False or a default value (1 Ang) will be used on True.

        """
        return self._nna_cutoff
        
    @nna_cutoff.setter
    def nna_cutoff(self, value: str | Method):
        self._nna_cutoff = value
        # reset atom properties
        self._reset_properties(include_properties=[
            "structure",
            "atom_edges",
            "atom_surface_areas",
            "atom_contact_surface_areas",
            "basin_atoms",
            "basin_atom_dists",
            "atom_labels",
            "atom_charges",
            "atom_volumes",
            "atom_min_surface_distances",
            "atom_avg_surface_distances",
            ])
        
    @property
    def maxima_persistence_tol(self) -> float:
        """

        Returns
        -------
        float
            It is common for false maxima to be found using only nearest neighbor
            points. To deal with this we combine pairs of basins that have low
            topological persistence.
            
            The persistence score is calculated as:
                
                score = (lower_maximum - connection_value) / connection_value


        """
        return self._maxima_persistence_tol
        
    @maxima_persistence_tol.setter
    def maxima_persistence_tol(self, value: str | Method):
        self._maxima_persistence_tol = value
        # reset atom properties
        self._reset_properties(exclude_properties=[
            "vacuum_mask",
            "num_vacuum",
            "vacuum_charge",
            "vacuum_volume",
            "structure",
            ])

    ###########################################################################
    # Maxima Basin Properties
    ###########################################################################

    @property
    def maxima_basin_labels(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            A 3D array of the same shape as the reference grid with entries
            representing the basin the voxel belongs to.

        """
        if self._maxima_basin_labels is None:
            self._run_bader()
        return self._maxima_basin_labels
    
    @property
    def maxima_basin_images(self) -> NDArray[int]:
        """

        Returns
        -------
        NDArray[int]
            a 3D array of the same shape as the reference grid with entries
            representing which periodic neighbor each point is assigned to. For
            example, a point may be assigned to atom 0, but following the gradient
            leads to atom zero in the unit cell at (1, 0, 0). Images are represented
            by integers to save memory and follow the values created by itertools:
                0: [-1, -1, -1]
                 1: [-1, -1,  0]
                 2: [-1, -1,  1]
                 3: [-1,  0, -1]
                 4: [-1,  0,  0]
                 5: [-1,  0,  1]
                 6: [-1,  1, -1]
                 7: [-1,  1,  0]
                 8: [-1,  1,  1]
                 9: [ 0, -1, -1]
                10: [ 0, -1,  0]
                11: [ 0, -1,  1]
                12: [ 0,  0, -1]
                13: [ 0,  0,  0]
                14: [ 0,  0,  1]
                15: [ 0,  1, -1]
                16: [ 0,  1,  0]
                17: [ 0,  1,  1]
                18: [ 1, -1, -1]
                19: [ 1, -1,  0]
                20: [ 1, -1,  1]
                21: [ 1,  0, -1]
                22: [ 1,  0,  0]
                23: [ 1,  0,  1]
                24: [ 1,  1, -1]
                25: [ 1,  1,  0]
                26: [ 1,  1,  1]

        """
        if self._maxima_basin_images is None:
            self._run_bader()
        return self._maxima_basin_images
    
    @property
    def maxima_frac(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The fractional coordinates of each attractor.

        """
        if self._maxima_frac is None:
            self._run_bader()
        return self._maxima_frac

    @property
    def maxima_charge_values(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The charge data value at each maximum. If the maximum is
            off grid, this value will be interpolated.

        """
        # TODO: change this to quadratic fit to match reference value method
        if self._maxima_charge_values is None:
            self._maxima_charge_values = self.charge_grid.values_at(
                self.maxima_frac
            )
        return self._maxima_charge_values.round(10)

    @property
    def maxima_ref_values(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The reference data value at each maximum. If the maximum is
            off grid, this value will be interpolated.

        """
        if self._maxima_ref_values is None:
            # we get these values during each bader method anyways, so
            # we run this here.
            self._run_bader()
        return self._maxima_ref_values.round(10)

    @property
    def maxima_vox(self) -> NDArray[int]:
        """

        Returns
        -------
        NDArray[int]
            The voxel coordinates of each attractor.

        """
        if self._maxima_vox is None:
            self._run_bader()
        return self._maxima_vox
    
    @property
    def maxima_voxel_groups(self) -> NDArray[int]:
        """

        Returns
        -------
        NDArray[int]
            In many systems multiple nearby points will be found to be maxima
            usually due to voxelation. We combine these maxima into one with a
            persistence metric. This property provides all of the "false" maxima
            that are associated with the final maxima list.
            
            For scalar fields like the ELF, LOL, or ELI-D, there may also be
            ring and cage-like maxima that are not well described by a single
            point. This also provides some indication of these maxima.

        """
        if self._maxima_voxel_groups is None:
            self._run_bader()
        return self._maxima_voxel_groups
    
    @property
    def maxima_persistence_values(self) -> NDArray[int]:
        """

        Returns
        -------
        NDArray[int]
            Each maxima may have been combined with several voxelated maxima
            (see maxima_voxel_groups). For each maxima group, this  is the 
            lowest value at which all of the maxima in the group are topologically 
            connected if one takes the all voxels at or above that value
        """
        if self._maxima_persistence_values is None:
            self._run_bader()
        return self._maxima_persistence_values

    @property
    def basin_charges(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The charges assigned to each attractor.

        """
        if self._basin_charges is None:
            self._run_bader()
        return self._basin_charges.round(10)

    @property
    def basin_volumes(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The volume assigned to each attractor.

        """
        if self._basin_volumes is None:
            self._run_bader()
        return self._basin_volumes.round(10)

    @property
    def basin_min_surface_distances(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The distance from each basin maxima to the nearest point on
            the basins surface

        """
        if self._basin_min_surface_distances is None:
            self._get_basin_surface_distances()
        return self._basin_min_surface_distances.round(10)

    @property
    def basin_avg_surface_distances(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The avg distance from each basin maxima to the edges of its basin

        """
        if self._basin_avg_surface_distances is None:
            self._get_basin_surface_distances()
        return self._basin_avg_surface_distances.round(10)

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
        return self._basin_atom_dists.round(10)
    
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
            self._basin_edges = get_edges(
                labeled_array=self.maxima_basin_labels,
                vacuum_mask=np.zeros(self.maxima_basin_labels.shape, dtype=np.bool_),
                neighbor_transforms=self.reference_grid.point_neighbor_transforms[0],
            )
        return self._basin_edges
    
    @property
    def basin_contact_surface_areas(self) -> NDArray[np.float64]:
        """

        Returns
        -------
        NDArray[np.float64]
            A 2D array with indices i, j where i is the basin index, j is the neighboring
            basin index, and the entry at i, j is the total area in contact between
            these labels. One extra index is added that stores the number of connections
            to the vacuum.

            This value is calculated using voronoi cells of the voxels to
            approximate the shared area between a voxel point and a neighbor in
            another basin.

        """
        if self._basin_contact_surface_areas is None:
            neighbor_transforms, _, neighbor_areas, _ = (
                self.reference_grid.point_neighbor_voronoi_transforms
            )
            self._basin_contact_surface_areas = get_neighboring_basin_surface_area(
                labeled_array=self.maxima_basin_labels,
                neighbor_transforms=neighbor_transforms,
                neighbor_areas=neighbor_areas,
                vacuum_mask=self.vacuum_mask,
                label_num=len(self.maxima_frac),
            )
        return self._basin_contact_surface_areas
    
    @property
    def basin_surface_areas(self) -> NDArray[np.float64]:
        """

        Returns
        -------
        NDArray[np.float64]
            The approximate surface area of each basin.

            This value is calculated using voronoi cells of the voxels to
            approximate the shared area between a voxel point and a neighbor in
            another basin.

        """
        if self._basin_surface_areas is None:
            # get the contact surface area of each basin
            contact_surfaces = self.basin_contact_surface_areas
            # sum across axis 0 to get the total
            self._basin_surface_areas = np.sum(contact_surfaces, axis=1)
        return self._basin_surface_areas
    
    ###########################################################################
    # Minima Basin Properties
    ###########################################################################
    @property
    def minima_basin_labels(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The equivalent of bader basins for the desending gradient to local
            minima. This is each minima's ascending manifold and can be used
            in combination with the bader basins to locate important topological
            features.

        """
        if self._minima_basin_labels is None:
            self._run_minima_bader()
        return self._minima_basin_labels
    
    @property
    def minima_basin_images(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The equivalent of the basin_images property for minima basins.

        """
        if self._minima_basin_images is None:
            self._run_minima_bader()
        return self._minima_basin_images

    @property
    def minima_frac(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The fractional coordinates of each local minimum.

        """
        if self._minima_frac is None:
            self._run_minima_bader()
        return self._minima_frac
    
    @property
    def minima_vox(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The voxel coordinates of each minima.

        """
        if self._minima_vox is None:
            self._run_minima_bader()
        return self._minima_vox
    
    @property
    def minima_charge_values(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The charge data value at each maximum. If the maximum is
            off grid, this value will be interpolated.

        """
        # TODO: change this to quadratic fit to match reference value method
        if self._minima_charge_values is None:
            self._minima_charge_values = self.charge_grid.values_at(
                self.minima_frac
            )
        return self._minima_charge_values.round(10)
    
    @property
    def minima_ref_values(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The reference data value at each maximum. If the maximum is
            off grid, this value will be interpolated.

        """
        if self._minima_ref_values is None:
            # we get these values during each bader method anyways, so
            # we run this here.
            self._run_minima_bader()
        return self._minima_ref_values.round(10)
    
    @property
    def minima_voxel_groups(self) -> NDArray[int]:
        """

        Returns
        -------
        NDArray[int]
            In many systems multiple nearby points will be found to be minima
            usually due to voxelation. We combine these minima into one with a
            persistence metric. This property provides all of the "false" minima
            that are associated with the final minima list.
            
            For scalar fields like the ELF, LOL, or ELI-D, there may also be
            ring and cage-like minima that are not well described by a single
            point. This also provides some indication of these minima.

        """
        if self._minima_voxel_groups is None:
            self._run_minima_bader()
        return self._minima_voxel_groups
    
    @property
    def minima_persistence_values(self) -> NDArray[int]:
        """

        Returns
        -------
        NDArray[int]
            Each minima may have been combined with several voxelated minima
            (see minima_voxel_groups). For each minima group, this  is the 
            highest value at which all of the minima in the group are topologically 
            connected if one takes the all voxels at or below that value
        """
        if self._minima_persistence_values is None:
            self._run_minima_bader()
        return self._minima_persistence_values
    
    @property
    def minima_persistence_tol(self) -> float:
        """

        Returns
        -------
        float
            It is common for false maxima to be found using only nearest neighbor
            points. To deal with this we combine pairs of basins that have low
            topological persistence.
            
            The persistence score is calculated as:
                
                score = (connection_value - higher_minimum) / connection_value


        """
        return self._minima_persistence_tol
    
    @minima_persistence_tol.setter
    def minima_persistence_tol(self, value: str | Method):
        self._minima_persistence_tol = value
        # reset atom properties
        self._reset_properties(exclude_properties=[
            "vacuum_mask",
            "num_vacuum",
            "vacuum_charge",
            "vacuum_volume",
            "structure",
            ])

    ###########################################################################
    # Atom Properties
    ###########################################################################
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
        return self._atom_charges.round(10)

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
        return self._atom_volumes.round(10)

    @property
    def atom_min_surface_distances(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The distance from each atom to the nearest point on the atoms surface.

        """
        if self._atom_min_surface_distances is None:
            self._get_atom_surface_distances()
        return self._atom_min_surface_distances.round(10)

    @property
    def atom_avg_surface_distances(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The avg distance from each atom to the edges of its basin

        """
        if self._atom_avg_surface_distances is None:
            self._get_basin_surface_distances()
        return self._atom_avg_surface_distances.round(10)

    @property
    def atom_edges(self) -> NDArray[np.bool_]:
        """

        Returns
        -------
        NDArray[np.bool_]
            A mask with the same shape as the input grids that is True at points
            on atom edges.

        """
        if self._atom_edges is None:
            self._atom_edges = get_edges(
                labeled_array=self.atom_labels,
                vacuum_mask=np.zeros(self.atom_labels.shape, dtype=np.bool_),
                neighbor_transforms=self.reference_grid.point_neighbor_transforms[0],
            )
        return self._atom_edges

    @property
    def atom_contact_surface_areas(self) -> NDArray[np.float64]:
        """

        Returns
        -------
        NDArray[np.float64]
            A 2D array with indices i, j where i is the atom index, j is the neighboring
            atom index, and the entry at i, j is the total area in contact between
            these labels. One extra index is added that stores the number of connections
            to the vacuum.

            This value is calculated using voronoi cells of the voxels to
            approximate the shared area between a voxel point and a neighbor in
            another atom.

        """
        if self._atom_contact_surface_areas is None:
            neighbor_transforms, _, neighbor_areas, _ = (
                self.reference_grid.point_neighbor_voronoi_transforms
            )
            self._atom_contact_surface_areas = get_neighboring_basin_surface_area(
                labeled_array=self.atom_labels,
                neighbor_transforms=neighbor_transforms,
                neighbor_areas=neighbor_areas,
                vacuum_mask=self.vacuum_mask,
                label_num=len(self.structure),
            )
        return self._atom_contact_surface_areas

    @property
    def atom_surface_areas(self) -> NDArray[np.float64]:
        """

        Returns
        -------
        NDArray[np.float64]
            The approximate surface area of each atom.

            This value is calculated using voronoi cells of the voxels to
            approximate the shared area between a voxel point and a neighbor in
            another atom.

        """
        if self._atom_surface_areas is None:
            # get the contact surface area of each atom
            contact_surfaces = self.atom_contact_surface_areas
            # sum across axis 0 to get the total
            self._atom_surface_areas = np.sum(contact_surfaces, axis=1)
        return self._atom_surface_areas
    
    ###########################################################################
    # Other Properties
    ###########################################################################
    
    @property
    def total_electron_number(self) -> float:
        """

        Returns
        -------
        float
            The total number of electrons in the system calculated from the
            atom charges and vacuum charge. If this does not match the true
            total electron number within reasonable floating point error,
            there is a major problem.

        """

        return round(self.atom_charges.sum() + self.vacuum_charge, 10)

    @property
    def total_volume(self):
        """

        Returns
        -------
        float
            The total volume integrated in the system. This should match the
            volume of the structure. If it does not there may be a serious problem.

        """

        return round(self.atom_volumes.sum() + self.vacuum_volume, 10)
    
    ###########################################################################
    # Methods
    ###########################################################################

    @staticmethod
    def all_methods() -> list[str]:
        """

        Returns
        -------
        list[str]
            A list of the available methods.

        """

        return [i.value for i in Method]

    def _run_bader(self) -> None:
        """
        Runs the entire Bader process and saves results to class variables.

        """
        t0 = time.time()
        logging.info(f"Beginning Bader Algorithm Using '{self.method.name}' Method")
        # Normalize the method name to a module and class name
        module_name = self.method.replace(
            "-", "_"
        )  # 'pseudo-neargrid' -> 'pseudo_neargrid'
        class_name = (
            "".join(part.capitalize() for part in module_name.split("_")) + "Method"
        )

        # import method
        mod = importlib.import_module(f"baderkit.core.bader.methods.{module_name}")
        Method = getattr(mod, class_name)

        # Instantiate and run the selected method
        method = Method(
            charge_grid=self.charge_grid,
            reference_grid=self.reference_grid,
            vacuum_mask=self.vacuum_mask,
            num_vacuum=self.num_vacuum,
            persistence_tol=self.maxima_persistence_tol,
        )
        if self._use_overdetermined:
            method._use_overdetermined = True
        results = method.run()

        for key, value in results.items():
            if "extrema" in key:
                new_key = key.replace("extrema", "maxima")
                setattr(self, f"_{new_key}", value)
            else:
                setattr(self, f"_{key}", value)

        t1 = time.time()
        logging.info("Bader Algorithm Complete")
        logging.info(f"Time: {round(t1-t0,2)}")
        
    def _run_minima_bader(self) -> None:
        """
        Runs the entire Bader process and saves results to class variables.

        """
        t0 = time.time()
        logging.info(f"Beginning Minima Bader Algorithm Using '{self.method.name}' Method")
        # Normalize the method name to a module and class name
        module_name = self.method.replace(
            "-", "_"
        )  # 'pseudo-neargrid' -> 'pseudo_neargrid'
        class_name = (
            "".join(part.capitalize() for part in module_name.split("_")) + "Method"
        )

        # import method
        mod = importlib.import_module(f"baderkit.core.bader.methods.{module_name}")
        Method = getattr(mod, class_name)

        # Instantiate and run the selected method
        method = Method(
            charge_grid=self.charge_grid,
            reference_grid=self.reference_grid,
            vacuum_mask=self.vacuum_mask,
            num_vacuum=self.num_vacuum,
            use_minima=True,
            persistence_tol=self.minima_persistence_tol,
        )
        if self._use_overdetermined:
            method._use_overdetermined = True
        results = method.run()
        # set related properties
        for key, value in results.items():
            if "extrema" in key:
                new_key = key.replace("extrema", "minima")
                setattr(self, f"_{new_key}", value)
        
        t1 = time.time()
        logging.info("Bader Algorithm Complete")
        logging.info(f"Time: {round(t1-t0,2)}")

    def assign_basins_to_structure(self, structure: Structure, nna_cutoff: bool | float = False):
        """
        Gets atom assignments for the provided structure.

        Parameters
        ----------
        structure : Structure
            The structure to assign basins to.
        nna_cutoff : bool | float, optional
            A distance cutoff. Any basins with maxima further from an atom than
            this value (in Angstroms) will be counted as its own species. Dummy
            atoms will be added to the structure. The default is False.

        Returns
        -------
        atom_labels : NDArray
            A 3D array with the same shape as the grid where each entry represents
            the atom that point belongs to.
        atom_charges : NDArray
            The charge assigned to each atom.
        atom_volumes : NDArray
            The volume assigned to each atom.
        basin_atoms : NDArray
            The atom each basin was assigned to.
        basin_atom_dists : NDArray
            The distance from each basin to the nearest atom.

        """
        if nna_cutoff is True:
            nna_cutoff = 1.0

        # Get basin and atom frac coords
        basins = self.maxima_frac  # (N_basins, 3)
        atoms = structure.frac_coords  # (N_atoms, 3)

        # get lattice matrix and number of atoms/basins
        L = structure.lattice.matrix  # (3, 3)
        N_basins = len(basins)
        def get_atom_basins(atoms, basins):
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
            return basin_atoms, basin_atom_dists
        basin_atoms, basin_atom_dists = get_atom_basins(atoms, basins)
        
        if nna_cutoff:
            # add dummy atoms at basin maxima far from atoms
            for coords, dist in zip(basins, basin_atom_dists):
                if dist > nna_cutoff:
                    # add a dummy atom to the structure
                    structure.append("X", coords)
            # recalculate distances
            atoms = structure.frac_coords
            basin_atoms, basin_atom_dists = get_atom_basins(atoms, basins)
        
        # vacuum is represented by an index one above the highest label. we add
        # that to our basin atoms temporarily
        basin_atoms = np.insert(basin_atoms, len(basin_atoms), len(structure))

        # Atom labels per grid point
        atom_labels = basin_atoms[self.maxima_basin_labels]
        
        # remove vacuum pointer in basin atoms
        basin_atoms = basin_atoms[:-1]

        atom_charges = np.bincount(
            basin_atoms, weights=self.basin_charges, minlength=len(structure)
        )
        atom_volumes = np.bincount(
            basin_atoms, weights=self.basin_volumes, minlength=len(structure)
        )

        return atom_labels, atom_charges, atom_volumes, basin_atoms, basin_atom_dists

    def run_atom_assignment(self):
        """
        Assigns bader basins to this Bader objects structure.

        """
        # ensure bader has run (otherwise our time will include the bader time)
        self.maxima_frac

        # Default structure
        structure = self.structure

        t0 = time.time()
        logging.info("Assigning Atom Properties")
        # get basin assignments for this bader objects structure
        atom_labels, atom_charges, atom_volumes, basin_atoms, basin_atom_dists = (
            self.assign_basins_to_structure(structure, self.nna_cutoff)
        )

        # Store everything
        self._basin_atoms = basin_atoms
        self._basin_atom_dists = basin_atom_dists
        self._atom_labels = atom_labels
        self._atom_charges = atom_charges
        self._atom_volumes = atom_volumes
        logging.info("Atom Assignment Finished")
        t1 = time.time()
        logging.info(f"Time: {round(t1-t0, 2)}")

    def get_oxidation_from_potcar(self, potcar_path: Path = "POTCAR"):
        """
        Calculates the oxidation state of each atom from the provided POTCAR
        file.

        Parameters
        ----------
        potcar_path : Path, optional
            The path to the POTCAR to calculate oxidation states from. The default is "POTCAR".

        Returns
        -------
        NDArray
            The oxidation state of each atom in the structure.

        """
        # convert to path
        potcar_path = Path(potcar_path)
        if not potcar_path.exists():
            logging.warning(
                "No POTCAR file found in the requested directory. Oxidation states cannot be calculated"
            )
            return
        # load
        with warnings.catch_warnings(record=True):
            potcars = Potcar.from_file(potcar_path)
        nelectron_data = {}
        # the result is a list because there can be multiple element potcars
        # in the file (e.g. for NaCl, POTCAR = POTCAR_Na + POTCAR_Cl)
        for potcar in potcars:
            nelectron_data.update({potcar.element: potcar.nelectrons})
        # calculate oxidation states
        oxi_state_data = []
        for site, site_charge in zip(self.structure, self.atom_charges):
            element_str = site.specie.name
            oxi_state = nelectron_data[element_str] - site_charge
            oxi_state_data.append(oxi_state)

        return np.array(oxi_state_data)
    
    def get_persistence_groups(self):
        """
        Gets the groups of voxels for each maximum and minimum that are within
        the extrema's basin and above/below the persistence value for that basin.
        The persistence value is defined as the value at which all voxelated
        maxima/minima (see maxima_voxel_groups/minima_voxel_groups) are connected

        Returns
        -------
        maxima_groups : list[NDArray[int64]]
            A list of Nx3 arrays where each array represents the grid indices of voxels 
            that are within each maximas persistence threshold.
        minima_groups : list[NDArray[int64]]
            A list of Nx3 arrays where each array represents the grid indices of voxels 
            that are within each minimas persistence threshold.

        """

        maxima_groups = get_persistence_groups(
            labels=self.maxima_basin_labels,
            data=self.reference_grid.total,
            persistence_cutoffs=self.maxima_persistence_values,
            extrema_vox=self.maxima_vox,
            use_minima=False,
            )

        minima_groups = get_persistence_groups(
            labels=self.minima_basin_labels,
            data=self.reference_grid.total,
            persistence_cutoffs=self.minima_persistence_values,
            extrema_vox=self.minima_vox,
            use_minima=True,
            )
        return maxima_groups, minima_groups
        
        

    def _get_atom_surface_distances(self):
        """
        Calculates the distance from each atom to the nearest surface. This is
        automatically called during the atom assignment and generally should
        not be called manually.

        """
        self._atom_min_surface_distances, self._atom_avg_surface_distances = (
            get_min_avg_surface_dists(
                labels=self.atom_labels,
                frac_coords=self.structure.frac_coords,
                edge_mask=self.atom_edges,
                matrix=self.reference_grid.matrix,
                max_value=np.max(self.structure.lattice.abc) * 2,
            )
        )

    def _get_basin_surface_distances(self):
        """
        Calculates the distance from each basin maxima to the nearest surface.
        This is automatically called during the atom assignment and generally
        should not be called manually.

        """
        # get the minimum distances
        self._basin_min_surface_distances, self._basin_avg_surface_distances = (
            get_min_avg_surface_dists(
                labels=self.maxima_basin_labels,
                frac_coords=self.maxima_frac,
                edge_mask=self.basin_edges,
                matrix=self.reference_grid.matrix,
                max_value=np.max(self.structure.lattice.abc) * 2,
            )
        )

    ###########################################################################
    # Write Methods
    ###########################################################################

    def write_basin_volumes(
        self,
        basin_indices: NDArray[int],
        **kwargs,
    ):
        """
        Writes bader basins to vasp-like files. Points belonging to the basin
        will have values from the charge or reference grid, and all other points
        will be 0.

        Parameters
        ----------
        basin_indices : NDArray
            The list of basin indices to write

        """
        for basin in basin_indices:
            # get a mask everywhere but the requested basin
            mask = self.maxima_basin_labels == basin
            kwargs["suffix"] = f"_b{basin}"

            self._write_volume(volume_mask=mask, **kwargs)

    def write_all_basin_volumes(
        self,
        basin_tol: float = 1e-03,
        **kwargs,
    ):
        """
        Writes all bader basins to vasp-like files. Points belonging to the basin
        will have values from the charge or reference grid, and all other points
        will be 0.

        Parameters
        ----------
        basin_tol : float, optional
            The total charge value below which a basin will not be considered written

        """
        basin_indices = np.where(self.basin_charges > basin_tol)[0]
        self.write_basin_volumes(
            basin_indices=basin_indices,
            **kwargs,
        )

    def write_basin_volumes_sum(
        self,
        basin_indices: NDArray[int],
        **kwargs,
    ):
        """
        Writes the union of the provided bader basins to vasp-like files.
        Points belonging to the basins will have values from the charge or
        reference grid, and all other points will be 0.

        Parameters
        ----------
        basin_indices : NDArray
            The list of basin indices to sum and write

        """
        # create a mask including each of the requested basins
        mask = np.isin(self.maxima_basin_labels, basin_indices)
        # write
        kwargs["suffix"] = "_bsum"
        self._write_volume(volume_mask=mask, **kwargs)

    def write_atom_volumes(
        self,
        atom_indices: NDArray,
        **kwargs,
    ):
        """
        Writes atomic basins to vasp-like files. Points belonging to the atom
        will have values from the charge or reference grid, and all other points
        will be 0.

        Parameters
        ----------
        atom_indices : NDArray
            The list of atom indices to write

        """

        for atom_index in atom_indices:
            # get a mask at the requested atoms
            mask = self.atom_labels == atom_index
            kwargs["suffix"] = "_a{atom_index}"
            self._write_volume(volume_mask=mask, **kwargs)

    def write_all_atom_volumes(
        self,
        **kwargs,
    ):
        """
        Writes all atomic basins to vasp-like files. Points belonging to the atom
        will have values from the charge or reference grid, and all other points
        will be 0.

        """
        atom_indices = np.array(range(len(self.structure)))
        self.write_atom_volumes(
            atom_indices=atom_indices,
            **kwargs,
        )

    def write_atom_volumes_sum(
        self,
        atom_indices: NDArray,
        **kwargs,
    ):
        """
        Writes the union of the provided atom basins to vasp-like files.
        Points belonging to the atoms will have values from the charge or
        reference grid, and all other points will be 0.

        Parameters
        ----------
        atom_indices : NDArray
            The list of atom indices to sum and write

        """

        mask = np.isin(self.atom_labels, atom_indices)
        # write
        kwargs["suffix"] = "_asum"
        self._write_volume(volume_mask=mask, **kwargs)

    def write_species_volume(
        self,
        species: str,
        **kwargs,
    ):
        """
        Writes the charge density or reference file for all atoms of the given
        species to a single file.

        Parameters
        ----------
        species : str, optional
            The species to write.

        """

        # add dummy atoms if desired
        indices = self.structure.indices_from_symbol(species)

        # Get mask where the grid belongs to requested species
        mask = np.isin(self.atom_labels, indices)
        kwargs["suffix"] = "_{species}"
        self._write_volume(volume_mask=mask, **kwargs)

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
                "surface_dist": self.atom_min_surface_distances,
            }
        )
        return atoms_df

    def get_basin_results_dataframe(self, basin_tol: float):
        """
        Collects a summary of results for the basins in a pandas DataFrame.

        Returns
        -------
        basin_df : pd.DataFrame
            A table summarizing the basins.
        basin_tol : float, optional
            The total charge value below which a basin will not be considered significant.

        """
        subset = self.basin_charges > basin_tol
        basin_frac_coords = self.maxima_frac[subset]
        basin_df = pd.DataFrame(
            {
                "atoms": np.array(self.structure.labels)[self.basin_atoms[subset]],
                "x": basin_frac_coords[:, 0],
                "y": basin_frac_coords[:, 1],
                "z": basin_frac_coords[:, 2],
                "charge": self.basin_charges[subset],
                "volume": self.basin_volumes[subset],
                "surface_dist": self.basin_min_surface_distances[subset],
                "atom_dist": self.basin_atom_dists[subset],
            }
        )
        return basin_df

    def write_atom_tsv(self, filepath: Path | str = "bader_atoms.tsv"):
        """
        Writes a summary of atom results to .tsv files.

        Parameters
        ----------
        filepath : str | Path
            The Path to write the results to. The default is "bader_atoms.tsv".


        """
        filepath = Path(filepath)

        # Get atom results summary
        atoms_df = self.get_atom_results_dataframe()
        formatted_atoms_df = atoms_df.copy()
        numeric_cols = formatted_atoms_df.select_dtypes(include="number").columns
        formatted_atoms_df[numeric_cols] = formatted_atoms_df[numeric_cols].map(
            lambda x: f"{x:.5f}"
        )

        # Determine max width per column including header
        col_widths = {
            col: max(len(col), formatted_atoms_df[col].map(len).max())
            for col in atoms_df.columns
        }

        # Note what we're writing in log
        logging.info(f"Writing Atom Summary to {filepath}")

        # write output summaries
        with open(filepath, "w") as f:
            # Write header
            header = "\t".join(
                f"{col:<{col_widths[col]}}" for col in formatted_atoms_df.columns
            )
            f.write(header + "\n")

            # Write rows
            for _, row in formatted_atoms_df.iterrows():
                line = "\t".join(
                    f"{val:<{col_widths[col]}}" for col, val in row.items()
                )
                f.write(line + "\n")
            # write vacuum summary to atom file
            f.write("\n")
            f.write(f"Vacuum Charge:\t\t{self.vacuum_charge:.5f}\n")
            f.write(f"Vacuum Volume:\t\t{self.vacuum_volume:.5f}\n")
            f.write(f"Total Electrons:\t{self.total_electron_number:.5f}\n")
            f.write(f"Total Volume:\t{self.total_volume:.5f}\n")

    def write_basin_tsv(self, filepath: Path | str = "bader_basins.tsv"):
        """
        Writes a summary of basin results to .tsv files.

        Parameters
        ----------
        filepath : str | Path
            The Path to write the results to. The default is "bader_basins.tsv".

        """
        filepath = Path(filepath)

        # Get basin results summary
        basin_df = self.get_basin_results_dataframe()
        formatted_basin_df = basin_df.copy()
        numeric_cols = formatted_basin_df.select_dtypes(include="number").columns
        formatted_basin_df[numeric_cols] = formatted_basin_df[numeric_cols].map(
            lambda x: f"{x:.5f}"
        )

        # Determine max width per column including header
        col_widths = {
            col: max(len(col), formatted_basin_df[col].map(len).max())
            for col in basin_df.columns
        }

        # Write to file with aligned columns using tab as separator

        # Note what we're writing in log

        logging.info(f"Writing Basin Summary to {filepath}")

        # write output summaries
        with open(filepath, "w") as f:
            # Write header
            header = "\t".join(f"{col:<{col_widths[col]}}" for col in basin_df.columns)
            f.write(header + "\n")

            # Write rows
            for _, row in formatted_basin_df.iterrows():
                line = "\t".join(
                    f"{val:<{col_widths[col]}}" for col, val in row.items()
                )
                f.write(line + "\n")

    def to_dict(
        self,
        potcar_path: Path | str = "POTCAR",
        use_json: bool = True,
    ) -> dict:
        """

        Gets a dictionary summary of the Bader analysis.

        Parameters
        ----------
        potcar_path : Path | str, optional
            The Path to a POTCAR file. This must be provided for oxidation states
            to be calculated, and they will be None otherwise. The default is "POTCAR".
        use_json : bool, optional
            Convert all entries to JSONable data types. The default is True.

        Returns
        -------
        dict
            A summary of the BadELF analysis in dictionary form.

        """
        # get charges first to ensure good logging
        self.atom_charges
        results = {}
        # collect method kwargs
        method_kwargs = {
            "method": self.method,
            "vacuum_tol": self.vacuum_tol,
        }
        results["method_kwargs"] = method_kwargs
        results["oxidation_states"] = self.get_oxidation_from_potcar(potcar_path)

        # split into basin and atom sections
        atom_results = {}
        for result in [
            "species",
            "atom_charges",
            "atom_volumes",
            "atom_min_surface_distances",
            "atom_avg_surface_distances",
        ]:
            atom_results[result] = getattr(self, result, None)

        basin_results = {}
        for result in [
            "basin_atoms",
            "basin_atom_dists",
            "basin_charges",
            "basin_volumes",
            "maxima_frac",
            "maxima_charge_values",
            "maxima_ref_values",
            "maxima_vox",
            "maxima_voxel_groups",
            "minima_frac",
            "minima_charge_values",
            "minima_ref_values",
            "minima_vox",
            "minima_voxel_groups",
            "basin_min_surface_distances",
            "basin_avg_surface_distances",
        ]:
            basin_results[result] = getattr(self, result, None)

        for result in [
            "structure",
            "vacuum_charge",
            "vacuum_volume",
            "num_vacuum",
            "total_electron_number",
            "total_volume",
        ]:
            results[result] = getattr(self, result, None)

        if use_json:
            # get serializable structure
            results["structure"] = results["structure"].to(fmt="POSCAR")

            # get serializable versions of each basin attribute
            for key in [
                "maxima_frac",
                "maxima_charge_values",
                "maxima_ref_values",
                "maxima_vox",
                "minima_frac",
                "minima_charge_values",
                "minima_ref_values",
                "minima_vox",
                "basin_charges",
                "basin_volumes",
                "basin_min_surface_distances",
                "basin_avg_surface_distances",
                "basin_atoms",
                "basin_atom_dists",
            ]:
                if basin_results[key] is None:
                    continue  # skip oxidation states if they fail
                basin_results[key] = basin_results[key].tolist()
            
            for key in [
                "maxima_voxel_groups",
                "minima_voxel_groups",
                    ]:
                basin_results[key] = [i.tolist() for i in basin_results[key]]

            # get serializable versions of each atom attribute
            for key in [
                "atom_charges",
                "atom_volumes",
                "atom_min_surface_distances",
                "atom_avg_surface_distances",
            ]:
                if atom_results[key] is None:
                    continue  # skip oxidation states if they fail
                atom_results[key] = atom_results[key].tolist()

            # get serializable oxidation states
            if results["oxidation_states"] is not None:
                results["oxidation_states"] = results["oxidation_states"].tolist()

        results["atom_results"] = atom_results
        results["basin_results"] = basin_results

        return results
