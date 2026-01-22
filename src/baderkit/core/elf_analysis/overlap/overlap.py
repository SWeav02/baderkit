# -*- coding: utf-8 -*-

import numpy as np
from numpy.typing import NDArray

from baderkit.core.toolkit import Structure
from .overlap_numba import get_overlap_counts

class BasinOverlap:
    """
    A convenience class for calculating the overlap between basins calculated
    in the charge density and a localization density such as ELF.
    
    Vacuum points should be represented by -1.
    """

    def __init__(
        self,
        atom_structure: Structure,
        local_structure: Structure,
        atom_labels: NDArray[np.int64],
        local_labels: NDArray[np.int64],
    ):

        self._atom_structure = atom_structure
        self._local_structure = local_structure
        self._atom_labels = atom_labels
        self._local_labels = local_labels
        
        assert np.all(np.equal(atom_labels.shape, local_labels.shape)), "Label arrays must have the same grid shape."
        
        self._vacuum_mask = (atom_labels == -1) | (local_labels == -1)
        self._num_local_basins = np.ptp(local_labels[~self.vacuum_mask])+1
        self._num_atom_basins = np.ptp(atom_labels[~self.vacuum_mask])+1
        
        self._reset_properties()


    ###########################################################################
    # Set Properties
    ###########################################################################
    def _reset_properties(
        self,
        include_properties: list[str] = None,
        exclude_properties: list[str] = [],
    ):
        # if include properties is not provided, we wnat to reset everything
        if include_properties is None:
            include_properties = [
                "atomicities",
                "overlap_atoms",
                "overlap_fractions",
                "overlap_matrix",
                "overlap_labels",
            ]
        # get our final list of properties
        reset_properties = [
            i for i in include_properties if i not in exclude_properties
        ]
        # set corresponding hidden variable to None
        for prop in reset_properties:
            setattr(self, f"_{prop}", None)
            
    @property
    def local_structure(self) -> Structure:
        """

        Returns
        -------
        Structure
            A PyMatGen structure object with dummy atoms representing attractors
            in the localization field.

        """
        return self._local_structure
    
    @property
    def atom_structure(self) -> Structure:
        """

        Returns
        -------
        Structure
            A PyMatGen structure object representing the atoms in the material.

        """
        return self._atom_structure
            
    @property
    def local_labels(self) -> NDArray[np.int64]:
        """

        Returns
        -------
        NDArray[np.int64]
            A 3D grid representing the points in space assigned to local basins
            such as those calculated from the ELF, ELI-D, LOL, etc.

        """
        return self._local_labels
            
    @property
    def atom_labels(self) -> NDArray[np.int64]:
        """

        Returns
        -------
        NDArray[np.int64]
            A 3D grid representing the points in space assigned to Bader atomic
            basins.

        """
        return self._atom_labels
            
    @property
    def vacuum_mask(self) -> NDArray[np.bool_]:
        """

        Returns
        -------
        NDArray[np.bool_]
            A 3D boolean array representing the points assigned to vacuum.

        """
        return self._vacuum_mask
    
    @property
    def num_local_basins(self) -> int:
        """

        Returns
        -------
        int
            The total number of local basins in the labeled localization grid

        """
        
        return self._num_local_basins
    
    @property
    def num_atom_basins(self) -> int:
        """

        Returns
        -------
        int
            The total number of atoms in the labeled atom grid

        """
        
        return self._num_atom_basins
    
    @property
    def atomicities(self) -> NDArray[np.int64]:
        """

        Returns
        -------
        NDArray[np.int64]
            The number of bader atoms that have at least some overlap with each
            local basin. Some of these may have very little overlap. For example,
            lone-pairs in solids often have a small contribution from neighboring
            atoms due to the periodicity requirement.

        """
        if self._atomicities is None:
            # The number of atoms contributing to each label is the number of
            # non-zero entries in each row of our overlap_matrix
            self._atomicities = np.count_nonzero(self.overlap_matrix, axis=1)
        return self._atomicities
    
    @property
    def overlap_atoms(self) -> list[NDArray[np.int64]]:
        """

        Returns
        -------
        list[NDArray[np.int64]]
            For each basin in the local Bader object, this returns the indices
            of the overlapping Bader atoms.

        """
        if self._overlap_atoms is None:
            self._get_overlap_portions()
        return self._overlap_atoms
    
    @property
    def overlap_fractions(self) -> NDArray[np.float64]:
        """

        Returns
        -------
        list[NDArray[np.int64]]
            For each basin in the local Bader object, this returns the fraction
            of overlap with overlapping Bader atoms.

        """
        if self._overlap_fractions is None:
            self._get_overlap_portions()
        return self._overlap_fractions

    
    @property
    def overlap_labels(self) -> NDArray[np.int64]:
        """

        Returns
        -------
        NDArray[np.int64]
            A 3D array with the same shape as the Bader grids representing the
            basin intersections. The entry number is the result of the szudzik 
            pairing function for indices i and j where i is the local basin index
            and j is the bader atom index.

        """
        if self._overlap_labels is None:
            self._get_overlap()
        return self._overlap_labels
    
    @property
    def overlap_matrix(self) -> NDArray[np.int64]:
        """

        Returns
        -------
        NDArray[np.int64]
            A 2D array with entries i, j where i is the local basin index, j is
            the bader atom index, and the entry is the number of voxels that overlap
            in these basins.

        """
        if self._overlap_matrix is None:
            self._get_overlap()
        return self._overlap_matrix
    
    def _get_overlap(self):
        self._overlap_matrix, self._overlap_labels = get_overlap_counts(
            local_labels=self.local_labels, # ELF basins
            atom_labels=self.atom_labels, # Bader Atoms
            vacuum_mask=self.vacuum_mask,
            num_local=self.num_local_basins,
            num_charge=self.num_atom_basins
            )
            
    def _get_overlap_portions(self):
        # get overlap matrix and normalize rows
        overlap_matrix = self.overlap_matrix / self.overlap_matrix.sum(axis=1, keepdims=True)
        
        # get the indices of the non-zero entries in our overlap matrix
        rows, cols = np.nonzero(overlap_matrix)
        # get the overlap counts at these points
        overlap_counts = overlap_matrix[rows, cols]
        # bin unique rows with overlap
        row_bins = np.bincount(rows)
        # get atom indices and overlap counts
        self._overlap_atoms = np.split(cols, np.cumsum(row_bins)[:-1])
        self._overlap_fractions = np.split(overlap_counts, np.cumsum(row_bins)[:-1])
            
    