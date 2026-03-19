# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from baderkit.core.bader.bader import Bader
from baderkit.core.bader.methods import Method
from baderkit.core.base.base_analysis import BaseAnalysis
from baderkit.core.toolkit import Grid
from .overlap_numba import get_overlaps


class BasinOverlap(BaseAnalysis):
    """
    A convenience class for calculating the overlap between basins calculated
    in the charge density and a localization density such as ELF.

    """
    
    _reset_props = [
        "atomicities",
        "overlap_counts",
        "overlap_labels",
        "local_overlap_fractions",
        "qtaim_overlap_fractions",
        "shared_local_basins",
        "unshared_local_basins",
        "polarization_indexes",
    ]

    def __init__(
        self,
        charge_grid: Grid,
        reference_grid: Grid,
        total_charge_grid: Grid | None = None,
        nna_cutoff: float = 1.0,
        method: str | Method = Method.neargrid,
        **kwargs,
    ):
        
        # create bader objects
        self.qtaim = Bader(
            charge_grid=charge_grid,
            total_charge_grid=total_charge_grid,
            reference_grid=total_charge_grid,
            nna_cutoff=nna_cutoff,
            method=method,
            **kwargs,
        )
        
        self.local_bader = Bader(
            charge_grid=charge_grid,
            total_charge_grid=total_charge_grid,
            reference_grid=reference_grid,
            method=method,
            **kwargs,
        )

        super().__init__(
            charge_grid=charge_grid,
            total_charge_grid=total_charge_grid,
            reference_grid=reference_grid,
            **kwargs,
        )

    ###########################################################################
    # Properties calculated by other classes
    ###########################################################################

            
    @property
    def vacuum_mask(self) -> NDArray[bool]:
        """

        Returns
        -------
        NDArray[bool]
            A mask representing the voxels that belong to the vacuum.

        """

        return self.qtaim.vacuum_mask

    @property
    def local_basin_labels(self) -> NDArray[np.int64]:
        """

        Returns
        -------
        NDArray[np.int64]
            A 3D grid representing the points in space assigned to local basins
            such as those calculated from the ELF, ELI-D, LOL, etc.

        """
        return self.local_bader.maxima_basin_labels
    
    @property
    def qtaim_basin_labels(self) -> NDArray[np.int64]:
        """

        Returns
        -------
        NDArray[np.int64]
            A 3D grid representing the points in space assigned to Bader atoms.

        """
        return self.qtaim.atom_labels
    
    @property
    def local_maxima_frac(self) -> NDArray[np.float64]:
        """

        Returns
        -------
        NDArray[np.float64]
            The fractional coordinates of each maximum in the localization function
            grid. The order corresponds to the basin labels.

        """
        return self.local_bader.maxima_frac
    
    @property
    def qtaim_maxima_frac(self) -> NDArray[np.float64]:
        """

        Returns
        -------
        NDArray[np.float64]
            The fractional coordinates of each maximum in the charge density
            grid. The order corresponds to the atom labels and structure.

        """
        return self.qtaim.maxima_frac
    
    ###########################################################################
    # Properties related to overlap
    ###########################################################################

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
            self._atomicities = np.array([len(i) for i in self.local_overlap_fractions])
        return self._atomicities
    
    @property
    def shared_local_basins(self) -> NDArray[bool]:
        if self._shared_local_basins is None:
            self._shared_local_basins = np.array([True if j>1 else False for j in self.atomicities])
        return self._shared_local_basins
            
    @property
    def overlap_counts(self) -> NDArray[np.int64]:
        """

        Returns
        -------
        NDArray[np.int64]
            An Nx3 array where the columns represent the qtaim basin, local basin,
            and number of overlapping grid points respectively.

        """
        if self._overlap_counts is None:
            self._get_overlap()
        return self._overlap_counts

    @property
    def local_overlap_fractions(self) -> list[NDArray[np.float64]]:
        """

        Returns
        -------
        list[NDArray[np.float64]]
            A list with the same length as the number of local basins. Each entry
            is an Nx2 array where N is the number of overlapping qtaim basins
            and each entry is the qtaim basin index and fraction of overlap
            respectively

        """
        if self._local_overlap_fractions is None:
            self._get_overlap()
        return self._local_overlap_fractions
    
    @property
    def qtaim_overlap_fractions(self) -> list[NDArray[np.float64]]:
        """

        Returns
        -------
        list[NDArray[np.float64]]
            A list with the same length as the number of qtaim basins. Each entry
            is an Nx2 array where N is the number of overlapping local basins
            and each entry is the local basin index and fraction of overlap
            respectively

        """
        if self._qtaim_overlap_fractions is None:
            self._get_overlap()
        return self._qtaim_overlap_fractions
    
    @property
    def polarization_indexes(self) -> NDArray[np.float64]:
        """
        Measure of polarization of a localized basin as defined here:
        https://pubs.acs.org/doi/10.1021/acs.inorgchem.5b00135

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if self._polarization_indexes is None:
            polarization_indexes = []
            for atom_fracs in self.local_overlap_fractions:
                fracs = atom_fracs[:,1]
                
        return self._polarization_indexes
    
    # TODO: 
        # move core basin finding to this class
        # get atoms with access to each basin
        # get atom access sets
        # access electron number and valence electron number
        # charge claim/average bond fraction of each atom
        # connection index (atomic?)
        # nearest neighbor sharing?
        # figure out bond polarity metric that works for multi-centered bonds

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


    def _get_overlap(self):
        (
        self._overlap_counts, 
        self._qtaim_overlap_fractions, 
        self._local_overlap_fractions, 
        self._overlap_labels,
        ) = get_overlaps(
            atom_labels=self.qtaim_basin_labels,  # Bader Atoms
            local_labels=self.local_basin_labels,  # ELF basins
            num_charge=len(self.qtaim_maxima_frac),
            num_local=len(self.local_maxima_frac),
        )
            
    def to_dict(self):
        pass
    
    @classmethod
    def from_vasp(
        cls,
        charge_filename: Path | str = "CHGCAR",
        reference_filename: Path | str = "ELFCAR",
        **kwargs,
    ):
        """
        Creates a Bader class object from VASP files.

        Parameters
        ----------
        charge_filename : Path | str, optional
            The path to the CHGCAR like file that will be used for integrating charge.
            The default is "CHGCAR".
        reference_filename : Path |  str
            The path to ELFCAR like file that will be used for partitioning.
        total_charge_filename : Grid | None, optional
            The path to the CHGCAR like file used for determining vacuum regions
            in the system. For pseudopotential codes this represents the total
            electron density and should be provided whenever possible.
            If None, defaults to the charge_grid.
        total_only: bool
            If true, only the first set of data in each file will be read. This
            increases speed and reduced memory usage as the other data is typically
            not used.
            Defaults to True.
        **kwargs : dict
            Keyword arguments to pass to the class.

        Returns
        -------
        Self
            A BaseAnalysis class object.

        """
        return super().from_vasp(
            charge_filename=charge_filename,
            reference_filename=reference_filename,
            **kwargs
            )

