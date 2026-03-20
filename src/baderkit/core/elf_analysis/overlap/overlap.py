# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from baderkit.core.bader.bader import Bader
from baderkit.core.base.base_analysis import BaseAnalysis
from baderkit.core.toolkit import Grid
from .overlap_numba import get_overlaps
from baderkit.core.utilities.coord_env import is_along_bond_all


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
        "core_basins",
        "shared_basins",
        "along_bond",
        "attractor_shapes",
        "polarization_indexes",
    ]

    def __init__(
        self,
        charge_grid: Grid,
        reference_grid: Grid,
        total_charge_grid: Grid | None = None,
        nna_cutoff: float = 1.0,
        min_covalent_angle: float = 135,
        polarization_cutoff: float = 0.5,
        **kwargs,
    ):
        self._polarization_cutoff = polarization_cutoff
        self._min_covalent_angle = min_covalent_angle
        
        # create bader objects
        self.qtaim_bader = Bader(
            charge_grid=charge_grid,
            total_charge_grid=total_charge_grid,
            reference_grid=total_charge_grid,
            nna_cutoff=nna_cutoff,
            **kwargs,
        )
        
        self.local_bader = Bader(
            charge_grid=charge_grid,
            total_charge_grid=total_charge_grid,
            reference_grid=reference_grid,
            **kwargs,
        )

        super().__init__(
            charge_grid=charge_grid,
            total_charge_grid=total_charge_grid,
            reference_grid=reference_grid,
            **kwargs,
        )

    ###########################################################################
    # Settings
    ###########################################################################
    @property
    def min_covalent_angle(self) -> float:
        return self._min_covalent_angle
    
    @min_covalent_angle.setter
    def min_covalent_angle(self, value: float):
        self._min_covalent_angle = value
        self._reset_properties(
            include_properties=[
                "core_basins",
                "along_bond",
                ], 
            )
        
    @property
    def polarization_cutoff(self) -> float:
        return self._polarization_cutoff
    
    @polarization_cutoff.setter
    def polarization_cutoff(self, value: float):
        self._polarization_cutoff = value
        self._reset_properties(
            include_properties=[
                "valence_basins",
                ], 
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

        return self.qtaim_bader.vacuum_mask

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
        return self.qtaim_bader.maxima_frac
    
    @property
    def along_bond(self):
        if self._along_bond is None:
            self._along_bond, _ = is_along_bond_all(
                feature_frac_coords=self.local_maxima_frac,
                atom_frac_coords=self.structure.frac_coords,
                atom_cart_coords=self.structure.cart_coords,
                matrix=self.reference_grid.matrix,
                min_covalent_angle=self.min_covalent_angle,
            )
        return self._along_bond
    
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
            and each entry is the qtaim basin index and fraction of charge in the
            overlapping basin respectively. For valence basins, this is equivalent 
            to the fraction of the bond belonging to each overlapping atom.

        """
        if self._local_overlap_fractions is None:
            self._get_overlap()
        return self._local_overlap_fractions
    
    @property
    def core_basins(self) -> NDArray[int]:
        """

        Returns
        -------
        NDArray[int]
            An array with an entry for each local basin corresponding to the
            index of the atom this basin is a core of. If the basin is not a
            core, the value is -1.

        """
        
        if self._core_basins is None:
            self._assign_cores()
        return self._core_basins
    
    @property
    def shared_basins(self) -> NDArray[bool]:
        """

        Returns
        -------
        NDArray[bool]
            An array with an entry for each local basin that is True for basins
            that are shared between multiple atoms. This is basically all basins
            that are not part of the atom core or lone-pairs

        """
        
        if self._shared_basins is None:
            self._assign_cores()
        return self._shared_basins
    
    # @property
    # def qtaim_overlap_fractions(self) -> list[NDArray[np.float64]]:
    #     """

    #     Returns
    #     -------
    #     list[NDArray[np.float64]]
    #         A list with the same length as the number of qtaim basins. Each entry
    #         is an Nx2 array where N is the number of overlapping local basins
    #         and each entry is the local basin index and fraction of overlap
    #         respectively

    #     """
    #     if self._qtaim_overlap_fractions is None:
    #         self._get_overlap()
    #     return self._qtaim_overlap_fractions
    
    @property
    def attractor_shapes(self) -> NDArray[str]:
        if self._attractor_shapes is None:
            betti_nums = self.local_bader.maxima_betti_numbers
            shapes = []
            for i,j,k in betti_nums:
                if i == 1 and j == 0 and k == 0:
                    shapes.append("point")
                elif i == 1 and j == 1 and k == 0:
                    shapes.append("ring")
                elif i == 1 and j == 0 and k == 1:
                    shapes.append("cage")
                else:
                    raise Exception("Unknown shape found for attractor. This is a bug!")
            self._attractor_shapes = np.array(shapes)
        return self._attractor_shapes
    
    @property
    def valence_basins(self) -> NDArray[bool]:
        if self._valence_basins is None:
            pass
        return self._valence_basins
    
    @property
    def polarization_indexes(self) -> NDArray[np.float64]:
        """
        Measure of polarization of a localized basin as defined here:
        10.1007/s002140100268
        
        For bonds with more than two intersecting atoms, the two largest 
        fractions are used for the calculation.

        Returns
        -------
        NDArray[np.float64]
            The degree of polarization for each basin

        """
        if self._polarization_indexes is None:
            polarization_indexes = []
            # loop over the atoms and fractions in each basin
            for atom_fracs in self.local_overlap_fractions:
                # if we have only one overlapped atom, this is a fully polarized
                # bond
                if len(atom_fracs) <= 1:
                    polarization_indexes.append(1.0)
                    continue
                # get fracs. These are already sorted from high to low
                fracs = atom_fracs[:,2]

                # calculate polarization index
                polarization_indexes.append((fracs[0] - fracs[1])/(fracs[0] + fracs[1]))
            
            self._polarization_indexes = np.array(polarization_indexes, dtype=np.float64).round(4)
                
                
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
        self._local_overlap_fractions, 
        self._overlap_labels,
        ) = get_overlaps(
            atom_labels=self.qtaim_bader.atom_labels,  # Bader Atoms
            atom_images=self.qtaim_bader.maxima_basin_images,
            local_labels=self.local_bader.maxima_basin_labels,  # ELF basins
            local_images=self.local_bader.maxima_basin_images,  # ELF basins
            charge_data=self.total_charge_grid.total,
            num_charge=len(self.qtaim_maxima_frac),
            num_local=len(self.local_maxima_frac),
            charge_frac=self.qtaim_maxima_frac,
            local_frac=self.local_maxima_frac
        )
    
    def _assign_cores(self, tol=0.02):
        # create tracker for which basins are part of each atoms core
        cores = np.full(len(self.local_maxima_frac), -1, dtype=np.int64)
        shared = np.zeros(len(self.local_maxima_frac), dtype=np.bool_)
        bonded_atoms = np.zeros(len(self.structure), dtype=bool)
        for feature_idx in range(len(self.local_maxima_frac)):
            shape = self.attractor_shapes[feature_idx]
            # basins that are fully polar are cores, unless they are a point
            # far from the atom center. These may be cores or lone-pairs and
            # we will need to determine between them later
            if self.polarization_indexes[feature_idx] >= 1.0 - tol:
                if not (shape == "point" and self.local_bader.basin_atom_dists[feature_idx] > 0.1):
                    cores[feature_idx] = int(self.local_overlap_fractions[feature_idx][:,0])
            
            # label atoms that have bonds
            if self.along_bond[feature_idx]:
                # mark the two most dominant atoms as bonded
                atoms = self.local_overlap_fractions[feature_idx][:2,0].astype(int)
                bonded_atoms[atoms] = True
                shared[feature_idx] = True
        
        for feature_idx in range(len(self.local_maxima_frac)):
            shape = self.attractor_shapes[feature_idx]
            # skip non-polarized features
            if self.polarization_indexes[feature_idx] < self.polarization_cutoff:
                shared[feature_idx] = True
                continue
            
            # check if the most dominant atom is bonded
            atom = int(self.local_overlap_fractions[feature_idx][0,0])
            # if the atom isn't bonded, this must be part of an ionic shell
            if not bonded_atoms[atom]:
                shared[feature_idx] = True
                continue
            
            # if the atom is bonded, and this is a point far from the atoms center,
            # it is a lone-pair
            if shape != "cage" and self.local_bader.basin_atom_dists[feature_idx] > 0.1:
                continue
            
            # otherwise, this is a core
            cores[feature_idx] = True
            
        self._shared_basins = shared
        self._core_basins = cores
            
            
            
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

