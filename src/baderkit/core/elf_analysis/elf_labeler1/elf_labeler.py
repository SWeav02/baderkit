# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from baderkit.core.base.base_analysis import BaseAnalysis
from baderkit.core.toolkit import Grid
from baderkit.core.elf_analysis.overlap import BasinOverlap
from baderkit.core.utilities.coord_env import is_along_bond_all

from .enum_and_styling import FeatureType


class ElfLabeler(BaseAnalysis):
    """
    A convenience class for calculating the overlap between basins calculated
    in the charge density and a localization density such as ELF.

    """
    
    _reset_props = [
        "basin_types",
        "attractor_shapes",
        "along_bond",
        "heavily_polarized",
        ]

    def __init__(
        self,
        charge_grid: Grid,
        reference_grid: Grid,
        total_charge_grid: Grid | None = None,
        min_covalent_angle: float = 135,
        max_covalent_polarization: float = 0.5,
        **kwargs,
    ):
        # create bader objects
        self.overlap = BasinOverlap(
            charge_grid=charge_grid,
            total_charge_grid=total_charge_grid,
            reference_grid=total_charge_grid,
            **kwargs,
        )
        
        self.elf_bader = self.overlap.local_bader
        
        self._min_covalent_angle = min_covalent_angle
        self._max_covalent_polarization = max_covalent_polarization

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
                "basin_types",
                "along_bond",
                ], 
            )
    
    @property
    def max_covalent_polarization(self) -> float:
        return self._max_covalent_polarization
    
    @max_covalent_polarization.setter
    def max_covalent_polarization(self, value: float):
        self._max_covalent_polarization = value
        self._reset_properties(
            include_properties=[
                "basin_types",
                "heavily_polarized",
                ], 
            )
    
    ###########################################################################
    # Properties
    ###########################################################################
     
    @property
    def maxima_frac(self) -> NDArray[np.float64]:
        return self.elf_bader.maxima_frac
    
    @property
    def attractor_shapes(self) -> NDArray[str]:
        if self._attractor_shapes is None:
            betti_nums = self.elf_bader.maxima_betti_numbers
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
    def basin_types(self) -> list[str]:
        if self._basin_types is None:
            self._label_basins()
        return self._basin_types
    
    @property
    def along_bond(self):
        if self._along_bond is None:
            self._along_bond, _ = is_along_bond_all(
                feature_frac_coords=self.maxima_frac,
                atom_frac_coords=self.structure.frac_coords,
                atom_cart_coords=self.structure.cart_coords,
                matrix=self.reference_grid.matrix,
                min_covalent_angle=self.min_covalent_angle,
            )
        return self._along_bond
    
    @property
    def heavily_polarized(self):
        if self._heavily_polarized is None:
            self._heavily_polarized = self.overlap.polarization_indexes > self.max_covalent_polarization
        return self._heavily_polarized
    
    def _label_basins(self):
        """
        Label scheme:
            shared:
                point/ring:
                    atom center -> ionic shell
                    along bond:
                        heavily shared -> covalent bond
                        barely shared -> ionic bond
                    not along bond:
                        heavily shared:
                            small -> metallic bond
                            medium -> multi-center bond?
                            large -> electride?
                        barely shared:
                            dominant atom has other bonds -> lone-pair
                            dominant atom has no bonds -> ionic shell
                cage -> ionic shell
            
            unshared:
                point:
                    atom center -> core
                    elsewhere -> lone-pair
                ring -> core?
                cage -> core
        """
        # check whether each basin is along a bond
        
        # create a list to store types
        types = []
        for feature_idx in range(len(self.maxima_frac)):
            is_shared = self.overlap.shared_local_basins[feature_idx]
            if is_shared:
                types.append(self._label_unshared(feature_idx))
            else:
                types.append(self._label_shared(feature_idx))
                
        # some feature types require information on other basins. First, we
        # need which atoms have bonds
        bonded_atoms = np.zeros(len(self.structure), dtype=np.bool)
        for feature_idx in range(len(self.maxima_frac)):
            # skip types that aren't bonds
            if not types[feature_idx].is_bonding:
                continue
            if types[feature_idx] != FeatureType.unknown:
                continue
            atoms = self.overlap.local_overlap_fractions[feature_idx][:,0].astype(int)
            bonded_atoms[atoms] = True
        # now we label ionic shells or lone-pairs that are highly polarized
        types[feature_idx] = self._label_polar_shared(feature_idx, bonded_atoms)
        
        self._basin_types = types

    def _label_unshared(self, feature_idx: int):
        shape = self.attractor_shapes[feature_idx]
        # if we have a ring or cage, this is a core basin
        if (
            shape == "ring" 
            or shape == "cage"
            or self.elf_bader.basin_atom_dists[feature_idx] < 0.1
            ):
            return FeatureType.core
        # otherwise, we have a point basin far from the core, indicating a
        # lone-pair
        return FeatureType.lone_pair
    
    def _label_shared(self, feature_idx: int):
        shape = self.attractor_shapes[feature_idx]
        # if we have a cage, this must be a shell with donated electrons that
        # are well separated from neighbors. Also, if the maximum is at the
        # center of an atom, this is a shell
        if shape == "cage" or self.elf_bader.basin_atom_dists[feature_idx] < 0.1:
            return FeatureType.ionic_shell
        
        # check if this is a highly polarized basin
        heavily_polarized = self.heavily_polarized[feature_idx]
        # check if this basin is situated along a bond
        if self.along_bond[feature_idx]:
            if heavily_polarized:
                return FeatureType.ionic
            else:
                return FeatureType.covalent
        
        # other features require info on all features
        if heavily_polarized:
            return FeatureType.nna
        else:
            return FeatureType.unknown
            
    def _label_polar_shared(self, feature_idx: int, bonded_atoms: NDArray[bool]):
        # get dominant atom
        contained_atoms = self.overlap.local_overlap_fractions[feature_idx]
        atoms = contained_atoms[:,0]
        fractions = contained_atoms[:,1]
        atom = atoms[np.argmax(fractions)]
        # if this atom has other bonds, this is a lone-pair thats part of a
        # vsepr structure
        if bonded_atoms[int(atom)]:
            return FeatureType.lone_pair
        # otherwise we have an ionic shell that is polar enough to not be along
        # atomic bonds
        else:
            return FeatureType.ionic_shell
            
                
            
    
    #properties:
        # labeled basins
        # radii
        
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