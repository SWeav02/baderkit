# -*- coding: utf-8 -*-

from pathlib import Path
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from baderkit.elf_analysis.base_elf_analysis import BaseElfAnalysis
from baderkit.bader.bader import Bader
from baderkit.elf_analysis.overlap.overlap import BasinOverlap
from baderkit.toolkit import Grid, Structure

from .elf_labeler_numba import (
    get_core_dist_ratios,
    get_core_dists
)
from .enum_and_styling import FeatureType

Self = TypeVar("Self", bound="ElfLabeler")

# TODO: Add useful write methods?


class ElfLabeler(BaseElfAnalysis):
    """

    A tool for labeling basins in a localization function (ELF, ELI-D, LOL, etc.)
    as various chemical features.

    """

    _method_kwargs = [
        "polarization_cutoff",
    ]

    _basin_results = [
        "basin_types",
        "attractor_shapes",
        "attractor_depths",
        "heavily_polarized",
        "basin_charges",
        "basin_volumes",
        "basin_atom_dists",
        "basin_dists_beyond_atoms",
        "maxima_frac",
        "maxima_center_frac",
        "maxima_elf_values",
        "nearest_atoms",
        "nearest_atom_species",
    ]

    _nna_results = [
        "nna_indices",
        "nna_formula",
        "num_nnas",
        "nnas_per_formula",
        "nnas_per_reduced_formula",
        "max_nna_dist",
        "species",
    ]

    _nonsummary_results = [
        "label_structure",
        "nna_structure",
        "along_bond",
    ]

    _reset_props = _basin_results + _nna_results + _nonsummary_results

    _summary_props = [
        "basin_results",
        "nna_results",
    ]

    _sub_methods = ["overlap"]

    def __init__(
        self,
        charge_grid: Grid,
        reference_grid: Grid,
        total_charge_grid: Grid | None = None,
        polarization_cutoff: float = 0.5,
        **kwargs,
    ):
        """
        Labels each basin in the ELF as various chemical features.

        This class is designed only for single spin or total spin charge densities
        and ELF. For spin-dependent systems, use the SpinElfLabeler instead.

        Parameters
        ----------
        charge_grid : Grid
            The charge density grid used for integrating charge.
        reference_grid : Grid
            The ELF grid used to partition volumes.
        total_charge_grid : Grid, optional
            The total charge density used for bader integrations and vacuum masks. If
            not provided, the charge_grid will be used instead.
        polarization_cutoff: float, optional
            The degree of polarization used for determining shared vs. unshared
            behavior in a basin. O is more non-polar and 1 is more polar. This
            is calculated from the two atoms that contribute the most to each
            ELF basin.


        **kwargs : dict
            Keyword arguments to pass to the Bader class.

        """

        super().__init__(
            charge_grid=charge_grid,
            total_charge_grid=total_charge_grid,
            reference_grid=reference_grid,
            **kwargs,
        )
        
        # create bader objects
        self._overlap = BasinOverlap(
            charge_grid=charge_grid,
            total_charge_grid=total_charge_grid,
            reference_grid=reference_grid,
            spin_system=self.spin_system,
            **kwargs,
        )

        self._elf_bader = self.overlap.local_bader

        self._polarization_cutoff = polarization_cutoff

    ###########################################################################
    # Settings
    ###########################################################################

    @property
    def polarization_cutoff(self) -> float:
        """

        Returns
        -------
        float
            The degree of polarization used for determining shared vs. unshared
            behavior in a basin. O is more non-polar and 1 is more polar. This
            is calculated from the two atoms that contribute the most to each
            ELF basin.

        """
        return self._polarization_cutoff

    @polarization_cutoff.setter
    def polarization_cutoff(self, value: float):
        self._polarization_cutoff = value
        self._reset_properties(
            include_properties=[
                "heavily_polarized",
                "basin_types",
            ],
        )

    ###########################################################################
    # Properties
    ###########################################################################
    @property
    def label_structure(self) -> Structure:
        """

        Returns
        -------
        Structure
            A PyMatGen Structure object made of dummy atoms representing each
            chemical feature found in the system.

        """
        if self._label_structure is None:
            structure = self.structure.copy()
            structure.remove_sites([i for i in range(len(structure))])
            for basin_type, basin_frac in zip(self.basin_types, self.maxima_frac):
                basin_type = FeatureType(basin_type)
                structure.append(basin_type.dummy_species, basin_frac)
            self._label_structure = structure
        return self._label_structure

    @property
    def nna_structure(self) -> Structure:
        """

        Returns
        -------
        Structure
            The original structure of the system with dummy atoms representing
            non-nuclear attractors appended at the end. Useful when anlyzing
            electride systems for example.

        """
        if self._nna_structure is None:
            structure = self.structure.copy()
            for idx in self.nna_indices:
                basin_type = FeatureType(self.basin_types[idx])
                basin_frac = self.maxima_frac[idx]
                structure.append(basin_type.dummy_species, basin_frac)
            self._nna_structure = structure
        return self._nna_structure

    @property
    def overlap(self) -> BasinOverlap:
        """

        Returns
        -------
        BasinOverlap
            The BasinOverlap class used for QTAIM/ELF overlap calculations.

        """
        return self._overlap

    @property
    def elf_bader(self) -> Bader:
        """

        Returns
        -------
        Bader
            The Bader class used to partition the ELF.

        """

        return self._elf_bader
    
    @property
    def basin_charges(self) -> NDArray[np.float64]:
        """

        Returns
        -------
        NDArray[np.float64]
            The charge contained in each ELF basin

        """
        return self.elf_bader.basin_charges
        
    @property
    def basin_volumes(self) -> NDArray[np.float64]:
        """

        Returns
        -------
        NDArray[np.float64]
            The volume contained in each ELF basin

        """
        return self.elf_bader.basin_volumes

    @property
    def maxima_frac(self) -> NDArray[np.float64]:
        """

        Returns
        -------
        NDArray[np.float64]
            The fractional coordinates of each maximum in the ELF.

        """
        return self.elf_bader.maxima_frac
    
    @property
    def maxima_center_frac(self) -> NDArray[np.float64]:
        """

        Returns
        -------
        NDArray[np.float64]
            The fractional coordinates of the "center of mass" for each maximum in
            the localization function grid. This is used when determining if a basin
            is along a bond, and is particularly necessary for ring shaped covalent bonds.

        """
        return self.overlap.local_maxima_center_frac
    
    @property
    def maxima_elf_values(self) -> NDArray[np.float64]:
        """

        Returns
        -------
        NDArray[np.float64]
            The ELF value at each basins maximum

        """
        return self.elf_bader.maxima_ref_values
    
    @property
    def attractor_shapes(self) -> NDArray[str]:
        """

        Returns
        -------
        NDArray[str]
            The shape of the maxima in the ELF.

        """
        return self.overlap.attractor_shapes
    
    @property
    def attractor_depths(self) -> NDArray[np.float64]:
        """
        Difference in value from the maximum to the first value an attractor
        connects to another.

        Returns
        NDArray[np.float64]
            The depth of each local basin
        """
        return self.overlap.attractor_depths

    @property
    def basin_types(self) -> list[str]:
        """

        Returns
        -------
        list[str]
            The type of chemical feature each basin is a part of.

        """
        if self._basin_types is None:
            self._label_basins()
        return [i.value for i in self._basin_types]
    
    @property
    def nearest_atoms(self) -> NDArray[int]:
        """

        Returns
        -------
        NDArray[int]
            The closest atom to each basin measured from the center of mass.

        """
        return self.elf_bader.basin_atoms
    
    @property
    def nearest_atom_species(self) -> list[str]:
        """

        Returns
        -------
        NDArray[int]
            The type of atom to each basin measured from the center of mass.

        """
        if self._nearest_atom_species is None:
            species = []
            for i in self.elf_bader.basin_atoms:
                species.append(self.elf_bader.structure[i].specie.symbol)
            self._nearest_atom_species = species
        return self._nearest_atom_species

    @property
    def nna_indices(self) -> NDArray[int]:
        """

        Returns
        -------
        NDArray[int]
            The basin indices assigned as non-nuclear attractors

        """
        if self._nna_indices is None:
            self._nna_indices = np.array(
                [
                    i
                    for i, j in enumerate(self.basin_types)
                    if j == FeatureType.nna.value
                ],
                dtype=np.int64,
            )
        return self._nna_indices

    @property
    def basin_dists_beyond_atoms(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The distance beyond each atoms radius at which each basin's maximum
            is located. For features at or below the atoms radius this is 0.0

        """

        if self._basin_dists_beyond_atoms is None:
            fracs = get_core_dist_ratios(
                labels=self.elf_bader.maxima_basin_labels,
                basin_frac_coords=self.elf_bader.maxima_frac,
                atom_frac_coords=self.elf_bader.structure.frac_coords,
                matrix=self.reference_grid.matrix,
                nna_indices=self.nna_indices,
                core_basins=self.overlap.core_basins,
                volume_bond_fracs=self.overlap.volume_bond_fractions,
            )
            self._basin_dists_beyond_atoms = fracs * self.basin_atom_dists
        return self._basin_dists_beyond_atoms
    
    @property
    def basin_atom_dists(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The distance from each basin to its nearest neighbors. This is a
            weighted average based on the degree of overlap the QTAIM atom's have
            with this basin.

        """

        if self._basin_atom_dists is None:
            dists = get_core_dists(
                labels=self.elf_bader.maxima_basin_labels,
                basin_frac_coords=self.elf_bader.maxima_frac,
                atom_frac_coords=self.elf_bader.structure.frac_coords,
                matrix=self.reference_grid.matrix,
                nna_indices=self.nna_indices,
                core_basins=self.overlap.core_basins,
                volume_bond_fracs=self.overlap.volume_bond_fractions,
            )
            self._basin_atom_dists = dists
        return self._basin_atom_dists
    
        
    @property
    def heavily_polarized(self) -> NDArray[bool]:
        """

        Returns
        -------
        NDArray[bool
            A boolean array representing which ELF basins are considered heavily
            polarized towards an atom. The results depend on the 'polarization_cutoff'
            parameter.

        """
        if self._heavily_polarized is None:
            self._heavily_polarized = (
                self.overlap.polarization_indexes > self.polarization_cutoff
            )
        return self._heavily_polarized
    
    ###########################################################################
    # NNA properties
    ###########################################################################
    
    @property
    def max_nna_dist(self) -> float:
        """

        Returns
        -------
        float
            The maximum distance that any NNA in the system sits from its
            neighboring atoms.

        """
        if self.num_nnas > 0:
            return self.basin_atom_dists.max()

    @property
    def num_nnas(self) -> int:
        """

        Returns
        -------
        int
            The number of non-nuclear attractor sites in the structure

        """
        return len(self.nna_indices)

    @property
    def nna_formula(self):
        """

        Returns
        -------
        str
            A string representation of the nna formula, rounding partial charge
            to the nearest integer.

        """
        return f"{self.structure.formula} e{round(self.nnas_per_formula)}"

    @property
    def nnas_per_formula(self):
        """

        Returns
        -------
        float
            The number of nna electrons for the full structure formula.

        """
        if self._nnas_per_formula is None:
            nnas_per_unit = 0
            for charge, basin_type in zip(
                self.elf_bader.basin_charges, self.basin_types
            ):
                if basin_type == FeatureType.nna.value:
                    nnas_per_unit += charge
            self._nnas_per_formula = nnas_per_unit
        return round(self._nnas_per_formula, 10)

    @property
    def nnas_per_reduced_formula(self):
        """

        Returns
        -------
        float
            The number of electrons in the reduced formula of the structure.

        """
        if self._nnas_per_reduced_formula is None:
            (
                _,
                formula_reduction_factor,
            ) = self.structure.composition.get_reduced_composition_and_factor()
            self._nnas_per_reduced_formula = (
                self.nnas_per_formula / formula_reduction_factor
            )
        return round(self._nnas_per_reduced_formula, 10)
    
    @property
    def species(self) -> list[str]:
        """

        Returns
        -------
        list[str]
            The species of each atom/dummy atom in the nna structure. Covalent
            and metallic features are not included.

        """
        return [i.specie.symbol for i in self.nna_structure]

    def _label_basins(self):
        """
        Label scheme:
            shared:
                point/ring:
                    along bond:
                        heavily shared -> covalent bond
                        barely shared -> ionic bond
                    not along bond:
                        heavily shared -> NNA
                        barely shared:
                            dominant atom has other bonds -> lone-pair
                            dominant atom has no bonds -> ionic shell
                cage -> ionic shell

            unshared:
                point:
                    atom center -> shell
                    elsewhere -> lone-pair
                cage -> core
        """

        # create a list to store types
        types = []
        for feature_idx in range(len(self.maxima_frac)):
            shape = self.overlap.attractor_shapes[feature_idx]
            # check for core
            is_core = self.overlap.core_basins[feature_idx] != -1
            if is_core:
                types.append(FeatureType.core)
                continue
            # check for lone-pair
            if not self.overlap.shared_basins[feature_idx]:
                types.append(FeatureType.lone_pair)
                continue

            along_bond = self.overlap.along_bond[feature_idx]
            heavily_polarized = self.heavily_polarized[feature_idx]

            # check for ionic/covalent bond
            if not shape == "cage" and along_bond:
                if heavily_polarized:
                    types.append(FeatureType.ionic)
                else:
                    types.append(FeatureType.covalent)

            # check for ionic shells and nnas
            else:
                if heavily_polarized:
                    types.append(FeatureType.ionic_shell)
                else:
                    types.append(FeatureType.nna)
        self._basin_types = types

    def write_features_by_type(
        self,
        basin_type: str | FeatureType,
        filename: str | Path = "ELFCAR",
        write_grid: str = "reference_grid",
        **kwargs,
    ):
        """
        Writes the charge density or reference file the requested
        chemical features.

        Parameters
        ----------
        basin_type : str | FeatureType
            The type of feature to write, e.g. metallic, electride, etc.

        """
        basin_type = FeatureType(basin_type)
        indices = [i for i, j in enumerate(self.basin_types) if j == basin_type.value]

        # get a mask at the requested feature
        up_mask = np.isin(self.elf_bader.maxima_basin_labels, indices)
        # write
        if not "suffix" in kwargs.keys():
            kwargs["suffix"] = f"_{basin_type.dummy_species}"
        self._write_volume(
            volume_mask=up_mask, write_grid=write_grid, filename=filename, **kwargs
        )

    def write_all_features(
        self,
        **kwargs,
    ):
        """
        Writes the charge density or reference file for all types
        of chemical features in the system.

        """
        feature_types = []
        for i in self.basin_types:
            basin_type = FeatureType(i)
            if basin_type not in feature_types:
                feature_types.append(basin_type)
                self.write_features_by_type(
                    basin_type,
                    **kwargs,
                )