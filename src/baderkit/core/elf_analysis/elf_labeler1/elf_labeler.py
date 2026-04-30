# -*- coding: utf-8 -*-

from pathlib import Path
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from baderkit.core.base.base_analysis import BaseAnalysis
from baderkit.core.bader.bader import Bader
from baderkit.core.toolkit import Grid, Structure
from baderkit.core.elf_analysis.overlap import BasinOverlap

from .enum_and_styling import FeatureType
from .elf_labeler_numba import (
    get_core_dist_ratios,
    get_core_dists,
    )

Self = TypeVar("Self", bound="ElfLabeler")

# TODO: Add useful write methods?

class ElfLabeler(BaseAnalysis):
    """

    A tool for labeling basins in a localization function (ELF, ELI-D, LOL, etc.)
    as various chemical features.

    """

    spin_system = "total"

    _summary_props = [
        "basin_types",
        "attractor_shapes",
        "heavily_polarized",
        "nna_bond_fracs",
        "nna_neighbor_dists",
        "nna_indices",
        "nna_formula",
        "nnas_per_formula",
        "nnas_per_reduced_formula",
        "label_structure",
        ]

    _reset_props = [
        "nna_structure",
        "along_bond",
        ] + _summary_props



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
        # create bader objects
        self._overlap = BasinOverlap(
            charge_grid=charge_grid,
            total_charge_grid=total_charge_grid,
            reference_grid=reference_grid,
            **kwargs,
        )

        self._elf_bader = self.overlap.local_bader

        super().__init__(
            charge_grid=charge_grid,
            total_charge_grid=total_charge_grid,
            reference_grid=reference_grid,
            **kwargs,
        )

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
    def maxima_frac(self) -> NDArray[np.float64]:
        """

        Returns
        -------
        NDArray[np.float64]
            The fractional coordinates of each maximum in the ELF.

        """
        return self.elf_bader.maxima_frac

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
    def nna_indices(self) -> NDArray[int]:
        """

        Returns
        -------
        NDArray[int]
            The indices of the basins in the structure that are assigned as
            non-nuclear attractors.

        """
        if self._nna_indices is None:
            self._nna_indices = np.array([i for i,j in enumerate(self.basin_types) if j == FeatureType.nna.value], dtype=np.int64)
        return self._nna_indices


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
            self._heavily_polarized = self.overlap.polarization_indexes > self.polarization_cutoff
        return self._heavily_polarized

    @property
    def nna_radii(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The radius of each non-nuclear attractor calculated as a weighted
            average of the radii between the NNA and the atoms that own some
            portion of its charge.

        """

        if self._nna_bond_fracs is None:
            fracs = get_core_dist_ratios(
                labels=self.elf_bader.maxima_basin_labels,
                basin_frac_coords=self.elf_bader.maxima_frac,
                atom_frac_coords=self.elf_bader.structure.frac_coords,
                matrix=self.reference_grid.matrix,
                nna_indices=self.nna_indices,
                core_basins=self.overlap.core_basins,
                volume_bond_fracs=self.overlap.volume_bond_fractions,
                    )
            self._nna_bond_fracs = fracs * self.nna_neighbor_dists
        return self._nna_bond_fracs

    @property
    def nna_neighbor_dists(self):
        """

        Returns
        -------
        NDArray[float]
            The distance to the nearest neighbors of each non-nuclear attractor
            calculated as a weighted average between the NNA and each atom that
            owns some portion of its charge

        """
        if self._nna_neighbor_dists is None:
            self._nna_neighbor_dists = get_core_dists(
                labels=self.elf_bader.maxima_basin_labels,
                basin_frac_coords=self.elf_bader.maxima_frac,
                atom_frac_coords=self.elf_bader.structure.frac_coords,
                matrix=self.reference_grid.matrix,
                nna_indices=self.nna_indices,
                core_basins=self.overlap.core_basins,
                volume_bond_fracs=self.overlap.volume_bond_fractions,
                    )
        return self._nna_neighbor_dists

    @property
    def num_nnas(self) -> int:
        """

        Returns
        -------
        int
            The number of non-nuclear attractor sites in the structure

        """
        return len(self.nna_structure) - len(self.structure)

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
            for charge, basin_type in zip(self.elf_bader.basin_charges, self.basin_types):
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
                            large -> nna?
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

    @classmethod
    def from_vasp(
        cls,
        charge_filename: Path | str = "CHGCAR",
        reference_filename: Path | str = "ELFCAR",
        pseudopotential_filename: Path | str = "POTCAR",
        **kwargs,
    ) -> Self:
        """
        Creates a Bader class object from VASP files.

        Parameters
        ----------
        charge_filename : Path | str
            The path to the CHGCAR like file that will be used for integrating charge.
            The default is "CHGCAR".
        reference_filename : Path |  str
            The path to ELFCAR like file that will be used for partitioning.
        total_charge_filename : Path |  str, optional
            The path to the CHGCAR like file used for determining vacuum regions
            in the system. For pseudopotential codes this represents the total
            electron density and should be provided whenever possible.
            If None, defaults to the charge_grid.
        pseudopotential_filename : Path |  str
            The path to the pseudopotentials used for calculating oxidation states. Alternatively,
            a dictionary representing the valence counts of each atom in the system
            where each entry is the species symbol and each value is the number
            of electrons used for that species in the calculation.
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
            pseudopotential_filename=pseudopotential_filename,
            **kwargs
            )