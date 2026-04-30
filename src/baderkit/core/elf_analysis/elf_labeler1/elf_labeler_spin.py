# -*- coding: utf-8 -*-


import logging
from pathlib import Path
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from baderkit.core.base.base_analysis import BaseAnalysis
from baderkit.core import Grid, Structure

from .elf_labeler import ElfLabeler

Self = TypeVar("Self", bound="ElfLabeler")

# TODO: Add useful write methods?

class SpinElfLabeler(BaseAnalysis):


    spin_system = "combined"

    _summary_props = [
        "basin_types",
        "nnas_per_formula",
        "nnas_per_reduced_formula",
        "label_structure",
        "nna_structure"

        ]

    _reset_props = [
        "along_bond",
        ] + _summary_props

    def __init__(
        self,
        charge_grid: Grid,
        reference_grid: Grid,
        total_charge_grid: Grid | None = None,
        **kwargs,
    ):
        """
        Labels each basin in the ELF as various chemical features. More detailed
        information can be found in the 'elf_labeler_up' and 'elf_labeler_down'
        properties.

        This class is designed only for spin separated calculations. For spin
        independent calculations, use the ElfLabeler instead.

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

        # First make sure the grids are actually spin polarized
        assert (
            reference_grid.is_spin_polarized and charge_grid.is_spin_polarized
        ), "ELF must be spin polarized. Use a spin polarized calculation or switch to the ElfLabeler class."

        # run base initialization
        super().__init__(
            charge_grid=charge_grid,
            total_charge_grid=total_charge_grid,
            reference_grid=reference_grid,
            **kwargs,
        )

        # split the grids to spin up and spin down
        self.reference_grid_up, self.reference_grid_down = (
            self.reference_grid.split_to_spin()
        )
        self.charge_grid_up, self.charge_grid_down = self.charge_grid.split_to_spin()

        if total_charge_grid is not None:
            self.total_charge_grid_up, self.total_charge_grid_down = self.total_charge_grid.split_to_spin()
        else:
            self.total_charge_grid_up = None
            self.total_charge_grid_down = None

        # check if spin up and spin down are the same
        if np.allclose(
            self.reference_grid_up.total,
            self.reference_grid_down.total,
            rtol=0,
            atol=1e-4,
        ):
            logging.info(
                "Spin grids are found to be equal. Only spin-up system will be used."
            )
            self.equal_spin = True
        else:
            self.equal_spin = False
        # create spin up and spin down elf analyzer instances
        self.elf_labeler_up = ElfLabeler(
            reference_grid=self.reference_grid_up,
            charge_grid=self.charge_grid_up,
            total_charge_grid=self.total_charge_grid_up,
            **kwargs,
        )
        if not self.equal_spin:
            self.elf_labeler_down = ElfLabeler(
                reference_grid=self.reference_grid_down,
                charge_grid=self.charge_grid_down,
                total_charge_grid=self.total_charge_grid_down,
                **kwargs,
            )
            self.elf_labeler_up.spin_system = "up"
            self.elf_labeler_down.spin_system = "down"
        else:
            self.elf_labeler_down = self.elf_labeler_up
            self.elf_labeler_up.spin_system = "half"  # same up/down

    ###########################################################################
    # Properties combining spin up and spin down systems
    ###########################################################################

    @property
    def label_structure(self) -> Structure:
        """

        Returns
        -------
        Structure
            The system's structure including dummy atoms representing nna
            sites and covalent/metallic bonds. Features unique to the spin-up/spin-down
            systems will have u or d appended to the species name respectively.
            Features that exist in both will have nothing appended.

        """
        if self._label_structure is None:
            # create trackers for basin types and coordinates
            basin_types = []
            new_coords = []

            # start with empty structure
            label_structure = self.structure.copy()
            label_structure.remove_sites([i for i in range(len(label_structure))])

            # get up and downs structures
            structure_up = self.elf_labeler_up.label_structure
            types_up = self.elf_labeler_up.basin_types

            structure_down = self.elf_labeler_down.label_structure
            types_down = self.elf_labeler_down.basin_types

            # get species from the spin up system
            new_species = []
            for site, basin_type in zip(structure_up, types_up):
                species = site.specie.symbol
                # add frac coords and type no matter what
                new_coords.append(site.frac_coords)
                basin_types.append(basin_type)
                # if this site is in the spin-down structure, it exists in both and
                # we add the site with the original species name
                if site in structure_down:
                    new_species.append(species)
                else:
                    # otherwise, we rename the species
                    new_species.append(species + "u")
            # do the same for the spin down system
            for site, basin_type in zip(structure_down, types_down):
                # only add the structure if it didn't exist in the spin up system
                if site not in structure_up:
                    species = site.specie.symbol
                    new_species.append(species + "d")
                    new_coords.append(site.frac_coords)
                    basin_types.append(basin_type)
            # add our sites
            for species, coords in zip(new_species, new_coords):
                label_structure.append(species, coords)
            self._label_structure = label_structure
            self._maxima_frac = np.array(new_coords, dtype=np.float64)
            self._basin_types = basin_types
        return self._label_structure

    @property
    def nna_structure(self) -> Structure:
        """

        Returns
        -------
        Structure
            The system's structure including dummy atoms representing nna
            sites. nnas unique to the spin-up/spin-down
            systems will have u or d appended to the species name respectively.
            nnas that exist in both will have nothing appended.

        """
        if self._nna_structure is None:
            # start with only atoms
            label_structure = self.structure.copy()
            # get up and downs structures
            structure_up = self.elf_labeler_up.nna_structure
            structure_down = self.elf_labeler_down.nna_structure
            # get species from the spin up system
            new_species = []
            new_coords = []
            for site in structure_up[len(self.structure):]:
                species = site.specie.symbol
                # add frac coords no matter what
                new_coords.append(site.frac_coords)
                # if this site is in the spin-down structure, it exists in both and
                # we add the site with the original species name
                if site in structure_down:
                    new_species.append(species)
                else:
                    # otherwise, we rename the species
                    new_species.append(species + "u")
            # do the same for the spin down system
            for site in structure_down[len(self.structure):]:
                # only add the structure if it didn't exist in the spin up system
                if site not in structure_up:
                    species = site.specie.symbol
                    new_species.append(species + "d")
                    new_coords.append(site.frac_coords)
            # add our sites
            for species, coords in zip(new_species, new_coords):
                label_structure.append(species, coords)
            self._nna_structure = label_structure

        return self._nna_structure

    @property
    def basin_types(self) -> list[str]:
        """

        Returns
        -------
        list[str]
            The type of chemical feature each basin is a part of. We include
            basins from both the spin-up and spin-down systems, combining any
            repeats.

        """
        if self._basin_types is None:
            self.label_structure
        return self._basin_types

    @property
    def maxima_frac(self) -> NDArray[np.float64]:
        """

        Returns
        -------
        NDArray[np.float64]
            The fractional coordinates of each maximum in the ELF. We include
            maxima from both the spin-up and spin-down systems, combining any
            repeats.

        """
        if self._maxima_frac is None:
            self.label_structure
        return self._maxima_frac

    @property
    def num_nnas(self) -> int:
        """

        Returns
        -------
        int
            The number of unique nna sites from both the spin-up and spin-down
            systems.

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
            nnas_per_unit = (
                self.elf_labeler_up.nnas_per_formula
                + self.elf_labeler_down.nnas_per_formula
            )
            self._nnas_per_formula = nnas_per_unit
        return self._nnas_per_formula

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
        return self._nnas_per_reduced_formula


    ###########################################################################
    # From methods
    ###########################################################################
    @classmethod
    def from_vasp(
        cls,
        charge_filename: Path | str = "CHGCAR",
        reference_filename: Path | str = "ELFCAR",
        total_only: bool = False,
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
            must be set to False for Spin methods.
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
            total_only=total_only,
            **kwargs
            )