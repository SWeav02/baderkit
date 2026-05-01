# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from baderkit._base_analysis import BaseAnalysis
from baderkit.elf_analysis.badelf.badelf import Badelf
from baderkit.elf_analysis.elf_labeler.elf_labeler_spin import SpinElfLabeler
from baderkit.toolkit import Grid, Structure

Self = TypeVar("Self", bound="SpinBadelf")


class SpinBadelf(BaseAnalysis):

    spin_system = "combined"

    _atom_results = [
        "atom_charges",
        "atom_volumes",
        "oxidation_states",
        "species",
    ]

    _nna_results = [
        "nna_dimensionality",
        "num_nnas",
        "nnas_per_formula",
        "nnas_per_reduced_formula",
    ]

    _nonsummary_results = [
        "nna_structure",
        "labeler",
    ]

    _reset_props = _atom_results + _nna_results + _nonsummary_results

    _summary_props = [
        "atom_results",
        "nna_results",
    ]

    _sub_methods = [
        "badelf_up",
        "badelf_down",
    ]

    def __init__(
        self,
        reference_grid: Grid,
        charge_grid: Grid,
        total_charge_grid: Grid | None = None,
        **kwargs,
    ):
        """
        Class for performing charge analysis using the electron localization function
        (ELF). For information on specific methods, see our [docs](https://sweav02.github.io/baderkit/).

        This class is designed only for spin separated calculations.
        For spin-independent systems, use the Badelf class instead.

        Parameters
        ----------
        reference_grid : Grid
            A Grid like object used for partitioning the unit cell volume. Should
            contain the ELF, ELI-D, LOL, or something similar.
        charge_grid : Grid
            A Grid like object used for summing charge. Should contain the charge
            density.
        total_charge_grid : Grid
            A Grid like object used for locating the vacuum. Should be set when using
            pseudopotential codes such as VASP.
        partition_method : Literal["badelf", "voronelf", "zero-flux"], optional
            The method to use for partitioning nnas from the nearby
            atoms.
                'badelf' (default)
                    Separates nnas using zero-flux surfaces then uses
                    planes at atom radii to separate atoms. This may give more reasonable
                    results for atoms, particularly in ionic solids. Radii are
                    calculated directly from the ELF.
                'voronelf'
                    Separates both nnas and atoms using planes at atomic/nna
                    radii. This is not recommended for nnas that are not
                    spherical, but may provide better results for those that are.
                    Radii are calculated directly from the ELF.
                'zero-flux'
                    Separates nnas and atoms using zero-flux surface. This
                    is the most traditional ELF analysis, but may display some
                    bias towards atoms with higher ELF values. Results for nna
                    sites are identical to BadELF, and the method can be significantly
                    faster.
        shared_feature_splitting_method : Literal["pauling", "equal", "dist", "nearest"], optional
            The method of assigning charge from shared ELF features
            such as covalent or metallic bonds. This parameter is only used with the
            zero-flux method.
                'weighted_dist' (default)
                    Fraction increases with decreasing distance to each atom. The
                    fraction is further weighted by the radius of each atom
                    calculated from the ELF
                'pauling'
                    Distributes charge to neighboring atoms (calculated using CrystalNN)
                    based on the pualing electronegativity of each species normalized
                    such that their sum is equal to 1. If no EN is found for the
                    atom a default of 2.2 is used (including for nnas).
                'equal'
                    Charge is distributed equaly to each neighboring atom/nna
                    (calculated using CrystalNN)
                'dist'
                    Charge is distributed such that more charge is given to the
                    closest atoms. Portions are determined by normalizing the sum
                    of (1/dist) to each neighboring atom.
                'nearest'
                    Gives all charge to the nearest atom or nna site.
        kwargs : dict, optional
            Any keywords to feed to the ElfRadii/ElfLabeler classes
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

        # create elf labeler
        labeler = SpinElfLabeler(
            charge_grid=charge_grid, reference_grid=reference_grid, **kwargs
        )

        self._labeler = labeler
        # link charge grids
        self.reference_grid_up = labeler.reference_grid_up
        self.reference_grid_down = labeler.reference_grid_down
        self.charge_grid_up = labeler.charge_grid_up
        self.charge_grid_down = labeler.charge_grid_down
        self.equal_spin = labeler.equal_spin
        # link labelers
        self.labeler_up = labeler.elf_labeler_up
        self.labeler_down = labeler.elf_labeler_down

        # Now check if we should run a spin polarized badelf calc or not
        if not self.equal_spin:
            self.badelf_up = Badelf(
                reference_grid=self.reference_grid_up,
                charge_grid=self.charge_grid_up,
                labeler=self.labeler_up,
                **kwargs,
            )
            self.badelf_down = Badelf(
                reference_grid=self.reference_grid_down,
                charge_grid=self.charge_grid_down,
                labeler=self.labeler_down,
                **kwargs,
            )
            self.badelf_up.spin_system = "up"
            self.badelf_down.spin_system = "down"
        else:
            self.badelf_up = Badelf(
                reference_grid=self.reference_grid_up,
                charge_grid=self.charge_grid_up,
                labeler=self.labeler_up,
                **kwargs,
            )
            self.badelf_up.spin_system = "half"
            self.badelf_down = self.badelf_up

    @property
    def labeler(self) -> SpinElfLabeler:
        """

        Returns
        -------
        ElfLabeler
            The ElfLabeler class used to locate non-nuclear attractors.

        """
        return self._labeler

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
        return self.labeler.nna_structure

    @property
    def num_nnas(self):
        """

        Returns
        -------
        int
            The number of electride sites (electride maxima) present in the system.

        """
        return len(self.nna_structure) - len(self.structure)

    @property
    def species(self) -> list[str]:
        """

        Returns
        -------
        list[str]
            The species of each atom/dummy atom in the electride structure. Covalent
            and metallic features are not included.

        """
        return [i.specie.symbol for i in self.nna_structure]

    @property
    def nna_dimensionality(self):
        """

        Returns
        -------
        int
            The dimensionality of the electride volume at a value of 0 ELF. If
            the dimensionality differes between the spin-up/spin-down results, the
            largest dimensionality is selected.

        """
        return max(
            self.badelf_up.nna_dimensionality,
            self.badelf_down.nna_dimensionality,
        )

    def _get_charges_and_volumes(self):
        """
        NOTE: Volumes may not have a physical meaning when differences are found
        between spin up/down systems. They are calculated as the average between
        the systems.
        """
        # get the initial charges/volumes from the spin up system
        charges = self.badelf_up.atom_charges.tolist()
        volumes = self.badelf_up.atom_volumes.tolist()

        # get the charges from the spin down system
        charges_down = self.badelf_down.atom_charges.tolist()
        volumes_down = self.badelf_down.atom_volumes.tolist()

        # get structures from each system
        structure_up = self.badelf_up.nna_structure
        structure_down = self.badelf_down.nna_structure

        # add charge from spin down structure
        for site, charge, volume in zip(structure_down, charges_down, volumes_down):
            if site in structure_up:
                index = structure_up.index(site)
                charges[index] += charge
                volumes[index] += volume
            else:
                charges.append(charge)
                volumes.append(volume)
        self._atom_charges = np.array(charges)
        self._atom_volumes = np.array(volumes) / 2

    @property
    def atom_charges(self):
        """

        Returns
        -------
        NDArray
            The charge associated with each atom and electride site in the system.
            If an electride site appears in both spin systems, the assigned charge
            is the sum.

        """
        if self._atom_charges is None:
            self._get_charges_and_volumes()
        return self._atom_charges.round(10)

    @property
    def atom_volumes(self):
        """

        Returns
        -------
        NDArray
            The volume associated with each atom and electride site in the system.
            The volume is taken as the average of the two systems, and may not have
            a physical meaning.

        """
        if self._atom_volumes is None:
            self._get_charges_and_volumes()
        return self._atom_volumes.round(10)

    @property
    def oxidation_states(self) -> NDArray[np.float64]:
        if not self.valence_counts:
            return None
        oxi_state_data = []
        for site, site_charge in zip(self.nna_structure, self.atom_charges):
            element_str = site.specie.name
            oxi_state = self.valence_counts.get(element_str, 0.0) - site_charge
            oxi_state_data.append(oxi_state)

        return np.array(oxi_state_data)

    @property
    def nnas_per_formula(self):
        """

        Returns
        -------
        float
            The number of electride electrons for the full structure formula.

        """
        return self.badelf_up.nnas_per_formula + self.badelf_down.nnas_per_formula

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
    def electride_formula(self):
        """

        Returns
        -------
        str
            A string representation of the electride formula, rounding partial charge
            to the nearest integer.

        """
        return f"{self.structure.formula} e{round(self.nnas_per_formula)}"

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
    # From methods
    ###########################################################################

    @classmethod
    def from_vasp(
        cls,
        charge_filename: Path | str = "CHGCAR",
        reference_filename: Path | str = "ELFCAR",
        pseudopotential_filename: Path | str = "POTCAR",
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
            pseudopotential_filename=pseudopotential_filename,
            total_only=total_only,
            **kwargs,
        )

    ###########################################################################
    # Write Methods
    ###########################################################################

    def write_atom_volumes(
        self,
        atom_indices: NDArray,
        filename: str | Path = "ELFCAR",
        write_grid: str = "reference_grid",
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
            self.badelf_up.write_atom_volumes([atom_index], suffix=f"_a{atom_index}_up")

            self.badelf_down.write_atom_volumes(
                [atom_index], suffix=f"_a{atom_index}_down"
            )

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
        filename: str | Path = "ELFCAR",
        write_grid: str = "reference_grid",
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
        self.badelf_up.write_atom_volumes_sum(atom_indices, suffix="_asum_up")

        self.badelf_down.write_atom_volumes_sum(atom_indices, suffix="_asum_down")

    def write_species_volume(
        self,
        species: str,
        filename: str | Path = "ELFCAR",
        write_grid: str = "reference_grid",
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

        self.badelf_up.write_species_volume(species, suffix=f"_{species}_up")

        self.badelf_down.write_species_volume(species, suffix=f"_{species}_down")

    def get_atom_results_dataframe(self) -> pd.DataFrame:
        """
        Collects a summary of results for the atoms in a pandas DataFrame.

        Returns
        -------
        atoms_df : pd.DataFrame
            A table summarizing the atomic basins.

        """
        # Get atom results summary
        atom_frac_coords = self.nna_structure.frac_coords
        atoms_df = pd.DataFrame(
            {
                "label": self.nna_structure.labels,
                "x": atom_frac_coords[:, 0],
                "y": atom_frac_coords[:, 1],
                "z": atom_frac_coords[:, 2],
                "charge": self.atom_charges,
                "volume": self.atom_volumes,
            }
        )
        return atoms_df

    def write_atom_tsv(self, filepath: Path | str = "badelf_atoms.tsv"):
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
