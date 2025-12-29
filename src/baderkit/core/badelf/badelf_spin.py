# -*- coding: utf-8 -*-

import json
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pymatgen.io.vasp import Potcar

from baderkit.core import Grid, SpinElfLabeler, Structure
from baderkit.core.badelf.badelf import Badelf
from baderkit.core.labelers.bifurcation_graph.enum_and_styling import FeatureType
from baderkit.core.utilities.file_parsers import Format


class SpinBadelf:
    """
    This class is a wrapper for the Badelf class adding the capability
    to individually handle spin-up and spin-down components of the
    ELF and charge density.
    """

    spin_system = "combined"

    def __init__(
        self,
        reference_grid: Grid,
        charge_grid: Grid,
        elf_labeler: SpinElfLabeler | dict = {},
        **kwargs,
    ):
        """
        An extension of the BadElfToolkit that performs separate calculations on
        the spin-up and spin-down systems.

        Parameters
        ----------
        reference_grid : Grid
            A badelf app Grid like object used for partitioning the unit cell
            volume. Usually contains ELF.
        charge_grid : Grid
            A badelf app Grid like object used for summing charge. Usually
            contains charge density.
        elf_labeler : dict | SpinElfLabeler, optional
            Keyword arguments to pass to the SpinElfLabeler class. This includes
            parameters controlling cutoffs for electrides. Alternatively, a
            SpinElfLabeler class can be passed directly. The default is {}.
        **kwargs : dict
            Any additional keyword arguments to pass to the ElfLabeler class.

        """
        # make sure our grids are spin polarized
        assert (
            reference_grid.is_spin_polarized
        ), "Provided grid is not spin polarized. Use the standard BadElfToolkit."

        self.reference_grid = reference_grid
        self.charge_grid = charge_grid

        # If no labeled structures are provided, we want to use the spin elf
        # labeler and link it to our badelf objects
        # we want to attach a SpinElfLabeler to our badelf objects
        if type(elf_labeler) is dict:
            elf_labeler = SpinElfLabeler(
                charge_grid=charge_grid, reference_grid=reference_grid, **elf_labeler
            )

        self.elf_labeler = elf_labeler
        # link charge grids
        self.reference_grid_up = elf_labeler.reference_grid_up
        self.reference_grid_down = elf_labeler.reference_grid_down
        self.charge_grid_up = elf_labeler.charge_grid_up
        self.charge_grid_down = elf_labeler.charge_grid_down
        self.equal_spin = elf_labeler.equal_spin
        # link labelers
        self.elf_labeler_up = elf_labeler.elf_labeler_up
        self.elf_labeler_down = elf_labeler.elf_labeler_down

        # Now check if we should run a spin polarized badelf calc or not
        if not self.equal_spin:
            self.badelf_up = Badelf(
                reference_grid=self.reference_grid_up,
                charge_grid=self.charge_grid_up,
                elf_labeler=self.elf_labeler_up,
                **kwargs,
            )
            self.badelf_down = Badelf(
                reference_grid=self.reference_grid_down,
                charge_grid=self.charge_grid_down,
                elf_labeler=self.elf_labeler_down,
                **kwargs,
            )
            self.badelf_up.spin_system = "up"
            self.badelf_down.spin_system = "down"
        else:
            self.badelf_up = Badelf(
                reference_grid=self.reference_grid_up,
                charge_grid=self.charge_grid_up,
                elf_labeler=self.elf_labeler_up,
                **kwargs,
            )
            self.badelf_up.spin_system = "half"
            self.badelf_down = self.badelf_up

        # Properties that will be calculated and cached
        self._electride_structure = None
        self._labeled_structure = None
        self._species = None

        self._electride_dim = None

        self._nelectrons = None
        self._charges = None
        self._volumes = None

        self._min_surface_distances = None
        self._avg_surface_distances = None

        self._electrides_per_formula = None
        self._electrides_per_reduced_formula = None

        self._total_charge = None
        self._total_volume = None
        # TODO: Add vacuum handling to Elf Analyzer and BadELF
        # self._vacuum_charge = None
        # self._vacuum_volume = None

        self._results_summary = None

    @property
    def structure(self):
        """

        Returns
        -------
        Structure
            The unlabeled structure representing the system, i.e. the structure
            with no dummy atoms.

        """
        return self.badelf_up.structure

    @property
    def labeled_structure(self):
        """

        Returns
        -------
        Structure
            The system's structure including dummy atoms representing electride
            sites and covalent/metallic bonds. Features unique to the spin-up/spin-down
            systems will have xu or xd appended to the species name respectively.
            Features that exist in both will have nothing appended.

        """
        if self._labeled_structure is None:
            # start with only atoms
            labeled_structure = self.structure.copy()
            # get up and downs structures
            structure_up = self.badelf_up.labeled_structure
            structure_down = self.badelf_down.labeled_structure
            # get species from the spin up system
            new_species = []
            new_coords = []
            for site in structure_up[len(self.structure) :]:
                species = site.specie.symbol
                # add frac coords no matter what
                new_coords.append(site.frac_coords)
                # if this site is in the spin-down structure, it exists in both and
                # we add the site with the original species name
                if site in structure_down:
                    new_species.append(species)
                else:
                    # otherwise, we rename the species
                    new_species.append(species + "xu")
            # do the same for the spin down system
            for site in structure_down[len(self.structure) :]:
                # only add the structure if it didn't exist in the spin up system
                if site not in structure_up:
                    species = site.specie.symbol
                    new_species.append(species + "xd")
                    new_coords.append(site.frac_coords)
            # add our sites
            for species, coords in zip(new_species, new_coords):
                labeled_structure.append(species, coords)
            self._labeled_structure = labeled_structure
        return self._labeled_structure

    @property
    def electride_structure(self) -> Structure:
        """

        Returns
        -------
        Structure
            The system's structure including dummy atoms representing electride
            sites. Electrides unique to the spin-up/spin-down
            systems will have xu or xd appended to the species name respectively.
            Electrides that exist in both will have nothing appended.

        """
        if self._electride_structure is None:
            # create our elecride structure from our labeled structure.
            # NOTE: We don't just use the structure from the elf labeler in
            # case the user provided their own
            electride_structure = self.structure.copy()
            # get bare species including up/down spin
            all_bare_species = []
            for species in FeatureType.bare_species:
                all_bare_species.append(species)
                all_bare_species.append(species + "xu")
                all_bare_species.append(species + "xd")
            # add any bare electron/electrides to our structure
            for site in self.labeled_structure:
                if site.specie.symbol in all_bare_species:
                    electride_structure.append(site.specie.symbol, site.frac_coords)
            self._electride_structure = electride_structure
        return self._electride_structure

    @property
    def nelectrides(self):
        """

        Returns
        -------
        int
            The number of electride sites (electride maxima) present in the system.

        """
        return len(self.electride_structure) - len(self.structure)

    @property
    def species(self) -> list[str]:
        """

        Returns
        -------
        list[str]
            The species of each atom/dummy atom in the electride structure. Covalent
            and metallic features are not included.

        """
        return [i.specie.symbol for i in self.electride_structure]

    @property
    def electride_dimensionality(self):
        """

        Returns
        -------
        int
            The dimensionality of the electride volume at a value of 0 ELF. If
            the dimensionality differes between the spin-up/spin-down results, the
            largest dimensionality is selected.

        """
        return max(
            self.badelf_up.electride_dimensionality,
            self.badelf_down.electride_dimensionality,
        )

    def _get_charges_and_volumes(self):
        """
        NOTE: Volumes may not have a physical meaning when differences are found
        between spin up/down systems. They are calculated as the average between
        the systems.
        """
        # get the initial charges/volumes from the spin up system
        charges = self.badelf_up.charges.tolist()
        volumes = self.badelf_up.volumes.tolist()

        # get the charges from the spin down system
        charges_down = self.badelf_down.charges.tolist()
        volumes_down = self.badelf_down.volumes.tolist()

        # get structures from each system
        structure_up = self.badelf_up.electride_structure
        structure_down = self.badelf_down.electride_structure

        # add charge from spin down structure
        for site, charge, volume in zip(structure_down, charges_down, volumes_down):
            if site in structure_up:
                index = structure_up.index(site)
                charges[index] += charge
                volumes[index] += volume
            else:
                charges.append(charge)
                volumes.append(volume)
        self._charges = np.array(charges)
        self._volumes = np.array(volumes) / 2

    @property
    def charges(self):
        """

        Returns
        -------
        NDArray
            The charge associated with each atom and electride site in the system.
            If an electride site appears in both spin systems, the assigned charge
            is the sum.

        """
        if self._charges is None:
            self._get_charges_and_volumes()
        return self._charges

    @property
    def volumes(self):
        """

        Returns
        -------
        NDArray
            The volume associated with each atom and electride site in the system.
            The volume is taken as the average of the two systems, and may not have
            a physical meaning.

        """
        if self._volumes is None:
            self._get_charges_and_volumes()
        return self._volumes

    @property
    def total_electron_number(self) -> float:
        """

        Returns
        -------
        float
            The total number of electrons in the system calculated from the
            atom charges. If this does not match the true
            total electron number within reasonable floating point error,
            there is a major problem.

        """

        return self.charges.sum()

    def get_oxidation_from_potcar(self, potcar_path: Path | str = "POTCAR"):
        """
        Calculates the oxidation state of each atom/electride using the
        electron counts of the neutral atoms provided in a POTCAR.

        Parameters
        ----------
        potcar_path : Path | str, optional
            The Path to the POTCAR file. The default is "POTCAR".

        Returns
        -------
        oxidation : list
            The oxidation states of each atom/electride.

        """
        # Check if POTCAR exists in path. If not, throw warning
        potcar_path = Path(potcar_path)
        if not potcar_path.exists():
            logging.warning(
                "No POTCAR file found in the requested directory. Oxidation states cannot be calculated"
            )
            return
        # get POTCAR info
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            potcars = Potcar.from_file(potcar_path)
        nelectron_data = {}
        # the result is a list because there can be multiple element potcars
        # in the file (e.g. for NaCl, POTCAR = POTCAR_Na + POTCAR_Cl)
        for potcar in potcars:
            nelectron_data[potcar.element] = potcar.nelectrons
        # get valence electrons for each site in the structure
        valence = np.zeros(len(self.electride_structure), dtype=np.float64)
        for i, site in enumerate(self.structure):
            valence[i] = nelectron_data[site.specie.symbol]
        # subtract charges from valence to get oxidation
        oxidation = valence - self.charges
        return oxidation

    @property
    def electrides_per_formula(self):
        """

        Returns
        -------
        float
            The number of electride electrons for the full structure formula.

        """
        if self._electrides_per_formula is None:
            electrides_per_unit = 0
            for i in range(len(self.structure), len(self.electride_structure)):
                electrides_per_unit += self.charges[i]
            self._electrides_per_formula = electrides_per_unit
        return self._electrides_per_formula

    @property
    def electrides_per_reduced_formula(self):
        """

        Returns
        -------
        float
            The number of electrons in the reduced formula of the structure.

        """
        if self._electrides_per_reduced_formula is None:
            (
                _,
                formula_reduction_factor,
            ) = self.structure.composition.get_reduced_composition_and_factor()
            self._electrides_per_reduced_formula = (
                self.electrides_per_formula / formula_reduction_factor
            )
        return self._electrides_per_reduced_formula

    @property
    def electride_formula(self):
        """

        Returns
        -------
        str
            A string representation of the electride formula, rounding partial charge
            to the nearest integer.

        """
        return f"{self.structure.formula} e{round(self.electrides_per_formula)}"

    @property
    def total_charge(self):
        """

        Returns
        -------
        float
            The total charge integrated in the system. This should match the
            number of electrons from the POTCAR. If it does not there may be a
            serious problem.

        """
        if self._total_charge is None:
            self._total_charge = self.charges.sum()
        return self._total_charge

    @property
    def total_volume(self):
        """

        Returns
        -------
        float
            The total volume integrated in the system. This should match the
            volume of the structure. If it does not there may be a serious problem.

        """
        if self._total_volume is None:
            self._total_volume = self.volumes.sum()
        return self._total_volume

    def to_dict(self, potcar_path: Path | str = "POTCAR", use_json: bool = True):
        """

        Gets a dictionary summary of the BadELF analysis.

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
        results = {}

        results["method_kwargs"] = self.badelf_up.to_dict()["method_kwargs"]

        results["oxidation_states"] = self.get_oxidation_from_potcar(potcar_path)

        for result in [
            "species",
            "structure",
            "labeled_structure",
            "electride_structure",
            "nelectrides",
            "electride_dimensionality",
            "charges",
            "volumes",
            "electride_formula",
            "electrides_per_formula",
            "electrides_per_reduced_formula",
            "total_charge",
            "total_volume",
            "spin_system",
        ]:
            results[result] = getattr(self, result, None)
        if use_json:
            # get serializable versions of each attribute
            for key in ["structure", "labeled_structure", "electride_structure"]:
                results[key] = results[key].to(fmt="POSCAR")
            for key in ["charges", "volumes", "oxidation_states"]:
                if results[key] is None:
                    continue
                results[key] = results[key].tolist()
        return results

    def to_json(self, **kwargs):
        """
        Creates a JSON string representation of the results, typically for writing
        results to file.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for the to_dict method.

        Returns
        -------
        str
            A JSON string representation of the BadELF results.

        """
        return json.dumps(self.to_dict(use_json=True, **kwargs))

    def write_json(
        self, filepath: Path | str = "badelf.json", write_spin: bool = False, **kwargs
    ):
        """
        Writes results of the analysis to file in a JSON format.

        Parameters
        ----------
        filepath : Path | str, optional
            The Path to write the results to. The default is "badelf_results_summary.json".
        write_spin : bool, optional
            Whether or not to write the spin up/down summary jsons as well
        **kwargs : dict
            keyword arguments for the to_dict method.

        """
        filepath = Path(filepath)

        # write total summary
        with open(filepath, "w") as json_file:
            json.dump(self.to_dict(use_json=True, **kwargs), json_file, indent=4)
        # write spin up and spin down summaries
        if write_spin:
            filepath_up = filepath.parent / f"{filepath.stem}_up{filepath.suffix}"
            filepath_down = filepath.parent / f"{filepath.stem}_down{filepath.suffix}"
            self.badelf_up.write_json(filepath=filepath_up)
            self.badelf_down.write_json(filepath=filepath_down)

    @classmethod
    def from_vasp(
        cls,
        reference_file: str | Path = "ELFCAR",
        charge_file: str | Path = "CHGCAR",
        **kwargs,
    ):
        """
        Creates a SpinBadElfToolkit instance from the requested partitioning file
        and charge file.

        Parameters
        ----------
        reference_file : str | Path, optional
            The path to the file to use for partitioning. Must be a VASP
            CHGCAR or ELFCAR type file. The default is "ELFCAR".
        charge_file : str | Path, optional
            The path to the file containing the charge density. Must be a VASP
            CHGCAR or ELFCAR type file. The default is "CHGCAR".
        **kwargs : any
            Additional keyword arguments for the SpinBadElfToolkit class.

        Returns
        -------
        SpinBadElfToolkit
            A SpinBadElfToolkit instance.
        """

        reference_grid = Grid.from_vasp(reference_file, total_only=False)
        charge_grid = Grid.from_vasp(charge_file, total_only=False)
        return cls(reference_grid=reference_grid, charge_grid=charge_grid, **kwargs)

    def write_atom_volumes(
        self,
        atom_indices: NDArray,
        directory: str | Path = None,
        write_reference: bool = True,
        include_dummy_atoms: bool = True,
        output_format: str | Format = None,
        prefix_override: str = None,
        **kwargs,
    ):
        """
        Writes the reference ELF or charge-density for the given atoms to
        separate files. Electrides found during the calculation are appended to
        the end of the structure. Note that non-atomic features of the same index
        in different spin systems may not correspond to the same feature.

        Parameters
        ----------
        atom_indices : NDArray
            The list of atom/electride indices to write
        directory : str | Path, optional
            The directory to write the result to. The default is None.
        write_reference : bool, optional
            Whether or not to write the reference data rather than the charge data.
            The default is True.
        include_dummy_atoms : bool, optional
            Whether or not to include . The default is True.
        output_format : str | Format, optional
            The format to write with. If None, writes to source format stored in
            the Grid objects metadata.
            Defaults to None.
        prefix_override : str, optional
            The string to add at the front of the output path. If None, defaults
            to the VASP file name equivalent to the data type stored in the
            grid.

        """

        if directory is None:
            directory = Path(".")

        # get prefix
        if prefix_override is None:
            if write_reference:
                prefix_override = self.reference_grid.data_type.prefix
            else:
                prefix_override = self.charge_grid.data_type.prefix

        # temporarily update prefix override to avoid overwriting
        if self.equal_spin:
            temp_prefix = f"{prefix_override}_temp"
        else:
            temp_prefix = prefix_override

        for atom_index in atom_indices:
            self.badelf_up.write_atom_volumes(
                atom_indices=[atom_index],
                directory=directory,
                prefix_override=temp_prefix,
                include_dummy_atoms=include_dummy_atoms,
                write_reference=write_reference,
                **kwargs,
            )
            if not self.equal_spin:
                # rename with "up" so we don't overwrite
                os.rename(
                    directory / f"{temp_prefix}_a{atom_index}",
                    directory / f"{prefix_override}_a{atom_index}_up",
                )
                # Write the spin down file and change the name
                self.badelf_down.write_atom_volumes(
                    atom_indices=[atom_index],
                    directory=directory,
                    prefix_override=temp_prefix,
                    include_dummy_atoms=include_dummy_atoms,
                    write_reference=write_reference,
                    **kwargs,
                )
                os.rename(
                    directory / f"{temp_prefix}_a{atom_index}",
                    directory / f"{prefix_override}_a{atom_index}_down",
                )

    def write_all_atom_volumes(
        self,
        directory: str | Path = None,
        write_reference: bool = True,
        include_dummy_atoms: bool = True,
        output_format: str | Format = None,
        prefix_override: str = None,
        **kwargs,
    ):
        """
        Writes the reference ELF or charge-density for the each atom to
        separate files. Electrides found during the calculation are appended to
        the end of the structure. Note that non-atomic features of the same index
        in different spin systems may not correspond to the same feature.

        Parameters
        ----------
        directory : str | Path, optional
            The directory to write the result to. The default is None.
        write_reference : bool, optional
            Whether or not to write the reference data rather than the charge data.
            The default is True.
        include_dummy_atoms : bool, optional
            Whether or not to include . The default is True.
        output_format : str | Format, optional
            The format to write with. If None, writes to source format stored in
            the Grid objects metadata.
            Defaults to None.
        prefix_override : str, optional
            The string to add at the front of the output path. If None, defaults
            to the VASP file name equivalent to the data type stored in the
            grid.

        """

        if directory is None:
            directory = Path(".")

        # get prefix
        if prefix_override is None:
            if write_reference:
                prefix_override = self.reference_grid.data_type.prefix
            else:
                prefix_override = self.charge_grid.data_type.prefix

        # temporarily update prefix override to avoid overwriting
        if self.equal_spin:
            temp_prefix = f"{prefix_override}_temp"
        else:
            temp_prefix = prefix_override

        for atom_index in range(len(self.electride_structure)):
            self.badelf_up.write_atom_volumes(
                atom_indices=[atom_index],
                directory=directory,
                write_reference=write_reference,
                include_dummy_atoms=include_dummy_atoms,
                prefix_override=temp_prefix,
                **kwargs,
            )
            if not self.equal_spin:
                # rename with "up" so we don't overwrite
                os.rename(
                    directory / f"{temp_prefix}_a{atom_index}",
                    directory / f"{prefix_override}_a{atom_index}_up",
                )
                # Write the spin down file and change the name
                self.badelf_down.write_atom_volumes(
                    atom_indices=[atom_index],
                    directory=directory,
                    write_reference=write_reference,
                    include_dummy_atoms=include_dummy_atoms,
                    prefix_override=temp_prefix,
                    **kwargs,
                )
                os.rename(
                    directory / f"{temp_prefix}_a{atom_index}",
                    directory / f"{prefix_override}_a{atom_index}_down",
                )

    def write_atom_volumes_sum(
        self,
        atom_indices: NDArray,
        directory: str | Path = None,
        write_reference: bool = True,
        output_format: str | Format = None,
        include_dummy_atoms: bool = True,
        prefix_override: str = None,
        **kwargs,
    ):
        """

        Writes the reference ELF or charge-density for the the union of the
        given atoms to a single file. Note that non-atomic features of the same index
        in different spin systems may not correspond to the same feature.

        Parameters
        ----------
        atom_indices : int
            The index of the atom/electride to write for.
        directory : str | Path
            The directory to write the files in. If None, the active directory
            is used.
        write_reference : bool, optional
            Whether or not to write the reference data rather than the charge data.
            Default is True.
        include_dummy_atoms : bool, optional
            Whether or not to add dummy files to the structure. The default is False.
        output_format : str | Format, optional
            The format to write with. If None, writes to source format stored in
            the Grid objects metadata.
            Defaults to None.
        prefix_override : str, optional
            The string to add at the front of the output path. If None, defaults
            to the VASP file name equivalent to the data type stored in the
            grid.

        """
        if directory is None:
            directory = Path(".")

        # get prefix
        if prefix_override is None:
            if write_reference:
                prefix_override = self.reference_grid.data_type.prefix
            else:
                prefix_override = self.charge_grid.data_type.prefix

        temp_prefix = f"{prefix_override}_temp"
        self.badelf_up.write_atom_volumes_sum(
            atom_indices=atom_indices,
            directory=directory,
            write_reference=write_reference,
            include_dummy_atoms=include_dummy_atoms,
            prefix_override=temp_prefix,
            **kwargs,
        )
        if not self.equal_spin:
            # rename with "up" so we don't overwrite
            os.rename(
                directory / f"{temp_prefix}_asum",
                directory / f"{prefix_override}_asum_up",
            )
            # Write the spin down file and change the name
            self.badelf_down.write_atom_volumes_sum(
                atom_indices=atom_indices,
                directory=directory,
                write_reference=write_reference,
                include_dummy_atoms=include_dummy_atoms,
                prefix_override=temp_prefix,
                **kwargs,
            )
            os.rename(
                directory / f"{temp_prefix}_asum",
                directory / f"{prefix_override}_asum_down",
            )

    def write_species_volume(
        self,
        directory: str | Path = None,
        species: str = FeatureType.bare_electron.dummy_species,
        write_reference: bool = True,
        output_format: str | Format = None,
        include_dummy_atoms: bool = True,
        prefix_override: str = None,
        **kwargs,
    ):
        """
        Writes the reference ELF or charge-density for all atoms of the given
        species to the same file.

        Parameters
        ----------
        directory : str | Path, optional
            The directory to write the result to. The default is None.
        species : str, optional
            The species to write. The default is "Le" (the electrides).
        write_reference : bool, optional
            Whether or not to write the reference data rather than the charge data.
            The default is True.
        output_format : str | Format, optional
            The format to write with. If None, writes to source format stored in
            the Grid objects metadata.
            Defaults to None.
        include_dummy_atoms : bool, optional
            Whether or not to include . The default is True.
        prefix_override : str, optional
            The string to add at the front of the output path. If None, defaults
            to the VASP file name equivalent to the data type stored in the
            grid.

        """

        if directory is None:
            directory = Path(".")

        # get prefix
        if prefix_override is None:
            if write_reference:
                prefix_override = self.reference_grid.data_type.prefix
            else:
                prefix_override = self.charge_grid.data_type.prefix

        self.badelf_up.write_species_volume(
            species=species,
            directory=directory,
            prefix_override=prefix_override,
            write_reference=write_reference,
            include_dummy_atoms=include_dummy_atoms,
            **kwargs,
        )
        if not self.equal_spin:
            # rename with "up" so we don't overwrite
            os.rename(
                directory / f"{prefix_override}_{species}",
                directory / f"{prefix_override}_{species}_up",
            )
            # Write the spin down file and change the name
            self.badelf_down.write_species_volume(
                species=species,
                directory=directory,
                prefix_override=prefix_override,
                write_reference=write_reference,
                include_dummy_atoms=include_dummy_atoms,
                **kwargs,
            )
            os.rename(
                directory / f"{prefix_override}_{species}",
                directory / f"{prefix_override}_{species}_down",
            )

    def get_atom_results_dataframe(self) -> pd.DataFrame:
        """
        Collects a summary of results for the atoms in a pandas DataFrame.

        Returns
        -------
        atoms_df : pd.DataFrame
            A table summarizing the atomic basins.

        """
        # Get atom results summary
        atom_frac_coords = self.electride_structure.frac_coords
        atoms_df = pd.DataFrame(
            {
                "label": self.electride_structure.labels,
                "x": atom_frac_coords[:, 0],
                "y": atom_frac_coords[:, 1],
                "z": atom_frac_coords[:, 2],
                "charge": self.charges,
                "volume": self.volumes,
                # "surface_dist": self.min_surface_distances,
            }
        )
        return atoms_df

    def write_atom_tsv(
        self,
        filepath: Path | str = "badelf_atoms.tsv",
        write_spin: bool = False,
    ):
        """
        Writes a summary of atom results to .tsv files.

        Parameters
        ----------
        filepath : str | Path, optional
            The Path to write the results to. The default is "badelf_atoms.tsv".
        write_spin : bool, optional
            Whether or not to write the spin up/down tsv files as well

        """
        if write_spin:
            # write spin up and spin down summaries
            filepath_up = filepath.parent / f"{filepath.stem}_up{filepath.suffix}"
            filepath_down = filepath.parent / f"{filepath.stem}_down{filepath.suffix}"
            self.badelf_up.write_atom_tsv(filepath=filepath_up)
            self.badelf_down.write_atom_tsv(filepath=filepath_down)
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

            f.write("\n")
            # f.write(f"Vacuum Charge:\t\t{self.vacuum_charge:.5f}\n")
            # f.write(f"Vacuum Volume:\t\t{self.vacuum_volume:.5f}\n")
            f.write(f"Total Electrons:\t{self.total_electron_number:.5f}\n")
