# -*- coding: utf-8 -*-

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Literal, TypeVar

import numpy as np
from numpy.typing import NDArray
from pymatgen.io.vasp import Potcar

from baderkit.core import Grid, Structure
from baderkit.core.elf_analysis.bifurcation_graph import FeatureType

from .elf_labeler import ElfLabeler

Self = TypeVar("Self", bound="ElfLabeler")


class SpinElfLabeler:
    """
    Labels chemical features present in the ELF and collects various properties
    e.g charge, volume, elf value, etc. The spin-up and spin-down systems are
    treated separately and results can be viewed combined or independently.

    Parameters
    ----------
    charge_grid : Grid
        A charge density Grid object. The total charge density (spin-up + spin-down)
        should be stored in the 'total' property and the difference (spin-up - spin-down)
        should be stored in the 'diff' property.
    reference_grid : Grid
        An ELF Grid object. The spin-up ELF should be stored in the 'total' property
        and the spin-down ELF should be stored in the 'diff' property.
    **kwargs : dict
        Any keyword argumetns to pass to the child ElfLabeler classes used for
        each spin.

    """

    spin_system = "combined"

    def __init__(
        self,
        charge_grid: Grid,
        reference_grid: Grid,
        **kwargs,
    ):

        # First make sure the grids are actually spin polarized
        assert (
            reference_grid.is_spin_polarized and charge_grid.is_spin_polarized
        ), "ELF must be spin polarized. Use a spin polarized calculation or switch to the ElfLabeler class."
        # store the original grid
        self.original_reference_grid = reference_grid
        self.original_charge_grid = charge_grid
        # split the grids to spin up and spin down
        self.reference_grid_up, self.reference_grid_down = (
            reference_grid.split_to_spin()
        )
        self.charge_grid_up, self.charge_grid_down = charge_grid.split_to_spin()
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
            **kwargs,
        )
        if not self.equal_spin:
            self.elf_labeler_down = ElfLabeler(
                reference_grid=self.reference_grid_down,
                charge_grid=self.charge_grid_down,
                **kwargs,
            )
            self.elf_labeler_up.spin_system = "up"
            self.elf_labeler_down.spin_system = "down"
        else:
            self.elf_labeler_down = self.elf_labeler_up
            self.elf_labeler_up.spin_system = "half"  # same up/down

        # calculated properties
        self._labeled_structure = None
        self._electride_structure = None
        self._atom_elf_radii = None
        self._atom_elf_radii_types = None
        self._atom_nn_elf_radii = None
        self._atom_nn_elf_radii_types = None
        self._nearest_neighbor_data = None
        self._electrides_per_formula = None
        self._electrides_per_reduced_formula = None

    ###########################################################################
    # Properties combining spin up and spin down systems
    ###########################################################################

    @property
    def structure(self) -> Structure:
        """

        Returns
        -------
        Structure
            The PyMatGen Structure representing the system.

        """
        structure = self.original_reference_grid.structure.copy()
        return structure

    @property
    def nearest_neighbor_data(self) -> list:
        """

        Returns
        -------
        tuple
            The nearest neighbor data for the atoms in the system represented as
            a tuple of arrays. The arrays represent, in order, the central
            atoms index, its neighbors index, the fractional coordinates of the
            neighbor, and the distance between the two sites.

        """
        if self._nearest_neighbor_data is None:
            # get nearest neighbors from spin up labeler
            nearest_neighbor_data = self.elf_labeler_up.nearest_neighbor_data
            # set spin down for speed
            self.elf_labeler_down._nearest_neighbor_data = nearest_neighbor_data
            self._nearest_neighbor_data = nearest_neighbor_data
        return self._nearest_neighbor_data

    @property
    def atom_elf_radii(self) -> NDArray[np.float64]:
        """

        Returns
        -------
        NDArray
            The radius of each atom calculated from the ELF using the closest
            neighboring atom in the structure. This is taken as the average value
            from both spin systems.

        """
        if self._atom_elf_radii is None:
            # get the atomic radii from the spin up/down systems
            spin_up_radii = self.elf_labeler_up.atom_elf_radii
            spin_down_radii = self.elf_labeler_down.atom_elf_radii
            self._atom_elf_radii = (spin_up_radii + spin_down_radii) / 2
        return self._atom_elf_radii

    @property
    def atom_elf_radii_types(self) -> NDArray[np.float64]:
        """

        Returns
        -------
        NDArray
            The type of radius of each elf radius. Covalent indicates that the
            bond crosses through some covalent or metallic region, and the radius
            is placed at the maximum in the ELF in this region. Ionic indicates
            that the bond does not pass through the covalent/metallic region and
            the radius is placed at the minimum between the two atoms.

        """
        if self._atom_elf_radii_types is None:
            # make sure spin up/down labelers have calculated radii
            self.atom_elf_radii
            # default to covalent
            self._atom_elf_radii_types = (
                self.elf_labeler_up._atom_elf_radii_types
                | self.elf_labeler_down._atom_elf_radii_types
            )
        # convert to strings and return
        return np.where(self._atom_elf_radii_types, "covalent", "ionic")

    @property
    def atom_nn_elf_radii(self) -> NDArray[np.float64]:
        """

        Returns
        -------
        NDArray
            The elf radii for each atom and its neighboring atoms in the same
            order as the nearest_neighbor_data property. Radii are taken as
            averages between the spin-up and spin-down systems.

        """
        if self._atom_nn_elf_radii is None:
            # get the atomic radii from the spin up/down systems
            spin_up_radii = self.elf_labeler_up.atom_nn_elf_radii
            spin_down_radii = self.elf_labeler_down.atom_nn_elf_radii
            self._atom_nn_elf_radii = (spin_up_radii + spin_down_radii) / 2
        return self._atom_nn_elf_radii

    @property
    def atom_nn_elf_radii_types(self) -> NDArray[np.float64]:
        """

        Returns
        -------
        NDArray
            The type of radius for each atom and its neighboring atoms in the same
            order as the nearest_neighbor_data property.

        """
        if self._atom_nn_elf_radii_types is None:
            # make sure spin up/down labelers have calculated radii
            self.atom_nn_elf_radii
            # default to covalent
            self._atom_nn_elf_radii_types = (
                self.elf_labeler_up._atom_nn_elf_radii_types
                | self.elf_labeler_down._atom_nn_elf_radii_types
            )
        # convert to strings and return
        return np.where(self._atom_nn_elf_radii_types, "covalent", "ionic")

    @property
    def labeled_structure(self) -> Structure:
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
            structure_up = self.elf_labeler_up.labeled_structure
            structure_down = self.elf_labeler_down.labeled_structure
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
            # start with only atoms
            labeled_structure = self.structure.copy()
            # get up and downs structures
            structure_up = self.elf_labeler_up.electride_structure
            structure_down = self.elf_labeler_down.electride_structure
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
            self._electride_structure = labeled_structure

        return self._electride_structure

    @property
    def nelectrides(self) -> int:
        """

        Returns
        -------
        int
            The number of electride sites in the structure

        """
        return len(self.electride_structure) - len(self.structure)

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
    def electrides_per_formula(self):
        """

        Returns
        -------
        float
            The number of electride electrons for the full structure formula.

        """
        if self._electrides_per_formula is None:
            electrides_per_unit = (
                self.elf_labeler_up.electrides_per_formula
                + self.elf_labeler_down.electrides_per_formula
            )
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

    def get_charges_and_volumes(
        self,
        use_electrides: bool = True,
        **kwargs,
    ):
        """
        Calculates charges and volumes by splitting feature charges/volumes to
        their neighboring atoms.

        NOTE: Volumes may not have a physical meaning when differences are found
        between spin up/down systems. They are calculated as the average between
        the systems.

        Parameters
        ----------
        use_electrides : bool, optional
            Whether or not to consider electrides as quasi-atoms. The default is True.
        **kwargs : dict
            Any keyword arguments to use in the corresponding ElfLabeler method
            for each spin system.

        Returns
        -------
        tuple
            The charges and volumes calculated for each atom.

        """

        # get the initial charges/volumes from the spin up system
        charges, volumes = self.elf_labeler_up.get_charges_and_volumes(
            use_electrides=use_electrides, **kwargs
        )
        # convert to lists
        charges = charges.tolist()
        volumes = volumes.tolist()

        # get the charges from the spin down system
        charges_down, volumes_down = self.elf_labeler_down.get_charges_and_volumes(
            use_electrides=use_electrides, **kwargs
        )
        # get structures from each system
        if use_electrides:
            structure_up = self.elf_labeler_up.electride_structure
            structure_down = self.elf_labeler_down.electride_structure
        else:
            structure_up = self.structure
            structure_down = self.structure
        # add charge from spin down structure
        for site, charge, volume in zip(structure_down, charges_down, volumes_down):
            if site in structure_up:
                index = structure_up.index(site)
                charges[index] += charge
                volumes[index] += volume
            else:
                charges.append(charge)
                volumes.append(volume)
        return np.array(charges), np.array(volumes) / 2

    def get_oxidation_and_volumes_from_potcar(
        self, potcar_path: Path = "POTCAR", use_electrides: bool = True, **kwargs
    ):
        """
        Calculates oxidation states, charges, and volumes by splitting feature
        charges/volumes to their neighboring atoms and comparing to the valence
        electrons in the POTCAR.

        NOTE: Volumes may not have a physical meaning when differences are found
        between spin up/down systems. They are calculated as the average between
        the systems.

        Parameters
        ----------
        potcar_path : Path | str, optional
            The Path to the POTCAR file. The default is "POTCAR".
        use_electrides : bool, optional
            Whether or not to consider electrides as quasi-atoms. The default is True.
        **kwargs : dict
            Any keyword arguments to use in the corresponding ElfLabeler method
            for each spin system.

        Returns
        -------
        tuple
            The oxidation states, charges, and volumes calculated for each atom.

        """

        # get the charges/volumes
        charges, volumes = self.get_charges_and_volumes(
            use_electrides=use_electrides, **kwargs
        )
        # convert to path
        potcar_path = Path(potcar_path)
        # check if potcar exists. If not, return None and a warning
        if not potcar_path.exists():
            logging.warning(
                "No POTCAR found at provided path. No oxidation states will be calculated."
            )
            return None, charges, volumes

        # load
        with warnings.catch_warnings(record=True):
            potcars = Potcar.from_file(potcar_path)
        nelectron_data = {}
        # the result is a list because there can be multiple element potcars
        # in the file (e.g. for NaCl, POTCAR = POTCAR_Na + POTCAR_Cl)
        for potcar in potcars:
            nelectron_data.update({potcar.element: potcar.nelectrons})
        # calculate oxidation states
        if use_electrides:
            structure = self.electride_structure
        else:
            structure = self.structure

        oxi_state_data = []
        for site, site_charge in zip(structure, charges):
            element_str = site.specie.symbol
            val_electrons = nelectron_data.get(element_str, 0.0)
            oxi_state = val_electrons - site_charge
            oxi_state_data.append(oxi_state)

        return np.array(oxi_state_data), charges, volumes

    def write_bifurcation_plot(self, filename: str | Path):
        """
        Writes an html plot representing the bifurcation graph.

        Parameters
        ----------
        filename : str | Path
            The file to write the bifurcation plot to.

        """
        filename = Path(filename)

        if filename.suffix:
            filename_up = filename.with_name(f"{filename.stem}_up{filename.suffix}")
            filename_down = filename.with_name(f"{filename.stem}_down{filename.suffix}")
        else:
            filename_up = filename.with_name(f"{filename.name}_up")
            filename_down = filename.with_name(f"{filename.name}_down")

        self.elf_labeler_up.write_bifurcation_plot(filename_up)
        self.elf_labeler_down.write_bifurcation_plot(filename_down)

    ###########################################################################
    # Vacuum Properties
    ###########################################################################
    @property
    def vacuum_charge(self) -> float:
        """

        Returns
        -------
        float
            The charge assigned to the vacuum.

        """
        return self.elf_labeler_up.vacuum_charge + self.elf_labeler_down.vacuum_charge

    @property
    def vacuum_volume(self) -> float:
        """

        Returns
        -------
        float
            The total volume assigned to the vacuum. This is an average between
            the spin up and spin down values.

        """
        return (
            self.elf_labeler_up.vacuum_volume + self.elf_labeler_down.vacuum_volume
        ) / 2

    @property
    def total_electron_number(self) -> float:
        """

        Returns
        -------
        float
            The total number of electrons in the system calculated from the
            spin-up and spin-down systems. If this does not match the true
            total electron number within reasonable floating point error,
            there is a major problem.

        """

        return round(
            self.elf_labeler_up.total_electron_number
            + self.elf_labeler_down.total_electron_number,
            10,
        )

    @property
    def total_volume(self):
        """

        Returns
        -------
        float
            The total volume integrated in the system. This should match the
            volume of the structure. If it does not there may be a serious problem.

            This is the average of the two systems

        """

        return (
            self.elf_labeler_up.total_volume + self.elf_labeler_down.total_volume
        ) / 2

    ###########################################################################
    # From methods
    ###########################################################################
    @classmethod
    def from_vasp(
        cls,
        charge_filename: Path | str = "CHGCAR",
        reference_filename: Path | str = "ELFCAR",
        **kwargs,
    ) -> Self:
        """
        Creates a SpinElfAnalysis class object from VASP files.

        Parameters
        ----------
        charge_filename : Path | str, optional
            The path to the CHGCAR like file that will be used for summing charge.
            The default is "CHGCAR".
        reference_filename : Path | str
            The path to ELFCAR like file that will be used for partitioning.
            If None, the charge file will be used for partitioning.
        total_only: bool
            If true, only the first set of data in the file will be read. This
            increases speed and reduced memory usage as the other data is typically
            not used.
            Defaults to True.
        **kwargs : dict
            Keyword arguments to pass to the Bader class.

        Returns
        -------
        Self
            A SpinElfAnalysis class object.

        """
        charge_grid = Grid.from_vasp(charge_filename, total_only=False)
        if reference_filename is None:
            reference_grid = None
        else:
            reference_grid = Grid.from_vasp(reference_filename, total_only=False)

        return cls(charge_grid=charge_grid, reference_grid=reference_grid, **kwargs)

    # TODO: Currently this class is only useful for VASP because .cube files
    # typically only contain a single grid. Is there a reason to create a convenience
    # function for cube files?

    def to_dict(
        self,
        potcar_path: Path | str = "POTCAR",
        use_json: bool = True,
        splitting_method: Literal[
            "equal", "pauling", "dist", "weighted_dist", "nearest"
        ] = "weighted_dist",
    ) -> dict:
        """

        Gets a dictionary summary of the ElfLabeler analysis.

        Parameters
        ----------
        potcar_path : Path | str, optional
            The Path to a POTCAR file. This must be provided for oxidation states
            to be calculated, and they will be None otherwise. The default is "POTCAR".
        use_json : bool, optional
            Convert all entries to JSONable data types. The default is True.
        splitting_method : Literal["equal", "pauling", "dist", "weighted_dist", "nearest"], optional
            See :meth:`write_feature_basins`.

        Returns
        -------
        dict
            A summary of the ElfLabeler analysis in dictionary form.

        """
        results = {}
        # collect method kwargs
        results["method_kwargs"] = {
            "splitting_method": splitting_method,
        }

        oxidation_states, charges, volumes = self.get_oxidation_and_volumes_from_potcar(
            potcar_path=potcar_path, use_electrides=False
        )
        oxidation_states_e, charges_e, volumes_e = (
            self.get_oxidation_and_volumes_from_potcar(
                potcar_path=potcar_path, use_electrides=True
            )
        )

        if oxidation_states is not None:
            oxidation_states = oxidation_states.tolist()
            oxidation_states_e = oxidation_states_e.tolist()

        results["oxidation_states"] = oxidation_states
        results["oxidation_states_e"] = oxidation_states_e
        results["charges"] = charges.tolist()
        results["charges_e"] = charges_e.tolist()
        results["volumes"] = volumes.tolist()
        results["volumes_e"] = volumes_e.tolist()

        # add objects that can convert to json
        for result in [
            "structure",
            "labeled_structure",
            "electride_structure",
        ]:
            result_obj = getattr(self, result, None)
            if result_obj is not None and use_json:
                result_obj = result_obj.to_json()
            results[result] = result_obj

        # add objects that are arrays
        for result in [
            "atom_elf_radii",
            "atom_elf_radii_types",
            "atom_elf_radii_e",
            "atom_elf_radii_types_e",
        ]:
            result_obj = getattr(self, result, None)
            if use_json and result_obj is not None:
                result_obj = result_obj.tolist()
            results[result] = result_obj

        # add other objects that are already jsonable
        for result in [
            "spin_system",
            "nelectrides",
            "feature_types",
            "electride_formula",
            "electrides_per_formula",
            "electrides_per_reduced_formula",
            "total_electron_number",
            "total_volume",
            "vacuum_charge",
            "vacuum_volume",
        ]:
            results[result] = getattr(self, result, None)

        return results

    def to_json(self, **kwargs) -> str:
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

    def write_json(self, filepath: Path | str = "elf_labeler.json", **kwargs) -> None:
        """
        Writes results of the analysis to file in a JSON format.

        Parameters
        ----------
        filepath : Path | str, optional
            The Path to write the results to. The default is "badelf_results_summary.json".
        **kwargs : dict
            keyword arguments for the to_dict method.

        """
        filepath = Path(filepath)
        # write spin up and spin down summaries
        filepath_up = filepath.parent / f"{filepath.stem}_up{filepath.suffix}"
        filepath_down = filepath.parent / f"{filepath.stem}_down{filepath.suffix}"
        self.elf_labeler_up.write_json(filepath=filepath_up, **kwargs)
        self.elf_labeler_down.write_json(filepath=filepath_down, **kwargs)

        # write total spin summary
        with open(filepath, "w") as json_file:
            json.dump(self.to_dict(use_json=True, **kwargs), json_file, indent=4)

    def write_all_features(
        self,
        directory: str | Path = Path("."),
        write_reference: bool = True,
        prefix_override: str = None,
        **kwargs,
    ):
        """
        Writes the bader basins associated with all features

        Parameters
        ----------
        directory : str | Path, optional
            The directory to write to. The default is None.
        write_reference : bool, optional
            Whether or not to write the reference data rather than the charge
            density. The default is True.
        prefix_override : str, optional
            The string to add at the front of the output path. If None, defaults
            to the VASP file name equivalent to the data type stored in the
            grid. The default is None.
        **kwargs : dict
            Keyword arguments to pass to the ElfLabeler write method.

        """

        if directory is None:
            directory = Path(".")

        # get prefix
        if prefix_override is None:
            if write_reference:
                prefix_override = self.original_reference_grid.data_type.prefix
            else:
                prefix_override = self.original_charge_grid.data_type.prefix

        # temporarily update prefix override to avoid overwriting
        if self.equal_spin:
            temp_prefix = f"{prefix_override}_temp"
        else:
            temp_prefix = prefix_override

        for feat_idx in range(len(self.elf_labeler_up.feature_charges)):
            self.elf_labeler_up.write_feature_basins(
                feature_indices=[feat_idx],
                directory=directory,
                write_reference=write_reference,
                prefix_override=temp_prefix,
                **kwargs,
            )
            if not self.equal_spin:
                # rename with "up" so we don't overwrite
                os.rename(
                    directory / f"{temp_prefix}_f{feat_idx}",
                    directory / f"{prefix_override}_f{feat_idx}_up",
                )
            else:
                os.rename(
                    directory / f"{temp_prefix}_f{feat_idx}",
                    directory / f"{prefix_override}_f{feat_idx}",
                )
        if self.equal_spin:
            return
        # Write the spin down file and change the name
        for feat_idx in range(len(self.elf_labeler_down.feature_charges)):
            self.elf_labeler_down.write_feature_basins(
                feature_indices=[feat_idx],
                directory=directory,
                write_reference=write_reference,
                prefix_override=temp_prefix,
                **kwargs,
            )
            if not self.equal_spin:
                # rename with "up" so we don't overwrite
                os.rename(
                    directory / f"{temp_prefix}_f{feat_idx}",
                    directory / f"{prefix_override}_f{feat_idx}_down",
                )

    def write_features_by_type(
        self,
        included_types: list[FeatureType],
        directory: str | Path = Path("."),
        prefix_override=None,
        write_reference: bool = True,
        **kwargs,
    ):
        """

        Writes the reference ELF or charge-density for the the union of the
        given atoms to a single file.

        Parameters
        ----------
        included_types : list[FeatureType]
            The types of features to include, e.g. metallic, lone-pair, etc.
        directory : str | Path
            The directory to write the files in. If None, the active directory
            is used.
        prefix_override : str, optional
            The string to add at the front of the output path. If None, defaults
            to the VASP file name equivalent to the data type stored in the
            grid.
        write_reference : bool, optional
            Whether or not to write the reference data rather than the charge data.
            Default is True.
        **kwargs :
            See :meth:`write_feature_basins`.

        """
        if directory is None:
            directory = Path(".")

        # get prefix
        if prefix_override is None:
            if write_reference:
                prefix_override = self.original_reference_grid.data_type.prefix
            else:
                prefix_override = self.original_charge_grid.data_type.prefix

        # temporarily update prefix override to avoid overwriting
        if self.equal_spin:
            temp_prefix = f"{prefix_override}_temp"
        else:
            temp_prefix = prefix_override

        for feat_type in included_types:
            feat_type = FeatureType(feat_type)

            self.elf_labeler_up.write_features_by_type(
                included_types=[feat_type],
                directory=directory,
                write_reference=write_reference,
                prefix_override=temp_prefix,
                **kwargs,
            )

            if not self.equal_spin:
                # rename with "up" so we don't overwrite
                os.rename(
                    directory / f"{temp_prefix}_{feat_type.dummy_species}",
                    directory / f"{prefix_override}_{feat_type.dummy_species}_up",
                )
            else:
                os.rename(
                    directory / f"{temp_prefix}_{feat_type.dummy_species}",
                    directory / f"{prefix_override}_{feat_type.dummy_species}",
                )
                return

                # Write the spin down file and change the name
            # temporarily update prefix override to avoid overwriting
            self.elf_labeler_down.write_features_by_type(
                included_types=[feat_type],
                directory=directory,
                write_reference=write_reference,
                prefix_override=temp_prefix,
                **kwargs,
            )
            if not self.equal_spin:
                # rename with "up" so we don't overwrite
                os.rename(
                    directory / f"{temp_prefix}_{feat_type.dummy_species}",
                    directory / f"{prefix_override}_{feat_type.dummy_species}_down",
                )

    def write_features_by_type_sum(
        self,
        included_types: list[FeatureType],
        directory: str | Path = Path("."),
        prefix_override=None,
        write_reference: bool = True,
        **kwargs,
    ):
        """

        Writes the reference ELF or charge-density for the the union of the
        given atoms to a single file.

        Parameters
        ----------
        included_types : list[FeatureType]
            The types of features to include, e.g. metallic, lone-pair, etc.
        directory : str | Path
            The directory to write the files in. If None, the active directory
            is used.
        prefix_override : str, optional
            The string to add at the front of the output path. If None, defaults
            to the VASP file name equivalent to the data type stored in the
            grid.
        write_reference : bool, optional
            Whether or not to write the reference data rather than the charge data.
            Default is True.
        **kwargs :
            See :meth:`write_feature_basins`.

        """
        if directory is None:
            directory = Path(".")

        # get prefix
        if prefix_override is None:
            if write_reference:
                prefix_override = self.original_reference_grid.data_type.prefix
            else:
                prefix_override = self.original_charge_grid.data_type.prefix

        # temporarily update prefix override to avoid overwriting
        if self.equal_spin:
            temp_prefix = f"{prefix_override}_temp"
        else:
            temp_prefix = prefix_override

        self.elf_labeler_up.write_features_by_type_sum(
            included_types=included_types,
            directory=directory,
            write_reference=write_reference,
            prefix_override=temp_prefix,
            **kwargs,
        )
        if not self.equal_spin:
            # rename with "up" so we don't overwrite
            os.rename(
                directory / f"{temp_prefix}_fsum",
                directory / f"{prefix_override}_fsum_up",
            )
        else:
            os.rename(
                directory / f"{temp_prefix}_fsum",
                directory / f"{prefix_override}_fsum",
            )
            return

        # Write the spin down file and change the name
        # temporarily update prefix override to avoid overwriting
        self.elf_labeler_down.write_features_by_type_sum(
            included_types=included_types,
            directory=directory,
            write_reference=write_reference,
            prefix_override=temp_prefix,
            **kwargs,
        )
        if not self.equal_spin:
            # rename with "up" so we don't overwrite
            os.rename(
                directory / f"{temp_prefix}_fsum",
                directory / f"{prefix_override}_fsum_down",
            )
