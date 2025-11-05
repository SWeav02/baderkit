# -*- coding: utf-8 -*-

"""
Extends the ElfLabeler to spin polarized calculations
"""

import logging
import warnings
from pathlib import Path
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray
from pymatgen.io.vasp import Potcar

from baderkit.core import Grid, Structure

from .elf_labeler import ElfLabeler

Self = TypeVar("Self", bound="ElfLabeler")


class SpinElfLabeler:

    _spin_system = "combined"

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
            self._equal_spin = True
        else:
            self._equal_spin = False
        # create spin up and spin down elf analyzer instances
        self.elf_labeler_up = ElfLabeler(
            reference_grid=self.reference_grid_up,
            charge_grid=self.charge_grid_up,
            **kwargs,
        )
        if not self._equal_spin:
            self.elf_labeler_down = ElfLabeler(
                reference_grid=self.reference_grid_down,
                charge_grid=self.charge_grid_down,
                **kwargs,
            )
            self.elf_labeler_up._spin_system = "up"
            self.elf_labeler_down._spin_system = "down"
        else:
            self.elf_labeler_down = self.elf_labeler_up
            self.elf_labeler_up._spin_system = "half"  # same up/down

        # calculated properties
        self._labeled_structure = None
        self._quasi_atom_structure = None
        self._atom_elf_radii = None
        self._atom_elf_radii_types = None
        self._atom_nn_elf_radii = None
        self._atom_nn_elf_radii_types = None
        self._nearest_neighbor_data = None

    ###########################################################################
    # Properties combining spin up and spin down systems
    ###########################################################################

    @property
    def structure(self) -> Structure:
        """
        Shortcut to grid's structure object
        """
        structure = self.original_reference_grid.structure.copy()
        return structure

    @property
    def nearest_neighbor_data(self) -> list:
        if self._nearest_neighbor_data is None:
            # get nearest neighbors from spin up labeler
            nearest_neighbor_data = self.elf_labeler_up.nearest_neighbor_data
            # set spin down for speed
            self.elf_labeler_down._nearest_neighbor_data = nearest_neighbor_data
            self._nearest_neighbor_data = nearest_neighbor_data
        return self._nearest_neighbor_data

    @property
    def atom_elf_radii(self) -> NDArray[np.float64]:
        if self._atom_elf_radii is None:
            # get the atomic radii from the spin up/down systems
            spin_up_radii = self.elf_labeler_up.atom_elf_radii
            spin_down_radii = self.elf_labeler_down.atom_elf_radii
            self._atom_elf_radii = (spin_up_radii + spin_down_radii) / 2
        return self._atom_elf_radii

    @property
    def atom_elf_radii_types(self) -> NDArray[np.float64]:
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
        if self._atom_nn_elf_radii is None:
            # get the atomic radii from the spin up/down systems
            spin_up_radii = self.elf_labeler_up.atom_nn_elf_radii
            spin_down_radii = self.elf_labeler_down.atom_nn_elf_radii
            self._atom_nn_elf_radii = (spin_up_radii + spin_down_radii) / 2
        return self._atom_nn_elf_radii

    @property
    def atom_nn_elf_radii_types(self) -> NDArray[np.float64]:
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
        The combined labeled structure from both the spin-up and spin-down system. Features
        found at the same fractional coordinates are combined, while those at
        different coordinates are labeled separately
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
    def quasi_atom_structure(self) -> Structure:
        """
        The combined quasi atom structure from both the spin-up and spin-down system. Features
        found at the same fractional coordinates are combined, while those at
        different coordinates are labeled separately
        """
        if self._quasi_atom_structure is None:
            # start with only atoms
            labeled_structure = self.structure.copy()
            # get up and downs structures
            structure_up = self.elf_labeler_up.quasi_atom_structure
            structure_down = self.elf_labeler_down.quasi_atom_structure
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
            self._quasi_atom_structure = labeled_structure

        return self._quasi_atom_structure

    def get_charges_and_volumes(
        self,
        use_quasi_atoms: bool = True,
        **kwargs,
    ):
        """
        NOTE: Volumes may not have a physical meaning when differences are found
        between spin up/down systems. They are calculated as the average between
        the systems.
        """
        # get the initial charges/volumes from the spin up system
        charges, volumes = self.elf_labeler_up.get_charges_and_volumes(
            use_quasi_atoms=use_quasi_atoms, **kwargs
        )
        # convert to lists
        charges = charges.tolist()
        volumes = volumes.tolist()

        # get the charges from the spin down system
        charges_down, volumes_down = self.elf_labeler_down.get_charges_and_volumes(
            use_quasi_atoms=use_quasi_atoms, **kwargs
        )
        # get structures from each system
        if use_quasi_atoms:
            structure_up = self.elf_labeler_up.quasi_atom_structure
            structure_down = self.elf_labeler_down.quasi_atom_structure
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
        self, potcar_path: Path = "POTCAR", use_quasi_atoms: bool = True, **kwargs
    ):
        """
        NOTE: Volumes may not have a physical meaning when differences are found
        between spin up/down systems
        """
        # get the charges/volumes
        charges, volumes = self.get_charges_and_volumes(
            use_quasi_atoms=use_quasi_atoms, **kwargs
        )
        # convert to path
        potcar_path = Path(potcar_path)
        # load
        with warnings.catch_warnings(record=True):
            potcars = Potcar.from_file(potcar_path)
        nelectron_data = {}
        # the result is a list because there can be multiple element potcars
        # in the file (e.g. for NaCl, POTCAR = POTCAR_Na + POTCAR_Cl)
        for potcar in potcars:
            nelectron_data.update({potcar.element: potcar.nelectrons})
        # calculate oxidation states
        if use_quasi_atoms:
            structure = self.quasi_atom_structure
        else:
            structure = self.structure

        oxi_state_data = []
        for site, site_charge in zip(structure, charges):
            element_str = site.specie.symbol
            val_electrons = nelectron_data.get(element_str, 0.0)
            oxi_state = val_electrons - site_charge
            oxi_state_data.append(oxi_state)

        return np.array(oxi_state_data), charges, volumes

    def write_bifurcation_plots(self, filename: str | Path):
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
