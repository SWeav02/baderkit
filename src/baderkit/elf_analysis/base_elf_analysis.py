# -*- coding: utf-8 -*-

from pathlib import Path
from typing import TypeVar

from baderkit._base_analysis import BaseAnalysis
from baderkit.toolkit import Grid

Self = TypeVar("Self", bound="BaseElfAnalysis")

class BaseElfAnalysis(BaseAnalysis):

    def __init__(
        self,
        charge_grid: Grid,
        reference_grid: Grid,
        **kwargs,
            ):
        """
        A wrapper class for elf analysis methods that makes the reference
        grid explicitly required.

        Parameters
        ----------
        charge_grid : Grid
            The Grid object with the charge density that will be integrated.
        reference_grid : Grid | None, optional
            The Grid object whose values will be used to construct the basins. This
            should typically only be set when partitioning functions other than the
            charge density (e.g. ELI-D, ELF, etc.).If None, defaults to the
            total_charge_grid.
        **kwargs : dict
            Additional arguments to feed to the BaseAnalysis class.

        """
        super().__init__(
            charge_grid=charge_grid,
            reference_grid=reference_grid,
            **kwargs
            )
        
        

    ###########################################################################
    # From Methods
    ###########################################################################

    @classmethod
    def from_vasp(
        cls,
        charge_filename: Path | str = "CHGCAR",
        reference_filename: Path | str = "ELFCAR",
        **kwargs,
    ) -> Self:
        """
        Creates a Bader class object from VASP files.

        Parameters
        ----------
        charge_filename : Path | str, optional
            The path to the CHGCAR like file that will be used for integrating charge.
            The default is "CHGCAR".
        reference_filename : Path | None | str
            The path to CHGCAR like file that will be used for partitioning.
            If None, the total charge file will be used for partitioning.
        total_charge_filename : Grid | None, optional
            The path to the CHGCAR like file used for determining vacuum regions
            in the system. For pseudopotential codes this represents the total
            electron density and should be provided whenever possible.
            If None, defaults to the charge_grid.
        pseudopotential_filename : Path | None | str | dict, optional
            The path to the POTCAR used for calculating oxidation states. Alternatively,
            a dictionary representing the valence counts of each atom in the system
            where each entry is the species symbol and each value is the number
            of electrons used for that species in the calculation. If None,
            any properties relying on valence counts will not be calculated.
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
        return cls.from_dynamic(charge_filename, reference_filename=reference_filename, format="vasp", **kwargs)

    @classmethod
    def from_cube(cls, **kwargs) -> Self:
        """
        Creates a Bader class object from .cube files.

        Parameters
        ----------
        charge_filename : Path | str, optional
            The path to the .cube like file that will be used for integrating charge.
        total_charge_filename : Grid | None, optional
            The path to the .cube like file used for determining vacuum regions
            in the system. For pseudopotential codes this represents the total
            electron density and should be provided whenever possible.
            If None, defaults to the charge_grid.
        reference_filename : Path | None | str, optional
            The path to .cube file that will be used for partitioning.
            If None, the total charge file will be used for partitioning.
        **kwargs : dict
            Keyword arguments to pass to the class.

        Returns
        -------
        Self
            A BaseAnalysis class object.

        """
        return cls.from_dynamic(format="cube", **kwargs)

    @classmethod
    def from_dynamic(
        cls,
        charge_filename: Path | str,
        reference_filename: Path | str,
        **kwargs,
    ) -> Self:
        """
        Creates a Bader class object from VASP or .cube files. If no format is
        provided the method will automatically try and determine the file type
        from the name

        Parameters
        ----------
        charge_filename : Path | str
            The path to the file containing the charge density that will be
            integrated.
        reference_filename : Path | None
            The path to the file that will be used for partitioning.
            If None, defaults to the total charge grid.
        total_charge_filename : Grid | None, optional
            The path to the file used for determining vacuum regions
            in the system. For pseudopotential codes this represents the total
            electron density and should be provided whenever possible.
            If None, defaults to the charge_grid.
        pseudopotential_filename : Path | None | str | dict, optional
            The path to the pseudopotentials used for calculating oxidation states. Alternatively,
            a dictionary representing the valence counts of each atom in the system
            where each entry is the species symbol and each value is the number
            of electrons used for that species in the calculation. If None,
            any properties relying on valence counts will not be calculated.
        format : Literal["vasp", "cube", None], optional
            The format of the grids to read in. If None, the formats will be
            guessed from the file names.
        total_only: bool
            If true, only the first set of data in the file will be read. This
            increases speed and reduced memory usage as the other data is typically
            not used. This is only used if the file format is determined to be
            VASP, as cube files are assumed to contain only one set of data.
            Defaults to True.
        **kwargs : dict
            Keyword arguments to pass to the class.

        Returns
        -------
        Self
            A BaseAnalysis class object.

        """
        return super().from_dynamic(
            charge_filename=charge_filename,
            reference_filename=reference_filename,
            **kwargs,
            )