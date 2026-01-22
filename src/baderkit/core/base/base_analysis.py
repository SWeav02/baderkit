# -*- coding: utf-8 -*-

import copy
import json
from pathlib import Path
from abc import ABC, abstractclassmethod
from typing import Literal, TypeVar
import logging

import numpy as np
from numpy.typing import NDArray

from baderkit.core.toolkit import Grid, Structure
from baderkit.core.utilities.file_parsers import Format

# # This allows for Self typing and is compatible with python 3.10
Self = TypeVar("Self", bound="BaseAnalysis")

# TODO:
# - Add handling of non-nuclear attractors (e.g. those in Li metal)


class BaseAnalysis(ABC):
    """
    This is the base class that most analysis classes pull from. It covers the absolute
    basic methods and ensures similar structure throughout the package.

    Parameters
    ----------
    charge_grid : Grid
        The Grid object with the charge density that will be integrated.
    total_charge_grid : Grid | None, optional
        The Grid object used for determining vacuum regions in the system. For
        pseudopotential codes this represents the total electron density and should
        be provided whenever possible. If None, defaults to the charge_grid.
    reference_grid : Grid | None, optional
        The Grid object whose values will be used to construct the basins. This
        should typically only be set when partitioning functions other than the 
        charge density (e.g. ELI-D, ELF, etc.).If None, defaults to the 
        total_charge_grid.
    vacuum_tol : float | bool, optional
        If a float is provided, this is the value below which a point will
        be considered part of the vacuum. If a bool is provided, no vacuum
        will be used on False, and the default tolerance (0.001) will be used on True.

    """
    
    _base_reset_props = [
        "vacuum_mask",
        "num_vacuum",
        "vacuum_charge",
        "vacuum_volume",
        "structure"
        ]
    _reset_props = []

    def __init__(
        self,
        charge_grid: Grid,
        total_charge_grid: Grid | None = None,
        reference_grid: Grid | None = None,
        vacuum_tol: float | bool = 1.0e-3,
        **kwargs,
    ):

        self._charge_grid = charge_grid
        
        # if no total charge is provided, use the base charge grid
        if total_charge_grid is None:
            total_charge_grid = charge_grid
        
        # if no reference grid is provided, use the total charge grid
        if reference_grid is None:
            reference_grid = total_charge_grid
            
        # ensure all grids have the same shape
        for x in range(3):
            assert charge_grid.shape[x] == total_charge_grid.shape[x] == reference_grid.shape[x], "Differing grid sizes found. All grids must have the same shape."

        self._total_charge_grid = total_charge_grid
        self._reference_grid = reference_grid

        # check if the total charge grid has values below 0
        if vacuum_tol is not False:
            assert total_charge_grid.total.min() >= 0, "The charge grid used to detect vacuum has values below 0. This typically results from too low of a grid density and causes incorrect partitions."

        # if vacuum tolerance is True, set it to the same default as above
        if vacuum_tol is True:
            self._vacuum_tol = 1.0e-3
        else:
            self._vacuum_tol = vacuum_tol

        # set hidden class variables. This allows us to cache properties and
        # still be able to recalculate them if needed, though that should only
        # be done by advanced users
        self._reset_properties()

    ###########################################################################
    # Set Properties
    ###########################################################################

    def _reset_properties(
        self,
        include_properties: list[str] = None,
        exclude_properties: list[str] = [],
    ):
        if include_properties is None:
            include_properties = self._reset_props + self._base_reset_props
        # get our final list of properties
        reset_properties = [
            i for i in include_properties if i not in exclude_properties
        ]
        # set corresponding hidden variable to None
        for prop in reset_properties:
            setattr(self, f"_{prop}", None)

    @property
    def structure(self) -> Structure:
        """

        Returns
        -------
        Structure
            The pymatgen structure basins are assigned to.

        """
        if self._structure is None:
            self._structure = self.reference_grid.structure.copy()
            self._structure.relabel_sites(ignore_uniq=True)
        return self._structure
    
    @property
    def species(self) -> list[str]:
        """

        Returns
        -------
        list[str]
            The species of each atom in the structure.
        """
        return [i.specie.symbol for i in self.structure]

    @property
    def charge_grid(self) -> Grid:
        """

        Returns
        -------
        Grid
            A Grid object with the charge density that will be integrated.

        """
        return self._charge_grid

    @charge_grid.setter
    def charge_grid(self, value: Grid):
        self._charge_grid = value
        self._reset_properties()
        
    @property
    def total_charge_grid(self) -> Grid:
        """

        Returns
        -------
        Grid
            A Grid object whose values are used to determine vacuum regions.

        """
        return self._total_charge_grid
    
    @total_charge_grid.setter
    def total_charge_grid(self, value: Grid):
        self._total_charge_grid = value
        self._reset_properties()

    @property
    def reference_grid(self) -> Grid:
        """

        Returns
        -------
        Grid
            A Grid object whose values are used to construct the basins.

        """
        return self._reference_grid

    @reference_grid.setter
    def reference_grid(self, value: Grid):
        self._reference_grid = value
        self._reset_properties()

    @property
    def vacuum_tol(self) -> float | bool:
        """

        Returns
        -------
        float
            The value below which a point will be considered part of the vacuum.
            The default is 0.001.

        """
        return self._vacuum_tol

    @vacuum_tol.setter
    def vacuum_tol(self, value: float | bool):
        self._vacuum_tol = value
        self._vacuum_mask = None

    ###########################################################################
    # Calculated Properties
    ###########################################################################

    @property
    def vacuum_charge(self) -> float:
        """

        Returns
        -------
        float
            The charge assigned to the vacuum.

        """
        if self._vacuum_charge is None:
            self._vacuum_charge = self.charge_grid.total[self.vacuum_mask].sum() / self.charge_grid.ngridpts
        return round(self._vacuum_charge, 10)

    @property
    def vacuum_volume(self) -> float:
        """

        Returns
        -------
        float
            The total volume assigned to the vacuum.

        """
        if self._vacuum_volume is None:
            self._vacuum_volume = (self.num_vacuum / self.reference_grid.ngridpts) * self.structure.volume
        return round(self._vacuum_volume, 10)

    @property
    def vacuum_mask(self) -> NDArray[bool]:
        """

        Returns
        -------
        NDArray[bool]
            A mask representing the voxels that belong to the vacuum.

        """
        if self._vacuum_mask is None:
            # if vacuum tolerance is set to False, ignore vacuum
            if self.vacuum_tol is False:
                self._vacuum_mask = np.zeros(self.total_charge_grid.shape, dtype=np.bool_)
            else:
                # get vacuum mask
                self._vacuum_mask = self.total_charge_grid.total < (
                    self.vacuum_tol * self.structure.volume  # normalize
                )
        return self._vacuum_mask

    @property
    def num_vacuum(self) -> int:
        """

        Returns
        -------
        int
            The number of vacuum points in the array

        """
        if self._num_vacuum is None:
            self._num_vacuum = np.count_nonzero(self.vacuum_mask)
        return self._num_vacuum

    ###########################################################################
    # From Methods
    ###########################################################################
    
    @classmethod
    def from_vasp(
        cls,
        charge_filename: Path | str = "CHGCAR",
        total_charge_filename: Path | None | str = None,
        reference_filename: Path | None | str = None,
        total_only: bool = True,
        **kwargs,
    ) -> Self:
        """
        Creates a Bader class object from VASP files.

        Parameters
        ----------
        charge_filename : Path | str, optional
            The path to the CHGCAR like file that will be used for integrating charge.
            The default is "CHGCAR".
        total_charge_filename : Grid | None, optional
            The path to the CHGCAR like file used for determining vacuum regions 
            in the system. For pseudopotential codes this represents the total 
            electron density and should be provided whenever possible. 
            If None, defaults to the charge_grid.
        reference_filename : Path | None | str, optional
            The path to CHGCAR like file that will be used for partitioning.
            If None, the total charge file will be used for partitioning.
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
        charge_grid = Grid.from_vasp(charge_filename, total_only=total_only)
        
        if total_charge_filename is None:
            total_charge_grid = None
        else:
            total_charge_grid = Grid.from_vasp(total_charge_filename, total_only=total_only)
            
        if reference_filename is None:
            reference_grid = None
        else:
            reference_grid = Grid.from_vasp(reference_filename, total_only=total_only)

        return cls(charge_grid=charge_grid, total_charge_grid=total_charge_grid, reference_grid=reference_grid, **kwargs)

    @classmethod
    def from_cube(
        cls,
        charge_filename: Path | str,
        total_charge_filename: Path | None | str = None,
        reference_filename: Path | None | str = None,
        **kwargs,
    ) -> Self:
        """
        Creates a Bader class object from .cube files.

        Parameters
        ----------
        charge_filename : Path | str, optional
            The path to the .cube like file that will be used for integrating charge.
            The default is "CHGCAR".
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
        charge_grid = Grid.from_cube(charge_filename)
        if total_charge_filename is None:
            total_charge_grid = None
        else:
            total_charge_grid = Grid.from_cube(total_charge_filename)
        if reference_filename is None:
            reference_grid = None
        else:
            reference_grid = Grid.from_cube(reference_filename)

        return cls(charge_grid=charge_grid, total_charge_grid=total_charge_grid, reference_grid=reference_grid, **kwargs)

    @classmethod
    def from_dynamic(
        cls,
        charge_filename: Path | str,
        total_charge_filename: Path | None | str = None,
        reference_filename: Path | None | str = None,
        format: Literal["vasp", "cube", None] = None,
        total_only: bool = True,
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
        total_charge_filename : Grid | None, optional
            The path to the file used for determining vacuum regions 
            in the system. For pseudopotential codes this represents the total 
            electron density and should be provided whenever possible. 
            If None, defaults to the charge_grid.
        reference_filename : Path | None | str, optional
            The path to the file that will be used for partitioning.
            If None, defaults to the total charge grid.
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
        charge_grid = Grid.from_dynamic(charge_filename, format=format, total_only=total_only)
        if total_charge_filename is None:
            total_charge_grid = None
        else:
            total_charge_grid = Grid.from_dynamic(total_charge_filename, format=format, total_only=total_only)
        if reference_filename is None:
            reference_grid = None
        else:
            reference_grid = Grid.from_dynamic(reference_filename, format=format, total_only=total_only)

        return cls(charge_grid=charge_grid, total_charge_grid=total_charge_grid, reference_grid=reference_grid, **kwargs)

    def copy(self) -> Self:
        """

        Returns
        -------
        Self
            A deep copy of this Bader object.

        """
        return copy.deepcopy(self)

    ###########################################################################
    # Summary Methods
    ###########################################################################

    @abstractclassmethod
    def to_dict(
        self,
        use_json: bool = True,
    ) -> dict:
        """

        Gets a summary dictionary of the analysis. Must be overwritten.

        """

        raise NotImplementedError()

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

    def write_json(self, filepath: Path | str, **kwargs) -> None:
        """
        Writes results of the analysis to file in a JSON format.

        Parameters
        ----------
        filepath : Path | str
            The Path to write the results to.
        **kwargs : dict
            keyword arguments for the to_dict method.

        """
        filepath = Path(filepath)
        with open(filepath, "w") as json_file:
            json.dump(self.to_dict(use_json=True, **kwargs), json_file, indent=4)
            
    ###########################################################################
    # Volume Writing Methods
    ###########################################################################
    
    def _write_volume(
        self,
        volume_mask: NDArray[bool],
        filename: str | Path = "CHGCAR",
        suffix: str = "",
        write_grid: Literal["charge_grid", "total_charge_grid", "reference_grid"] = "charge_grid",
        output_format: str | Format = None,
        **writer_kwargs,
    ):
        """
        Writes the values in the provided mask for the requested grid.
        
        Parameters
        ----------
        volume_mask : NDArray[bool]
            A 3D array of the same shape as the grids that is True where values
            should be included.
        filepath : str | Path, optional
            The path to write the file to. Defaults to 'CHGCAR'
        suffix : str, optional
            A suffix to add to the path. Defaults to ''.
        write_grid : Literal["charge_grid", "total_charge_grid", "reference_grid"], optional
            The property name of the grid to write to file. Defaults to 'charge_grid'
        output_format : str | Format, optional
            The format to write with. If None, writes to source format stored in
            the Grid objects metadata.
            Defaults to None.

        """
        # ensure filename is a Path object
        filename = Path(filename)
        # add suffix
        filename = filename.with_name(
            filename.stem + suffix + "".join(filename.suffixes)
        )
        
        # get the data to use
        grid = getattr(self, write_grid, None)
        if grid is None:
            logging.warning("Provided grid name does not exist. Defaulting to 'charge_grid'")
            grid = self.charge_grid
            
        data_array = grid.total
        data_type = grid.data_type
        
        # copy data to avoid overwriting. Set data off of basin to 0
        data_array_copy = data_array.copy()
        data_array_copy[~volume_mask] = 0.0
        
        # create a new grid object with the masked data
        grid = Grid(
            structure=self.structure,
            data={"total": data_array_copy},
            data_type=data_type,
        )

        # write file
        grid.write(filename=filename, output_format=output_format, **writer_kwargs)

