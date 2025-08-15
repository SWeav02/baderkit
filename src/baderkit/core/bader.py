# -*- coding: utf-8 -*-

import copy
import importlib
import logging
from pathlib import Path
from typing import Literal, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from rich.progress import track

from baderkit.core.methods import method_names
from baderkit.core.methods.shared_numba import get_edges
from baderkit.core.toolkit import Grid, Structure

# This allows for Self typing and is compatible with python 3.10
Self = TypeVar("Self", bound="Bader")


class Bader:
    """
    Class for running Bader analysis on a regular grid. For information on each
    method, see our [docs](https://sweav02.github.io/baderkit/)
    """

    def __init__(
        self,
        charge_grid: Grid,
        reference_grid: Grid,
        method: str = "neargrid",
        vacuum_tol: float = 1.0e-3,
        normalize_vacuum: bool = True,
        basin_tol: float = 1.0e-3,
    ):
        """

        Parameters
        ----------
        charge_grid : Grid
            A Grid object with the charge density that will be integrated.
        reference_grid : Grid
            A grid object whose values will be used to construct the basins.
        method : str, optional
            The algorithm to use for generating bader basins. If None, defaults
            to neargrid.
        vacuum_tol: float, optional
            The value below which a point will be considered part of the vacuum.
            The default is 0.001.
        normalize_vacuum: bool, optional
            Whether or not the reference data needs to be converted to real space
            units for vacuum tolerance comparison. This should be set to True if
            the data follows VASP's CHGCAR standards, but False if the data should
            be compared as is (e.g. in ELFCARs)
        basin_tol: float, optional
            The value below which a basin will not be considered significant. This
            is used to avoid writing out data that is likely not valuable.
            The default is 0.001.

        Returns
        -------
        None.

        """
        # ensure th method is valid
        if method not in method_names:
            raise ValueError(
                f"Invalid method '{method}'. Available options are: {method_names}"
            )

        self._charge_grid = charge_grid
        self._reference_grid = reference_grid
        self._method = method
        self._vacuum_tol = vacuum_tol
        self._normalize_vacuum = normalize_vacuum
        self._basin_tol = basin_tol

        # set hidden class variables. This allows us to cache properties and
        # still be able to recalculate them if needed, though that should only
        # be done by advanced users
        self._reset_properties()
        self._use_overdetermined = False

    ###########################################################################
    # Set Properties
    ###########################################################################
    def _reset_properties(
        self,
        include_properties: list[str] = None,
        exclude_properties: list[str] = [],
    ):
        # if include properties is not provided, we wnat to reset everything
        if include_properties is None:
            include_properties = [
                # assigned by run_bader
                "basin_labels",
                "basin_maxima_frac",
                "basin_maxima_vox",
                "basin_charges",
                "basin_volumes",
                "vacuum_charge",
                "vacuum_volume",
                "significant_basins",
                "vacuum_mask",
                "num_vacuum",
                # Assigned by calling the property
                "basin_surface_distances",
                "basin_edges",
                "atom_edges",
                "structure",
                # Assigned by run_atom_assignment
                "basin_atoms",
                "basin_atom_dists",
                "atom_labels",
                "atom_charges",
                "atom_volumes",
                "atom_surface_distances",
                "total_electron_number",
            ]
        # get our final list of properties
        reset_properties = [
            i for i in include_properties if i not in exclude_properties
        ]
        # set corresponding hidden variable to None
        for prop in reset_properties:
            setattr(self, f"_{prop}", None)

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
    def reference_grid(self) -> Grid:
        """

        Returns
        -------
        Grid
            A grid object whose values will be used to construct the basins.

        """
        return self._reference_grid

    @reference_grid.setter
    def reference_grid(self, value: Grid):
        self._reference_grid = value
        self._reset_properties()

    @property
    def method(self) -> str:
        """

        Returns
        -------
        str
            The algorithm to use for generating bader basins. If None, defaults
            to neargrid.

        """
        return self._method

    @method.setter
    def method(self, value: str):
        self._method = value
        self._reset_properties(exclude_properties=["vacuum_mask", "num_vacuum"])

    @property
    def vacuum_tol(self) -> float:
        """

        Returns
        -------
        float
            The value below which a point will be considered part of the vacuum.
            The default is 0.001.

        """
        return self._vacuum_tol

    @vacuum_tol.setter
    def vacuum_tol(self, value: float):
        self._vacuum_tol = value
        self._reset_properties()
        # TODO: only reset everything if the vacuum actually changes

    @property
    def normalize_vacuum(self) -> bool:
        """

        Returns
        -------
        bool
            Whether or not the reference data needs to be converted to real space
            units for vacuum tolerance comparison. This should be set to True if
            the data follows VASP's CHGCAR standards, but False if the data should
            be compared as is (e.g. in ELFCARs)

        """
        return self._normalize_vacuum

    @normalize_vacuum.setter
    def normalize_vacuum(self, value: bool) -> bool:
        self._normalize_vacuum = value
        self._reset_properties()
        # TODO: only reset everything if the vacuum actually changes

    @property
    def basin_tol(self) -> float:
        """

        Returns
        -------
        float
            The value below which a basin will not be considered significant. This
            is used to avoid writing out data that is likely not valuable.
            The default is 0.001.

        """
        return self._basin_tol

    @basin_tol.setter
    def basin_tol(self, value: float):
        self._basin_tol = value
        self._reset_properties(include_properties=["significant_basins"])

    ###########################################################################
    # Calculated Properties
    ###########################################################################

    @property
    def basin_labels(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            A 3D array of the same shape as the reference grid with entries
            representing the basin the voxel belongs to. Note that for some
            methods (e.g. weight) the voxels have weights for each basin.
            These will be stored in the basin_weights property.

        """
        if self._basin_labels is None:
            self.run_bader()
        return self._basin_labels

    @property
    def basin_maxima_frac(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The fractional coordinates of each attractor.

        """
        if self._basin_maxima_frac is None:
            self.run_bader()
        return self._basin_maxima_frac

    @property
    def basin_maxima_vox(self) -> NDArray[int]:
        """

        Returns
        -------
        NDArray[int]
            The voxel coordinates of each attractor. There may be more of these
            than the fractional coordinates, as some maxima sit exactly between
            several voxels.

        """
        if self._basin_maxima_vox is None:
            self.run_bader()
        return self._basin_maxima_vox

    @property
    def basin_charges(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The charges assigned to each attractor.

        """
        if self._basin_charges is None:
            self.run_bader()
        return self._basin_charges

    @property
    def basin_volumes(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The volume assigned to each attractor.

        """
        if self._basin_volumes is None:
            self.run_bader()
        return self._basin_volumes

    @property
    def basin_surface_distances(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The distance from each basin maxima to the nearest point on
            the basins surface

        """
        if self._basin_surface_distances is None:
            self._get_basin_surface_distances()
        return self._basin_surface_distances

    @property
    def basin_atoms(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The atom index of each basin is assigned to.

        """
        if self._basin_atoms is None:
            self.run_atom_assignment()
        return self._basin_atoms

    @property
    def basin_atom_dists(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The distance from each attractor to the nearest atom

        """
        if self._basin_atom_dists is None:
            self.run_atom_assignment()
        return self._basin_atom_dists

    @property
    def significant_basins(self) -> NDArray[bool]:
        """

        Returns
        -------
        NDArray[bool]
            A 1D mask with an entry for each basin that is True where basins
            are significant.

        """
        if self._significant_basins is None:
            self._significant_basins = self.basin_charges > self.basin_tol
        return self._significant_basins

    @property
    def atom_labels(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            A 3D array of the same shape as the reference grid with entries
            representing the atoms the voxel belongs to.

            Note that for some methods (e.g. weight) some voxels have fractional
            assignments for each basin and this will not represent exactly how
            charges are assigned.

        """
        if self._atom_labels is None:
            self.run_atom_assignment()
        return self._atom_labels

    @property
    def atom_charges(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The charge assigned to each atom

        """
        if self._atom_charges is None:
            self.run_atom_assignment()
        return self._atom_charges

    @property
    def atom_volumes(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The volume assigned to each atom

        """
        if self._atom_volumes is None:
            self.run_atom_assignment()
        return self._atom_volumes

    @property
    def atom_surface_distances(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The distance from each atom to the nearest point on the atoms surface.

        """
        if self._atom_surface_distances is None:
            self._get_atom_surface_distances()
        return self._atom_surface_distances

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
    def basin_edges(self) -> NDArray[np.bool_]:
        """

        Returns
        -------
        NDArray[np.bool_]
            A mask with the same shape as the input grids that is True at points
            on basin edges.

        """
        if self._basin_edges is None:
            self._basin_edges = get_edges(
                labeled_array=self.basin_labels,
                vacuum_mask=np.zeros(self.basin_labels.shape, dtype=np.bool_),
                neighbor_transforms=self.reference_grid.voxel_26_neighbors[0],
            )
        return self._basin_edges

    @property
    def atom_edges(self) -> NDArray[np.bool_]:
        """

        Returns
        -------
        NDArray[np.bool_]
            A mask with the same shape as the input grids that is True at points
            on atom edges.

        """
        if self._atom_edges is None:
            self._atom_edges = get_edges(
                labeled_array=self.atom_labels,
                vacuum_mask=np.zeros(self.atom_labels.shape, dtype=np.bool_),
                neighbor_transforms=self.reference_grid.voxel_26_neighbors[0],
            )
        return self._atom_edges

    @property
    def vacuum_charge(self) -> float:
        """

        Returns
        -------
        float
            The charge assigned to the vacuum.

        """
        if self._vacuum_charge is None:
            self.run_bader()
        return self._vacuum_charge

    @property
    def vacuum_volume(self) -> float:
        """

        Returns
        -------
        float
            The total volume assigned to the vacuum.

        """
        if self._vacuum_volume is None:
            self.run_bader()
        return self._vacuum_volume

    @property
    def vacuum_mask(self) -> NDArray[bool]:
        """

        Returns
        -------
        NDArray[bool]
            A mask representing the voxels that belong to the vacuum.

        """
        if self._vacuum_mask is None:
            if self.normalize_vacuum:
                self._vacuum_mask = self.reference_grid.total < (
                    self.vacuum_tol * self.structure.volume
                )
            else:
                self._vacuum_mask = self.reference_grid.total < self.vacuum_tol
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

        return self.atom_charges.sum() + self.vacuum_charge

    @staticmethod
    def all_methods() -> list[str]:
        """

        Returns
        -------
        list[str]
            A list of the available methods.

        """

        return method_names

    @property
    def results_summary(self) -> dict:
        """

        Returns
        -------
        results_dict : dict
            A dictionary summary of all results

        """
        results_dict = {
            "method": self.method,
            "basin_maxima_frac": self.basin_maxima_frac,
            "basin_maxima_vox": self.basin_maxima_vox,
            "basin_charges": self.basin_charges,
            "basin_volumes": self.basin_volumes,
            "basin_surface_distances": self.basin_surface_distances,
            "basin_atoms": self.basin_atoms,
            "basin_atom_dists": self.basin_atom_dists,
            "atom_charges": self.atom_charges,
            "atom_volumes": self.atom_volumes,
            "atom_surface_distances": self.atom_surface_distances,
            "structure": self.structure,
            "vacuum_charge": self.vacuum_charge,
            "vacuum_volume": self.vacuum_volume,
            "significant_basins": self.significant_basins,
            "total_electron_num": self.total_electron_number,
        }
        return results_dict

    def run_bader(self) -> None:
        """
        Runs the entire Bader process and saves results to class variables.

        Returns
        -------
        None

        """
        # Normalize the method name to a module and class name
        module_name = self.method.replace(
            "-", "_"
        )  # 'pseudo-neargrid' -> 'pseudo_neargrid'
        class_name = (
            "".join(part.capitalize() for part in module_name.split("_")) + "Method"
        )

        # import method
        mod = importlib.import_module(f"baderkit.core.methods.{module_name}")
        Method = getattr(mod, class_name)

        # Instantiate and run the selected method
        method = Method(
            charge_grid=self.charge_grid,
            reference_grid=self.reference_grid,
            vacuum_mask=self.vacuum_mask,
            num_vacuum=self.num_vacuum,
        )
        if self._use_overdetermined:
            method._use_overdetermined = True
        results = method.run()

        for key, value in results.items():
            setattr(self, f"_{key}", value)

    def run_atom_assignment(self, structure: Structure = None):
        """
        Assigns bader basins to the atoms in the provided structure.

        Parameters
        ----------
        structure : Structure, optional
            If provided, basins will be assigned to the atoms in this structure.
            The default is None.

        Returns
        -------
        None.

        """
        # Default structure
        structure = structure or self.structure
        self._structure = structure

        # Shorthand access
        basins = self.basin_maxima_frac  # (N_basins, 3)
        atoms = structure.frac_coords  # (N_atoms, 3)
        L = structure.lattice.matrix  # (3, 3)
        N_basins, N_atoms = len(basins), len(atoms)

        logging.info("Assigning atom properties")

        # Vectorized deltas, minimum‑image wrapping
        diffs = atoms[None, :, :] - basins[:, None, :]
        diffs += np.where(diffs <= -0.5, 1, 0)
        diffs -= np.where(diffs >= 0.5, 1, 0)

        # Cartesian diffs & distances
        cart = np.einsum("bij,jk->bik", diffs, L)
        dists = np.linalg.norm(cart, axis=2)

        # Basin→atom assignment & distances
        basin_atoms = np.argmin(dists, axis=1)  # (N_basins,)
        basin_atom_dists = dists[np.arange(N_basins), basin_atoms]  # (N_basins,)

        # Atom labels per grid point
        # NOTE: append -1 so that vacuum gets assigned to -1 in the atom_labels
        # array
        basin_atoms = np.insert(basin_atoms, len(basin_atoms), -1)
        atom_labels = basin_atoms[self.basin_labels]

        # Sum up charges/volumes per atom in one shot. slice with -1 is necessary
        # to prevent no negative value error
        try:
            atom_charges = np.bincount(
                basin_atoms[:-1], weights=self.basin_charges, minlength=N_atoms
            )
            atom_volumes = np.bincount(
                basin_atoms[:-1], weights=self.basin_volumes, minlength=N_atoms
            )
        except:
            breakpoint()
        # Store everything
        self._basin_atoms = basin_atoms[:-1]
        self._basin_atom_dists = basin_atom_dists
        self._atom_labels = atom_labels
        self._atom_charges = atom_charges
        self._atom_volumes = atom_volumes

    def _get_atom_surface_distances(self):
        """
        Calculates the distance from each atom to the nearest surface. This is
        automatically called during the atom assignment and generally should
        not be called manually.

        Returns
        -------
        None.

        """
        atom_labeled_voxels = self.atom_labels
        atom_radii = []
        # NOTE: We don't use the atom_edges property because its usually not
        # needed by users and takes up more space in memory
        edge_mask = self.atom_edges
        for atom_index in track(
            range(len(self.structure)), description="Calculating atom radii"
        ):
            # get the voxels corresponding to the interior edge of this basin
            atom_edge_mask = (atom_labeled_voxels == atom_index) & edge_mask
            edge_vox_coords = np.argwhere(atom_edge_mask)
            # convert to frac coords
            edge_frac_coords = self.reference_grid.get_frac_coords_from_vox(
                edge_vox_coords
            )
            atom_frac_coord = self.structure.frac_coords[atom_index]
            # Get the difference in coords between atom and edges
            coord_diff = atom_frac_coord - edge_frac_coords
            # Wrap any coords that are more than 0.5 or less than -0.5
            coord_diff -= np.round(coord_diff)
            # Convert to cartesian coordinates
            cart_coords = self.reference_grid.get_cart_coords_from_frac(coord_diff)
            # Calculate distance of each
            norm = np.linalg.norm(cart_coords, axis=1)
            if len(norm) == 0:
                logging.warning(f"No volume assigned to atom at site {atom_index}.")
                atom_radii.append(0)
            else:
                atom_radii.append(norm.min())
        atom_radii = np.array(atom_radii)
        self._atom_surface_distances = atom_radii

    def _get_basin_surface_distances(self):
        """
        Calculates the distance from each basin maxima to the nearest surface.
        This is automatically called during the atom assignment and generally
        should not be called manually.

        Returns
        -------
        None.

        """
        basin_labeled_voxels = self.basin_labels
        basin_radii = []
        # NOTE: We don't use the basin_edges property because its usually not
        # needed by users and takes up more space in memory
        edge_mask = self.basin_edges
        for basin in track(
            range(len(self.basin_maxima_frac)), description="Calculating feature radii"
        ):
            # We only calculate the edges for significant basins
            if not self.significant_basins[basin]:
                basin_radii.append(0.0)
                continue
            basin_edge_mask = (basin_labeled_voxels == basin) & edge_mask
            edge_vox_coords = np.argwhere(basin_edge_mask)
            edge_frac_coords = self.reference_grid.get_frac_coords_from_vox(
                edge_vox_coords
            )
            basin_frac_coord = self.basin_maxima_frac[basin]

            coord_diff = basin_frac_coord - edge_frac_coords
            coord_diff -= np.round(coord_diff)
            cart_coords = self.reference_grid.get_cart_coords_from_frac(coord_diff)
            norm = np.linalg.norm(cart_coords, axis=1)
            basin_radii.append(norm.min())
        basin_radii = np.array(basin_radii)
        self._basin_surface_distances = basin_radii

    @classmethod
    def from_vasp(
        cls,
        charge_filename: Path | str = "CHGCAR",
        reference_filename: Path | None | str = None,
        **kwargs,
    ) -> Self:
        """
        Creates a Bader class object from VASP files.

        Parameters
        ----------
        charge_filename : Path | str, optional
            The path to the CHGCAR like file that will be used for summing charge.
            The default is "CHGCAR".
        reference_filename : Path | None | str, optional
            The path to CHGCAR like file that will be used for partitioning.
            If None, the charge file will be used for partitioning.
        **kwargs : dict
            Keyword arguments to pass to the Bader class.

        Returns
        -------
        Self
            A Bader class object.

        """
        charge_grid = Grid.from_vasp(charge_filename)
        if reference_filename is None:
            reference_grid = charge_grid.copy()
        else:
            reference_grid = Grid.from_vasp(reference_filename)
        return cls(charge_grid=charge_grid, reference_grid=reference_grid, **kwargs)

    @classmethod
    def from_cube(
        cls,
        charge_filename: Path | str,
        reference_filename: Path | None | str = None,
        **kwargs,
    ) -> Self:
        """
        Creates a Bader class object from .cube files.

        Parameters
        ----------
        charge_filename : Path | str, optional
            The path to the .cube file that will be used for summing charge.
        reference_filename : Path | None | str, optional
            The path to .cube file that will be used for partitioning.
            If None, the charge file will be used for partitioning.
        **kwargs : dict
            Keyword arguments to pass to the Bader class.

        Returns
        -------
        Self
            A Bader class object.

        """
        charge_grid = Grid.from_cube(charge_filename)
        if reference_filename is None:
            reference_grid = charge_grid.copy()
        else:
            reference_grid = Grid.from_cube(reference_filename)
        return cls(charge_grid=charge_grid, reference_grid=reference_grid, **kwargs)

    @classmethod
    def from_dynamic(
        cls,
        charge_filename: Path | str,
        reference_filename: Path | None | str = None,
        format: Literal["vasp", "cube", None] = None,
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
        reference_filename : Path | None | str, optional
            The path to the file that will be used for partitioning.
            If None, the charge file will be used for partitioning.
        format : Literal["vasp", "cube", None], optional
            The format of the grids to read in. If None, the formats will be
            guessed from the file names.
        **kwargs : dict
            Keyword arguments to pass to the Bader class.

        Returns
        -------
        Self
            A Bader class object.

        """

        charge_grid = Grid.from_dynamic(charge_filename, format=format)
        if reference_filename is None:
            reference_grid = charge_grid.copy()
        else:
            reference_grid = Grid.from_dynamic(reference_filename, format=format)
        return cls(charge_grid=charge_grid, reference_grid=reference_grid, **kwargs)

    def copy(self) -> Self:
        """

        Returns
        -------
        Self
            A deep copy of this Bader object.

        """
        return copy.deepcopy(self)

    def write_basin_volumes(
        self,
        basin_indices: NDArray,
        directory: str | Path = None,
        file_prefix: str = "CHGCAR",
        data_type: Literal["charge", "reference"] = "charge",
    ):
        """
        Writes bader basins to vasp-like files. Points belonging to the basin
        will have values from the charge or reference grid, and all other points
        will be 0. Filenames are written as {file_prefix}_b{i} where i is the
        basin index.

        Parameters
        ----------
        basin_indices : NDArray
            The list of basin indices to write
        directory : str | Path
            The directory to write the files in. If None, the active directory
            is used.
        file_prefix : str, optional
            The string to append to each file name. The default is "CHGCAR".
        data_type : Literal["charge", "reference"], optional
            Which file to write from. The default is "charge".

        Returns
        -------
        None.

        """
        if data_type == "charge":
            grid = self.charge_grid.copy()
        elif data_type == "reference":
            grid = self.reference_grid.copy()

        data_array = grid.total
        if directory is None:
            directory = Path(".")
        for basin in basin_indices:
            mask = self.basin_labels == basin
            data_array_copy = data_array.copy()
            data_array_copy[~mask] = 0
            data = {"total": data_array_copy}
            grid = Grid(structure=self.structure, data=data)
            grid.write_file(directory / f"{file_prefix}_b{basin}")

    def write_all_basin_volumes(
        self,
        directory: str | Path = None,
        file_prefix: str = "CHGCAR",
        data_type: Literal["charge", "reference"] = "charge",
    ):
        """
        Writes all bader basins to vasp-like files. Points belonging to the basin
        will have values from the charge or reference grid, and all other points
        will be 0. Filenames are written as {file_prefix}_b{i} where i is the
        basin index.

        Parameters
        ----------
        directory : str | Path
            The directory to write the files in. If None, the active directory
            is used.
        file_prefix : str, optional
            The string to append to each file name. The default is "CHGCAR".
        data_type : Literal["charge", "reference"], optional
            Which file to write from. The default is "charge".

        Returns
        -------
        None.

        """
        basin_indices = np.where(self.significant_basins)[0]
        self.write_basin_volumes(
            basin_indices=basin_indices,
            directory=directory,
            file_prefix=file_prefix,
            data_type=data_type,
        )

    def write_basin_volumes_sum(
        self,
        basin_indices: NDArray,
        directory: str | Path = None,
        file_prefix: str = "CHGCAR",
        data_type: Literal["charge", "reference"] = "charge",
    ):
        """
        Writes the union of the provided bader basins to vasp-like files.
        Points belonging to the basins will have values from the charge or
        reference grid, and all other points will be 0. Filenames are written
        as {file_prefix}_bsum.

        Parameters
        ----------
        basin_indices : NDArray
            The list of basin indices to sum and write
        directory : str | Path
            The directory to write the files in. If None, the active directory
            is used.
        file_prefix : str, optional
            The string to append to each file name. The default is "CHGCAR".
        data_type : Literal["charge", "reference"], optional
            Which file to write from. The default is "charge".

        Returns
        -------
        None.

        """
        if data_type == "charge":
            grid = self.charge_grid.copy()
        elif data_type == "reference":
            grid = self.reference_grid.copy()

        data_array = grid.total
        if directory is None:
            directory = Path(".")
        mask = np.isin(self.basin_labels, basin_indices)
        data_array_copy = data_array.copy()
        data_array_copy[~mask] = 0
        data = {"total": data_array_copy}
        grid = Grid(structure=self.structure, data=data)
        grid.write_file(directory / f"{file_prefix}_bsum")

    def write_atom_volumes(
        self,
        atom_indices: NDArray,
        directory: str | Path = None,
        file_prefix: str = "CHGCAR",
        data_type: Literal["charge", "reference"] = "charge",
    ):
        """
        Writes atomic basins to vasp-like files. Points belonging to the atom
        will have values from the charge or reference grid, and all other points
        will be 0. Filenames are written as {file_prefix}_a{i} where i is the
        atom index.

        Parameters
        ----------
        atom_indices : NDArray
            The list of atom indices to write
        directory : str | Path
            The directory to write the files in. If None, the active directory
            is used.
        file_prefix : str, optional
            The string to append to each file name. The default is "CHGCAR".
        data_type : Literal["charge", "reference"], optional
            Which file to write from. The default is "charge".

        Returns
        -------
        None.

        """
        if data_type == "charge":
            grid = self.charge_grid.copy()
        elif data_type == "reference":
            grid = self.reference_grid.copy()

        data_array = grid.total
        if directory is None:
            directory = Path(".")
        for atom_index in atom_indices:
            mask = self.atom_labels == atom_index
            data_array_copy = data_array.copy()
            data_array_copy[~mask] = 0
            data = {"total": data_array_copy}
            grid = Grid(structure=self.structure, data=data)
            grid.write_file(directory / f"{file_prefix}_a{atom_index}")

    def write_all_atom_volumes(
        self,
        directory: str | Path = None,
        file_prefix: str = "CHGCAR",
        data_type: Literal["charge", "reference"] = "charge",
    ):
        """
        Writes all atomic basins to vasp-like files. Points belonging to the atom
        will have values from the charge or reference grid, and all other points
        will be 0. Filenames are written as {file_prefix}_a{i} where i is the
        atom index.

        Parameters
        ----------
        directory : str | Path
            The directory to write the files in. If None, the active directory
            is used.
        file_prefix : str, optional
            The string to append to each file name. The default is "CHGCAR".
        data_type : Literal["charge", "reference"], optional
            Which file to write from. The default is "charge".

        Returns
        -------
        None.

        """
        atom_indices = np.array(range(len(self.structure)))
        self.write_atom_volumes(
            atom_indices=atom_indices,
            directory=directory,
            file_prefix=file_prefix,
            data_type=data_type,
        )

    def write_atom_volumes_sum(
        self,
        atom_indices: NDArray,
        directory: str | Path = None,
        file_prefix: str = "CHGCAR",
        data_type: Literal["charge", "reference"] = "charge",
    ):
        """
        Writes the union of the provided atom basins to vasp-like files.
        Points belonging to the atoms will have values from the charge or
        reference grid, and all other points will be 0. Filenames are written
        as {file_prefix}_asum.

        Parameters
        ----------
        atom_indices : NDArray
            The list of atom indices to sum and write
        directory : str | Path
            The directory to write the files in. If None, the active directory
            is used.
        file_prefix : str, optional
            The string to append to each file name. The default is "CHGCAR".
        data_type : Literal["charge", "reference"], optional
            Which file to write from. The default is "charge".

        Returns
        -------
        None.

        """
        if data_type == "charge":
            grid = self.charge_grid.copy()
        elif data_type == "reference":
            grid = self.reference_grid.copy()

        data_array = grid.total
        if directory is None:
            directory = Path(".")
        mask = np.isin(self.atom_labels, atom_indices)
        data_array_copy = data_array.copy()
        data_array_copy[~mask] = 0
        data = {"total": data_array_copy}
        grid = Grid(structure=self.structure, data=data)
        grid.write_file(directory / f"{file_prefix}_asum")

    def get_atom_results_dataframe(self) -> pd.DataFrame:
        """
        Collects a summary of results for the atoms in a pandas DataFrame.

        Returns
        -------
        atoms_df : pd.DataFrame
            A table summarizing the atomic basins.

        """
        # Get atom results summary
        atom_frac_coords = self.structure.frac_coords
        atoms_df = pd.DataFrame(
            {
                "label": self.structure.labels,
                "x": atom_frac_coords[:, 0],
                "y": atom_frac_coords[:, 1],
                "z": atom_frac_coords[:, 2],
                "charge": self.atom_charges,
                "volume": self.atom_volumes,
                "surface_dist": self.atom_surface_distances,
            }
        )
        return atoms_df

    def get_basin_results_dataframe(self):
        """
        Collects a summary of results for the basins in a pandas DataFrame.

        Returns
        -------
        basin_df : pd.DataFrame
            A table summarizing the basins.

        """
        subset = self.significant_basins
        basin_frac_coords = self.basin_maxima_frac[subset]
        basin_df = pd.DataFrame(
            {
                "atoms": np.array(self.structure.labels)[self.basin_atoms[subset]],
                "x": basin_frac_coords[:, 0],
                "y": basin_frac_coords[:, 1],
                "z": basin_frac_coords[:, 2],
                "charge": self.basin_charges[subset],
                "volume": self.basin_volumes[subset],
                "surface_dist": self.basin_surface_distances[subset],
                "atom_dist": self.basin_atom_dists[subset],
            }
        )
        return basin_df

    def write_results_summary(
        self,
        directory: Path | str | None = None,
    ):
        """
        Writes a summary of atom and basin results to .tsv files.

        Parameters
        ----------
        directory : str | Path
            The directory to write the files in. If None, the active directory
            is used.

        Returns
        -------
        None.

        """
        if directory is None:
            directory = Path(".")

        # Get atom results summary
        atoms_df = self.get_atom_results_dataframe()
        formatted_atoms_df = atoms_df.copy()
        numeric_cols = formatted_atoms_df.select_dtypes(include="number").columns
        formatted_atoms_df[numeric_cols] = formatted_atoms_df[numeric_cols].map(
            lambda x: f"{x:.5f}"
        )

        # Get basin results summary
        basin_df = self.get_basin_results_dataframe()
        formatted_basin_df = basin_df.copy()
        numeric_cols = formatted_basin_df.select_dtypes(include="number").columns
        formatted_basin_df[numeric_cols] = formatted_basin_df[numeric_cols].map(
            lambda x: f"{x:.5f}"
        )

        # Determine max width per column including header
        atom_col_widths = {
            col: max(len(col), formatted_atoms_df[col].map(len).max())
            for col in atoms_df.columns
        }
        basin_col_widths = {
            col: max(len(col), formatted_basin_df[col].map(len).max())
            for col in basin_df.columns
        }

        # Write to file with aligned columns using tab as separator
        for df, col_widths, name in zip(
            [formatted_atoms_df, formatted_basin_df],
            [atom_col_widths, basin_col_widths],
            ["bader_atom_summary.tsv", "bader_basin_summary.tsv"],
        ):
            with open(directory / name, "w") as f:
                # Write header
                header = "\t".join(f"{col:<{col_widths[col]}}" for col in df.columns)
                f.write(header + "\n")

                # Write rows
                for _, row in df.iterrows():
                    line = "\t".join(
                        f"{val:<{col_widths[col]}}" for col, val in row.items()
                    )
                    f.write(line + "\n")
                # write vacuum summary to atom file
                if name == "bader_atom_summary.tsv":
                    f.write("\n")
                    f.write(f"Vacuum Charge:\t\t{self.vacuum_charge:.5f}\n")
                    f.write(f"Vacuum Volume:\t\t{self.vacuum_volume:.5f}\n")
                    f.write(f"Total Electrons:\t{self.total_electron_number:.5f}\n")
