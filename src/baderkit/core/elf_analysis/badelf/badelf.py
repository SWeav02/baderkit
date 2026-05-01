# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Literal
from typing import TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.ndimage import label
from tqdm import tqdm
from pymatgen.analysis.local_env import CrystalNN

from baderkit.core.base.base_analysis import BaseAnalysis

from .badelf_numba import (
    get_badelf_assignments,
)

from baderkit.core.bader import Bader
from baderkit.core.elf_analysis import ElfLabeler
from baderkit.core.elf_analysis import ElfRadii
from baderkit.core.toolkit import Grid, Structure
from baderkit.core.utilities.basins import (
    get_edges,
    get_min_avg_surface_dists,
)
from baderkit.core.utilities.voronoi import get_cell_wrapped_voronoi

Self = TypeVar("Self", bound="Badelf")

class Badelf(BaseAnalysis):

    spin_system = "total"

    _summary_props = [
        "atom_charges",
        "atom_volumes",
        "oxidation_states",
        "nna_structure",
        "species",
        "nna_dim",
        "all_nna_dims",
        "all_nna_dim_cutoffs",
        "min_surface_distances",
        "avg_surface_distances",
        "num_nnas",
        "nnas_per_formula",
        "nnas_per_reduced_formula",
        "maxima_elf_values",
        ]

    _reset_props = [
        "partitioning_planes",
        "atom_labels",
        "elf_radii",
        "labeler",
        "bader",
        ] + _summary_props

    def __init__(
        self,
        reference_grid: Grid,
        charge_grid: Grid,
        total_charge_grid: Grid = None,
        method: Literal["badelf", "voronelf", "zero-flux"] = "zero-flux",
        shared_feature_splitting_method: Literal[
            "weighted_dist", "pauling", "equal", "dist", "nearest"
        ] = "weighted_dist",
        **kwargs,
    ):
        """
        Class for performing charge analysis using the electron localization function
        (ELF). For information on specific methods, see our [docs](https://sweav02.github.io/baderkit/).

        This class is designed only for single spin or total spin charge densities
        and ELF. For spin-dependent systems, use the SpinBadelf class instead.

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
        method : Literal["badelf", "voronelf", "zero-flux"], optional
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

        if method not in ["badelf", "voronelf", "zero-flux"]:
            raise ValueError(
                """The method setting you chose does not exist. Please select
                  either 'badelf', 'voronelf', or 'zero-flux'.
                  """
            )
        self._method = method
        self._shared_feature_splitting_method = shared_feature_splitting_method
        self._kwargs = kwargs

        super().__init__(
            charge_grid=charge_grid,
            total_charge_grid=total_charge_grid,
            reference_grid=reference_grid,
            **kwargs,
        )

    ###########################################################################
    # Settings
    ###########################################################################

    @property
    def method(self) -> str:
        """

        Returns
        -------
        str
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

        """
        return self._method

    @method.setter
    def method(self, value: str):
        self._method = value
        self._reset_properties()

    @property
    def shared_feature_splitting_method(self) -> str:
        """

        Returns
        -------
        str
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


        """
        return self._shared_feature_splitting_method

    @shared_feature_splitting_method.setter
    def shared_feature_splitting_method(self, value: str):
        self._shared_feature_splitting_method = value
        self._reset_properties()

    ###########################################################################
    # Convenience Properites
    ###########################################################################

    @property
    def bader(self) -> Bader:
        """

        Returns
        -------
        Bader
            The Bader class used to partition the ELF.

        """
        if self._bader is None:
            self._bader = self.labeler.elf_bader
        return self._bader

    @property
    def labeler(self) -> ElfLabeler:
        """

        Returns
        -------
        ElfLabeler
            The ElfLabeler class used to locate non-nuclear attractors.

        """
        if self._labeler is None:
            self._labeler = self.elf_radii.labeler
        return self._labeler

    @property
    def elf_radii(self) -> ElfRadii:
        """

        Returns
        -------
        ElfRadii
            The ElfRadii class used to calculate radii in the system.

        """
        if self._elf_radii is None:
            if self.method == "badelf":
                include_nnas = False
            elif self.method == "voronelf":
                include_nnas = False
            else:
                include_nnas = False

            self._elf_radii = ElfRadii(
                charge_grid=self.charge_grid,
                total_charge_grid=self.total_charge_grid,
                reference_grid=self.reference_grid,
                include_nnas = include_nnas,
                **self._kwargs,
            )
        return self._elf_radii

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
    def num_nnas(self) -> int:
        """

        Returns
        -------
        int
            The number of nna sites (nna maxima) present in the system.

        """
        return self.labeler.num_nnas

    @property
    def species(self) -> list[str]:
        """

        Returns
        -------
        list[str]
            The species of each atom/dummy atom in the nna structure. Covalent
            and metallic features are not included.

        """
        return [i.specie.symbol for i in self.labeler.nna_structure]

    ###########################################################################
    # Properties
    ###########################################################################

    @property
    def atom_charges(self) -> NDArray:
        """

        Returns
        -------
        NDArray
            The charge associated with each atom and nna site in the system.

        """
        if self._atom_charges is None:
            self._get_voxel_assignments()
        return self._atom_charges.round(10)

    @property
    def atom_volumes(self) -> NDArray:
        """

        Returns
        -------
        NDArray
            The volume associated with each atom and nna site in the system.

        """
        if self._atom_volumes is None:
            self._get_voxel_assignments()
        return self._atom_volumes.round(10)

    @property
    def maxima_elf_values(self) -> NDArray:
        """

        Returns
        -------
        NDArray
            The maximum ELF value for each atom and nna in the system.

        """
        if self._maxima_elf_values is None:
            mapping = self.elf_radii.label_atom_map
            max_vals = self.bader.maxima_ref_values
            final_vals = np.zeros(len(self.labeler.nna_structure), dtype=np.float64)
            for i in range(len(max_vals)):
                if max_vals[i] > final_vals[mapping[i]]:
                    final_vals[mapping[i]] = max_vals[i]
            self._maxima_elf_values = final_vals

        return self._maxima_elf_values

    @property
    def partitioning_planes(self) -> tuple | None:
        """

        Returns
        -------
        tuple | None
            The partitioning planes for each site in the structure as a tuple of
            arrays. The first array is the site the plane belongs to, the second
            is a point on the plane, and the third is the vector normal to the plane.
            None if the zero-flux method is selected.

        """

        if self.method == "zero-flux":
            return None, None, None

        if self._partitioning_planes is None:
            logging.info("Finding partitioning planes")

            # get bonding information
            bond_pairs = self.elf_radii.bonding_pairs
            site_indices = bond_pairs[0][:,0]
            # neigh_indices = bond_pairs[:,1]
            plane_points, plane_vectors = self.elf_radii.voronoi_planes


            # we want to transform our planes to the 26 nearest neighbor cells
            # to ensure that we cover our unit cell.
            # For speed, we can remove planes that contain the entire unit cell
            # and we can remove a full set of planes if none of them contain any
            # part of the unit cell.
            # Finally, we can sort each plane by how much of the unit cell it slices
            # such that planes that are likely to reject a grid point come first

            # first we get wrapped planes
            (
                site_indices,
                transforms,
                plane_points,
                plane_vectors,
                plane_atom_volumes,
            ) = get_cell_wrapped_voronoi(
                site_indices=site_indices,
                plane_points=plane_points,
                plane_vectors=plane_vectors,
            )

            # sort planes by site, transform, and volume.
            combined_sort = np.column_stack((plane_atom_volumes, transforms, site_indices))
            sorted_indices = np.lexsort(combined_sort.T)
            transforms = transforms[sorted_indices]
            plane_points = plane_points[sorted_indices]
            plane_vectors = plane_vectors[sorted_indices]
            plane_atom_volumes = plane_atom_volumes[sorted_indices]

            # get plane equations in cartesian coordinates
            plane_vectors = self.reference_grid.frac_to_cart(plane_vectors)
            plane_points = self.reference_grid.frac_to_cart(plane_points)

            # normalize vectors
            plane_vectors = (plane_vectors.T / np.linalg.norm(plane_vectors, axis=1)).T

            # calculate plane equations
            b = -np.einsum("ij,ij->i", plane_vectors, plane_points)

            plane_equations = np.column_stack((plane_vectors, b))

            self._partitioning_planes = (
                site_indices,
                transforms,
                plane_equations,
            )
        return self._partitioning_planes

    @property
    def atom_labels(self) -> NDArray:
        """

        Returns
        -------
        NDArray
            A 3D array with the same shape as the charge grid indicating
            which atom/nna each grid point is assigned to.

        """
        if self._atom_labels is None:
            self._get_voxel_assignments()
        return self._atom_labels

    @property
    def all_nna_dims(self) -> list | None:
        """

        Returns
        -------
        list
            The possible dimensions the nna takes on from an ELF value of
            0 to 1. If no nnas are present the value will be None.

        """
        if self._all_nna_dims is None:
            self._get_nna_dimensionality()
        # if there are no nnas we want to return None, but we don't want
        # to rerun the search each time. I mark the dims as -1 to avoid this
        if self._all_nna_dims == -1:
            return None
        return self._all_nna_dims

    @property
    def all_nna_dim_cutoffs(self) -> list:
        """

        Returns
        -------
        list
            The highest ELF value where each dimensionality in the "all_nna_dims"
            property exists.

        """
        if self._all_nna_dim_cutoffs is None:
            self._get_nna_dimensionality()
        if self._all_nna_dim_cutoffs == -1:
            return None
        return self._all_nna_dim_cutoffs

    @property
    def nna_dimensionality(self) -> int:
        """

        Returns
        -------
        int
            The dimensionality of the nna volume at a value of 0 ELF.

        """
        if self._nna_dim is None and self.all_nna_dims is not None:
            self._nna_dim = self.all_nna_dims[0]

        return self._nna_dim

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
    def min_surface_distances(self) -> NDArray:
        """

        Returns
        -------
        NDArray
            The minimum distance from each atom or nna center to the partioning
            surface.

        """
        if self._min_surface_distances is None:
            self._get_min_avg_surface_dists()
        return self._min_surface_distances.round(10)

    @property
    def avg_surface_distances(self) -> NDArray:
        """

        Returns
        -------
        NDArray
            The average distance from each atom or nna center to the partitioning
            surface.

        """
        if self._avg_surface_distances is None:
            self._get_min_avg_surface_dists()
        return self._avg_surface_distances.round(10)

    @property
    def nnas_per_formula(self) -> float:
        """

        Returns
        -------
        float
            The number of nna electrons for the full structure formula.

        """
        if self._nnas_per_formula is None:
            nnas_per_unit = 0
            for i in range(len(self.structure), len(self.labeler.nna_structure)):
                nnas_per_unit += self.charges[i]
            self._nnas_per_formula = nnas_per_unit
        return round(self._nnas_per_formula, 10)

    @property
    def nnas_per_reduced_formula(self) -> float:
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
    def nna_formula(self) -> str:
        """

        Returns
        -------
        str
            A string representation of the nna formula, rounding partial charge
            to the nearest integer.

        """
        return f"{self.structure.formula} e{round(self.nnas_per_formula)}"


    ###########################################################################
    # Assignment methods
    ###########################################################################

    def _get_zero_flux_assignments(self) -> tuple[NDArray]:
        """
        Assign charge from each feature to their associated atoms. For features
        that have multiple neighbors (e.g. covalent bonds), several options
        are provided for how to divide the charge to nearby neighbors.
        If use_electrides is set to True, features with electride like character
        will be treated as atoms and

        Parameters
        ----------
        splitting_method : Literal["equal", "pauling", "dist", "weighted_dist", "nearest"], optional
            The method used to divide charge and volume of shared features
            betweeen their coordinating atoms.

                'weighted_dist' (default)
                    Fraction increases with decreasing distance to each atom. The
                    fraction is further weighted by the radius of each atom
                    calculated from the ELF
                'pauling'
                    Fraction increases with decreasing pauling electronegativity.
                    If an atom has no recorded EN a value of 2.2 is used which
                    may be incorrect in many cases.
                'equal'
                    Each neighboring atom receives an equal fraction.
                'dist'
                    Fraction increases with decreasing distance to the center
                    of each atom
                'nearest'
                    All charge is assigned to the features nearest atom.

        use_electrides : bool, optional
            If True, features labeled as bare electrons will be treated as electride
            atoms. They will receive partial charge from other shared features
            and their charge/volume will be appended after the atoms'.

        Returns
        -------
        tuple[NDArray]
            Two arrays representing the charges and volumes respectively.

        """
        structure = self.labeler.nna_structure
        mapping = self.elf_radii.label_atom_map
        splitting_method = self.shared_feature_splitting_method

        # create an array to store atom charges and volumes
        atom_charge = np.zeros(len(structure), dtype=np.float64)
        atom_volume = np.zeros(len(structure), dtype=np.float64)

        # if using pauling, get all electronegativities
        if splitting_method == "pauling":
            pauling_ens = np.array([i.specie.X for i in structure])
            pauling_ens = np.nan_to_num(pauling_ens, nan=2.2)

        cnn = CrystalNN(
            distance_cutoffs= None,
            x_diff_weight= 0.0,
            porous_adjustment= False,
        )

        for basin_idx in range(len(self.labeler.basin_types)):
            charge = self.bader.basin_charges[basin_idx]
            volume = self.bader.basin_volumes[basin_idx]
            # get atom label
            label = mapping[basin_idx]
            # Labels under the structures length belong to a single atom.
            if label < len(structure):
                atom_charge[label] += charge
                atom_volume[label] += volume
                continue
            # otherwise, this is a shared basin
            temp_structure = structure.copy()
            temp_structure.append("H", coord=self.labeler.maxima_frac[basin_idx])
            coord_sites = cnn.get_nn(temp_structure, n=len(structure))
            coord_atoms = [i.index for i in coord_sites]

            # get unique atoms and counts (correction for small cells)
            unique_atoms, unique_indices, atom_counts = np.unique(
                coord_atoms, return_index=True, return_counts=True
            )

            if len(coord_atoms) == 0:
                # This shouldn't happen, but could if CrystalNN failed
                # to find neighbors.
                logging.warning(
                    f"No neighboring atoms found for feature with index {basin_idx}. Feature assigned to nearest atom."
                )
                # assign all charge/volume to the closest atom
                nearest = np.argmin(temp_structure.distance_matrix[len(structure)])
                atom_charge[nearest] += charge
                atom_volume[nearest] += volume

            elif len(coord_atoms) == 1:
                # all methods will add charge and volume to this atom.
                # assigning it here potentially avoids divide by zeros for
                # core features
                atom_charge[coord_atoms[0]] += charge
                atom_volume[coord_atoms[0]] += volume

            elif splitting_method == "equal":
                # evenly split the feature to each neighboring atom
                atom_charge[unique_atoms] += (charge / len(coord_atoms)) * atom_counts
                atom_volume[unique_atoms] += (volume / len(coord_atoms)) * atom_counts

            elif splitting_method == "pauling":
                # get the pauling ens for coordinated atoms
                ens = pauling_ens[coord_atoms]
                # normalize to the total en
                ens /= ens.sum()
                # get the weights for each unique atom
                ens = ens[unique_indices]
                atom_charge[unique_atoms] += charge * ens * atom_counts
                atom_volume[unique_atoms] += volume * ens * atom_counts

            elif splitting_method == "dist":
                # get the dist to each coordinated atom
                all_dists = temp_structure.distance_matrix[len(structure)]
                dists = all_dists[coord_atoms]
                # invert and normalize
                dists = 1 / dists
                dists /= dists.sum()
                # add for each atom
                for coord_idx, atom in enumerate(coord_atoms):
                    atom_charge[atom] += charge * dists[coord_idx]
                    atom_volume[atom] += volume * dists[coord_idx]

            elif splitting_method == "weighted_dist":
                # get the dist to each coordinated atom
                all_dists = temp_structure.distance_matrix[len(structure)]
                dists = all_dists[coord_atoms]
                atom_radii = self.elf_radii.atom_radii[coord_atoms]

                # calculate the weighted contribution to each atom and normalize
                weight = atom_radii / dists
                weight /= weight.sum()
                # add for each atom
                for coord_idx, atom in enumerate(coord_atoms):
                    atom_charge[atom] += charge * weight[coord_idx]
                    atom_volume[atom] += volume * weight[coord_idx]

            elif splitting_method == "nearest":
                # assign all charge/volume to the closest atom
                # get the dist to each coordinated atom
                all_dists = temp_structure.distance_matrix[len(structure)]
                dists = all_dists[coord_atoms]
                nearest = np.argmin(dists)
                atom_charge[nearest] += charge
                atom_volume[nearest] += volume
            else:
                raise ValueError(
                    f"'{splitting_method}' is not a valid splitting method"
                )

        return atom_charge.round(10), atom_volume.round(10)

    def _get_voxel_assignments(self) -> None:
        """

        Returns
        -------
        None
            Gets a dataframe of voxel assignments. The dataframe has columns
            [x, y, z, charge, sites].

        """
        # make sure we've run our partitioning (for logging clarity)
        (
            site_indices,
            site_transforms,
            plane_equations,
        ) = self.partitioning_planes

        logging.info("Beginning voxel assignment")


        if self.method == "zero-flux":
            # we are done here and can assign charges/volumes immediately
            self._atom_labels = None
            self._atom_charges, self._atom_volumes = self._get_zero_flux_assignments()

        else:
            # get bader basin labels
            basin_labels = self.bader.maxima_basin_labels
            num_basins = len(self.labeler.maxima_frac)
            structure_len = len(self.labeler.nna_structure)
            # get vacuum
            vacuum = basin_labels == num_basins
            # initialize badelf labels
            labels = np.full(basin_labels.shape, structure_len, dtype=basin_labels.dtype)

            # create a mask at nna indices
            if self.method == "badelf":
                # create map from basin index to nna structure index
                nna_indices = self.labeler.nna_indices
                label_map = np.empty(len(self.labeler.maxima_frac), dtype=np.int64)
                label_map[nna_indices] = np.arange(len(nna_indices)) + len(self.structure)

                # get labels at electride sites
                electride_mask = np.isin(basin_labels, nna_indices)
                labels[electride_mask] = label_map[basin_labels[electride_mask]]
            else:
                electride_mask = np.zeros(labels.shape, dtype=np.bool_)

            # get mask to ignore
            exclude_mask = vacuum | electride_mask

            # calculate the maximum distance in fractional coords from the center of
            # a voxel to its edges
            voxel_dist = self.reference_grid.max_point_dist + 1e-12

            # get the transforms within a set radius
            min_radius = voxel_dist * 2
            max_radius = (np.array(self.structure.lattice.abc) / 2).min()
            max_radius = min(max_radius, 3.0)  # cap radius for large cells
            sphere_transforms, transform_dists = (
                self.reference_grid.get_radial_neighbor_transforms(r=max_radius)
            )
            valid_mask = transform_dists >= min_radius
            sphere_transforms = sphere_transforms[np.where(valid_mask)[0]]
            transform_dists = transform_dists[valid_mask]

            # get the indices at which new transform dists occur
            transform_breaks = np.where(transform_dists[:-1] != transform_dists[1:])[0]

            # Now calculate labels, charges, and volumes assigned to each feature
            labels, charges, volumes = get_badelf_assignments(
                data=self.charge_grid.total,
                labels=labels,
                site_indices=site_indices,
                site_transforms=site_transforms,
                plane_equations=plane_equations,
                exclude_mask=exclude_mask,
                min_plane_dist=voxel_dist,
                lattice_matrix=self.reference_grid.matrix,
                sphere_transforms=sphere_transforms,
                transform_dists=transform_dists,
                transform_breaks=transform_breaks,
                max_val=structure_len
            )

            # convert charges/volumes to correct units
            charges /= self.charge_grid.ngridpts
            volumes = volumes * self.structure.volume / self.charge_grid.ngridpts

            # write feature charges/volumes
            self._atom_labels = labels
            self._atom_charges = charges
            self._atom_volumes = volumes

        logging.info("Finished voxel assignment")

    def _get_ELF_dimensionality(
        self,
        nna_mask: NDArray,
        cutoff: float,
    ) -> int:
        """

        This algorithm works by checking if the voxels with values above the cutoff
        are connected to the equivalent voxel in the unit cell one transformation
        over. This is done primarily using scipy.ndimage.label which determines
        which voxels are connected. To do this rigorously, the unit cell is repeated
        to make a (2,2,2) super cell and the connections are checked going from
        the original unit cell to the unit cells connected at the faces, edges,
        and corners. If a connection in that direction is found, the total number
        of connections increases. Dimensionalities of 0,1,2, and 3 are represented
        by 0,1,4,and 13 connections respectively.

        NOTE: This can be made much faster with numba using an algorithm similar
        to that used in BaderKit to determine which atoms are surrounded. However,
        this would require a lot of time.

        Parameters
        ----------
        nna_mask : np.array
            The ELF Grid object with only values associated with nnas.
        cutoff : float
            The minimum elf value to consider as a connection.

        Returns
        -------
        int
            The dimensionality at the ELF cutoff.

        """
        # Remove data below our cutoff
        mask = nna_mask & (self.reference_grid.total >= cutoff)

        # if we have no features, return 0 immediately
        if not np.any(mask):
            return 0

        # get the features that sit in the mask at this value
        feature_indices = self.labeler.nna_structure.frac_coords[len(self.structure) :]
        feature_indices = (
            np.round(self.charge_grid.frac_to_grid(feature_indices)).astype(int)
            % self.reference_grid.shape
        )
        # only use indices that are not 0
        feature_indices = [i for i in feature_indices if mask[i[0], i[1], i[2]]]

        # if we have no nna features in the mask, immediately return 0
        if len(feature_indices) == 0:
            return 0

        # create a supercell mask and label it
        supercell_mask = np.tile(mask, [2, 2, 2])
        labels, num_features = label(supercell_mask, structure=np.ones([3, 3, 3]))

        # We are going to need to translate the above voxels and the entire unit
        # cell so we create a list of desired transformations
        transformations = [
            [0, 0, 0],  # -
            [1, 0, 0],  # x
            [0, 1, 0],  # y
            [0, 0, 1],  # z
            [1, 1, 0],  # xy
            [1, 0, 1],  # xz
            [0, 1, 1],  # yz
            [1, 1, 1],  # xyz
        ]
        transformations = np.array(transformations)
        transformations = self.charge_grid.frac_to_grid(transformations)

        # The unit cell can be connected to neighboring unit cells in 26 directions.
        # however, we only need to consider half of these as the others are symmetrical.
        connections = [
            # surfaces (3)
            [0, 1],  # x
            [0, 2],  # y
            [0, 3],  # z
            # edges (6)
            [0, 4],  # xy
            [0, 5],  # xz
            [0, 6],  # yz
            [3, 1],  # x-z
            [3, 2],  # y-z
            [1, 2],  # -xy
            # corners (4)
            [0, 7],  # x,y,z
            [1, 6],  # -x,y,z
            [2, 5],  # x,-y,z
            [3, 4],  # x,y,-z
        ]
        # Using these connections we can determine the dimensionality of the system.
        # 1 connection is 1D, 2-4 connections is 2D and 5-13 connections is 3D.
        # !!! These may need to be updated if I'm wrong. The idea comes from
        # the fact that the connections should be 1, 4, and 13, but sometimes
        # voxelation issues result in a connection not working in one direction
        # while it would in the reverse direction (which isn't possible with
        # true symmetry). The range accounts for this possibility. The problem
        # might be if its possible to have for example a 2D connecting structure
        # with 5 connections. However, I'm pretty sure that immediately causes
        # an increase to 3D dimensionality.
        # First we create a list to store potential dimensionalites based off of
        # each feature. We will take the highest dimensionality.
        dimensionalities = []
        for coord in feature_indices:
            # get the labels at each transformation
            trans_labels = []
            for trans in transformations:
                x, y, z = trans + coord
                trans_labels.append(labels[x, y, z])

            # count number of connections
            connections_num = 0
            for connection in connections:
                # get the feature label at each voxel
                label1 = trans_labels[connection[0]]
                label2 = trans_labels[connection[1]]
                # If the labels are the same, the unit cell is connected in this
                # direction
                if label1 == label2:
                    connections_num += 1
            if connections_num == 0:
                dimensionalities.append(0)
            elif connections_num == 1:
                dimensionalities.append(1)
            elif 1 < connections_num <= 4:
                dimensionalities.append(2)
            elif 5 < connections_num <= 13:
                dimensionalities.append(3)

        return max(dimensionalities)

    def _get_nna_dimensionality(self) -> None:
        """

        Gets the nna dimensionalities and range of ELF values that they
        exist at.

        """
        # TODO: This whole method should probably be rewritten in Numba
        # If we have no nnas theres no reason to continue so we stop here
        logging.info("Finding nna dimensionality cutoffs")
        if self.num_nnas == 0:
            self._all_nna_dims = -1
            self._all_nna_dim_cutoffs = -1

        ###############################################################################
        # This section preps an ELF grid that only contains values from the nna
        # sites and is zero everywhere else.
        ###############################################################################

        # Create a mask at nnas
        nna_indices = [
            i for i in range(len(self.structure), len(self.labeler.nna_structure))
        ]
        # NOTE: even if we have shared features, these indices are still correct
        # so long as the nna sites come first
        nna_mask = np.isin(self.atom_labels, nna_indices)

        #######################################################################
        # This section scans across different cutoffs to determine what dimensionalities
        # exist in the nna ELF
        #######################################################################
        logging.info("Calculating dimensionality at 0 ELF")
        highest_dimension = self._get_ELF_dimensionality(nna_mask, 0)
        dimensions = [i for i in range(0, highest_dimension)]
        dimensions.reverse()
        logging.info(f"Max nna dimensionality: {highest_dimension}")
        # Create lists for the refined dimensions
        final_dimensions = [highest_dimension]
        final_connections = [0]
        amounts_to_change = []
        # We refine by guessing the cutoff is 0.5 then increasing or decreasing by
        # 0.25, then 0.125 etc. down to 0.000015259.
        for i in range(1, 16):
            amounts_to_change.append(1 / (2 ** (i + 1)))
        for dimension in dimensions:
            guess = 0.5
            # assume this dimension is not found
            found_dimension = False
            logging.info(f"Refining cutoff for dimension {dimension}")
            for i in tqdm(amounts_to_change, total=len(amounts_to_change)):
                # check what our current dimension is. If we are at a higher dimension
                # we need to raise the cutoff. If we are at a lower dimension or at
                # the dimension we need to lower it
                current_dimension = self._get_ELF_dimensionality(nna_mask, guess)
                if current_dimension > dimension:
                    guess += i
                elif current_dimension < dimension:
                    guess -= i
                elif current_dimension == dimension:
                    # We have found the dimension so we add it to our lists.
                    guess -= i
                    found_dimension = True
            if found_dimension:
                final_connections.append(round(guess, 4))
                final_dimensions.append(dimension)
        self._all_nna_dims = final_dimensions
        self._all_nna_dim_cutoffs = final_connections

    def _get_min_avg_surface_dists(self) -> None:
        """

        Calculates the minimum and average distance from each atom and nna
        to the partitioning surface.

        """
        neigh_transforms, _ = self.charge_grid.point_neighbor_transforms
        edges = get_edges(
            labeled_array=self.atom_labels,
            neighbor_transforms=neigh_transforms,
            vacuum_label=-1,
        )

        self._min_surface_distances, self._avg_surface_distances = (
            get_min_avg_surface_dists(
                labels=self.atom_labels,
                frac_coords=self.labeler.nna_structure.frac_coords,
                edge_mask=edges,
                matrix=self.charge_grid.matrix,
                max_value=np.max(self.structure.lattice.abc) * 2,
            )
        )

    ###########################################################################
    # From methods
    ###########################################################################

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

    ###########################################################################
    # Write Methods
    ###########################################################################

    def write_atom_volumes(
        self,
        atom_indices: NDArray,
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
            mask = self.atom_labels == atom_index
            kwargs["suffix"] = f"_a{atom_index}"
            self._write_volume(volume_mask=mask, **kwargs)

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

        mask = np.isin(self.atom_labels, atom_indices)
        # write
        kwargs["suffix"] = "_asum"
        self._write_volume(volume_mask=mask, **kwargs)

    def write_species_volume(
        self,
        species: str,
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

        # add dummy atoms if desired
        indices = self.structure.indices_from_symbol(species)

        # Get mask where the grid belongs to requested species
        mask = np.isin(self.atom_labels, indices)
        kwargs["suffix"] = f"_{species}"
        self._write_volume(volume_mask=mask, **kwargs)

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
                "surface_dist": self.min_surface_distances,
            }
        )
        return atoms_df

    def write_atom_tsv(self, filepath: Path | str = "bader_atoms.tsv"):
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