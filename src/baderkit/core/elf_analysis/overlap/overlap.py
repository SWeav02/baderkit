# -*- coding: utf-8 -*-

from pathlib import Path
from typing import TypeVar
import math

import numpy as np
from numpy.typing import NDArray

from baderkit.core.bader.bader import Bader
from baderkit.core.base.base_analysis import BaseAnalysis
from baderkit.core.toolkit import Grid, Structure
from .overlap_numba import (
    get_atom_charge_claims,
    get_overlap_table,
    get_overlap_charge_volume,
    get_overlap_fractions,
    get_atom_shell_groups
    )
from baderkit.core.utilities.coord_env import is_along_bond_all

Self = TypeVar("Self", bound="BasinOverlap")

class BasinOverlap(BaseAnalysis):
    """
    A convenience class for calculating the overlap between basins calculated
    in the charge density and a localization density such as ELF.

    """

    _summary_props = [
        "atomicities",
        "bond_fractions",
        "core_basins",
        "shared_basins",
        "along_bond",
        "attractor_shapes",
        "polarization_indexes",
        "atom_core_populations",
        "atom_core_volumes",
        "atom_valence_populations",
        "atom_charge_claims",
        "atom_connection_indices",
        "atom_connection_index_labels",
        ]

    _reset_props = [
        "overlap_table",
        "overlap_charges",
        "overlap_volumes",
        "overlap_labels",
        "volume_bond_fractions",
        "qtaim_overlap_groups",
        "along_bond",
        "atom_access_sets",
        "atom_access_electron_numbers",
        "atom_shell_groups",
    ] + _summary_props



    def __init__(
        self,
        charge_grid: Grid,
        reference_grid: Grid,
        total_charge_grid: Grid | None = None,
        nna_cutoff: float = 1.0,
        min_bond_angle: float = 135,
        **kwargs,
    ):
        self._min_bond_angle = min_bond_angle

        # create bader objects
        self.qtaim_bader = Bader(
            charge_grid=charge_grid,
            total_charge_grid=total_charge_grid,
            reference_grid=total_charge_grid,
            nna_cutoff=nna_cutoff,
            **kwargs,
        )

        self.local_bader = Bader(
            charge_grid=charge_grid,
            total_charge_grid=total_charge_grid,
            reference_grid=reference_grid,
            **kwargs,
        )

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
    def min_bond_angle(self) -> float:
        return self._min_bond_angle

    @min_bond_angle.setter
    def min_bond_angle(self, value: float):
        self._min_bond_angle = value
        self._reset_properties(
            include_properties=[
                "core_basins",
                "along_bond",
                ],
            )

    ###########################################################################
    # Properties calculated by other classes
    ###########################################################################

    @property
    def structure(self) -> Structure:
        """

        Returns
        -------
        Structure
            The pymatgen structure basins are assigned to.

        """
        return self.qtaim_bader.structure

    @property
    def vacuum_mask(self) -> NDArray[bool]:
        """

        Returns
        -------
        NDArray[bool]
            A mask representing the voxels that belong to the vacuum.

        """

        return self.qtaim_bader.vacuum_mask

    @property
    def local_maxima_frac(self) -> NDArray[np.float64]:
        """

        Returns
        -------
        NDArray[np.float64]
            The fractional coordinates of each maximum in the localization function
            grid. The order corresponds to the basin labels.

        """
        return self.local_bader.maxima_frac

    @property
    def qtaim_maxima_frac(self) -> NDArray[np.float64]:
        """

        Returns
        -------
        NDArray[np.float64]
            The fractional coordinates of each maximum in the charge density
            grid. The order corresponds to the atom labels and structure.

        """
        return self.qtaim_bader.structure.frac_coords

    ###########################################################################
    # Properties
    ###########################################################################
    @property
    def along_bond(self):
        if self._along_bond is None:
            self._along_bond, _ = is_along_bond_all(
                feature_frac_coords=self.local_maxima_frac,
                atom_frac_coords=self.reference_grid.structure.frac_coords,
                atom_cart_coords=self.reference_grid.structure.cart_coords,
                matrix=self.reference_grid.matrix,
                min_bond_angle=self.min_bond_angle * math.pi / 180,
            )
        return self._along_bond

    @property
    def atomicities(self) -> NDArray[np.int64]:
        """

        Returns
        -------
        NDArray[np.int64]
            The number of bader atoms that have at least some overlap with each
            local basin. Some of these may have very little overlap. For example,
            lone-pairs in solids often have a small contribution from neighboring
            atoms due to the periodicity requirement.

        """
        # TODO: Update with tolerance used for detecting cores/shared
        if self._atomicities is None:
            # The number of atoms contributing to each label is the number of
            # non-zero entries in each row of our overlap_matrix
            self._atomicities = np.array([len(i) for i in self.bond_fractions])
        return self._atomicities

    @property
    def overlap_table(self) -> NDArray[np.int64]:
        """

        Returns
        -------
        NDArray[np.int64]
            An Nx4 array where each row represents a single overlap basin. The
            columns represent the atom index, atom image, local basin index
            and local basin image.

        """
        if self._overlap_table is None:
            self._get_overlap_table()
        return self._overlap_table

    @property
    def overlap_charges(self) -> NDArray[np.float64]:
        """

        Returns
        -------
        NDArray[np.float64]
            The charges assigned to each overlap basins.

        """
        if self._overlap_charges is None:
            self._get_overlap_charges_volumes()
        return self._overlap_charges

    @property
    def overlap_volumes(self) -> NDArray[np.float64]:
        """

        Returns
        -------
        NDArray[np.float64]
            The volumes assigned to each overlap basins.

        """
        if self._overlap_volumes is None:
            self._get_overlap_charges_volumes()
        return self._overlap_volumes

    @property
    def overlap_labels(self) -> NDArray[np.int64]:
        """

        Returns
        -------
        NDArray[np.int64]
            A 3D array representing the overlapping atom and local basins. The
            labels correspond to the rows of the `overlap_table` property.

        """
        if self._overlap_labels is None:
            self._get_overlap_charges_volumes()
        return self._overlap_labels

    @property
    def bond_fractions(self) -> list[NDArray[np.float64]]:
        """

        Returns
        -------
        list[NDArray[np.float64]]
            A list with the same length as the number of local basins. Each entry
            is an Nx3 array where N is the number of overlapping qtaim basins
            and each entry is the qtaim atom index, the qtaim atom's periodic
            image (relative to the local basin's maximum), and the fractional
            charge claim of the atom.

        """
        if self._bond_fractions is None:
            self._get_overlap_fractions()
        return self._bond_fractions

    @property
    def volume_bond_fractions(self) -> list[NDArray[np.float64]]:
        """

        Returns
        -------
        list[NDArray[np.float64]]
            A list with the same length as the number of local basins. Each entry
            is an Nx3 array where N is the number of overlapping qtaim basins
            and each entry is the qtaim atom index, the qtaim atom's periodic
            image (relative to the local basin's maximum), and the fractional
            volume claim of the atom.

        """
        if self._volume_bond_fractions is None:
            self._get_overlap_fractions()
        return self._volume_bond_fractions

    @property
    def qtaim_overlap_groups(self) -> list[NDArray[np.float64]]:
        """

        Returns
        -------
        list[NDArray[np.float64]]
            A list with the same length as the number of atoms. Each entry
            represents the local basins that overlap with the qtaim atomic basin.
            The entries are an Nx2 array where the columns are the local basin's
            index and the local basin's periodic image (relative to the atoms
            coordinates).

        """
        if self._qtaim_overlap_groups is None:
            self._get_overlap_fractions()
        return self._qtaim_overlap_groups

    @property
    def core_basins(self) -> NDArray[int]:
        """

        Returns
        -------
        NDArray[int]
            An array with an entry for each local basin corresponding to the
            index of the atom this basin is a core of. If the basin is not a
            core, the value is -1. Note that for pseudopotential codes, atoms
            will often not have a core basin in the ELF.

        """

        if self._core_basins is None:
            self._assign_cores()
        return self._core_basins

    @property
    def shared_basins(self) -> NDArray[bool]:
        """

        Returns
        -------
        NDArray[bool]
            An array with an entry for each local basin. True for basins that
            have significant sharing between multiple atoms. This is basically
            all basins that are not part of the atom core or lone-pairs

        """

        if self._shared_basins is None:
            self._assign_cores()
        return self._shared_basins

    @property
    def attractor_shapes(self) -> NDArray[str]:
        """

        Returns
        -------
        NDArray[str]
            The shape of the attractor (maximum) for each basin in the local
            grid. Typically these are points but may form rings or cages in
            high symmetry environments..

        """
        if self._attractor_shapes is None:
            betti_nums = self.local_bader.maxima_betti_numbers
            shapes = []
            for i,j,k in betti_nums:
                if i == 1 and j == 0 and k == 0:
                    shapes.append("point")
                elif i == 1 and j == 1 and k == 0:
                    shapes.append("ring")
                elif i == 1 and j == 0 and k == 1:
                    shapes.append("cage")
                else:
                    raise Exception("Unknown shape found for attractor. This is a bug!")
            self._attractor_shapes = np.array(shapes)
        return self._attractor_shapes

    @property
    def polarization_indexes(self) -> NDArray[np.float64]:
        """
        Measure of polarization of a localized basin as defined by S. Raub and
        G. Jansen: 10.1007/s002140100268

        A value of 0 indicates fully non-polar while a value of 1 indicates
        fully polar. Anything between is a polar-covalent bond.

        As this measure was originally only designed for bonds between two
        atoms, the two largest fractions are used for the calculation even
        when there are other significant bond fractions.

        Returns
        -------
        NDArray[np.float64]
            The degree of polarization for each basin

        """
        if self._polarization_indexes is None:
            polarization_indexes = []
            # loop over the atoms and fractions in each basin
            for atom_fracs in self.bond_fractions:
                # if we have only one overlapped atom, this is a fully polarized
                # bond
                if len(atom_fracs) <= 1:
                    polarization_indexes.append(1.0)
                    continue
                # get fracs. These are already sorted from high to low
                fracs = atom_fracs[:,2]

                # calculate polarization index
                polarization_indexes.append((fracs[0] - fracs[1])/(fracs[0] + fracs[1]))

            self._polarization_indexes = np.array(polarization_indexes, dtype=np.float64).round(4)


        return self._polarization_indexes

    @property
    def atom_core_populations(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The total charge assigned to core basins for each atom.

            WARNING: for pseudopotential codes this will often be 0 and meaningless

        """
        if self._atom_core_populations is None:
            core_charges = np.zeros(len(self.qtaim_bader.structure))
            for feature_idx in range(len(self.local_maxima_frac)):
                # skip shared basins
                if self.core_basins[feature_idx] == -1:
                    continue
                # otherwise, assign the charge to the dominant atom
                atom = int(self.bond_fractions[feature_idx][:,0])
                core_charges[atom] += self.local_bader.basin_charges[feature_idx]
            self._atom_core_populations = core_charges
        return self._atom_core_populations.round(6)

    @property
    def atom_core_volumes(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The total volume assigned to core basins for each atom.

            WARNING: for pseudopotential codes this will often be 0 and meaningless

        """
        if self._atom_core_volumes is None:
            core_volumes = np.zeros(len(self.qtaim_bader.structure))
            for feature_idx in range(len(self.local_maxima_frac)):
                # skip shared basins
                if self.core_basins[feature_idx] == -1:
                    continue
                # otherwise, assign the charge to the dominant atom
                atom = int(self.bond_fractions[feature_idx][:,0])
                core_volumes[atom] += self.local_bader.basin_volumes[feature_idx]
            self._atom_core_volumes = core_volumes
        return self._atom_core_volumes.round(6)

    @property
    def atom_valence_populations(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            The total valence charge for each atom. For example, in NaCl this
            should be approximately 0 and 8 respectively.

            WARNING: for pseudopotential codes this may be incorrect depending
            on the pseudopotential used.

        """
        if self._atom_valence_populations is None:
            self._atom_valence_populations = self.qtaim_bader.atom_charges[:len(self.atom_core_populations)] - self.atom_core_populations
        return self._atom_valence_populations

    @property
    def atom_access_sets(self) -> list[NDArray[int]]:
        """

        Returns
        -------
        list[NDArray[int]]
            The local valence basins that border each atoms core. This is
            essentially the group of basins that include at least some of this
            atoms valence.

        """
        if self._atom_access_sets is None:
            access_sets = []
            for atom_fracs in self.qtaim_overlap_groups:
                access_set = []
                for i in atom_fracs:
                    # skip core
                    if not self.core_basins[int(i[0])] == -1:
                        continue
                    access_set.append(i[:2])
                access_set = np.array(access_set, dtype=int)
                access_sets.append(access_set)
            self._atom_access_sets = access_sets
        return self._atom_access_sets

    @property
    def atom_access_electron_numbers(self) -> NDArray[np.float64]:
        """

        Returns
        -------
        NDArray[float]
            The total charge in each atoms access set. This is the total number
            of electrons the atom has access to, but is typically higher than
            the atoms valence count.

        """
        if self._atom_access_electron_numbers is None:
            self._get_charge_claims()
        return self._atom_access_electron_numbers

    @property
    def atom_charge_claims(self) -> list[NDArray[float]]:
        """

        Returns
        -------
        list[NDArray[float]]
            A list of Nx3 arrays where each list entry represents a qtaim atom.
            Each array represents the charge claims for that atoms access set.
            The first column is the index of the atom that has a claim to part of the
            set, the second column is the periodic image of that atom, and the
            third column is the fractional charge claim the atom has to the set.
            The total accessessible charge can be obtained from the atom_access_sets
            property and the integrated charge of each local basin.

        """
        if self._atom_charge_claims is None:
            self._get_charge_claims()
        return self._atom_charge_claims

    @property
    def atom_connection_indices(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            This value was originally proposed by [Grin et. al.](http://dx.doi.org/10.1021/acs.inorgchem.5b00135)
            The connection index for each atom can be thought
            of as a condensed representation of all bonding for each atom. A
            value of 1 is non-polar while a value of 0 is polar.

            Our implementation differs slightly from what is described in Grin et. al.
            They suggest using the charge claims of each neighboring atom that
            has some charge claim in each atoms access set of basins. In reality,
            they sum the charge claims for symmetrically identical species, effectively
            reducing the number of "neighboring atoms". This would break under
            small perturbations to the system, so we sum the charge claims of
            all neighboring atoms of the same element regardless of if they
            are symmetrically equivalent.

        """
        if self._atom_connection_indices is None:
            self._get_charge_claims()
        return self._atom_connection_indices.round(6)

    @property
    def atom_connection_index_labels(self) -> NDArray[float]:
        """
        The symbol representation of each atoms connection index. These follow
        the format CInX where CI stands for connection index, n is the number
        of unique species with some charge claim to the atoms access set, and
        X is the central atom's species. Note that the central atom is considered
        a separate species (e.g. A Si atom in silicon would still have n=2).

        """
        if self._atom_connection_index_labels is None:
            self._get_charge_claims()
        return self._atom_connection_index_labels

    @property
    def atom_shell_groups(self) -> list[NDArray[int]]:
        if self._atom_shell_groups is None:
            # get atom shells
            all_atom_shells, basin_dists = get_atom_shell_groups(
                atom_local_groups=self.qtaim_overlap_groups,
                atom_frac_coords=self.qtaim_maxima_frac,
                local_frac_coords=self.local_maxima_frac,
                matrix=self.reference_grid.matrix,
                tol=0.1
                    )
            self._atom_shell_groups = all_atom_shells
            self._atom_average_shell_dists = basin_dists

        return self._atom_shell_groups

    @property
    def atom_average_shell_dists(self) -> list[NDArray[float]]:
        if self._atom_average_shell_dists is None:
            self.atom_shell_groups
        return self._atom_average_shell_dists

###############################################################################
# Methods to generate properties
###############################################################################

    def _get_charge_claims(self):
        _, index, inverse = np.unique(self.qtaim_bader.structure.atomic_numbers, return_inverse=True, return_index=True)
        order = np.argsort(index)
        equiv_species = order[inverse]

        (
            self._atom_charge_claims,
            self._atom_access_electron_numbers,
            self._atom_connection_indices,
            species_nums
            ) = get_atom_charge_claims(
            access_sets=self.atom_access_sets,
            bond_fractions=self.bond_fractions,
            local_basin_charges=self.local_bader.basin_charges,
            equiv_species=equiv_species,
            num_atoms=len(self.qtaim_maxima_frac),
            num_local=len(self.local_maxima_frac),
            )
        # get labels
        symbols = []
        for idx, site in enumerate(self.qtaim_bader.structure):
            species_num = species_nums[idx]
            species = site.specie.symbol
            symbols.append(f"CI{int(species_num)}{species}")
        self._atom_connection_index_labels = symbols

    def _get_overlap_table(self):
        overlap_table = get_overlap_table(
                atom_labels=self.qtaim_bader.atom_labels,  # Bader Atoms
                atom_images=self.qtaim_bader.atom_images,
                local_labels=self.local_bader.maxima_basin_labels,  # ELF basins
                local_images=self.local_bader.maxima_basin_images,  # ELF basins
                num_atoms=len(self.qtaim_maxima_frac),
                num_local=len(self.local_maxima_frac),
                )
        # sort lexographically
        overlap_table = overlap_table[np.lexsort((
            overlap_table[:, 3],  # lowest priority
            overlap_table[:, 1],
            overlap_table[:, 2],
            overlap_table[:, 0],  # highest priority
        ))]
        self._overlap_table = overlap_table

    def _get_overlap_charges_volumes(self):
        (
        self._overlap_charges,
        self._overlap_volumes,
        self._overlap_labels,
            ) = get_overlap_charge_volume(
            unique_overlaps=self.overlap_table,
            atom_labels=self.qtaim_bader.atom_labels,  # Bader Atoms
            atom_images=self.qtaim_bader.atom_images,
            local_labels=self.local_bader.maxima_basin_labels,  # ELF basins
            local_images=self.local_bader.maxima_basin_images,  # ELF basins
            charge_data=self.total_charge_grid.total,
            cell_volume=self.structure.volume,
            )

    def _get_overlap_fractions(self, tol=0.001):
        self._bond_fractions, self._volume_bond_fractions, self._qtaim_overlap_groups = get_overlap_fractions(
            self.overlap_table,
            self.overlap_charges,
            self.overlap_volumes,
            num_atoms=len(self.qtaim_maxima_frac),
            num_local=len(self.local_maxima_frac),
            tol=tol,
            )

    def _assign_cores(
            self,
            core_tol=0.01,
            lone_pair_tol=0.05,
            ):
        # create tracker for which basins are part of each atoms core
        cores = np.full(len(self.local_maxima_frac), -1, dtype=np.int64)
        lone_pairs = np.full(len(self.local_maxima_frac), -1, dtype=np.int64)
        shared = np.zeros(len(self.local_maxima_frac), dtype=np.bool_)

        for atom_idx in range(len(self.atom_shell_groups)):
            atom_shells = self.atom_shell_groups[atom_idx]
            local_overlap = self.qtaim_overlap_groups[atom_idx]
            for shell in atom_shells:
                overlap_fracs = local_overlap[shell][:,2]
                local_indices = local_overlap[shell][:,0].astype(int)
                # if all members of this shell are almost entirely owned by this
                # atom, we have a core
                if np.all(overlap_fracs > 1.0 - core_tol):
                    cores[local_indices] = atom_idx
                    continue

                # if none of the basins are above our threshold for lone-pairs
                # they must all be shared. If all of them are above the threshold
                # we have highly ionic shared basins.
                max_frac = overlap_fracs.max()
                min_frac = overlap_fracs.min()
                if (
                    max_frac < 1.0 - lone_pair_tol
                    or min_frac > 1.0 - lone_pair_tol
                    ):
                    continue
                # otherwise, me may have a lone-pairs or shared basins.
                # We only accept basins as lone-pairs if they have a significantly
                # higher value than the other basins in this shell
                # lone-pairs if there is also a basin that is significantly
                # shared. This helps avoid misassigning highly ionic shared
                # basins
                for local_idx, local_frac in zip(local_indices, overlap_fracs):
                    if (local_frac-min_frac) / min_frac > 0.2:
                        lone_pairs[local_idx] = atom_idx

                # anything left over is a shared basin
        shared = (cores == -1) & (lone_pairs == -1)

        self._shared_basins = shared
        self._core_basins = cores

    @classmethod
    def from_vasp(
        cls,
        charge_filename: Path | str = "CHGCAR",
        reference_filename: Path | str = "ELFCAR",
        **kwargs,
    ) -> Self:
        """
        Creates a BasinOverlap class object from VASP files.

        Parameters
        ----------
        charge_filename : Path | str, optional
            The path to the CHGCAR like file that will be used for integrating charge.
            The default is "CHGCAR".
        reference_filename : Path |  str
            The path to ELFCAR like file that will be used for partitioning.
        total_charge_filename : Grid | None, optional
            The path to the CHGCAR like file used for determining vacuum regions
            in the system. For pseudopotential codes this represents the total
            electron density and should be provided whenever possible.
            If None, defaults to the charge_grid.
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
            **kwargs
            )