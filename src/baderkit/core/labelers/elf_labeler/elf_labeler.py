# -*- coding: utf-8 -*-

import json
import logging
import math
import os
import warnings
from pathlib import Path
from typing import Literal, TypeVar

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.io.vasp import Potcar

from baderkit.core import Bader, Grid, Structure
from baderkit.core.labelers.bifurcation_graph import (
    BifurcationGraph,
    DomainSubtype,
    FeatureType,
)
from baderkit.core.utilities.coord_env import check_all_covalent
from baderkit.core.utilities.file_parsers import Format

from .elf_labeler_numba import get_feature_edges, get_min_avg_feat_surface_dists
from .elf_radii import ElfRadiiTools

Self = TypeVar("Self", bound="ElfLabeler")

# TODO:
# - Add distinction between core and valence shells using POTCAR if desired
# - Add modified feature volume normalized to structure volume (likely better
# metric for hp-electrides)
# - Add option for calculating atom radius from volume rather than bonds
# - Add electride dimensionality calculation here. Ideally use numba method
# - add a method to print a more traditional bifurcation plot?


class ElfLabeler:
    """
    Labels chemical features present in the ELF and collects various properties
    e.g charge, volume, elf value, etc.
    This is originally designed for analyzing electride materials, but could be
    quite useful for bond analysis as well.

    This class is designed only for single spin or total spin charge densities
    and ELF. For spin-dependent systems, use the SpinElfLabeler instead.

    Parameters
    ----------
    charge_grid : Grid
        The charge density grid used for integrating charge.
    reference_grid : Grid
        The ELF grid used to partition volumes.
    ignore_low_pseudopotentials : bool, optional
        Whether or not to ignore errors associated with the use of pseudopotentials
        with limited valence electrons. The default is False.
    shared_shell_ratio : float, optional
        The ratio used to determine if shallow nodes surrounding an atom should
        be considered as a single shell feature.

        Highly symmetric features such
        as atom shells will often split into smaller features at a high isosurface
        value very close to the maximum value in these smaller features. This
        ratio refers to the range of values the ELF feature exists as a
        single feature divided by the highest ELF value in the feature.

        As covalency increases, this ratio will decrease as the small features
        grow and eventually form large covalent features that align along a bond.
        Thus, this parameter is in some ways a cutoff for considering a
        feature a covalent bond versus an ionic shell. The default is 0.75.
    covalent_molecule_ratio : float, optional
        The ratio used to determine if an ELF feature belongs to one or multiple
        atomic shell systems.

        The covalent-ionic spectrum can be visualized in the ELF by comparing
        the ELF values at which Shell-like features form and fully surround
        atoms. "Fully surround" in this case means there is a void space
        within the shell that is not the same surface as the outside of the shell.
        In ionic bonds, the outermost shells will fully surround a single atom
        at a relatively high ELF value and will not connect to other features
        to surround additional atoms until considerably lower ELF values.
        In contrast, homogenous covalent bonds will never fully surround 1 atom,
        instead surrounding at least two when they first form a shell-like
        shape. Heterogenous covalent bonds sit somewhere between the two,
        surrounding a single atom for a small range of values before connecting
        to a feature surrounding additional atoms.

        The ratio here is the range of ELF values where the feature belongs
        to a parent feature surrounding multiple atoms divided by the range of values
        where it belongs to a parent feature surrounding only 1 atoms. This
        ratio can be fairly low even for slight EN differences, though it is
        usually significantly lower in clear ionic cases.
        The default is 0.2.
    combine_shells : bool, optional
        Whether or not to combine the Bader basins making up a shell into a
        single feature. This is likely a more reasonable chemical perspective
        and reduces differences that arise from the inability of a regular
        cartesian grid to represent spherical shells. The default is True.
    min_covalent_charge : float, optional
        The minimum charge that a feature must have to be considered covalent.
        This exists primarily to distinguish between highly covalent features
        and metallic features. The default is 0.6.
    min_covalent_angle : float, optional
        The minimum angle between the two atoms and potential covalent feature for
        the feature to be considered a covalent bond. For example, most single
        covalent bonds sit exactly along the bond with an angle of 180. Triple
        bonds and metallic bonds with covalent-like behavior may form ring
        like shapes due to digeneracy with maxima slightly of the bond, decreasing
        this angle. The default is 135.
    max_metal_depth : float, optional
        The maximum depth a feature can have and be considered metallic. Features
        between multiple atoms with a depth lower than this value will be marked
        as metallic. The default is 0.1.
    min_electride_elf_value : float, optional
        The minimum ELF value for a feature to be considered an electride rather
        than a multi-centered bond. The default is 0.5.
    min_electride_depth : float, optional
        The minimum range of ELF values a feature must exist distinctly to
        be considered an electride rather than a multi-centered bond. The default is 0.2.
    min_electride_charge : float, optional
        The minimum charge a feature must have to be considered an electride
        rather than a multi-centered bond. The default is 0.5.
    min_electride_volume : float, optional
        The minimum volume a feature must have to be considered an electride
        rather than a multi-centered bond. The default is 10.
    min_electride_dist_beyond_atom : float, optional
        The minimum distance beyond the atoms radius that an electride must
        sit to be considered an electride rather than a multi-centered bond. The radius
        is calculated as the minimum in the ELF between two atoms. If a covalent
        or metallic bond is present, the maximum is used instead. The default is 0.3.
    crystalnn_kwargs : dict, optional
        The keyword arguments used to create the CrystalNN object used for
        nearest neighbor calculations. This is only important for calculating
        charges on atoms when splitting covalent/metallic features to their
        nearest neighbors.
    vacuum_tol : float | bool, optional
        The tolerance for considering a region to be part of the vacuum.
        WARNING: This is set to False for now as we have not implemented
        vacuum handling for the ELF.
        The default is False.
    **kwargs : dict
        Keyword arguments to pass to the Bader class.

    """

    _labeled_covalent = False
    _labeled_multi = False
    spin_system = "total"

    def __init__(
        self,
        charge_grid: Grid,
        reference_grid: Grid,
        ignore_low_pseudopotentials: bool = False,
        shared_shell_ratio: float = 0.75,
        covalent_molecule_ratio: float = 0.2,
        combine_shells: bool = True,
        min_covalent_charge: float = 0.6,
        min_covalent_angle: float = 135,
        max_metal_depth: float = 0.1,
        min_electride_elf_value: float = 0.5,
        min_electride_depth: float = 0.2,
        min_electride_charge: float = 0.5,
        min_electride_volume: float = 10,
        min_electride_dist_beyond_atom: float = 0.3,
        crystalnn_kwargs: dict = {
            "distance_cutoffs": None,
            "x_diff_weight": 0.0,
            "porous_adjustment": False,
        },
        vacuum_tol=False,
        **kwargs,
    ):

        # ensure the reference file is ELF
        if reference_grid.data_type != "elf":
            logging.warning(
                "A non-ELF reference file has been detected. Results may not be valid."
            )

        self.charge_grid = charge_grid
        self.reference_grid = reference_grid

        self.ignore_low_pseudopotentials = ignore_low_pseudopotentials
        self.crystalnn_kwargs = crystalnn_kwargs
        self.cnn = CrystalNN(**crystalnn_kwargs)
        # BUGFIX: We use a separate cnn with very loose rules to find neighbors
        # that may potentially have the smallest radius. This is intentionally
        # separate from the cnn used to calculate NNs for features
        self._radii_cnn = CrystalNN(
            weighted_cn=True,
            distance_cutoffs=None,
            x_diff_weight=0.0,
            porous_adjustment=False,
        )

        # define cutoff variables
        # TODO: These should be hidden variables to allow for setter methods
        self.shared_shell_ratio = shared_shell_ratio
        self.covalent_molecule_ratio = covalent_molecule_ratio
        self.combine_shells = combine_shells
        self.min_covalent_charge = min_covalent_charge
        self.min_covalent_angle = min_covalent_angle

        # electride cutoffs
        self.min_electride_elf_value = min_electride_elf_value
        self.min_electride_charge = min_electride_charge
        self.min_electride_depth = min_electride_depth
        self.min_electride_charge = min_electride_charge
        self.min_electride_volume = min_electride_volume
        self.min_electride_dist_beyond_atom = min_electride_dist_beyond_atom

        # define properties that will be updated by running the method
        self._bifurcations = None
        self._bifurcation_graph = None
        self._bifurcation_plot = None
        self._atom_elf_radii = None
        self._atom_elf_radii_types = None
        self._atom_elf_radii_e = None
        self._atom_elf_radii_types_e = None
        self._atom_nn_elf_radii = None
        self._atom_nn_elf_radii_e = None
        self._atom_nn_elf_radii_types = None
        self._atom_nn_elf_radii_types_e = None
        self._nearest_neighbor_data = None
        self._nearest_neighbor_data_e = None
        self._atom_feature_indices_e = None
        self._atom_max_values_e = None
        self._atom_nn_planes = None
        self._atom_nn_planes_e = None

        self._electrides_per_formula = None
        self._electrides_per_reduced_formula = None

        # create a bader object
        self.bader = Bader(
            charge_grid=charge_grid,
            reference_grid=reference_grid,
            vacuum_tol=vacuum_tol,
            **kwargs,
        )

    # TODO: Make these reset on a change similar to the Bader class. Add docs

    ###########################################################################
    # Bifurcation Properties
    ###########################################################################

    @property
    def bifurcation_graph(self) -> BifurcationGraph:
        """

        Returns
        -------
        BifurcationGraph
            A BifurcationGraph class representing features and bifurcations in the
            ELF.

        """
        if self._bifurcation_graph is None:
            self._get_bifurcation_graph()
        return self._bifurcation_graph

    @property
    def bifurcation_plot(self) -> go.Figure:
        """

        Returns
        -------
        go.Figure
            A plotly graph object representing the bifurcation graph.

        """
        if self._bifurcation_plot is None:
            self._bifurcation_plot = self.bifurcation_graph.get_plot()
        return self._bifurcation_plot

    ###########################################################################
    # Structure Properties
    ###########################################################################
    @property
    def structure(self) -> Structure:
        """

        Returns
        -------
        Structure
            The PyMatGen Structure representing the system.

        """
        return self.reference_grid.structure

    @property
    def electride_structure(self) -> Structure:
        """

        Returns
        -------
        Structure
            The system's structure including dummy atoms representing electride
            sites.

        """
        return self.bifurcation_graph.electride_structure

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
        return self.bifurcation_graph.labeled_structure

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
            electrides_per_unit = 0
            for i, node in enumerate(self.bifurcation_graph.irreducible_nodes):
                if node.feature_type in FeatureType.bare_types:
                    electrides_per_unit += self.feature_charges[i]
            self._electrides_per_formula = electrides_per_unit
        return round(self._electrides_per_formula, 10)

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
        return round(self._electrides_per_reduced_formula, 10)

    ###########################################################################
    # Coordination Environment
    ###########################################################################

    @property
    def nearest_neighbor_data(self) -> tuple:
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
            # run bifurcation assignment
            self.bifurcation_graph
        return self._nearest_neighbor_data

    @property
    def nearest_neighbor_data_e(self):
        """

        Returns
        -------
        tuple
            The nearest neighbor data for the atoms AND the electrides
            in the system represented as a tuple of arrays. The arrays represent,
            in order, the central atoms index, its neighbors index, the fractional
            coordinates of the neighbor, and the distance between the two sites.

        """
        if self._nearest_neighbor_data_e is None:
            # run assignment
            self.atom_nn_elf_radii_e
        return self._nearest_neighbor_data_e

    ###########################################################################
    # Atom and Electride quasi-atom Properties
    ###########################################################################

    @property
    def atom_elf_radii(self) -> NDArray[np.float64]:
        """

        Returns
        -------
        NDArray
            The radius of each atom calculated from the ELF using the closest
            neighboring atom in the structure.

        """
        if self._atom_elf_radii is None:
            self._atom_elf_radii, self._atom_elf_radii_types = self._get_atom_elf_radii(
                self.structure,
                self.atom_nn_elf_radii,
                self._atom_nn_elf_radii_types,
                self.nearest_neighbor_data,
            )

        return self._atom_elf_radii.round(10)

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
            # run labeling and radii calc by calling our bifurcation graph
            self._atom_elf_radii = None
            self.atom_elf_radii
        return np.where(self._atom_elf_radii_types, "covalent", "ionic")

    @property
    def atom_elf_radii_e(self) -> NDArray[np.float64]:
        """

        Returns
        -------
        NDArray
            The radius of each atom and electride site calculated from the ELF
            using the closest neighboring atom/electride in the structure.

        """
        if self._atom_elf_radii_e is None:
            self._atom_elf_radii_e, self._atom_elf_radii_types_e = (
                self._get_atom_elf_radii(
                    self.electride_structure,
                    self.atom_nn_elf_radii_e,
                    self._atom_nn_elf_radii_types_e,
                    self.nearest_neighbor_data_e,
                )
            )
        return self._atom_elf_radii_e.round(10)

    @property
    def atom_elf_radii_types_e(self) -> NDArray[np.float64]:
        """

        Returns
        -------
        NDArray
            The type of radius of each atom/electride elf radius. Covalent indicates that the
            bond crosses through some covalent or metallic region, and the radius
            is placed at the maximum in the ELF in this region. Ionic indicates
            that the bond does not pass through the covalent/metallic region and
            the radius is placed at the minimum between the two atoms.

        """
        if self._atom_elf_radii_types_e is None:
            # run labeling and radii calc by calling our bifurcation graph
            self.atom_elf_radii_e
        return np.where(self._atom_elf_radii_types_e, "covalent", "ionic")

    @property
    def atom_nn_elf_radii(self) -> NDArray[np.float64]:
        """

        Returns
        -------
        NDArray
            The elf radii for each atom and its neighboring atoms in the same
            order as the nearest_neighbor_data property.

        """
        if self._atom_nn_elf_radii is None:
            self.bifurcation_graph
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
            # call radii method
            self.atom_nn_elf_radii
        return np.where(self._atom_nn_elf_radii_types, "covalent", "ionic")

    @property
    def atom_nn_elf_radii_e(self) -> NDArray[np.float64]:
        """

        Returns
        -------
        NDArray
            The elf radii for each atom/electride and its neighboring atoms in the same
            order as the nearest_neighbor_data_e property.

        """
        if self._atom_nn_elf_radii_e is None:
            # make sure labeled bifurcation graph exists
            if self._labeled_covalent is None:
                self.bifurcation_graph
            # if there are no electride atoms, just return the results for the base
            # structure (avoid repeat calc)
            if len(self.structure) == len(self.electride_structure):
                self._atom_nn_elf_radii_e = self.atom_nn_elf_radii
                self._atom_nn_elf_radii_types_e = self._atom_nn_elf_radii_types
                self._nearest_neighbor_data_e = self.nearest_neighbor_data
                self._atom_nn_planes_e = self._atom_nn_planes
            else:
                (
                    site_indices,
                    neigh_indices,
                    neigh_coords,
                    radii,
                    dists,
                    bond_types,
                    plane_points,
                    plane_vectors,
                ) = self._get_nn_atom_elf_radii(use_electrides=True)
                self._atom_nn_elf_radii_e = radii
                self._atom_nn_elf_radii_types_e = bond_types
                self._nearest_neighbor_data_e = (
                    site_indices,
                    neigh_indices,
                    neigh_coords,
                    dists,
                )
                self._atom_nn_planes_e = (plane_points, plane_vectors)

        return self._atom_nn_elf_radii_e

    @property
    def atom_nn_elf_radii_types_e(self) -> NDArray[np.float64]:
        """

        Returns
        -------
        NDArray
            The type of radius for each atom/electride and its neighboring atoms in the same
            order as the nearest_neighbor_data property.

        """
        if self._atom_nn_elf_radii_types_e is None:
            # call radii method
            self.atom_nn_elf_radii_e
        return np.where(self._atom_nn_elf_radii_types, "covalent", "ionic")

    @property
    def atom_feature_indices_e(self) -> NDArray[np.int64]:
        """

        Returns
        -------
        list
            The feature indices assigned to each atom/electride in the electride
            structure. Features assigned to multiple atoms are not included.

        """
        # For each atom, a list of feature indices that belong solely to that
        # atom
        if self._atom_feature_indices_e is None:
            atom_features = [[] for i in range(len(self.electride_structure))]
            # add atom indices
            for feat_idx, node in enumerate(self.bifurcation_graph.irreducible_nodes):
                if (
                    node.coord_number == 1
                    and node.feature_type not in FeatureType.bare_types
                ):
                    atom_features[node.coord_indices_e[0]].append(feat_idx)
            # add electride indices
            electride_num = 0
            for i, node in enumerate(self.bifurcation_graph.irreducible_nodes):
                if node.feature_type in FeatureType.bare_types:
                    atom_features[len(self.structure) + electride_num].append(i)
                    electride_num += 1
            self._atom_feature_indices_e = atom_features
        return self._atom_feature_indices_e

    @property
    def atom_max_values_e(self) -> NDArray[np.float64]:
        """

        Returns
        -------
        NDArray
            The maximum value that each atom has an existing feature.

        """
        if self._atom_max_values_e is None:
            feature_max_values = self.feature_max_values
            max_values = []
            for i, feature_indices in enumerate(self.atom_feature_indices_e):
                max_values.append(np.max(feature_max_values[feature_indices]))

            self._atom_max_values_e = np.array(max_values)
        return self._atom_max_values_e.round(10)

    ###########################################################################
    # Feature Properties
    ###########################################################################

    def _get_feature_properties(self, property_name: str) -> list:
        """
        Collects the request property for each feature in the BifurcationGraph

        Parameters
        ----------
        property_name : str
            The name of the property to collect

        Returns
        -------
        features : list
            The properties for each feature.

        """
        features = []
        for node in self.bifurcation_graph.irreducible_nodes:
            features.append(getattr(node, property_name, None))
        return features

    @property
    def feature_types(self) -> list[str]:
        """

        Returns
        -------
        list[str]
            The type of each ELF feature in the system.

        """
        return self._get_feature_properties("feature_type")

    @property
    def feature_basins(self) -> list[NDArray]:
        """

        Returns
        -------
        list[NDArray]
            The Bader basins associated with each feature. For features with
            multiple basins, the basins are separated by very shallow minima
            and are better understood as a single feature.

        """
        return self._get_feature_properties("basins")

    @property
    def feature_frac_coords(self) -> list[NDArray]:
        """

        Returns
        -------
        list[NDArray]
            The fractional coordinates of the maxima in each Bader basin
            for each feature in the structure.

        """
        return self._get_feature_properties("frac_coords")

    @property
    def feature_average_frac_coords(self) -> NDArray:
        """

        Returns
        -------
        NDArray
            The average fractional coordinates of the Bader basin maxima for each
            feature. Note that it is possible for this coordinate to be outside
            of the feature, particularly for atom shells which form sphere-like
            shapes.

        """
        return np.array(self._get_feature_properties("average_frac_coords"))

    @property
    def feature_max_values(self) -> NDArray:
        """

        Returns
        -------
        NDArray
            The maximum value at which each feature exists.

        """
        return np.array(self._get_feature_properties("max_value")).round(10)

    @property
    def feature_min_values(self) -> NDArray:
        """

        Returns
        -------
        NDArray
            The minimum value at which each feature exists.

        """
        return np.array(self._get_feature_properties("min_value")).round(10)

    @property
    def feature_charges(self) -> NDArray:
        """

        Returns
        -------
        NDArray
            The charge associated with each feature.

        """
        return np.array(self._get_feature_properties("charge")).round(10)

    @property
    def feature_volumes(self) -> NDArray:
        """

        Returns
        -------
        NDArray
            The volume associated with each feature.

        """
        return np.array(self._get_feature_properties("volume")).round(10)

    @property
    def feature_coord_atoms(self) -> list:
        """

        Returns
        -------
        list
            The atoms coordinated with each feature.

        """
        return self._get_feature_properties("coord_indices")

    @property
    def feature_coord_nums(self) -> list:
        """

        Returns
        -------
        list
            The number of atoms coordinated with each feature.

        """
        return np.array(self._get_feature_properties("coord_number"), dtype=np.int64)

    @property
    def feature_coord_dists(self) -> list:
        """

        Returns
        -------
        list
            The distance to each coordinated atom from each feature.

        """
        return self._get_feature_properties("coord_dists")

    @property
    def feature_coord_atoms_e(self) -> list:
        """

        Returns
        -------
        list
            The coordinated atoms to each feature, including electrides as quasi-atoms

        """
        return self._get_feature_properties("coord_indices_e")

    @property
    def feature_coord_nums_e(self) -> list:
        """

        Returns
        -------
        list
            The number of coordinated atoms for each featuree, including
            electrides as quasi-atoms.

        """
        return np.array(self._get_feature_properties("coord_number_e"), dtype=np.int64)

    @property
    def feature_coord_atoms_dists_e(self):
        """

        Returns
        -------
        list
            The distance to each coordinated atom/electride from each feature.

        """
        return self._get_feature_properties("coord_dists_e")

    @property
    def feature_min_surface_dists(self) -> list:
        """

        Returns
        -------
        list
            The minimum distance from the average frac coord of each feature to the
            partitioning surface.

        """
        return np.array(self._get_feature_properties("min_surface_dist")).round(10)

    @property
    def feature_avg_surface_dists(self) -> NDArray:
        """

        Returns
        -------
        list
            The average distance from the average frac coord of each feature to the
            partitioning surface.

        """
        return np.array(self._get_feature_properties("avg_surface_dist")).round(10)

    ###########################################################################
    # Helpful Methods
    ###########################################################################
    def feature_indices_by_type(self, feature_types: list[FeatureType | str]):
        """

        Gets a list of feature indices from a list of types of Features
        (e.g. bare electron, metallic, covalent)

        Parameters
        ----------
        feature_types : list[FeatureType | str]
            The list of feature types to find indices for.

        Returns
        -------
        NDArray
            A list of feature indices corresponding to all feature types in the
            provided list. These correspond to the order in which feature properties
            (e.g. ElfLabeler.feature_charges) appear in.

        """
        return np.array(
            [i for i, feat in enumerate(self.feature_types) if feat in feature_types],
            dtype=np.int64,
        )

    def get_oxidation_and_volumes_from_potcar(
        self, potcar_path: Path = "POTCAR", use_electrides: bool = True, **kwargs
    ) -> tuple[NDArray]:
        """
        Calculates the oxidation states, charges, and volumes associated with each
        atom/electride using the information from a provided POTCAR.

        Parameters
        ----------
        potcar_path : Path, optional
            The path to the POTCAR file. The default is "POTCAR".
        use_electrides : bool, optional
            Whether or not to treat electrides as quasi atoms. The default is True.
        **kwargs : dict
            Any keyword arguments to pass to the 'get_charges_and_volumes' method.

        Returns
        -------
        tuple[NDArray]
            three arrays representing the oxidation states, charges, and volumes
            respectively.

        """
        charges, volumes = self.get_charges_and_volumes(
            use_electrides=use_electrides, **kwargs
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

        return np.array(oxi_state_data).round(10), charges.round(10), volumes.round(10)

    def get_charges_and_volumes(
        self,
        splitting_method: Literal[
            "weighted_dist", "pauling", "equal", "dist", "nearest"
        ] = "weighted_dist",
        use_electrides: bool = True,
        **kwargs,
    ) -> tuple[NDArray]:
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
        if use_electrides:
            structure = self.electride_structure
        else:
            structure = self.structure

        # create an array to store atom charges and volumes
        atom_charge = np.zeros(len(structure), dtype=np.float64)
        atom_volume = np.zeros(len(structure), dtype=np.float64)

        # if using pauling, get all electronegativities
        if splitting_method == "pauling":
            pauling_ens = np.array([i.specie.X for i in structure])
            pauling_ens = np.nan_to_num(pauling_ens, nan=2.2)

        electride_num = 0
        for feature_idx in range(len(self.feature_types)):
            charge = self.feature_charges[feature_idx]
            volume = self.feature_volumes[feature_idx]

            if use_electrides:
                # check if this is an electride feature
                if self.feature_types[feature_idx] in FeatureType.bare_types:
                    # assign charge/volume to self
                    struc_idx = len(self.structure) + electride_num
                    atom_charge[struc_idx] += charge
                    atom_volume[struc_idx] += volume
                    electride_num += 1
                    continue

                # get coordination with electrides
                coord_atoms = self.feature_coord_atoms_e[feature_idx]
            else:
                # get coordination without electrides
                coord_atoms = self.feature_coord_atoms[feature_idx]
            # get unique atoms and counts (correction for small cells)
            unique_atoms, unique_indices, atom_counts = np.unique(
                coord_atoms, return_index=True, return_counts=True
            )

            if len(coord_atoms) == 0:
                # This shouldn't happen, but could if CrystalNN failed
                # to find neighbors.
                logging.warning(
                    f"No neighboring atoms found for feature with index {feature_idx}. Feature assigned to nearest atom."
                )
                # assign all charge/volume to the closest atom
                nearest = np.argmin(self.feature_coord_dists[feature_idx])
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
                if use_electrides:
                    dists = self.feature_coord_atoms_dists_e[feature_idx].copy()
                else:
                    dists = self.feature_coord_dists[feature_idx].copy()
                # invert and normalize
                dists = 1 / dists
                dists /= dists.sum()
                # add for each atom
                for coord_idx, atom in enumerate(coord_atoms):
                    atom_charge[atom] += charge * dists[coord_idx]
                    atom_volume[atom] += volume * dists[coord_idx]

            elif splitting_method == "weighted_dist":
                # get the dist to each coordinated atom and their radii
                if use_electrides:
                    dists = self.feature_coord_atoms_dists_e[feature_idx].copy()
                    atom_radii = self.atom_elf_radii_e[coord_atoms]
                else:
                    dists = self.feature_coord_dists[feature_idx].copy()
                    atom_radii = self.atom_elf_radii[coord_atoms]

                # calculate the weighted contribution to each atom and normalize
                try:
                    weight = atom_radii / dists
                except:
                    breakpoint()
                weight /= weight.sum()
                # add for each atom
                for coord_idx, atom in enumerate(coord_atoms):
                    atom_charge[atom] += charge * weight[coord_idx]
                    atom_volume[atom] += volume * weight[coord_idx]

            elif splitting_method == "nearest":
                # assign all charge/volume to the closest atom
                nearest = np.argmin(self.feature_coord_dists[feature_idx])
                atom_charge[nearest] += charge
                atom_volume[nearest] += volume
            else:
                raise ValueError(
                    f"'{splitting_method}' is not a valid splitting method"
                )

        return atom_charge.round(10), atom_volume.round(10)

    def get_feature_labels(
        self,
        included_features: list[str] = FeatureType.valence_types,
        return_structure: bool = True,
        return_feat_indices: bool = False,
        return_charge_volume: bool = False,
        order_by_type: bool = True,
    ) -> tuple:
        """

        Assigns each grid point to atoms and features included in the 'included_features'
        tag. The assignments are represented by an array with the same dimensions
        as the charge/reference grids with integers representing the atom/feature
        index in the Structure object including requested feature dummy atoms.
        By default this method orders requested features so that atoms
        come first, followed by electrides, and then any other feature types. This
        is for methods such as BadELF which rely on this ordering scheme.

        Atoms are included by default and it is generally not recommended to include
        core/shell features. If these are of interest to you, reach out to us
        on our [github](https://github.com/SWeav02/baderkit)

        Parameters
        ----------
        included_features : list[str], optional
            The features to include in addition to the atoms. The default is FeatureType.valence_types.
        return_structure : bool, optional
            Whether or not to return the structure that the index labels correspond
            to. The default is True.
        return_feat_indices : bool, optional
            Whether or not to return the original feature indices in their new
            order. The default is False.
        order_by_type : bool, optional
            Whether or not to reorder the structure prior to assigning to
            grid points. The default is True.
        return_charge_volume : bool, optional
            Whether or not to return the corresponding charge/volume associated
            with each feature

        Returns
        -------
        tuple
            An array representing grid point assignments. If return_structure or
            return_feat_indices is set to True, these are returned as well. The
            returned result is always a tuple, even if only the feature assignments
            are requested.

        """

        # Get the original basin atom assignments
        basin_atoms = self.bader.basin_atoms.copy()

        # get indices of requested type
        feature_indices = self.feature_indices_by_type(included_features)

        # reorder features so that electrides are first if requested
        if order_by_type:
            bare_feature_indices = []
            other_feature_indices = []
            for feat_idx in feature_indices:
                if self.feature_types[feat_idx] in FeatureType.bare_types:
                    bare_feature_indices.append(feat_idx)
                else:
                    other_feature_indices.append(feat_idx)
            feature_indices = bare_feature_indices + other_feature_indices

        # reassign basin atoms and get structure
        feature_structure = self.structure.copy()
        new_atom_idx = len(self.structure)
        for sorted_idx, feat_idx in enumerate(feature_indices):
            # update basin labels
            basins = self.feature_basins[feat_idx]
            basin_atoms[basins] = new_atom_idx
            new_atom_idx += 1
            # add feature to structure
            species = self.feature_types[feat_idx].dummy_species
            coords = self.feature_average_frac_coords[feat_idx]
            feature_structure.append(species, coords)

        # NOTE: append -1 so that vacuum gets assigned to -1 in the atom_labels
        # array
        basin_atoms = np.insert(basin_atoms, len(basin_atoms), -1)

        # reassign labels
        feature_labels = basin_atoms[self.bader.basin_labels]

        # get requested results
        if not any((return_structure, return_feat_indices, return_charge_volume)):
            return feature_labels
        results = [feature_labels]
        if return_structure:
            results.append(feature_structure)

        if return_feat_indices:
            results.append(feature_indices)

        if return_charge_volume:
            basin_atoms = basin_atoms[:-1]

            atom_charges = np.bincount(
                basin_atoms,
                weights=self.bader.basin_charges,
                minlength=len(feature_structure),
            )
            atom_volumes = np.bincount(
                basin_atoms,
                weights=self.bader.basin_volumes,
                minlength=len(feature_structure),
            )
            results.append(atom_charges)
            results.append(atom_volumes)

        # return labels
        return tuple(results)

    def get_feature_structure(
        self,
        included_features: list[str] = FeatureType.valence_types,
        return_feat_indices: bool = False,
        order_by_type: bool = True,
    ) -> Structure:
        """

        Generates a PyMatGen Structure object with dummy atoms for each requested
        feature. By default this method orders requested features so that atoms
        come first, followed by electrides, and then any other feature types.

        Parameters
        ----------
        included_features : list[str], optional
            The features to include in addition to the atoms. The default is FeatureType.valence_types.
        return_feat_indices : bool, optional
            Whether or not to return the original feature indices in their new
            order. The default is False.
        order_by_type : bool, optional
            Whether or not to reorder the structure. The default is True.

        Returns
        -------
        Structure
            DESCRIPTION.

        """
        # get indices of requested type
        feature_indices = self.feature_indices_by_type(included_features)

        # reorder features so that electrides are first if requested
        if order_by_type:
            bare_feature_indices = []
            other_feature_indices = []
            for feat_idx in feature_indices:
                if self.feature_types[feat_idx] in FeatureType.bare_types:
                    bare_feature_indices.append(feat_idx)
                else:
                    other_feature_indices.append(feat_idx)
            feature_indices = bare_feature_indices + other_feature_indices

        # get structure from indices
        structure = self.get_feature_structure_by_index(feature_indices)

        if return_feat_indices:
            return structure, feature_indices

        return structure

    def get_feature_structure_by_index(self, feature_indices: list[int]) -> Structure:
        """

        Generates a PyMatGen Structure object with dummy atoms for each requested
        feature. This method does not reorder features.

        Parameters
        ----------
        feature_indices : list[int]
            The indices of features to include.

        Returns
        -------
        structure : Structure
            The system's PyMatGen Structure including dummy atoms representing
            requested features.

        """

        # Create a new structure without oxidation states
        structure = self.structure.copy()
        structure.remove_oxidation_states()

        # add each feature
        for feat_idx in feature_indices:
            structure.append(
                self.feature_types[feat_idx].dummy_species,
                self.feature_average_frac_coords[feat_idx],
            )

        return structure

    ###########################################################################
    # Hidden Utility Methods
    ###########################################################################

    def _get_nn_atom_elf_radii(
        self,
        use_electrides: bool = False,
    ) -> tuple:
        """

        Calculate the ELF radius for all atom neighbor pairs that result in partitioning
        planes lying on the voronoi surface

        Parameters
        ----------
        use_electrides : bool, optional
            Whether or not to treat electrides as quasi-atoms. The default is False.

        Returns
        -------
        tuple
            The radius for each atom/neighbor pair and their bond type (0=ionic, 1=covalent)

        """
        if not self._labeled_covalent:
            raise Exception("Covalent features must be labeled for reliable radii.")
        # get appropriate structure and neighbor data
        if use_electrides:
            structure = self.electride_structure
            # we don't treat electride atoms as metals in this case, as we are
            # treating them like quasi atoms
            covalent_types = [
                i for i in FeatureType.valence_types if i not in FeatureType.bare_types
            ]
        else:
            structure = self.structure
            # we do treat electride atoms as metals/covalent features
            covalent_types = FeatureType.valence_types

        # Get a labeled structure including covalent, metallic, and bare
        # electron features
        included_types = FeatureType.valence_types.copy()
        feature_labels, feature_structure = self.get_feature_labels(
            included_features=included_types
        )
        covalent_symbols = np.unique([i.dummy_species for i in covalent_types])
        # Calculate all radii on the voronoi surface
        radii_tools = ElfRadiiTools(
            grid=self.reference_grid,
            feature_labels=feature_labels,
            feature_structure=feature_structure,
            override_structure=structure,
            covalent_symbols=covalent_symbols,
        )
        return radii_tools.get_voronoi_radii()
        # return radii_tools.get_crystalnn_radii()

    @staticmethod
    def _get_atom_elf_radii(structure, all_radii, all_radii_types, nn_data) -> tuple:
        """
        Gets the smallest radius for each atom from the provided data.

        Parameters
        ----------
        structure : Structure
            The structure corresponding to the atoms to find radii for.
        all_radii : NDArray
            All radii between all atoms and their nearest neighbors.
        all_radii_types : NDArray
            All radii types for all atom/neighbor pairs.
        nn_data : NDArray
            Four arrays representing the coordination environment of the provided
            structure.

        Returns
        -------
        tuple
            The radii and radius types for each atom (0=ionic, 1=covalent)

        """
        # NOTE: This is the smallest radius of each atom which is not always
        # along the bond to the nearest neighbor, but typically is. For
        # example CdPt3 has 2 Pt atoms with a Cd an dPt atom tied for the
        # nearest.
        site_indices, neigh_indices, neigh_coords, dists = nn_data

        # sort radii
        sorted_indices = np.argsort(all_radii)
        sorted_sites = site_indices[sorted_indices]

        # get first instance of each atom
        radii = np.empty(len(structure), dtype=np.float64)
        try:
            radii_types = np.empty(len(structure), dtype=all_radii_types.dtype)
        except:
            breakpoint()
        for i in range(len(structure)):
            first_index = np.argmax(sorted_sites == i)
            radii[i] = all_radii[sorted_indices[first_index]]
            radii_types[i] = all_radii_types[sorted_indices[first_index]]
        return radii, radii_types

    def _calculate_feature_surface_dists(self):
        """

        Calculates the distance from the average coordinates to the surface grid
        points for each feature node.

        """
        # Calculate the minimum and average distance from each irreducible features
        # fractional coordinate to its edges. This is often different from the
        # original basins as we may combine some of them.

        nodes = self.bifurcation_graph.irreducible_nodes

        # collect frac coords and map basin labels to features
        frac_coords = [i.average_frac_coords for i in nodes]
        feature_map = np.empty(len(self.bader.basin_maxima_frac), dtype=np.uint32)
        for node_idx, node in enumerate(nodes):
            feature_map[node.basins] = node_idx

        # get feature edges
        neighbor_transforms, _ = self.reference_grid.point_neighbor_transforms

        edge_mask = get_feature_edges(
            labeled_array=self.bader.basin_labels,
            feature_map=feature_map,
            neighbor_transforms=neighbor_transforms,
            vacuum_mask=self.bader.vacuum_mask,
        )

        # calculate the minimum and average distance to each features surface
        min_dists, avg_dists = get_min_avg_feat_surface_dists(
            labels=self.bader.basin_labels,
            feature_map=feature_map,
            frac_coords=np.array(frac_coords, dtype=np.float64),
            edge_mask=edge_mask,
            matrix=self.reference_grid.matrix,
            max_value=np.max(self.structure.lattice.abc) * 2,
        )

        # set surface distances
        for node, min_dist, avg_dist in zip(nodes, min_dists, avg_dists):
            node._min_surface_dist = min_dist
            node._avg_surface_dist = avg_dist

    ###########################################################################
    # Read Methods
    ###########################################################################
    @classmethod
    def from_vasp(
        cls,
        charge_filename: Path | str = "CHGCAR",
        reference_filename: Path | str = "ELFCAR",
        total_only: bool = True,
        **kwargs,
    ) -> Self:
        """
        Creates an ElfLabeler class object from VASP files.

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
            Keyword arguments to pass to the Labeler class.

        Returns
        -------
        Self
            An ElfLabeler class object.

        """
        # This is just a wrapper of the Bader class to update the default to
        # load the ELFCAR
        charge_grid = Grid.from_vasp(charge_filename, total_only=total_only)
        if reference_filename is None:
            reference_grid = None
        else:
            reference_grid = Grid.from_vasp(reference_filename, total_only=total_only)

        return cls(charge_grid=charge_grid, reference_grid=reference_grid, **kwargs)

    ###########################################################################
    # Write Methods
    ###########################################################################

    def write_bifurcation_plot(
        self,
        filename: str | Path,
    ):
        """

        Writes the BifurcationPlot to an html file. This is just a shortcut for
        ElfLabeler.bifurcation_plot.write_html.

        Parameters
        ----------
        filename : str | Path
            The Path to write the plot to. 'html' will be appended if it is not
            present in the path.

        """
        plot = self.bifurcation_plot
        # make sure path is a Path object
        filename = Path(filename)
        # add .html if filename doesn't include it
        filename_html = filename.with_suffix(".html")
        plot.write_html(filename_html)

    def write_feature_basins(
        self,
        feature_indices: list[int],
        directory: str | Path = Path("."),
        include_dummy_atoms: bool = True,
        write_reference: bool = True,
        output_format: str | Format = None,
        prefix_override: str = None,
        **writer_kwargs,
    ):
        """
        For a give list of node keys, writes the bader basins associated with
        each.

        Parameters
        ----------
        feature_indices : list[int]
            Which features to write basin volumes for.
        directory : str | Path, optional
            The directory to write the files in. If None, the active directory
            is used.
        include_dummy_atoms : bool, optional
            Whether or not to add dummy files to the structure. The default is False.
        write_reference : bool, optional
            Whether or not to write the reference data rather than the charge data.
            Default is False.
        output_format : str | Format, optional
            The format to write with. If None, writes to source format stored in
            the Grid objects metadata.
            Defaults to None.
        prefix_override : str, optional
            The string to add at the front of the output path. If None, defaults
            to the VASP file name equivalent to the data type stored in the
            grid.

        """
        # get the data to use
        if write_reference:
            data_array = self.reference_grid.total
            data_type = self.reference_grid.data_type
        else:
            data_array = self.charge_grid.total
            data_type = self.charge_grid.data_type

        if directory is None:
            directory = Path(".")

        # get structure
        if include_dummy_atoms:
            structure = self.get_feature_structure_by_index(feature_indices)
        else:
            structure = self.structure

        # get prefix
        if prefix_override is None:
            prefix_override = data_type.prefix

        for feat_idx in feature_indices:
            basins = self.feature_basins[feat_idx]
            # get mask where this feature is NOT
            mask = np.isin(self.bader.basin_labels, basins, invert=True)
            # copy data to avoid overwriting. Set data off of basin to 0
            data_array_copy = data_array.copy()
            data_array_copy[mask] = 0.0
            grid = Grid(
                structure=structure,
                data={"total": data_array_copy},
                data_type=data_type,
            )
            file_path = directory / f"{prefix_override}_f{feat_idx}"
            # write file
            grid.write(filename=file_path, output_format=output_format, **writer_kwargs)

    def write_feature_basins_sum(
        self,
        feature_indices: list[int],
        directory: str | Path = Path("."),
        include_dummy_atoms: bool = False,
        write_reference: bool = True,
        output_format: str | Format = None,
        prefix_override: str = None,
        **writer_kwargs,
    ):
        """
        For a give list of node keys, writes the union of the bader basins
        associated with each.

        Parameters
        ----------
        feature_indices : list[int]
            Which features to include in the volume.
        directory : str | Path, optional
            The directory to write the files in. If None, the active directory
            is used.
        include_dummy_atoms : bool, optional
            Whether or not to add dummy files to the structure. The default is False.
        write_reference : bool, optional
            Whether or not to write the reference data rather than the charge data.
            Default is False.
        output_format : str | Format, optional
            The format to write with. If None, writes to source format stored in
            the Grid objects metadata.
            Defaults to None.
        prefix_override : str, optional
            The string to add at the front of the output path. If None, defaults
            to the VASP file name equivalent to the data type stored in the
            grid.
        """
        # get the data to use
        if write_reference:
            data_array = self.reference_grid.total
            data_type = self.reference_grid.data_type
        else:
            data_array = self.charge_grid.total
            data_type = self.charge_grid.data_type

        if directory is None:
            directory = Path(".")

        # get structure
        if include_dummy_atoms:
            structure = self.get_feature_structure_by_index(feature_indices)
        else:
            structure = self.structure

        # get prefix
        if prefix_override is None:
            prefix_override = data_type.prefix

        # get all basin indices to include
        basin_list = []
        for feat_idx in feature_indices:
            basin_list.extend(self.feature_basins[feat_idx])

        # get mask where features are not
        mask = np.isin(self.bader.basin_labels, basin_list, invert=True)
        # copy data to avoid overwriting. Set data off of basin to 0
        data_array_copy = data_array.copy()
        data_array_copy[mask] = 0.0
        grid = Grid(
            structure=structure,
            data={"total": data_array_copy},
            data_type=data_type,
        )
        file_path = directory / f"{prefix_override}_fsum"
        # write file
        grid.write(filename=file_path, output_format=output_format, **writer_kwargs)

    def write_all_features(self, **kwargs):
        """
        Writes the bader basins associated with all features

        Parameters
        ----------
        **kwargs :
            See :meth:`write_feature_basins`.

        """
        self.write_feature_basins(
            feature_indices=np.arange(len(self.feature_charges), dtype=int), **kwargs
        )

    def write_features_by_type(
        self,
        included_types: list[FeatureType],
        prefix_override=None,
        write_reference: bool = True,
        directory: str | Path = Path("."),
        **kwargs,
    ):
        """
        Writes the bader basins associated with all features of the selected type

        Parameters
        ----------
        included_types : list[FeatureType]
            The types of features to include, e.g. metallic, lone-pair, etc.
        prefix_override : str
            The string to add at the front of the output path. If None, defaults
            to the VASP file name equivalent to the data type stored in the
            grid.
        write_reference : bool, optional
            Whether or not to write the reference data rather than the charge data.
            Default is True.
        **kwargs :
            See :meth:`write_feature_basins`.

        """
        # get the data to use
        if write_reference:
            data_type = self.reference_grid.data_type
        else:
            data_type = self.charge_grid.data_type
        # get prefix
        if prefix_override is None:
            prefix_override = data_type.prefix
        for feature_type in included_types:
            feature_type = FeatureType(feature_type)
            feature_indices = self.feature_indices_by_type([feature_type])
            prefix = prefix_override + f"_{feature_type.dummy_species}"
            self.write_feature_basins_sum(
                feature_indices=feature_indices,
                prefix_override=prefix,
                directory=directory,
                **kwargs,
            )

            # rename to remove fsum
            os.rename(
                directory / f"{prefix}_fsum",
                directory / f"{prefix}",
            )

    def write_features_by_type_sum(
        self,
        included_types: list[FeatureType],
        **kwargs,
    ):
        """
        Writes the union of the bader basins associated with all features of the
        selected type

        Parameters
        ----------
        **kwargs :
            See :meth:`write_feature_basins_sum`.

        """
        feature_indices = self.feature_indices_by_type(included_types)
        self.write_feature_basins_sum(feature_indices=feature_indices, **kwargs)

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
        method_kwargs = {
            "splitting_method": splitting_method,
            "crystalnn_kwargs": self.crystalnn_kwargs,
            "ignore_low_pseudopotentials": self.ignore_low_pseudopotentials,
            "shared_shell_ratio": self.shared_shell_ratio,
            "covalent_molecule_ratio": self.covalent_molecule_ratio,
            "min_covalent_charge": self.min_covalent_charge,
            "min_covalent_angle": self.min_covalent_angle,
            "min_electride_elf_value": self.min_electride_elf_value,
            "min_electride_depth": self.min_electride_depth,
            "min_electride_charge": self.min_electride_charge,
            "min_electride_volume": self.min_electride_volume,
            "min_electride_dist_beyond_atom": self.min_electride_dist_beyond_atom,
        }
        results["method_kwargs"] = method_kwargs
        results["spin_system"] = self.spin_system

        # only try to calculate oxidation state if this was not a half spin system
        potcar_path = Path(potcar_path)
        if self.spin_system == "total" and potcar_path.exists():
            oxidation_states, charges, volumes = (
                self.get_oxidation_and_volumes_from_potcar(
                    potcar_path=potcar_path, use_electrides=False
                )
            )
            oxidation_states_e, charges_e, volumes_e = (
                self.get_oxidation_and_volumes_from_potcar(
                    potcar_path=potcar_path, use_electrides=True
                )
            )
        else:
            oxidation_states = None
            oxidation_states_e = None
            charges, volumes = self.get_charges_and_volumes(use_electrides=False)
            charges_e, volumes_e = self.get_charges_and_volumes(use_electrides=True)
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
            "bifurcation_graph",
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
            "atom_max_values_e",
            "feature_max_values",
            "feature_min_values",
            "feature_charges",
            "feature_volumes",
            "feature_coord_nums",
            "feature_coord_nums_e",
            "feature_min_surface_dists",
            "feature_avg_surface_dists",
        ]:
            result_obj = getattr(self, result, None)
            if use_json and result_obj is not None:
                result_obj = result_obj.tolist()
            results[result] = result_obj

        # add objects that are lists with arrays
        for result in [
            "feature_coord_dists",
            "feature_coord_atoms_dists_e",
        ]:
            result_obj = getattr(self, result, None)
            if use_json and result_obj is not None:
                result_obj = [i.tolist() for i in result_obj]
            results[result] = result_obj

        # add other objects that are already jsonable
        for result in [
            "nelectrides",
            "electride_formula",
            "electrides_per_formula",
            "electrides_per_reduced_formula",
            "feature_types",
            "feature_coord_atoms",
            "feature_coord_atoms_e",
            "atom_feature_indices_e",
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
            The Path to write the results to. The default is "elf_labeler.json".
        **kwargs : dict
            keyword arguments for the to_dict method.

        """
        filepath = Path(filepath)
        with open(filepath, "w") as json_file:
            json.dump(self.to_dict(use_json=True, **kwargs), json_file, indent=4)

    ###########################################################################
    # Core Graph Construction
    ###########################################################################

    def _get_bifurcation_graph(self):
        """

        This constructs a BifurcationGraph class and labels each irreducible feature
        as a chemical feature. This is the core of the method and is not meant to
        be called directly.

        """
        # run bader for nice looking logging purposes
        # NOTE: I call a property rather than run_bader/run_atom_assignment to
        # avoid repeat calcs if we've already run
        self.bader.atom_labels

        logging.info("Beginning ELF Analysis")

        # get an initial graph connecting bifurcations and final basins
        self._initialize_bifurcation_graph()

        # Now we have a graph with information associated with each basin. We want
        # to label each node. First, we label cores as they are the simplest
        self._mark_cores()

        # Next, we mark shells. This step distinguishes ionic and covalent
        # features, the most ambiguous step
        self._mark_shells()

        # Next we label covalent bonds. These must lie along an atomic bond and
        # have a reasonably large charge
        self._mark_covalent()

        # Next we mark lone pairs. These split off from covalent bonds or rarely
        # from atomic shells (e.g. SnO)
        self._mark_lonepairs()

        # get atomic radii before checking for metal/bare.
        # NOTE: This intentionally may underestimates radii in metallic systems.
        # For metal systems, we would treat metal maxima similar to covalent
        # bonds. For electrides, we would treat maxima as part of a separate
        # "electride atom"
        logging.info("Calculating atomic radii")
        (
            site_indices,
            neigh_indices,
            neigh_coords,
            radii,
            dists,
            bond_types,
            plane_points,
            plane_vectors,
        ) = self._get_nn_atom_elf_radii(use_electrides=False)
        self._atom_nn_elf_radii = radii
        self._atom_nn_elf_radii_types = bond_types
        self._nearest_neighbor_data = (site_indices, neigh_indices, neigh_coords, dists)
        self._atom_nn_planes = (plane_points, plane_vectors)

        # Next we mark our metallic/bare electrons. These currently have a set
        # of rather arbitrary cutoffs to distinguish between them. In the future
        # I would like to perform a comprehensive study.
        self._mark_metallic_or_bare()

        # BUGFIX: The atomic radii are somewhat dependent on if we consider a
        # feature to be metallic or a qausi-atom. We first calculate the radii
        # assuming electride atoms to allow for the best chance of a feature getting
        # through the "dist_beyond_atom" cutoff. Then we recalculate and relabel
        # so that metallic systems have the proper label
        # Recalculate radii with our updated markers
        # logging.info("Re-calculating atomic radii")
        # self._atom_elf_radii = None
        # self._atom_elf_radii_types = None
        # self._atom_nn_elf_radii, self._atom_nn_elf_radii_types = self._get_nn_atom_elf_radii(self.structure, self.nearest_neighbor_data)

        # # Re-mark metallic/electrides
        # self._mark_metallic_or_bare()

        # In some cases, the user may not have used a pseudopotential with enough core electrons.
        # This can result in an atom having no assigned core/shell, which will
        # result in nonsense later. We check for this here and throw an error
        assigned_atoms = []
        for node in self.bifurcation_graph.get_feature_nodes(FeatureType.atomic_types):
            assigned_atoms.append(node.nearest_atom)
        if (
            len(np.unique(assigned_atoms)) != len(self.structure)
            and not self.ignore_low_pseudopotentials
        ):

            raise Exception(
                "At least one atom was not assigned a zero-flux basin. This typically results"
                "from pseudo-potentials (PPs) with only valence electrons (e.g. the defaults for Al, Si, B in VASP 5.X.X)."
                "Try using PPs with more valence electrons such as VASP's GW potentials"
            )
        # Finally, we ensure that all nodes have an assignment
        if len(self.bifurcation_graph.unassigned_nodes) > 0:
            raise Exception(
                "At least one ELF feature was not assigned. This is a bug!!! Please report it to our github:"
                "https://github.com/SWeav02/baderkit/issues"
            )

        # calculate feature surface distances
        self._calculate_feature_surface_dists()
        logging.info("Finished labeling ELF")

    def _initialize_bifurcation_graph(self):
        """

        Creates an initial unlabeled bifurcation graph using information from
        Bader analysis of the ELF.

        """
        self._bifurcation_graph = BifurcationGraph.from_labeler(self)

    ###############################################################################
    # Core feature labeling methods
    ###############################################################################

    def _mark_cores(self):
        logging.info("Marking atomic shells")
        # cores are reducible domains that contain an atom. They should be within
        # one voxel of the atom. We must check for this as it is possible for
        # a different type of basin to contain the atom if too few valence electrons
        # are in the pseudopotential
        max_dist = self.reference_grid.max_point_dist * 2
        for node in self.bifurcation_graph.unassigned_nodes:
            if node.domain_subtype == DomainSubtype.irreducible_cage:
                continue
            if len(node.contained_atoms) != 1:
                continue
            if node.atom_distance <= max_dist:
                node.feature_type = FeatureType.shell

    def _mark_shells(self):
        # shells are reducible domains that surround exactly one atom.
        # In a vacuum, an atoms shells are spherical due to symmetry. In a
        # molecule/solid they will warp due to interactions with neighboring
        # atoms. If a neighbor has a strong enough attraction, the shell will
        # break into multiple child domains (covalent/lone-pairs). If its even
        # stronger, the shell will move fully to the neighbor (ionic bond) and
        # form a shell there.

        # Our criteria for a shell domain is as follows:
        # 1. Surrounds 1 atom
        # 2. Is 0D (finite)
        # 3. Exists as a shell in a much larger range than as individual child
        # domains

        # First, we label any nodes that surround 1 atom but don't have a maximum
        # at the atoms nucleus. This often happens for particularly shallow shells
        # if they are combined when our graph is first generated
        for node in self.bifurcation_graph.unassigned_nodes:
            if len(node.contained_atoms) == 1:
                node.feature_type = "shell"

        # Now we label shells that may be slightly deeper. We compare the range
        # where the domain surrounds 1 atom to the range itself or any of its children
        # exist. The ratio is compared to our cutoff

        shell_nodes = []
        for node in self.bifurcation_graph.reducible_nodes:
            # skip nodes that don't surround a single atom or have a dimensionality above 0
            if node.is_infinite or len(node.contained_atoms) != 1:
                continue
            # skip nodes that have children that also contain the single atom
            if any([len(i.contained_atoms) == 1 for i in node.children]):
                continue

            # Get the depth of the domain
            shared_shell_depth = node.depth

            # Get the total range this node or its children exist
            max_elf = 0.0
            for child in node.deep_children:
                if child.max_value > max_elf:
                    max_elf = child.max_value
            total_depth = max_elf - node.min_value

            ratio = shared_shell_depth / total_depth
            if ratio > self.shared_shell_ratio:
                shell_nodes.append(node)

        # mark nodes as shells
        for node in shell_nodes:
            if self.combine_shells:
                # convert to an irreducible node
                node = node.make_irreducible()
                node.feature_type = "shell"
            else:
                for child in node.deep_children:
                    if not child.is_reducible:
                        child.subtype = "shell"

    def _mark_covalent(self):
        """
        Takes in a bifurcation graph and labels valence features that
        are obviously metallic or covalent
        """
        logging.info("Marking covalent features")
        graph = self.bifurcation_graph

        # get frac coords of unassigned nodes
        valence_nodes = []
        valence_frac = []
        for node in graph.unassigned_nodes:
            valence_frac.append(node.average_frac_coords)
            valence_nodes.append(node)

        # Convert our cutoff angle to radians
        min_covalent_angle = self.min_covalent_angle * math.pi / 180

        # get our atom frac coords
        atom_frac_coords = self.structure.frac_coords
        atom_cart_coords = self.structure.cart_coords

        # check which nodes are within our tolerance
        # if/then is to avoid numba disliking empty lists
        if len(valence_frac) > 0:
            nodes_in_tolerance, atom_neighs = check_all_covalent(
                valence_frac,
                atom_frac_coords,
                atom_cart_coords,
                frac2cart=self.structure.lattice.matrix,
                min_covalent_angle=min_covalent_angle,
            )
        else:
            nodes_in_tolerance, atom_neighs = [], []

        for node, in_tolerance, (atom0, atom1) in zip(
            valence_nodes, nodes_in_tolerance, atom_neighs
        ):
            # skip nodes that aren't within our angle tolerance
            if not in_tolerance:
                continue

            # set backup
            contained_atoms = node.ancestors[-1].contained_atoms
            if atom0 in contained_atoms and atom1 in contained_atoms:
                is_covalent = True
            else:
                is_covalent = False  # could happen if there are multiple roots

            # Sometimes a lone pair happens to align well with an atom that
            # is not part of our covalent system (e.g. CaC2). Similar to shells,
            # the range of values containing both atoms should be comparable to
            # the rane of values containing at least one of them.
            # POSSIBLE BUG: Atoms with translational symmetry might be counted
            # as part of multiple covalent systems and break this in small
            # unit cells. This would require knowing which atom image is contained
            # which we currently don't track and would require a large rework

            # find range of values containing at least one and both of the atoms
            one_atom = None
            both_atoms = None
            for parent in node.ancestors:
                contained_atoms = parent.contained_atoms
                if atom0 in contained_atoms or atom1 in contained_atoms:
                    if one_atom is None:
                        one_atom = parent.max_value
                if atom0 in contained_atoms and atom1 in contained_atoms:
                    if both_atoms is None:
                        both_atoms = parent.max_value
                        break

            is_covalent = (both_atoms / one_atom) >= self.covalent_molecule_ratio

            # check that we truly only have two nearest neighbors
            if is_covalent and not node.coord_number == 2:
                is_covalent = False

            if is_covalent:
                # label as covalent or metallic covalent depending on the species'
                # type and charge
                species1 = self.structure[int(atom0)].specie
                species2 = self.structure[int(atom1)].specie
                if species1.is_metal or species2.is_metal:
                    if node.charge > self.min_covalent_charge:
                        node.feature_type = FeatureType.covalent_metallic
                    else:
                        # this is a metal bond but it has very low charge
                        node.feature_type = FeatureType.shallow_covalent_metallic
                    # we also go ahead and set the neighboring atoms to avoid some
                    # CrystalNN calculations
                    node._coord_indices = [int(atom0), int(atom1)]
                elif node.charge > self.min_covalent_charge:
                    node.feature_type = FeatureType.covalent
                    node._coord_indices = [int(atom0), int(atom1)]

        # note we've labeled our covalent features
        self._labeled_covalent = True

    def _mark_lonepairs(self):
        logging.info("Marking lone-pair features")
        # lone-pairs separate off from covalent bonds or rarely from an ionic
        # core/shell. When breaking from covalent bonds, the lone-pair will only
        # have one nearest neighbor in the molecule. When breaking off of
        # an ionic core/shell, the lone-pair will at some point be part of a
        # domain surrounding a single atom.

        # POSSIBLE BUG:
        # In small cells, if a feature neighbors the same atoms at different
        # translations and both belong to the same molecule/domain, it may
        # be mislabeled as a lone-pair instead of a metal bond
        assigned_nodes = []
        for node in self.bifurcation_graph.irreducible_nodes:
            if not node.feature_type in [
                None,
                FeatureType.unknown,
                FeatureType.shallow_covalent_metallic,
            ]:
                # NOTE: exclude shallow covalent metal features in case these end up being
                # misassigned deep shells. Not sure if this is necessary
                assigned_nodes.append(node.key)

        nodes = self.bifurcation_graph.sorted_reducible_nodes.copy()
        nodes.reverse()
        for node in nodes:
            # skip nodes that don't contain atoms
            if len(node.contained_atoms) == 0:
                continue

            # if none of this nodes children have an assignment, they are mislabeled
            # shells
            all_labeled = True
            none_labeled = True
            some_covalent = False
            some_atom = False
            for child in node.deep_children:
                if child.is_reducible:
                    continue
                # note if we have a covalent feature
                if child.feature_type in [
                    FeatureType.covalent,
                    FeatureType.covalent_metallic,
                ]:
                    some_covalent = True
                # note if we have some shell feature
                if child.feature_type in FeatureType.atomic_types:
                    some_atom = True
                # If this node is assigned, note that not all are unlabeled
                if child.key in assigned_nodes:
                    none_labeled = False
                # If this node isn't assigned, note not all are labeled
                else:
                    all_labeled = False

            # skip if all children are already labeled
            if all_labeled:
                continue

            # if we have exactly one atom in this domain, we may have an atom
            # that's ionically bonded with a lone-pair (e.g. SnO). We may also
            # have mislabeled a shell earlier due to a particularly deep feature
            if len(node.contained_atoms) == 1 and not node.is_infinite:
                # if none are labeled these are mislabeled shells
                if none_labeled:
                    for child in node.deep_children:
                        child.feature_type = FeatureType.deep_shell
                        assigned_nodes.append(child.key)
                # if some are atomic and none are covalent, we have ionic
                # lone-pairs or small metallic features. We check that they
                # have reasonable charge
                elif some_atom and not some_covalent:
                    for child in node.deep_children:
                        if child.is_reducible or child.key in assigned_nodes:
                            continue
                        if child.charge > self.min_covalent_charge:
                            child.feature_type = FeatureType.lone_pair
                            assigned_nodes.append(child.key)
                # otherwise these features are some form of metallic and we just
                # continue
                continue

            # if nothing is labeled and we surround more than one atom, we have
            # a group of metal features. We mark them as assigned and continue
            if none_labeled:
                for child in node.deep_children:
                    assigned_nodes.append(child.key)
                continue

            # Any other lone-pairs must come along with a covalent bond. If we
            # don't have one we continue
            # NOTE: Should I mark these as checked?
            if not some_covalent:
                continue

            # otherwise, check each unassigned node to see if it has exactly
            # one neighbor in this domain
            for child in node.deep_children:
                # skip reducible nodes and nodes with assignments
                if child.is_reducible or child.key in assigned_nodes:
                    continue
                # check how many neighs are in this domain
                neighs_in_domain = 0
                for atom_idx in child.coord_indices:
                    if atom_idx in node.contained_atoms:
                        neighs_in_domain += 1
                    if neighs_in_domain > 1:
                        break
                # if theres more than one neighbor in the molecule, continue
                if neighs_in_domain > 1:
                    continue
                # if we have a node with reasonable charge, mark it at a lone-pair
                if child.charge > self.min_covalent_charge:
                    child.feature_type = FeatureType.lone_pair
                    assigned_nodes.append(child.key)
                    # reset coord env so that it gets calculated as 1
                    child._coord_indices = None

    def _mark_metallic(self):
        logging.info("Marking metallic features")
        # we mark metallic features simply based on their depth
        for node in self.bifurcation_graph.unassigned_nodes:
            if node.depth < self.max_metal_depth:
                node.feature_type = FeatureType.metallic

    def _mark_metallic_or_bare(self):
        if not self._labeled_multi:
            logging.info("Marking multi-centered and bare electron features")
        else:
            logging.info("Re-marking multi-centered and bare electron features")
        self._labeled_multi = True
        # The remaining features are various types of non-nuclear attractors.
        # We separate them into metallic or "bare electrons" based on a series
        # of cutoffs

        # first we need to calculate how far each feature is beyond nearby
        # atom radii
        for node in self.bifurcation_graph.irreducible_nodes:
            # get dists to atoms
            atom_dists = node.atom_dists
            # get atom radii
            atom_radii = self.atom_elf_radii
            # get the lowest distance beyond an atom
            dist_beyond_atoms = atom_dists - atom_radii
            node._nearest_atom_sphere = np.argmin(dist_beyond_atoms)
            node._dist_beyond_atom = dist_beyond_atoms.min()

        # NOTE: These are very likely to change in the future
        conditions = np.array(
            [
                self.min_electride_elf_value,
                self.min_electride_depth,
                self.min_electride_charge,
                self.min_electride_volume,
                self.min_electride_dist_beyond_atom,
            ]
        )
        for node in self.bifurcation_graph.irreducible_nodes:
            # skip nodes that aren't metallic/bare/unlabeled
            if node.feature_type not in [
                FeatureType.metallic,
                FeatureType.bare_electron,
                FeatureType.unknown,
            ]:
                continue
            condition_test = np.array(
                [
                    node.max_value,
                    node.depth_to_infinite,  # Note we use the depth to an infinite connection rather than true depth
                    node.charge,
                    node.volume,
                    node.dist_beyond_atom,
                ]
            )
            if np.all(condition_test > conditions):
                node.feature_type = FeatureType.bare_electron
            else:
                node.feature_type = FeatureType.multi_centered

    # This is a method aimed at giving a feature a score on how "bare" it is
    # with the goal of distinguishing metals from electrides. We will leave it
    # out until we have a better metric
    # def _mark_bare_electron_indicator(self):
    #     """
    #     Takes in a bifurcation graph and calculates an electride character
    #     score for each valence feature. Electride character ranges from
    #     0 to 1 and is the combination of several different metrics:
    #     ELF value, charge, depth, volume, and atom distance.
    #     """
    #     # create a structure object with oxidation states for improved
    #     # crystalnn
    #     temp_structure = self.structure.copy()
    #     temp_structure.add_oxidation_state_by_guess()

    #     nodes = self.bifurcation_graph.get_feature_nodes(["metallic", "bare electron"])
    #     for node in track(nodes, description="Calculating bare electron character"):

    #         # We want to get a metric of how "bare" each feature is. To do this,
    #         # we need a value that ranges from 0 to 1 for each attribute we have
    #         # available. We can combine these later with or without weighting to
    #         # get a final value from 0 to 1.
    #         # First, the ELF value already ranges from 0 to 1, with 1 being more
    #         # localized. We don't need to alter this in any way.
    #         elf_contribution = node.max_elf

    #         # next, we look at the charge. If we are using a spin polarized result
    #         # the maximum amount should be 1. Otherwise, the value could be up
    #         # to 2. We make a guess at what the value should be here
    #         charge = node.charge
    #         if self._spin_polarized:
    #             max_value = 1
    #         else:
    #             if 0 < charge <= 1.1:
    #                 max_value = 1
    #             else:
    #                 max_value = 2
    #         # Anything significantly below our indicates metallic character and
    #         # anything above indicates a feature like a covalent bond with pi contribution.
    #         # we use a symmetric linear equation around our max value that maxes out at 1
    #         # where the charge exactly matches and decreases moving away.
    #         if charge <= max_value:
    #             charge_contribution = charge / max_value
    #         else:
    #             # If somehow our charge is much greater than the value, we will
    #             # get a negative value, so we use a max function to prevent this
    #             charge_contribution = max(-charge / max_value + 2, 0)

    #         # Now we look at the depth of our feature. Like the ELF value, this
    #         # can only be from 0 to 1, and bare electrons tend to take on higher
    #         # values. Therefore, we leave this as is.
    #         # NOTE: The depth here is the depth to the first irreducible feature
    #         # that extends infinitely in at least one direction. This is different
    #         # from the technical "depth" used in ELF topology analysis, but is
    #         # more related to how isolated a feature is.
    #         depth_contribution = node.depth_to_infinite

    #         # Next is the volume. Bare electrons are usually thought of as being
    #         # similar to a free s-orbital with a similar size to a hydride. Therefore
    #         # we use the hydride crystal radius to calculate an ideal volume and set
    #         # this contribution as a fraction of this, capping at 1.
    #         hydride_radius = 1.34  # Taken from wikipedia and subject to change
    #         hydride_volume = 4 / 3 * 3.14159 * (hydride_radius**3)
    #         volume_contribution = min(node.volume / hydride_volume, 1)

    #         # Next is the radius which is based on the average distance to the
    #         # features surface. We need
    #         # to set an ideal distance corresponding to 1 and a minimum distance
    #         # corresponding to 0. The ideal distance is the sum of the atoms radius
    #         # plus the radius of a true bare electron (approx the H- radius). The
    #         # minimum radius should be 0, corresponding to the radius of the atom.
    #         # Thus covalent bonds should have a value of 0 and lone-pairs may
    #         # be slightly within this radius, also recieving a value of 0.
    #         radius = node.average_surface_dist

    #         # Now that we have a radius, we need to get a metric of 0-1.
    #         dist_contribution = radius / hydride_radius
    #         # limit to a range of 0 to 1
    #         dist_contribution = min(max(dist_contribution, 0), 1)

    #         # We want to keep track of the full values in a convenient way
    #         unnormalized_contributors = np.array(
    #             [
    #                 elf_contribution,
    #                 charge,
    #                 depth_contribution,
    #                 node.volume,
    #                 radius,
    #             ]
    #         )
    #         # Finally, our bare electron indicator is a linear combination of
    #         # the indicator above. The contributions are somewhat arbitrary, but
    #         # are based on chemical intuition. The ELF and charge contributions
    #         contributers = np.array(
    #             [
    #                 elf_contribution,
    #                 charge_contribution,
    #                 depth_contribution,
    #                 volume_contribution,
    #                 dist_contribution,
    #             ]
    #         )
    #         weights = np.array(
    #             [
    #                 0.2,
    #                 0.2,
    #                 0.2,
    #                 0.2,
    #                 0.2,
    #             ]
    #         )
    #         bare_electron_indicator = np.sum(contributers * weights)

    #         # we update our node to include this information
    #         node.unnormalized_bare_electron_indicator = unnormalized_contributors
    #         node.bare_electron_indicator = bare_electron_indicator
    #         node.bare_electron_scores = contributers
