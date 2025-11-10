# -*- coding: utf-8 -*-

"""
Contains the main class for running elf topology analysis.
"""

import logging
import math
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
from baderkit.core.utilities.basic import mutiple_dists
from baderkit.core.utilities.coord_env import check_all_covalent
from baderkit.core.utilities.file_parsers import Format

from .elf_labeler_numba import get_feature_edges, get_min_avg_feat_surface_dists
from .elf_radii import ElfRadiiTools

Self = TypeVar("Self", bound="ElfLabeler")
# TODO:

# - Add more bond types: metal and metal/covalent

# - add a method to print a more traditional bifurcation plot?
# - create method to get all radii around each atom using new radii method:
# 1. get atom neighbors (up to distance or dynamically)
# 2. find radii for closest and add to official neighbors
# 3. go to next closest and get radii. See if they fit under planes made
# by other radii
# 4. Continue until all neighbors found. (Whats the best stop condition?)
# - rework BadELF voxel assignment using Numba. That should just be so much
# faster. I could probably even reimplement exact voxel splitting
# - Update simmate workflows (and badelf -_- )


class ElfLabeler:
    """
    A class for finding electride sites from an ELFCAR.
    """

    _labeled_covalent = False
    _labeled_metallic = False
    _spin_system = "total"

    def __init__(
        self,
        charge_grid: Grid,
        reference_grid: Grid,
        ignore_low_pseudopotentials: bool = False,
        shared_shell_ratio: float = 1 / 3,
        combine_shells: bool = True,
        min_covalent_charge: float = 0.6,
        min_covalent_angle: float = 135,
        min_electride_elf_value: float = 0.5,
        min_electride_depth: float = 0.2,
        min_electride_charge: float = 0.5,
        min_electride_volume: float = 10,
        min_electride_dist_beyond_atom: float = 0.3,
        crystalnn_kwargs: dict = {"distance_cutoffs": None},
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

        # define cutoff variables
        # TODO: These should be hidden variables to allow for setter methods
        self.shared_shell_ratio = shared_shell_ratio
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
        self._quasi_atom_elf_radii = None
        self._quasi_atom_elf_radii_types = None
        self._atom_nn_elf_radii = None
        self._quasi_atom_nn_elf_radii = None
        self._atom_nn_elf_radii_types = None
        self._quasi_atom_nn_elf_radii_types = None
        self._nearest_neighbor_data = None
        self._quasi_atom_nearest_neighbor_data = None

        self._feature_labels = None
        self._feature_structure = None

        # create a bader object
        self.bader = Bader(
            charge_grid=charge_grid, reference_grid=reference_grid, **kwargs
        )

    ###########################################################################
    # Calculated Properties
    ###########################################################################
    # TODO: Make these reset on a change similar to the Bader class. Add docs

    ###########################################################################
    # Bifurcation Properties
    ###########################################################################

    @property
    def bifurcation_graph(self) -> BifurcationGraph:
        if self._bifurcation_graph is None:
            self._get_bifurcation_graph()
        return self._bifurcation_graph

    @property
    def bifurcation_plot(self) -> go.Figure:
        if self._bifurcation_plot is None:
            self._bifurcation_plot = self.bifurcation_graph.get_plot()
        return self._bifurcation_plot

    ###########################################################################
    # Atom Properties
    ###########################################################################
    @property
    def structure(self) -> Structure:
        return self.reference_grid.structure

    @property
    def quasi_atom_structure(self) -> Structure:
        return self.bifurcation_graph.quasi_atom_structure

    @property
    def labeled_structure(self) -> Structure:
        return self.bifurcation_graph.labeled_structure

    @property
    def nearest_neighbor_data(self):
        if self._nearest_neighbor_data is None:
            self._nearest_neighbor_data = self._get_nearest_neighbor_data(self.structure)
        return self._nearest_neighbor_data
    
    @property
    def quasi_atom_nearest_neighbor_data(self):
        if self._quasi_atom_nearest_neighbor_data is None:
            self._quasi_atom_nearest_neighbor_data = self._get_nearest_neighbor_data(self.quasi_atom_structure)
        return self._quasi_atom_nearest_neighbor_data

    @property
    def atom_elf_radii(self) -> NDArray[np.float64]:
        if self._atom_elf_radii is None:            
            self._atom_elf_radii, self._atom_elf_radii_types = self._get_atom_elf_radii(
                self.structure, 
                self.atom_nn_elf_radii, 
                self._atom_nn_elf_radii_types, 
                self.nearest_neighbor_data,
                )

        return self._atom_elf_radii

    @property
    def atom_elf_radii_types(self) -> NDArray[np.float64]:
        if self._atom_elf_radii_types is None:
            # run labeling and radii calc by calling our bifurcation graph
            self.atom_elf_radii
        return np.where(self._atom_elf_radii_types, "covalent", "ionic")

    @property
    def quasi_atom_elf_radii(self) -> NDArray[np.float64]:
        if self._quasi_atom_elf_radii is None:
            self._quasi_atom_elf_radii, self._quasi_atom_elf_radii_types = self._get_atom_elf_radii(
                self.quasi_atom_structure, 
                self.quasi_atom_nn_elf_radii, 
                self._quasi_atom_nn_elf_radii_types, 
                self.quasi_atom_nearest_neighbor_data,
                )
        return self._quasi_atom_elf_radii
    
    @property
    def quasi_atom_elf_radii_types(self) -> NDArray[np.float64]:
        if self._quasi_atom_elf_radii_types is None:
            # run labeling and radii calc by calling our bifurcation graph
            self.quasi_atom_elf_radii
        return np.where(self._quasi_atom_elf_radii_types, "covalent", "ionic")

    @property
    def atom_nn_elf_radii(self) -> NDArray[np.float64]:
        if self._atom_nn_elf_radii is None:
            if self._labeled_covalent is None:
                self.bifurcation_graph
            self._atom_nn_elf_radii, self._atom_nn_elf_radii_types = self._get_nn_atom_elf_radii(self.structure)
        return self._atom_nn_elf_radii

    @property
    def atom_nn_elf_radii_types(self) -> NDArray[np.float64]:
        if self._atom_nn_elf_radii_types is None:
            # call radii method
            self.atom_nn_elf_radii
        return np.where(self._atom_nn_elf_radii_types, "covalent", "ionic")
    
    @property
    def quasi_atom_nn_elf_radii(self) -> NDArray[np.float64]:
        if self._quasi_atom_nn_elf_radii is None:
            # make sure labeled bifurcation graph exists
            if self._labeled_covalent is None:
                self.bifurcation_graph
            # if there are no quasi atoms, just return the results for the base
            # structure (avoid repeat calc)
            if len(self.structure) == len(self.quasi_atom_structure):
                self._quasi_atom_nn_elf_radii = self.atom_nn_elf_radii
                self._quasi_atom_nn_elf_radii_types = self._atom_nn_elf_radii_types
            else:
                self._quasi_atom_nn_elf_radii, self._quasi_atom_nn_elf_radii_types = self._get_nn_atom_elf_radii(self.quasi_atom_structure)
        return self._quasi_atom_nn_elf_radii
    
    @property
    def quasi_atom_nn_elf_radii_types(self) -> NDArray[np.float64]:
        if self._quasi_atom_nn_elf_radii_types is None:
            # call radii method
            self.quasi_atom_nn_elf_radii
        return np.where(self._atom_nn_elf_radii_types, "covalent", "ionic")

    ###########################################################################
    # Feature Properties
    ###########################################################################

    def _get_feature_properties(self, property_name: str):
        features = []
        for node in self.bifurcation_graph.irreducible_nodes:
            features.append(getattr(node, property_name, None))
        return features

    @property
    def feature_types(self):
        return self._get_feature_properties("feature_type")

    @property
    def feature_basins(self):
        return self._get_feature_properties("basins")

    @property
    def feature_frac_coords(self):
        return self._get_feature_properties("frac_coords")

    @property
    def feature_average_frac_coords(self):
        return np.array(self._get_feature_properties("average_frac_coords"))

    @property
    def feature_charges(self):
        return np.array(self._get_feature_properties("charge"))

    @property
    def feature_volumes(self):
        return np.array(self._get_feature_properties("volume"))

    @property
    def feature_coord_atoms(self):

        return self._get_feature_properties("coord_atom_indices")

    @property
    def feature_coord_atom_dists(self):

        return self._get_feature_properties("coord_atom_dists")

    @property
    def feature_quasi_coord_atoms(self):
        return self._get_feature_properties("coord_quasi_atom_indices")

    @property
    def feature_quasi_coord_atom_dists(self):
        return self._get_feature_properties("coord_quasi_atom_dists")

    @property
    def feature_min_surface_dists(self):
        return np.array(self._get_feature_properties("min_surface_dist"))

    @property
    def feature_avg_surface_dists(self):
        return np.array(self._get_feature_properties("avg_surface_dist"))

    ###########################################################################
    # Helpful Methods
    ###########################################################################
    def feature_indices_by_type(self, feature_types: list[FeatureType | str]):
        return np.array(
            [i for i, feat in enumerate(self.feature_types) if feat in feature_types],
            dtype=np.int64,
        )

    def get_oxidation_and_volumes_from_potcar(
        self, potcar_path: Path = "POTCAR", use_quasi_atoms: bool = True, **kwargs
    ):
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

    def get_charges_and_volumes(
        self,
        splitting_method: Literal[
            "weighted_dist", "pauling", "equal", "dist", "nearest"
        ] = "pauling",
        use_quasi_atoms: bool = True,
        **kwargs,
    ):
        """
        Assign charge from each feature to their associated atoms. For features
        that have multiple neighbors (e.g. covalent bonds), several options
        are provided for how to divide the charge to nearby neighbors.
        If use_quasi_atoms is set to True, features with electride like character
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

        use_quasi_atoms : bool, optional
            If True, features labeled as bare electrons will be treated as quasi
            atoms. They will receive partial charge from other shared features
            and their charge/volume will be appended after the atoms'.

        Returns
        -------
        None.

        """
        if use_quasi_atoms:
            structure = self.quasi_atom_structure
        else:
            structure = self.structure

        # create an array to store atom charges and volumes
        atom_charge = np.zeros(len(structure), dtype=np.float64)
        atom_volume = np.zeros(len(structure), dtype=np.float64)

        # if using pauling, get all electronegativities
        if splitting_method == "pauling":
            pauling_ens = np.array([i.specie.X for i in structure])
            pauling_ens = np.nan_to_num(pauling_ens, nan=2.2)

        quasi_num = 0
        for feature_idx in range(len(self.feature_types)):
            charge = self.feature_charges[feature_idx]
            volume = self.feature_volumes[feature_idx]
            if (
                use_quasi_atoms
                and self.feature_types[feature_idx] in FeatureType.bare_types
            ):
                # assign charge/volume to self
                struc_idx = len(self.structure) + quasi_num
                atom_charge[struc_idx] += charge
                atom_volume[struc_idx] += volume
                quasi_num += 1
                continue

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
                nearest = np.argmin(self.feature_coord_atom_dists[feature_idx])
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
                if use_quasi_atoms:
                    dists = self.feature_quasi_coord_atom_dists[feature_idx].copy()
                else:
                    dists = self.feature_coord_atom_dists[feature_idx].copy()
                # invert and normalize
                dists = 1 / dists
                dists /= dists.sum()
                # add for each atom
                for coord_idx, atom in enumerate(coord_atoms):
                    atom_charge[atom] += charge * dists[coord_idx]
                    atom_volume[atom] += volume * dists[coord_idx]

            elif splitting_method == "weighted_dist":
                # get the dist to each coordinated atom and their radii
                if use_quasi_atoms:
                    dists = self.feature_quasi_coord_atom_dists[feature_idx].copy()
                    atom_radii = self.quasi_atom_elf_radii[coord_atoms]
                else:
                    dists = self.feature_coord_atom_dists[feature_idx].copy()
                    atom_radii = self.atom_elf_radii[coord_atoms]

                # calculate the weighted contribution to each atom and normalize
                weight = atom_radii / dists
                weight /= weight.sum()
                # add for each atom
                for coord_idx, atom in enumerate(coord_atoms):
                    atom_charge[atom] += charge * weight[coord_idx]
                    atom_volume[atom] += volume * weight[coord_idx]

            elif splitting_method == "nearest":
                # assign all charge/volume to the closest atom
                nearest = np.argmin(self.feature_coord_atom_dists[feature_idx])
                atom_charge[nearest] += charge
                atom_volume[nearest] += volume
            else:
                raise ValueError(
                    f"'{splitting_method}' is not a valid splitting method"
                )

        return atom_charge, atom_volume

    def get_feature_labels(
        self,
        included_features: list[str] = [
            "covalent",
            "metallic",
            "bare electron",
            "electride",
        ],
        return_structure: bool = True,
    ):

        # Get the original basin atom assignments
        basin_atoms = self.bader.basin_atoms

        # get indices of requested type
        feature_indices = self.feature_indices_by_type(included_features)

        # reassign basin atoms
        new_atom_idx = len(self.structure)
        for feat_idx in feature_indices:
            basins = self.feature_basins[feat_idx]
            basin_atoms[basins] = new_atom_idx
            new_atom_idx += 1

        # NOTE: append -1 so that vacuum gets assigned to -1 in the atom_labels
        # array
        basin_atoms = np.insert(basin_atoms, len(basin_atoms), -1)

        # reassign labels
        feature_labels = basin_atoms[self.bader.basin_labels]

        # get corresponding structure if requested
        if return_structure:
            feature_structure = self.get_feature_structure(included_features)
            return feature_labels, feature_structure

        # return labels
        return feature_labels

    def get_feature_structure(
        self,
        included_features: list[str] = [
            "covalent",
            "metallic",
            "bare electron",
            "electride",
        ],
    ):
        """
        Generates a PyMatGen Structure object with dummy atoms for each requested
        feature.
        """

        # Create a new structure without oxidation states
        structure = self.structure.copy()
        structure.remove_oxidation_states()

        # Add dummy atoms of each type in list
        for feature_type, frac_coords in zip(
            self.feature_types, self.feature_average_frac_coords
        ):
            if feature_type in included_features:
                structure.append(feature_type.dummy_species, frac_coords)

        return structure

    def get_feature_structure_by_index(self, feature_indices: list[int]):
        """
        Generates a PyMatGen Structure object with dummy atoms for each requested
        feature.
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
    # Utilities
    ###########################################################################
    
    def _get_nearest_neighbor_data(self, structure: Structure):
        nearest_neighs = self.cnn.get_all_nn_info(structure)
        # we just want the species and frac coords of each atom's neighbors
        sites = []
        neighs = []
        frac_coords = []
        cart_coords = []
        for site_idx, neigh_list in enumerate(nearest_neighs):
            for neigh_dict in neigh_list:
                sites.append(site_idx)
                neighs.append(neigh_dict["site_index"])
                frac_coords.append(neigh_dict["site"].frac_coords)
                cart_coords.append(neigh_dict["site"].coords)
        # convert to arrays
        sites = np.array(sites, dtype=np.uint32)
        neighs = np.array(neighs, dtype=np.uint32)
        frac_coords = np.array(frac_coords, dtype=np.float64)
        cart_coords = np.array(cart_coords, dtype=np.float64)
        # calculate distances (since pymatgen doesn't include these in the
        # summary for some reason)
        site_cart_coords = structure.cart_coords[sites]
        dists = mutiple_dists(site_cart_coords, cart_coords)

        return (sites, neighs, frac_coords, dists)
    
    def _get_nn_atom_elf_radii(self, structure: Structure):
        if not self._labeled_covalent:
            raise Exception("Covalent features must be labeled for reliable radii.")
        
        # First we get our neighbor arrays
        site_indices, neigh_indices, neigh_frac_coords, neigh_dists = (
            self.quasi_atom_nearest_neighbor_data
        )

        # Now we get a labeled structure including covalent, metallic, and bare
        # electron features
        included_types = FeatureType.valence_types.copy()
        feature_labels, feature_structure = self.get_feature_labels(
            included_features=included_types
        )

        # Calculate all radii
        radii_tools = ElfRadiiTools(
            grid=self.reference_grid,
            feature_labels=feature_labels,
            feature_structure=feature_structure,
            override_structure=structure,
        )
        return (
            radii_tools.get_all_neigh_elf_radii_and_type(
                site_indices,
                neigh_indices,
                neigh_frac_coords,
                neigh_dists,
            )
        )
        
    @staticmethod
    def _get_atom_elf_radii(structure, all_radii, all_radii_types, nn_data):
        # NOTE: This is the smallest radius of each atom which is not always
        # along the bond to the nearest neighbor, but typically is. For
        # example CdPt3 has 2 Pt atoms with a Cd an dPt atom tied for the
        # nearest.

        site_indices, neigh_indices, neigh_coords, neigh_dists = nn_data

        # sort radii
        sorted_indices = np.argsort(all_radii)
        sorted_sites = site_indices[sorted_indices]

        # get first instance of each atom
        radii = np.empty(len(structure), dtype=np.float64)
        radii_types = np.empty(len(structure), dtype=all_radii_types.dtype)
        for i in range(len(structure)):
            first_index = np.argmax(sorted_sites == i)
            radii[i] = all_radii[sorted_indices[first_index]]
            radii_types[i] = all_radii_types[sorted_indices[first_index]]
        return radii, radii_types


    def _calculate_feature_surface_dists(self):
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
    # Read methods
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
    # Methods for writing results
    ###########################################################################
    def write_bifurcation_plot(
        self,
        filename: str | Path,
    ):
        plot = self.bifurcation_plot
        # make sure path is a Path object
        filename = Path(filename)
        # add .html if filename doesn't include it
        filename_html = filename.with_suffix(".html")
        plot.write_html(filename_html)

    def write_feature_basins(
        self,
        feature_indices: list[int],
        directory: str | Path = None,
        include_dummy_atoms: bool = False,
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

        Returns
        -------
        None.

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
            basins = self.bader.feature_basins[feat_idx]
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
        directory: str | Path = None,
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

        Returns
        -------
        None.

        """
        self.write_feature_basins(
            feature_indices=np.arange(len(self.feature_charges), dtype=int), **kwargs
        )

    def write_features_by_type(
        self,
        included_types: list[FeatureType],
        **kwargs,
    ):
        """
        Writes the bader basins associated with all features of the selected type

        Parameters
        ----------
        **kwargs :
            See :meth:`write_feature_basins`.

        Returns
        -------
        None.

        """
        feature_indices = self.feature_indices_by_type(included_types)
        self.write_feature_basins(
            feature_indices=feature_indices,
            **kwargs,
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

        Returns
        -------
        None.

        """
        feature_indices = self.feature_indices_by_type(included_types)
        self.write_feature_basins_sum(feature_indices=feature_indices, **kwargs)

    ###########################################################################
    # Core Graph Construction
    ###########################################################################

    def _get_bifurcation_graph(self):
        """
        This will construct a BifurcationGraph class.
        Each node will contain information on whether it is
        reducible/irreducible, atomic/valent, etc.

        This method is largely meant to be called through the get_bifurcation_graphs
        method.
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
        # "quasi atom"
        logging.info("Calculating atomic radii")
        self._atom_nn_elf_radii, self._atom_nn_elf_radii_types = self._get_nn_atom_elf_radii(self.structure)

        # Next we mark our metallic/bare electrons. These currently have a set
        # of rather arbitrary cutoffs to distinguish between them. In the future
        # I would like to perform a comprehensive study.
        self._mark_metallic_or_bare()
        
        # BUGFIX: The atomic radii are somewhat dependent on if we consider a
        # feature to be metallic or a qausi-atom. We first calculate the radii
        # assuming quasi atoms to allow for the best chance of a feature getting
        # through the "dist_beyond_atom" cutoff. Then we recalculate and relabel
        # so that metallic systems have the proper label
        # Recalculate radii with our updated markers
        logging.info("Re-calculating atomic radii")
        self._atom_elf_radii = None
        self._atom_nn_elf_radii, self._atom_nn_elf_radii_types = self._get_nn_atom_elf_radii(self.structure)
        
        # Re-mark metallic/electrides
        self._mark_metallic_or_bare()

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

    def _initialize_bifurcation_graph(self):
        self._bifurcation_graph = BifurcationGraph.from_labeler(self)

    def _mark_cores(self):
        logging.info("Marking atomic cores")
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
                node.feature_type = FeatureType.core

    def _mark_shells(self):
        logging.info("Marking atomic shells")
        # shells are "reducible" features that surround exactly one atom. 
        # Often they shouldn't really be reducible, but appear to be so due to
        # the inability of the voxel grid to represent a sphere well
        # The main difficulty is distinguishing them from heterogeneous covalent
        # bonds.

        # as a first step, we label any features that were combined when the
        # graph was generated due to being very shallow. These will be marked
        # as irreducible cages and contain one atom.
        # BUG-FIX: There is also a possibility if the bader basin reduction tolerance
        # is too high that a cage will be combined during the bader algorithm and
        # marked as a point. We mark this as well
        for node in self.bifurcation_graph.unassigned_nodes:
            if (
                node.domain_subtype in [DomainSubtype.irreducible_cage, DomainSubtype.irreducible_point]
                and len(node.contained_atoms) == 1
            ):
                node.feature_type = "shell"

        # Now we label shells that may be slightly deeper. Regardless of depth,
        # shells will always surround a single atom. To distinguish them from
        # heterogenous covalent bonds (which also may surround 1 atom), we gauge
        # the degree to which the potential shell features belong to the atom's
        # shell vs. another atoms shell. This can be thought of as a measure of
        # ionicity. To measure this, we compare the highest ELF value where
        # the feature fully surrounds the single atom to the highest value where
        # it surrounds at least one other atom

        shell_nodes = []
        for node in self.bifurcation_graph.reducible_nodes:
            # skip nodes that don't surround a single atom or have a dimensionality above 0
            if node.is_infinite or len(node.contained_atoms) != 1:
                continue
            # skip nodes that have children that also contain the single atom
            if any([len(i.contained_atoms) == 1 for i in node.children]):
                continue

            # we want to find the highest ELF value where these features contribute
            # to another atoms outer shell. This isn't always the first
            # parent that contains multiple atoms, as shells and heterogenous
            # covalent bonds usually connect to each other before surrounding
            # a different atom. Instead we want the first ancestor that surrounds
            # a new atom without being in contact with that atoms shells/core

            found_parent = False
            included_atoms = node.contained_atoms
            for parent in node.ancestors:
                if len(parent.contained_atoms) == len(included_atoms):
                    continue
                # check if all atoms are included in at least one child
                distinct_atoms = []
                for child in parent.children:
                    distinct_atoms.extend(child.contained_atoms)
                if not all([i in distinct_atoms for i in parent.contained_atoms]):
                    highest_neighbor_shell = parent.max_value
                    found_parent = True
                    break
                included_atoms = parent.contained_atoms

            if not found_parent:
                # There is no appropriate parent and this must be a shell
                shell_nodes.append(node)
                continue
            # get the highest point where this feature contains the atom
            highest_nearest_shell = node.max_value
            # check if the ratio is below our cutoff
            if (
                highest_neighbor_shell / highest_nearest_shell
            ) < self.shared_shell_ratio:
                shell_nodes.append(node)

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
            # These atoms won't connect until a much lower value
            # POSSIBLE BUG: Atoms with translational symmetry might be counted
            # as part of multiple covalent systems and break this in small
            # unit cells. This would require knowing which atom image is contained
            # which we currently don't track and would require a large rework

            # find the first parent containing at least one of the neighboring
            # atoms and containing both
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
            if not one_atom is None and not both_atoms is None:
                if (both_atoms / one_atom) < self.shared_shell_ratio:
                    is_covalent = False
                else:
                    is_covalent = True

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
                    node._coord_atom_indices = [int(atom0), int(atom1)]
                elif node.charge > self.min_covalent_charge:
                    node.feature_type = FeatureType.covalent
                    node._coord_atom_indices = [int(atom0), int(atom1)]

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
            for child in node.deep_children:
                if child.is_reducible:
                    continue
                if child.key in assigned_nodes:
                    none_labeled = False
                else:
                    all_labeled = False

            # skip if all children are already labeled
            if all_labeled:
                continue

            # if none are labeled, check if this should be a shell. If not, these
            # are some form of metal bond so we continue
            if none_labeled:
                if len(node.contained_atoms) == 1 and not node.is_infinite:
                    for child in node.deep_children:
                        child.feature_type = FeatureType.deep_shell
                for child in node.deep_children:
                    assigned_nodes.append(child.key)
                continue

            # otherwise, check each unassigned node to see if it has exactly
            # one neighbor in this domain
            for child in node.deep_children:
                # skip reducible nodes and nodes with assignments
                if child.is_reducible or child.key in assigned_nodes:
                    continue
                # check how many neighs are in this domain
                neighs_in_domain = 0
                for atom_idx in child.coord_atom_indices:
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
                    child._coord_atom_indices = None

    def _mark_metallic_or_bare(self):
        if not self._labeled_metallic:
            logging.info("Marking metallic and bare electron features")
        else:
            logging.info("Re-marking metallic and bare electron features")
        self._labeled_metallic = True
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
            if node.feature_type not in [FeatureType.metallic, FeatureType.bare_electron, FeatureType.unknown]:
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
                node.feature_type = FeatureType.metallic

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
