# -*- coding: utf-8 -*-

import json
import logging
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pymatgen.io.vasp import Potcar
from scipy.ndimage import label
from tqdm import tqdm

from baderkit.core import ElfLabeler, Grid, Structure
from baderkit.core.badelf.badelf_numba import get_badelf_assignments
from baderkit.core.bader.methods.shared_numba import (
    get_edges,
    get_min_avg_surface_dists,
)
from baderkit.core.labelers.bifurcation_graph.enum_and_styling import (
    FEATURE_DUMMY_ATOMS,
    FeatureType,
)
from baderkit.core.utilities.file_parsers import Format
from baderkit.core.utilities.voronoi import get_cell_wrapped_voronoi

# TODO:

# 4. push new release and update warrenapp info
# 5. update simmate workflows/database
# 6. update simmate docs
# 7. request new release from Jack


class Badelf:
    """
    Class for performing charge analysis using the electron localization function
    (ELF). For information on specific methods, see our [docs](https://sweav02.github.io/baderkit/).

    For more in-depth ELF analysis we recommend using the ElfLabeler class.

    This class only performs analysis on one spin system.

    Parameters
    ----------
    reference_grid : Grid
        A badelf app Grid like object used for partitioning the unit cell
        volume. Usually contains ELF.
    charge_grid : Grid
        A badelf app Grid like object used for summing charge. Usually
        contains charge density.
    method : Literal["badelf", "voronelf", "zero-flux"], optional
        The method to use for partitioning electrides from the nearby
        atoms.
            'badelf' (default)
                Separates electrides using zero-flux surfaces then uses
                planes at atom radii to separate atoms. This may give more reasonable
                results for atoms, particularly in ionic solids. Radii are
                calculated directly from the ELF.
            'voronelf'
                Separates both electrides and atoms using planes at atomic/electride
                radii. This is not recommended for electrides that are not
                spherical, but may provide better results for those that are.
                Radii are calculated directly from the ELF.
            'zero-flux'
                Separates electrides and atoms using zero-flux surface. This
                is the most traditional ELF analysis, but may display some
                bias towards atoms with higher ELF values. Results for electride
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
                atom a default of 2.2 is used (including for electrides).
            'equal'
                Charge is distributed equaly to each neighboring atom/electride
                (calculated using CrystalNN)
            'dist'
                Charge is distributed such that more charge is given to the
                closest atoms. Portions are determined by normalizing the sum
                of (1/dist) to each neighboring atom.
            'nearest'
                Gives all charge to the nearest atom or electride site.
    elf_labeler : dict | ElfLabeler, optional
        Keyword arguments to pass to the ElfLabeler class. This includes
        parameters controlling cutoffs for electrides as well as parameters
        controlling the Bader algorithm. Alternatively, an
        ElfLabeler class can be passed directly. The default is {}.

    """

    spin_system = "total"

    def __init__(
        self,
        reference_grid: Grid,
        charge_grid: Grid,
        method: Literal["badelf", "voronelf", "zero-flux"] = "zero-flux",
        shared_feature_splitting_method: Literal[
            "weighted_dist", "pauling", "equal", "dist", "nearest"
        ] = "weighted_dist",
        elf_labeler: dict | ElfLabeler = {},
        **kwargs,
    ):
        assert (
            reference_grid.structure == charge_grid.structure
        ), "Grid structures must be the same."

        if method not in ["badelf", "voronelf", "zero-flux"]:
            raise ValueError(
                """The method setting you chose does not exist. Please select
                  either 'badelf', 'voronelf', or 'zero-flux'.
                  """
            )

        self.reference_grid = reference_grid
        self.charge_grid = charge_grid
        self.method = method
        self.shared_feature_splitting_method = shared_feature_splitting_method

        # We want to use the ElfLabeler. We check if an ElfLabeler class is
        # provided or a dict of kwargs
        self._labeled_structure = None
        if type(elf_labeler) == dict:
            self.elf_labeler_kwargs = elf_labeler
            self.elf_labeler = ElfLabeler(
                charge_grid=charge_grid, reference_grid=reference_grid, **elf_labeler
            )
        else:
            # use provided elf labeler
            self.elf_labeler_kwargs = None
            self.elf_labeler = elf_labeler
        # connect the same bader class.
        self.bader = self.elf_labeler.bader

        # Properties that will be calculated and cached
        self._structure = None
        self._electride_structure = None
        self._species = None

        self._partitioning_planes = None
        self._zero_flux_feature_labels_cache = None
        self._atom_labels = None

        self._electride_dim = None
        self._all_electride_dims = None
        self._all_electride_dim_cutoffs = None

        self._nelectrons = None
        self._charges = None
        self._volumes = None

        self._min_surface_distances = None
        self._avg_surface_distances = None

        self._electrides_per_formula = None
        self._electrides_per_reduced_formula = None

        # TODO: Add vacuum handling to Elf Analyzer and BadELF
        # self._vacuum_charge = None
        # self._vacuum_volume = None

        self._results_summary = None

    ###########################################################################
    # Convenient Properites
    ###########################################################################

    @staticmethod
    def _get_sorted_structure(structure: Structure) -> Structure:
        """
        Sorts a labeled structure such that atoms come first followed by electrides
        and then covalent/metallic features.

        Parameters
        ----------
        structure : Structure
            The labeled structure to sort.

        Returns
        -------
        Structure
            The sorted structure.

        """
        # For our partitioning scheme, we need the structure to be ordered as
        # atoms, electrides, other. This is so that the labeled grid points map
        # to structure indices.
        bare_species = FeatureType.bare_species
        shared_species = FeatureType.shared_species
        atom_sites = []
        bare_electron_sites = []
        shared_sites = []
        for site in structure:
            symbol = site.specie.symbol
            if symbol in bare_species:
                bare_electron_sites.append(site)
            elif symbol in shared_species:
                shared_sites.append(site)
            else:
                atom_sites.append(site)
        # get empty structure
        new_structure = structure.copy()
        new_structure.remove_sites([i for i in range(len(structure))])
        # add back sites in appropriate order
        for sites_list in [atom_sites, bare_electron_sites, shared_sites]:
            for site in sites_list:
                symbol = site.specie.symbol
                coord = site.frac_coords
                new_structure.append(symbol, coord)
        return new_structure

    @property
    def labeled_structure(self) -> Structure:
        """

        Returns
        -------
        Structure
            The system's structure including dummy atoms representing electride
            sites and covalent/metallic bonds.

        """
        if self._labeled_structure is None:
            labeled_structure = self.elf_labeler.get_feature_structure(
                included_features=FeatureType.valence_types
            )
            self._labeled_structure = labeled_structure
        return self._labeled_structure

    @property
    def structure(self) -> Structure:
        """

        Returns
        -------
        Structure
            The unlabeled structure representing the system, i.e. the structure
            with no dummy atoms.

        """
        if self._structure is None:
            # NOTE: We don't just use the structure from one of the grids in
            # case for some reason they differ from a provided structure from
            # the user
            structure = self.labeled_structure.copy()
            # remove all non-atomic sites
            for symbol in FEATURE_DUMMY_ATOMS.values():
                if symbol in structure.symbol_set:
                    structure.remove_species([symbol])
            structure.relabel_sites(ignore_uniq=True)
            self._structure = structure
        return self._structure

    @property
    def electride_structure(self) -> Structure:
        """

        Returns
        -------
        Structure
            The system's structure including dummy atoms representing electride
            sites.

        """
        if self._electride_structure is None:
            # create our elecride structure from our labeled structure.
            # NOTE: We don't just use the structure from the elf labeler in
            # case the user provided their own
            electride_structure = self.structure.copy()
            for site in self.labeled_structure:
                if site.specie.symbol in FeatureType.bare_species:
                    electride_structure.append(
                        FeatureType.bare_electron.dummy_species, site.frac_coords
                    )

            electride_structure.relabel_sites(ignore_uniq=True)
            self._electride_structure = electride_structure
        return self._electride_structure

    @property
    def nelectrides(self) -> int:
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
    def charges(self) -> NDArray:
        """

        Returns
        -------
        NDArray
            The charge associated with each atom and electride site in the system.

        """
        if self._charges is None:
            self._get_voxel_assignments()
        return self._charges.round(10)

    @property
    def volumes(self) -> NDArray:
        """

        Returns
        -------
        NDArray
            The volume associated with each atom and electride site in the system.

        """
        if self._volumes is None:
            self._get_voxel_assignments()
        return self._volumes.round(10)

    @property
    def elf_maxima(self) -> NDArray:
        """

        Returns
        -------
        NDArray
            The maximum ELF value for each atom and electride in the system.

        """

        return self.elf_labeler.atom_max_values_e

    @property
    def _zero_flux_feature_labels(self) -> NDArray:
        """

        Returns
        -------
        NDArray
            An array representing which atoms/dummy atoms each voxel point is
            assigned to.

        """
        if self._zero_flux_feature_labels_cache is None:
            # Use the ElfLabeler's assignments.
            self._zero_flux_feature_labels_cache = self.elf_labeler.get_feature_labels(
                included_features=FeatureType.valence_types,
                return_structure=False,
                return_charge_volume=False,
            )

        return self._zero_flux_feature_labels_cache

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

            # if we have an elf labeler, use its results to get partitioning
            if self.method == "badelf":
                site_indices, neigh_indices, _, _ = (
                    self.elf_labeler.nearest_neighbor_data
                )
                plane_points, plane_vectors = self.elf_labeler._atom_nn_planes
            elif self.method == "voronelf":
                site_indices, neigh_indices, _, _ = (
                    self.elf_labeler.nearest_neighbor_data_e
                )
                plane_points, plane_vectors = self.elf_labeler._atom_nn_planes_e

            # we want to transform our planes to the 26 nearest neighbor cells
            # to ensure that we cover our unit cell.
            # For speed, we can remove planes that contain the entire unit cell
            # and we can remove a full set of planes if none of them contain any
            # part of the unit cell.
            # Finally, we can sort each plane by how much of the unit cell it slices
            # such that planes that are likely to reject a grid point come first

            # first we get wrapped planes
            site_indices, transforms, plane_points, plane_vectors, plane_volumes = (
                get_cell_wrapped_voronoi(
                    site_indices=site_indices,
                    plane_points=plane_points,
                    plane_vectors=plane_vectors,
                )
            )

            # sort planes by site, transform, and volume.
            combined_sort = np.column_stack((plane_volumes, transforms, site_indices))
            sorted_indices = np.lexsort(combined_sort.T)
            transforms = transforms[sorted_indices]
            plane_points = plane_points[sorted_indices]
            plane_vectors = plane_vectors[sorted_indices]
            plane_volumes = plane_volumes[sorted_indices]

            # get plane equations in cartesian coordinates
            plane_vectors = self.reference_grid.frac_to_cart(plane_vectors)
            plane_points = self.reference_grid.frac_to_cart(plane_points)

            # normalize vectors
            plane_vectors = (plane_vectors.T / np.linalg.norm(plane_vectors, axis=1)).T

            # calculate plane equations
            b = -np.einsum("ij,ij->i", plane_vectors, plane_points)

            plane_equations = np.column_stack((plane_vectors, b))

            self._partitioning_planes = (site_indices, transforms, plane_equations)
        return self._partitioning_planes

    @property
    def atom_labels(self) -> NDArray:
        """

        Returns
        -------
        NDArray
            A 3D array with the same shape as the charge grid indicating
            which atom/electride each grid point is assigned to.

        """
        if self._atom_labels is None:
            self._get_voxel_assignments()
        return self._atom_labels

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

        # get the zero-flux labels as a starting point
        labels = self._zero_flux_feature_labels.copy()

        if self.method == "zero-flux":
            # we are done here and can assign charges/volumes immediately
            self._atom_labels = labels
            self._charges, self._volumes = self.elf_labeler.get_charges_and_volumes(
                splitting_method=self.shared_feature_splitting_method,
                use_electrides=True,
            )
        else:
            # In badelf, we want to label our electride basins ahead of time
            if self.method == "badelf":
                # get a mask only at electride indices
                indices = np.array(
                    [
                        i
                        for i in range(
                            len(self.structure), len(self.electride_structure)
                        )
                    ]
                )
                mask = np.isin(labels, indices, invert=True)
                # set regions where we don't want to use zero-flux results to -1
                labels[mask] = -1
                # get the number of atoms in the partitioning structure
                structure_len = len(self.structure)
            elif self.method == "voronelf":
                # we are using the voronoi method with the plane method and don't want
                # to override anything
                structure_len = len(self.electride_structure)

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
                vacuum_mask=self.vacuum_mask,
                min_plane_dist=voxel_dist,
                num_assignments=len(self.electride_structure),
                lattice_matrix=self.reference_grid.matrix,
                sphere_transforms=sphere_transforms,
                transform_dists=transform_dists,
                transform_breaks=transform_breaks,
                max_label=structure_len,
            )

            # convert charges/volumes to correct units
            charges /= self.charge_grid.ngridpts
            volumes = volumes * self.structure.volume / self.charge_grid.ngridpts

            # overwrite zero-flux feature charges/volumes
            self._atom_labels = labels
            self._charges = charges
            self._volumes = volumes

        logging.info("Finished voxel assignment")

    @property
    def all_electride_dims(self) -> list | None:
        """

        Returns
        -------
        list
            The possible dimensions the electride takes on from an ELF value of
            0 to 1. If no electrides are present the value will be None.

        """
        if self._all_electride_dims is None:
            self._get_electride_dimensionality()
        # if there are no electrides we want to return None, but we don't want
        # to rerun the search each time. I mark the dims as -1 to avoid this
        if self._all_electride_dims == -1:
            return None
        return self._all_electride_dims

    @property
    def all_electride_dim_cutoffs(self) -> list:
        """

        Returns
        -------
        list
            The highest ELF value where each dimensionality in the "all_electride_dims"
            property exists.

        """
        if self._all_electride_dim_cutoffs is None:
            self._get_electride_dimensionality()
        if self._all_electride_dim_cutoffs == -1:
            return None
        return self._all_electride_dim_cutoffs

    @property
    def electride_dimensionality(self) -> int:
        """

        Returns
        -------
        int
            The dimensionality of the electride volume at a value of 0 ELF.

        """
        if self._electride_dim is None and self.all_electride_dims is not None:
            self._electride_dim = self.all_electride_dims[0]

        return self._electride_dim

    def _get_ELF_dimensionality(
        self,
        electride_mask: NDArray,
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
        electride_mask : np.array
            The ELF Grid object with only values associated with electrides.
        cutoff : float
            The minimum elf value to consider as a connection.

        Returns
        -------
        int
            The dimensionality at the ELF cutoff.

        """
        # Remove data below our cutoff
        mask = electride_mask & (self.reference_grid.total >= cutoff)

        # if we have no features, return 0 immediately
        if not np.any(mask):
            return 0

        # get the features that sit in the mask at this value
        feature_indices = self.electride_structure.frac_coords[len(self.structure) :]
        feature_indices = np.round(
            self.charge_grid.frac_to_grid(feature_indices)
        ).astype(int)
        # only use indices that are not 0
        feature_indices = [i for i in feature_indices if mask[i[0], i[1], i[2]]]

        # if we have no electride features in the mask, immediately return 0
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

    def _get_electride_dimensionality(self) -> None:
        """

        Gets the electride dimensionalities and range of ELF values that they
        exist at.

        """
        # TODO: This whole method should probably be rewritten in Numba
        # If we have no electrides theres no reason to continue so we stop here
        logging.info("Finding electride dimensionality cutoffs")
        if self.nelectrides == 0:
            self._all_electride_dims = -1
            self._all_electride_dim_cutoffs = -1

        ###############################################################################
        # This section preps an ELF grid that only contains values from the electride
        # sites and is zero everywhere else.
        ###############################################################################

        # Create a mask at electrides
        electride_indices = [
            i for i in range(len(self.structure), len(self.electride_structure))
        ]
        # NOTE: even if we have shared features, these indices are still correct
        # so long as the electride sites come first
        electride_mask = np.isin(self.atom_labels, electride_indices)

        #######################################################################
        # This section scans across different cutoffs to determine what dimensionalities
        # exist in the electride ELF
        #######################################################################
        logging.info("Calculating dimensionality at 0 ELF")
        highest_dimension = self._get_ELF_dimensionality(electride_mask, 0)
        dimensions = [i for i in range(0, highest_dimension)]
        dimensions.reverse()
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
                current_dimension = self._get_ELF_dimensionality(electride_mask, guess)
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
        self._all_electride_dims = final_dimensions
        self._all_electride_dim_cutoffs = final_connections

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

    def _get_min_avg_surface_dists(self) -> None:
        """

        Calculates the minimum and average distance from each atom and electride
        to the partitioning surface.

        """
        neigh_transforms, _ = self.charge_grid.point_neighbor_transforms
        edges = get_edges(
            labeled_array=self.atom_labels,
            neighbor_transforms=neigh_transforms,
            vacuum_mask=self.vacuum_mask,
        )
        self._min_surface_distances, self._avg_surface_distances = (
            get_min_avg_surface_dists(
                labels=self.atom_labels,
                frac_coords=self.electride_structure.frac_coords,
                edge_mask=edges,
                matrix=self.charge_grid.matrix,
                max_value=np.max(self.structure.lattice.abc) * 2,
            )
        )

    @property
    def min_surface_distances(self) -> NDArray:
        """

        Returns
        -------
        NDArray
            The minimum distance from each atom or electride center to the partioning
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
            The average distance from each atom or electride center to the partitioning
            surface.

        """
        if self._avg_surface_distances is None:
            self._get_min_avg_surface_dists()
        return self._avg_surface_distances.round(10)

    @property
    def electrides_per_formula(self) -> float:
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
        return round(self._electrides_per_formula, 10)

    @property
    def electrides_per_reduced_formula(self) -> float:
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

    @property
    def electride_formula(self) -> str:
        """

        Returns
        -------
        str
            A string representation of the electride formula, rounding partial charge
            to the nearest integer.

        """
        return f"{self.structure.formula} e{round(self.electrides_per_formula)}"

    @property
    def total_volume(self):
        """

        Returns
        -------
        float
            The total volume integrated in the system. This should match the
            volume of the structure. If it does not there may be a serious problem.

        """

        return round(self.volumes.sum() + self.vacuum_volume, 10)
    
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
        return self.elf_labeler.vacuum_charge

    @property
    def vacuum_volume(self) -> float:
        """

        Returns
        -------
        float
            The total volume assigned to the vacuum.

        """
        return self.elf_labeler.vacuum_volume

    @property
    def vacuum_mask(self) -> NDArray[bool]:
        """

        Returns
        -------
        NDArray[bool]
            A mask representing the voxels that belong to the vacuum.

        """
        return self.elf_labeler.vacuum_mask

    @property
    def num_vacuum(self) -> int:
        """

        Returns
        -------
        int
            The number of vacuum points in the array

        """
        return self.elf_labeler.num_vacuum
    
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

        return round(self.charges.sum() + self.vacuum_charge, 10)

    def to_dict(
        self, potcar_path: Path | str = "POTCAR", use_json: bool = True
    ) -> dict:
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
        # collect method kwargs
        method_kwargs = {
            "method": self.method,
            "shared_feature_splitting_method": self.shared_feature_splitting_method,
            "elf_labeler_kwargs": self.elf_labeler_kwargs,
        }
        results["method_kwargs"] = method_kwargs

        # only try to calculate oxidation state if this was a spin dependent system
        if self.spin_system == "total":
            results["oxidation_states"] = self.get_oxidation_from_potcar(potcar_path)
        else:
            results["oxidation_states"] = None

        # get charges first to ensure good logging
        self.charges

        for result in [
            "species",
            "structure",
            "labeled_structure",
            "electride_structure",
            "nelectrides",
            "all_electride_dims",
            "all_electride_dim_cutoffs",
            "electride_dimensionality",
            "charges",
            "volumes",
            "elf_maxima",
            "min_surface_distances",
            "avg_surface_distances",
            "electride_formula",
            "electrides_per_formula",
            "electrides_per_reduced_formula",
            "total_electron_number",
            "total_volume",
            "spin_system",
            "vacuum_charge",
            "vacuum_volume",
        ]:
            results[result] = getattr(self, result, None)
        if use_json:
            # get serializable versions of each attribute
            for key in ["structure", "labeled_structure", "electride_structure"]:
                results[key] = results[key].to(fmt="POSCAR")
            for key in [
                "charges",
                "volumes",
                "elf_maxima",
                "oxidation_states",
                "min_surface_distances",
                "avg_surface_distances",
            ]:
                if results[key] is None:
                    continue  # skip oxidation states if they fail
                results[key] = results[key].tolist()
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

    def write_json(self, filepath: Path | str = "badelf.json", **kwargs) -> None:
        """
        Writes results of the analysis to file in a JSON format.

        Parameters
        ----------
        filepath : Path | str, optional
            The Path to write the results to. The default is "badelf.json".
        **kwargs : dict
            keyword arguments for the to_dict method.

        """
        filepath = Path(filepath)
        with open(filepath, "w") as json_file:
            json.dump(self.to_dict(use_json=True, **kwargs), json_file, indent=4)

    @classmethod
    def from_vasp(
        cls,
        reference_file: str | Path = "ELFCAR",
        charge_file: str | Path = "CHGCAR",
        **kwargs,
    ):
        """
        Creates a BadElfToolkit instance from the requested partitioning file
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
            Additional keyword arguments for the BadElfToolkit class.

        Returns
        -------
        BadElfToolkit
            A BadElfToolkit instance.
        """

        reference_grid = Grid.from_vasp(reference_file, **kwargs)
        charge_grid = Grid.from_vasp(charge_file, **kwargs)
        return cls(reference_grid=reference_grid, charge_grid=charge_grid, **kwargs)

    def write_atom_volumes(
        self,
        atom_indices: list[int],
        directory: str | Path = None,
        write_reference: bool = True,
        include_dummy_atoms: bool = True,
        output_format: str | Format = None,
        prefix_override: str = None,
    ):
        """

        Writes an the reference ELF or charge-density for the given atoms to
        separate files. Electrides found during the calculation are appended to
        the end of the structure.

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

        # Get voxel assignments and data
        voxel_assignment_array = self.atom_labels
        if write_reference:
            grid = self.reference_grid.copy()
        else:
            grid = self.charge_grid.copy()

        # add dummy atoms if desired
        if include_dummy_atoms:
            grid.structure = self.electride_structure

        # get prefix
        if prefix_override is None:
            prefix_override = grid.data_type.prefix

        # Get mask where the grid belongs to requested species
        for atom_index in atom_indices:
            mask = voxel_assignment_array == atom_index
            grid.total[mask] = 0
            if grid.diff is not None:
                grid.diff[mask] = 0

            file_path = directory / f"{prefix_override}_a{atom_index}"
            # write file
            grid.write(filename=file_path, output_format=output_format)

    def write_all_atom_volumes(
        self,
        directory: str | Path = None,
        write_reference: bool = True,
        output_format: str | Format = None,
        include_dummy_atoms: bool = True,
        prefix_override: str = None,
        **writer_kwargs,
    ):
        """
        Writes all atomic basins.

        Parameters
        ----------
        directory : str | Path
            The directory to write the files in. If None, the active directory
            is used.
        directory : str | Path
            The directory to write the files in. If None, the active directory
            is used.
        write_reference : bool, optional
            Whether or not to write the reference data rather than the charge data.
            Default is False.
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

        Returns
        -------
        None.

        """
        atom_indices = np.array(range(len(self.electride_structure)))
        self.write_volumes(
            atom_indices=atom_indices,
            directory=directory,
            write_reference=write_reference,
            include_dummy_atoms=include_dummy_atoms,
            output_format=output_format,
            prefix_override=prefix_override,
            **writer_kwargs,
        )

    def write_atom_volumes_sum(
        self,
        atom_indices: NDArray,
        directory: str | Path = None,
        write_reference: bool = True,
        output_format: str | Format = None,
        include_dummy_atoms: bool = True,
        prefix_override: str = None,
        **writer_kwargs,
    ):
        """

        Writes the reference ELF or charge-density for the the union of the
        given atoms to a single file.

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
        # get the data to use
        if write_reference:
            data_array = self.reference_grid.total
            data_type = self.reference_grid.data_type
        else:
            data_array = self.charge_grid.total
            data_type = self.charge_grid.data_type

        if directory is None:
            directory = Path(".")
        mask = np.isin(self.atom_labels, atom_indices)
        data_array_copy = data_array.copy()
        data_array_copy[~mask] = 0.0
        grid = Grid(
            structure=self.structure,
            data={"total": data_array_copy},
            data_type=data_type,
        )
        # add dummy atoms if desired
        if include_dummy_atoms:
            grid.structure = self.electride_structure

        # get prefix
        if prefix_override is None:
            prefix_override = grid.data_type.prefix

        file_path = directory / f"{prefix_override}_asum"
        # write file
        grid.write(filename=file_path, output_format=output_format, **writer_kwargs)

    def write_species_volume(
        self,
        directory: str | Path = None,
        write_reference: bool = True,
        species: str = FeatureType.bare_electron.dummy_species,
        include_dummy_atoms: bool = True,
        output_format: str | Format = None,
        prefix_override: str = None,
    ):
        """
        Writes an ELFCAR or CHGCAR for a given species.

        Parameters
        ----------
        directory : str | Path, optional
            The directory to write the result to. The default is None.
        write_reference : bool, optional
            Whether or not to write the reference data rather than the charge data.
            The default is True.
        species : str, optional
            The species to write. The default is "Le" (the electrides).
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

        # Get voxel assignments and data
        voxel_assignment_array = self.atom_labels
        if write_reference:
            grid = self.reference_grid.copy()
        else:
            grid = self.charge_grid.copy()

        # add dummy atoms if desired
        indices = self.electride_structure.indices_from_symbol(species)
        if include_dummy_atoms:
            grid.structure = self.electride_structure
        # Get mask where the grid belongs to requested species
        mask = np.isin(voxel_assignment_array, indices, invert=True)
        grid.total[mask] = 0
        if grid.diff is not None:
            grid.diff[mask] = 0

        # get prefix
        if prefix_override is None:
            prefix_override = grid.data_type.prefix

        file_path = directory / f"{prefix_override}_{species}"
        # write file
        grid.write(filename=file_path, output_format=output_format)

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
                "surface_dist": self.min_surface_distances,
            }
        )
        return atoms_df

    def write_atom_tsv(self, filepath: Path | str = "badelf_atoms.tsv"):
        """
        Writes a summary of atom results to .tsv files.

        Parameters
        ----------
        filepath : str | Path
            The Path to write the results to. The default is "badelf_atoms.tsv".

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

            f.write("\n")
            # f.write(f"Vacuum Charge:\t\t{self.vacuum_charge:.5f}\n")
            # f.write(f"Vacuum Volume:\t\t{self.vacuum_volume:.5f}\n")
            f.write(f"Total Electrons:\t{self.total_electron_number:.5f}\n")
