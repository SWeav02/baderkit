# -*- coding: utf-8 -*-

from pathlib import Path
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray
from pymatgen.analysis.local_env import CrystalNN
from scipy.spatial import HalfspaceIntersection

from baderkit._base_analysis import BaseAnalysis
from baderkit.elf_analysis.elf_labeler.elf_labeler import ElfLabeler
from baderkit.elf_analysis.elf_labeler.enum_and_styling import FeatureType
from baderkit.global_numba.voronoi import (
    generate_symmetric_bonds,
    get_canonical_bonds,
    get_planes_on_surface,
)
from baderkit.toolkit import Grid, Structure

from .elf_radii_numba import (
    get_all_atom_elf_radii,
)

Self = TypeVar("Self", bound="ElfRadii")


class ElfRadii(BaseAnalysis):
    """
    A tool for calculating ionic/covalent radii based on a localization function
    (ELF, ELI-D, LOL, etc.).

    """

    _method_kwargs = ["include_nnas", "cnn_kwargs"]

    _radii_results = [
        "bonding_pairs",
        "all_radii",
        "atom_radii",
        "all_bond_types",
        "atom_bond_types",
    ]

    _nonsummary_results = [
        "structure",
        "local_basin_labels",
        "label_atom_map",
        "voronoi_planes",
    ]

    _reset_props = _radii_results + _nonsummary_results

    _summary_props = [
        "radii_results",
    ]

    _sub_methods = ["labeler"]

    def __init__(
        self,
        charge_grid: Grid,
        reference_grid: Grid,
        total_charge_grid: Grid | None = None,
        include_nnas: bool = False,
        cnn_kwargs: dict | None = None,
        **kwargs,
    ):
        """
        Calculates the radius of each atom using a localization function
        (e.g. ELF, ELI-D, LOL). Atom-neighbor pairs are chosen such that the
        planes perpendicular to the bond, placed at the radius, form a weighted
        voronoi surface. The method for determining the radius depends on the
        bond type:

            unshared    - The minimum point representing where the atoms are
                          separated by a voronoi surface

            shared      - The maximum point in the covalent/metallic shared basin
                          that separates the two atoms

            non-bonding - The maximum point in whatever atomic basins separate
                          the atom pair. These are included only to complete the
                          voronoi surface.

        Parameters
        ----------
        charge_grid : Grid
            The charge density grid used for integrating charge.
        reference_grid : Grid
            The ELF grid used to partition volumes.
        total_charge_grid : Grid, optional
            The total charge density used for bader integrations and vacuum masks. If
            not provided, the charge_grid will be used instead.
        include_nnas : bool, optional
            Whether or not to treat non-nuclear attractors as quasi atoms. If
            set to true, they will be included as central points for the generated
            weighted voronoi surface and be given calculated radii.
        cnn_kwargs : dict | None, optional
            If provided, the nearest neighbors will be determined using PyMatGen's CrystalNN
            class using the keyword arguments in this argument. If not provided, all neighbors
            that share a weighted voronoi facet (constructed from the ELF radii) will be
            included.
        **kwargs : dict
            Keyword arguments to pass to the ElfLabeler class.

        """

        # create bader objects
        self._labeler = ElfLabeler(
            charge_grid=charge_grid,
            total_charge_grid=total_charge_grid,
            reference_grid=reference_grid,
            cnn_kwargs=None,
            **kwargs,
        )

        self._include_nnas = include_nnas

        if cnn_kwargs is not None:
            self._use_cnn = True
            self._cnn_kwargs = cnn_kwargs
            self._cnn = CrystalNN(**cnn_kwargs)
        else:
            self._use_cnn = False
            self._cnn_kwargs = None
            self._cnn = None

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
    def include_nnas(self) -> bool:
        """

        Returns
        -------
        bool
            Whether or not to treat non-nuclear attractors as quasi atoms. If
            set to true, they will be included as central points for the generated
            weighted voronoi surface and be given calculated radii.

        """
        return self._include_nnas

    @include_nnas.setter
    def include_nnas(self, value: bool):
        self._include_nnas = value
        self._reset_properties()

    @property
    def cnn_kwargs(self) -> dict | None:
        """

        Returns
        -------
        dict
            The keyword arguments used to construct the CrystalNN
            object.

        """
        return self._cnn_kwargs

    @cnn_kwargs.setter
    def cnn_kwargs(self, value: dict | None):
        if value is not None:
            self._cnn_kwargs = value
            self._cnn = CrystalNN(**value)
            self._use_cnn = True
        else:
            self._cnn_kwargs = value
            self._cnn = value
            self._use_cnn = False
        self._reset_properties()

    ###########################################################################
    # Helper Properties
    ###########################################################################
    @property
    def labeler(self) -> ElfLabeler:
        """

        Returns
        -------
        ElfLabeler
            The ElfLabeler class used to determine the type of each bond.

        """
        return self._labeler

    @property
    def structure(self) -> Structure:
        """

        Returns
        -------
        Structure
            The Structure object used in the calculation. If include_nnas is
            set to True, non-nuclear attractors will be appended to the original
            structure as dummy-atoms.

        """
        if self._structure is None:
            structure = self.reference_grid.structure.copy()
            # add nnas if requested
            if self.include_nnas:
                frac_coords = self.labeler.maxima_frac[self.labeler.nna_indices]
                for frac in frac_coords:
                    structure.append("x", frac)
            self._structure = structure
        return self._structure

    @property
    def local_basin_labels(self) -> NDArray[int]:
        """

        Returns
        -------
        NDArray[int]
            The labeled grid assigning each grid point to a basin in the
            localization function.

        """
        return self.labeler.elf_bader.maxima_basin_labels

    @property
    def label_atom_map(self) -> NDArray[int]:
        """

        Returns
        -------
        NDArray[int]
            A 1D array connecting each local basin to its corresponding atom
            or basin type. This is used internally to determine the bond types.

        """
        if self._label_atom_map is None:
            # get feature types
            basin_types = self.labeler.basin_types
            # get overlapping atoms
            atom_fracs = self.labeler.overlap.bond_fractions
            # get nna indices
            nna_indices = self.labeler.nna_indices

            num_basins = len(basin_types)

            label_map = np.empty(num_basins + 1, dtype=np.int64)
            for idx, (basin_type, frac) in enumerate(zip(basin_types, atom_fracs)):
                # point core/shell/lone-pairs to corresponding atom
                if len(frac) == 1 or basin_type in FeatureType.unshared:
                    # get atoms index in structure
                    label_map[idx] = int(frac[0, 0])
                # point nnas to corresponding atom
                elif self.include_nnas and basin_type == FeatureType.nna.value:
                    # get nna index in structure
                    label_map[idx] = np.searchsorted(nna_indices, idx) + len(
                        self.reference_grid.structure
                    )
                # point covalent to len(structure) + 1
                elif basin_type == FeatureType.covalent.value:
                    label_map[idx] = len(self.structure) + 1
                # point metallic to len(structure) + 2
                elif basin_type in FeatureType.metal_like:
                    label_map[idx] = len(self.structure) + 2
                # point remainder to len(structure)
                else:
                    # assign to value above possible structure lengths
                    label_map[idx] = len(self.structure)
            # set vacuum map
            label_map[-1] = len(self.structure) + 3
            self._label_atom_map = label_map

        return self._label_atom_map

    @property
    def cnn(self) -> CrystalNN:
        """

        Returns
        -------
        CrystalNN
            If cnn_kwargs were provided, the CrystalNN object used to
            determine coordination environments.

        """
        return self._cnn

    ###########################################################################
    # Radii Properties
    ###########################################################################
    @property
    def bonding_pairs(self) -> (NDArray[int], NDArray[int]):
        """

        Returns
        -------
        (NDArray[int], NDArray[int])
            A tuple where the first object is a Nx2 array representing the
            site/neighbor atom indices involved in each bond and the second object
            is a Nx3 array representing the periodic image the neighbor atom
            sits in.

        """
        if self._bonding_pairs is None:
            self._get_radii()
        return self._bonding_pairs

    @property
    def all_radii(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            A 1D array of length N where N is the total number of bonds found,
            that lists all bond radii found in the system (in Å).

        """
        if self._all_radii is None:
            self._get_radii()
        return self._all_radii

    @property
    def atom_radii(self) -> NDArray[float]:
        """

        Returns
        -------
        NDArray[float]
            A 1D array of length M where M is the number of atoms in the system,
            that lists the shortest bonding radius found for each atom in Å.

        """
        if self._atom_radii is None:
            all_radii = self.all_radii
            all_types = self.all_bond_types
            atom_radii = np.empty(len(self.structure), dtype=np.float64)
            atom_bond_types = []
            site_indices = self.bonding_pairs[0][:, 0]
            for i in range(len(self.structure)):
                idx = np.searchsorted(site_indices, i)
                atom_radii[i] = all_radii[idx]
                atom_bond_types.append(all_types[idx])
            self._atom_radii = atom_radii
            self._atom_bond_types = np.array(atom_bond_types)
        return self._atom_radii

    @property
    def atom_bond_types(self) -> NDArray[str]:
        """

        Returns
        -------
        NDArray[str]
            The primary bonding type for each atom. The options are determined
            based on what types of basins are found along the bond.
                ionic - No shared basins
                covalent - Covelent basin
                metallic - Metallic, nna, multi-centered etc.
                non-bonding - Atomic basin belonging to either atom in the pair

        """
        if self._atom_bond_types is None:
            self.atom_radii
        return self._atom_bond_types

    @property
    def all_bond_types(self) -> NDArray[str]:
        """

        Returns
        -------
        NDArray[str]
            The type of each bond in the system. The options are determined
            based on what types of basins are found along the bond.
                ionic - No shared basins
                covalent - Covelent basin
                metallic - Metallic, nna, multi-centered etc.
                non-bonding - Atomic basin belonging to either atom in the pair

        """
        if self._all_bond_types is None:
            self._get_radii()
        mapping = np.array(["ionic", "covalent", "metallic", "non-bonding"])
        return mapping[self._all_bond_types]

    @property
    def voronoi_planes(self) -> (NDArray[float], NDArray[float]):
        """

        Returns
        -------
        (NDArray[float], NDArray[float])
            A tuple containing the plane points and normal vectors representing
            the voronoi surface made by the bonds found in the system.

        """
        if self._voronoi_planes is None:
            self._get_radii()
        return self._voronoi_planes

    ###########################################################################
    # General Methods
    ###########################################################################

    def _get_elf_radii_and_type(
        self,
        site_indices: NDArray[int],
        neigh_indices: NDArray[int],
        neigh_frac_coords: NDArray[float],
        pair_dists: NDArray[float],
        reversed_bonds: NDArray[bool],
    ):
        """
        Wrapper for the numba method used to calculate radii information
        given a set of bonding pairs.

        Parameters
        ----------
        site_indices : NDArray[int]
            The indices of the first atom in each bond.
        neigh_indices : NDArray[int]
            The index of the neighboring atom in each bond.
        neigh_frac_coords : NDArray[float]
            The fractional coordinates of the neighboring atom.
        pair_dists : NDArray[float]
            The length of each bond.
        reversed_bonds : NDArray[bool]
            Whether or not each bond has been reversed.

        """

        # calculate radii
        return get_all_atom_elf_radii(
            site_indices=site_indices,
            neigh_indices=neigh_indices,
            site_frac_coords=self.structure.frac_coords,
            neigh_frac_coords=neigh_frac_coords,
            neigh_dists=pair_dists,
            reversed_bonds=reversed_bonds,
            data=self.reference_grid.cubic_spline_coeffs,
            labels=self.local_basin_labels,
            label_map=self.label_atom_map,
            equivalent_atoms=self.structure.equivalent_atoms,
        )

    ###########################################################################
    # Voronoi Methods
    ###########################################################################

    def _get_neigh_info_from_cnn(self, neigh_info: list):
        """
        Converts the output from CrystalNN.get_all_nn_info() to arrays.

        Parameters
        ----------
        neigh_info : list
            The output from a CrystalNN.get_all_nn_info() call.

        Returns
        -------
        neigh_indices : NDArray[int]
            The site index of each neighbor.
        neigh_images : NDArray[int]
            The periodic image of each neighbor.
        pair_dists : NDArray[float]
            The distance to each neighbor.

        """
        # for each site, get all neighbors within a sphere of twice the largest
        # CrystalNN neighbor distance. Get relavent information
        neigh_indices = []
        neigh_images = []
        pair_dists = []
        for i, neighs in enumerate(neigh_info):
            # collect information
            neigh_indices1 = np.array([j["site_index"] for j in neighs], dtype=int)
            neigh_images1 = np.array([j["site"].image for j in neighs], dtype=float)
            pair_dists1 = np.array([j["site"].nn_distance for j in neighs], dtype=float)

            # sort by neighbor distance
            sorted_indices = np.argsort(pair_dists1)

            # remove any indices that correspond to a distance of zero
            mask = pair_dists1[sorted_indices] != 0
            sorted_indices = sorted_indices[mask]

            # add to neighbors lists
            neigh_indices.append(neigh_indices1[sorted_indices])
            neigh_images.append(neigh_images1[sorted_indices])
            pair_dists.append(pair_dists1[sorted_indices])

        return neigh_indices, neigh_images, pair_dists

    def _get_plane_points_vectors(
        self,
        site_coords: NDArray,
        neigh_coords: NDArray,
        fracs: NDArray,
    ):
        """
        Calculates the plane points/vectors along an atomic bond

        Parameters
        ----------
        site_coords : NDArray
            The fractional coordinates of the first site in the bond.
        neigh_coords : NDArray
            The fractional coordinates of the second site in the bond.
        fracs : NDArray
            The fraction radius.

        Returns
        -------
        plane_points : NDArray[float]
            A point on the plane.
        plane_vectors : NDArray[float]
            The vector from the first atom to the second.

        """
        # calculate vectors from sites to neighs
        plane_vectors = neigh_coords - site_coords

        # calculate points on each plane
        plane_points = site_coords + (plane_vectors.T * fracs).T
        # normalize vectors
        magnitudes = np.linalg.norm(plane_vectors, axis=1, keepdims=True)
        plane_vectors /= magnitudes
        return plane_points, plane_vectors

    def _get_plane_equations(
        self,
        site_coord,
        neigh_coords,
        fracs: NDArray | float,
    ):
        """
        Calculates the plane equation along an atomic bond.

        Parameters
        ----------
        site_coords : NDArray
            The fractional coordinates of the first site in the bond.
        neigh_coords : NDArray
            The fractional coordinates of the second site in the bond.
        fracs : NDArray
            The fraction radius.


        Returns
        -------
        normals : NDArray[float]
            The vector from the first atom to the second.
        b : float
            The b part of the plane equation

        """

        # Get normal vector (A)
        normals = neigh_coords - site_coord  # (N,3)

        # get point on plane
        if type(fracs) in (float, int):
            points = (normals * fracs) + site_coord
        else:
            points = (normals.T * fracs).T + site_coord

        # Normalize
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        # Get b (b = -n · neighbor)
        b = -np.einsum("ij,ij->i", normals, points)

        return normals, b

    def _get_possible_voronoi_planes(
        self,
        unique_atoms: NDArray[int],
        neigh_coords: NDArray[float],
        pair_dists: NDArray[float],
        neigh_indices: NDArray[int],
        neigh_images: NDArray[int],
    ):
        """

        Parameters
        ----------
        unique_atoms : NDArray[int]
            The set of atoms that are symmetrically unique.
        neigh_coords : NDArray[float]
            The neighbor coordinates determined by CrystalNN.
        pair_dists : NDArray[float]
            The distance to each neighbor.
        neigh_indices : NDArray[int]
            The structure indices of each neighbor.
        neigh_images : NDArray[int]
            The periodic image of each neighbor.

        Returns
        -------
        neigh_indices : NDArray[int]
            The structure indices of each neighbor after expansion.
        neigh_images : NDArray[int]
            The periodic image of each neighbor after expansion.
        neigh_coords : NDArray[float]
            The neighbor coordinates determined by expanding the results from CrystalNN.
        pair_dists : NDArray[float]
            The distance to each neighbor after expansion.

        """
        # for each unique site, we want to get a set of neighbors that may be part
        # of our voronoi surface. To do this, we want to expand our current set
        # of atoms slightly
        for unique_idx, site_idx in enumerate(unique_atoms):
            site_coord = self.structure.frac_coords[site_idx]
            neigh_coords1 = neigh_coords[unique_idx]

            # get plane equations
            cutoff = 1.5
            A, b = self._get_plane_equations(
                site_coord,
                neigh_coords1,
                fracs=cutoff,
            )
            # get a set of neighbor points that definitely includes all points
            # inside our planes
            dist = pair_dists[unique_idx].max()
            (
                _,
                max_neigh_indices,
                max_neigh_images,
                max_pair_dists,
            ) = self.structure.get_neighbor_list(
                dist * cutoff * 2, sites=[self.structure[site_idx]], exclude_self=True
            )
            # get neighbor frac coords
            max_neigh_coords = (
                self.structure.frac_coords[max_neigh_indices] + max_neigh_images
            )
            # get the neighbors that lie within our voronoi surface
            vals = A @ max_neigh_coords.T + b[:, None]
            important_mask = np.all(vals <= -1e-12, axis=0) & (max_pair_dists != 0)
            # also filter for neighbors that correspond to the central site.
            # Pymatgen seems to miss these sometimes
            self_neighs = max_pair_dists < 1e-12

            important_indices = np.where(important_mask & ~self_neighs)[0]
            # write our new neighbor info
            neigh_indices[unique_idx] = max_neigh_indices[important_indices]
            neigh_images[unique_idx] = max_neigh_images[important_indices]
            neigh_coords[unique_idx] = max_neigh_coords[important_indices]
            pair_dists[unique_idx] = max_pair_dists[important_indices]
        return neigh_indices, neigh_images, neigh_coords, pair_dists

    def _get_voronoi_contributors(
        self,
        site_indices,
        unique_atoms,
        neigh_coords,
        fracs,
        inverse,
        neigh_indices,
        neigh_images,
        pair_dists,
        all_bond_types,
    ):
        """

        Reduces the bonds by determining which are involved in the
        voronoi surface.

        """
        # get ranges for each site
        site_ranges = np.where(site_indices[:-1] != site_indices[1:])[0] + 1
        site_ranges = np.insert(
            site_ranges, [0, len(site_ranges)], [0, len(site_indices)]
        )

        # calculate all of the plane equations
        halfspaces = []
        for unique_idx, site_idx in enumerate(unique_atoms):
            # get range of planes
            lower = site_ranges[unique_idx]
            upper = site_ranges[unique_idx + 1]

            current_neigh_coords = neigh_coords[lower:upper]

            # calculate plane equations
            A, b = self._get_plane_equations(
                self.structure.frac_coords[site_idx],
                current_neigh_coords,
                fracs[lower:upper],
            )
            halfspaces.append(np.column_stack((A, b)))
        halfspaces = np.concatenate(halfspaces)

        # for each unique atom we check the unique halfspaces against all halfspaces
        # for that atom
        important_plane_mask = np.zeros(len(site_indices), dtype=np.bool_)

        # for each unique atom, get the planes making up the voronoi surface using
        # scipy's HalfspaceIntersection combined with convexHull
        for unique_idx, site_idx in enumerate(unique_atoms):
            # get range of planes
            lower = site_ranges[unique_idx]
            upper = site_ranges[unique_idx + 1]

            current_halfspaces = halfspaces[lower:upper]
            halfspace = HalfspaceIntersection(
                current_halfspaces,
                self.structure.frac_coords[site_idx],
                incremental=False,
            )
            vertices = halfspace.intersections

            # Get one plane for each unique bond with this atom at the center.
            # This reduces the number of calculations we need to perform
            unique_equiv = inverse[lower:upper]
            unique_equiv_, unique_indices, unique_inverse = np.unique(
                unique_equiv, return_index=True, return_inverse=True
            )

            current_unique_halfspaces = halfspaces[lower:upper][unique_indices]

            important_planes = get_planes_on_surface(
                current_unique_halfspaces, vertices
            )
            # note important halfspaces
            important_plane_mask[lower:upper] = important_planes[unique_inverse]

        # expand important unique planes
        important_plane_mask = np.where(important_plane_mask)[0]
        site_indices = site_indices[important_plane_mask]
        neigh_indices = neigh_indices[important_plane_mask]
        neigh_images = neigh_images[important_plane_mask]
        neigh_coords = neigh_coords[important_plane_mask]
        pair_dists = pair_dists[important_plane_mask]
        fracs = fracs[important_plane_mask]
        all_bond_types = all_bond_types[important_plane_mask]

        return (
            site_indices,
            neigh_indices,
            neigh_images,
            neigh_coords,
            pair_dists,
            fracs,
            all_bond_types,
        )

    def _get_radii(self):
        """
        Calculates the voronoi planes making up the dividing polyhedra between
        atoms. Planes are placed at atom radii and include any that contribute to
        the surface of the polyhedron.

        Returns
        -------
        site_indices : NDArray[int]
            The site indices of each first atom in all bonds.
        neigh_indices : NDArray[int]
            The site indices of each second atom in all bonds.
        neigh_coords : NDArray[float]
            The fractional coordinates of each neighboring site.
        radii : NDArray[float]
            The radius from the central atom in each bond.
        all_bond_types : NDArray[bool]
            The type of each bond, either True for covalent or False for ionic.
        plane_points : NDArray[float]
            A point on each partitioning plane. The point is also along the bond
            line positioned at the radius.
        plane_vectors : NDArray[float]
            The vector normal to each plane.

        """

        # create crystalNN that will get all nearby atoms
        # NOTE: This is distinct from the optional cnn provided by the
        # user and is exclusively geometric to ensure a closed surface
        if self.cnn is None:
            cnn = CrystalNN(
                weighted_cn=True,
                distance_cutoffs=None,
                x_diff_weight=0.0,
                porous_adjustment=False,
            )
        else:
            cnn = self.cnn

        # get symmetric atoms
        equivalent_atoms = self.structure.equivalent_atoms
        unique_atoms = np.unique(equivalent_atoms)

        # get symmetry operations. convert to c contiguous for speed in numba
        symm_ops = self.structure.spacegroup_analyzer.get_symmetry_operations(
            cartesian=False
        )
        rotation_matrices = [np.ascontiguousarray(i.rotation_matrix) for i in symm_ops]
        translation_vectors = [
            np.ascontiguousarray(i.translation_vector) for i in symm_ops
        ]

        # get neighbors
        neigh_info = cnn.get_all_nn_info(self.structure)

        # get neighbor info for unique atoms
        unique_neigh_info = [neigh_info[i] for i in unique_atoms]

        # convert cnn output to array representation
        neigh_indices, neigh_images, pair_dists = self._get_neigh_info_from_cnn(
            neigh_info=unique_neigh_info
        )
        neigh_coords = [
            self.structure.frac_coords[neigh_indices[i]] + neigh_images[i]
            for i in range(len(neigh_indices))
        ]

        if not self._use_cnn:
            # Get possible valid neighbors by expanding out beyond the
            # geometric coordination env.
            (
                neigh_indices,
                neigh_images,
                neigh_coords,
                pair_dists,
            ) = self._get_possible_voronoi_planes(
                unique_atoms,
                neigh_coords,
                pair_dists,
                neigh_indices,
                neigh_images,
            )

        # Reduce to geometrically unique bonding pairs

        # move all bond info to a single array
        site_indices = np.concatenate(
            [[k for i in range(len(j))] for k, j in zip(unique_atoms, neigh_indices)],
            dtype=np.int32,
        )
        neigh_indices = np.concatenate(neigh_indices)
        neigh_images = np.concatenate(neigh_images)
        neigh_coords = np.concatenate(neigh_coords)
        pair_dists = np.concatenate(pair_dists)

        canonical_bonds = get_canonical_bonds(
            site_indices=site_indices,
            neigh_indices=neigh_indices,
            neigh_coords=neigh_coords,
            equivalent_atoms=equivalent_atoms,
            all_frac_coords=self.structure.frac_coords,
            rotation_matrices=rotation_matrices,
            translation_vectors=translation_vectors,
            pair_dists=pair_dists,
            shape=self.reference_grid.shape,
            tol=1,
        )

        # get a mask for reversed bonds
        reversed_mask = canonical_bonds[:, 0] == 1

        # get unique bonds. Reverse bonds are counted as the same (i.e. Ca-N == N-Ca)
        unique_bonds, indices, inverse = np.unique(
            canonical_bonds[:, 1:], return_index=True, return_inverse=True, axis=0
        )

        unique_site_indices = site_indices[indices]
        unique_neigh_indices = neigh_indices[indices]
        unique_neigh_images = neigh_images[indices]
        unique_neigh_coords = neigh_coords[indices]
        unique_pair_dists = pair_dists[indices]
        is_reverse = reversed_mask[indices]

        unique_neigh_coords = (
            self.structure.frac_coords[unique_neigh_indices] + unique_neigh_images
        )
        # get radii for each unique bond
        radii, fracs, all_bond_types = self._get_elf_radii_and_type(
            unique_site_indices,
            unique_neigh_indices,
            unique_neigh_coords,
            unique_pair_dists,
            is_reverse,
        )

        # assign fractions back to each bond
        fracs = fracs[inverse]
        all_bond_types = all_bond_types[inverse]
        # reverse any that need it
        fracs[reversed_mask] = 1 - fracs[reversed_mask]

        if not self._use_cnn:
            # remove bonds that do not contribute to the voronoi
            # surface.
            (
                site_indices,
                neigh_indices,
                neigh_images,
                neigh_coords,
                pair_dists,
                fracs,
                all_bond_types,
            ) = self._get_voronoi_contributors(
                site_indices,
                unique_atoms,
                neigh_coords,
                fracs,
                inverse,
                neigh_indices,
                neigh_images,
                pair_dists,
                all_bond_types,
            )

        # Regenerate all bonds using symmetry operations
        all_bonds = generate_symmetric_bonds(
            site_indices=site_indices,
            neigh_indices=neigh_indices,
            neigh_coords=neigh_coords,
            bond_types=all_bond_types,
            all_frac_coords=self.structure.frac_coords,
            fracs=fracs,
            rotation_matrices=rotation_matrices,
            translation_vectors=translation_vectors,
            shape=self.reference_grid.shape,
            frac2cart=self.reference_grid.matrix,
            tol=1,
        )

        # remove repeats
        all_bonds, indices = np.unique(all_bonds, return_index=True, axis=0)

        # Return partitioning information
        site_indices = all_bonds[:, 0].astype(int)
        neigh_indices = all_bonds[:, 1].astype(int)
        radii = all_bonds[:, 2]
        pair_dists = all_bonds[:, 3]
        all_bond_types = all_bonds[:, 4].astype(int)
        plane_points = all_bonds[:, 5:8]
        plane_vectors = all_bonds[:, 8:11]
        neigh_coords = all_bonds[:, 11:]

        if -1 in site_indices:
            raise Exception(
                "Bond generation failed. This is a bug! Please report to our github: https://github.com/SWeav02/baderkit/issues"
            )

        # sort radii by site index and radius
        sorted_indices = np.lexsort((radii, site_indices))

        site_indices = site_indices[sorted_indices]
        neigh_indices = neigh_indices[sorted_indices]
        radii = radii[sorted_indices]
        pair_dists = pair_dists[sorted_indices]
        all_bond_types = all_bond_types[sorted_indices]
        plane_points = plane_points[sorted_indices]
        plane_vectors = plane_vectors[sorted_indices]
        neigh_coords = neigh_coords[sorted_indices]

        self._bonding_pairs = np.column_stack((site_indices, neigh_indices)), (
            neigh_coords // 1
        ).astype(int)
        self._pair_dists = pair_dists
        self._all_radii = radii
        self._all_bond_types = all_bond_types
        self._voronoi_planes = plane_points, plane_vectors

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
        # this is just a wrapper to set the ELFCAR as a default
        return super().from_vasp(
            charge_filename=charge_filename,
            reference_filename=reference_filename,
            **kwargs,
        )
