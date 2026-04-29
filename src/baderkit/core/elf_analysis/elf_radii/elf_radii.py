# -*- coding: utf-8 -*-

from pathlib import Path
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray
from pymatgen.analysis.local_env import CrystalNN
from scipy.spatial import HalfspaceIntersection

from baderkit.core.base.base_analysis import BaseAnalysis
from baderkit.core.toolkit import Grid, Structure
from baderkit.core.elf_analysis.elf_labeler1.elf_labeler import ElfLabeler

from baderkit.core.elf_analysis.elf_labeler1.enum_and_styling import FeatureType
from .elf_radii_numba import (
    get_all_atom_elf_radii,
    )
from baderkit.core.utilities.voronoi import (
    generate_symmetric_bonds,
    get_canonical_bonds,
    get_planes_on_surface,
)


Self = TypeVar("Self", bound="ElfRadii")


class ElfRadii(BaseAnalysis):
    """
    A tool for calculating ionic/covalent radii based on a localization function
    (ELF, ELI-D, LOL, etc.).

    """

    _summary_props = [
        "bonding_pairs",
        "all_radii",
        "atom_radii",
        "all_bond_types",
        "atom_bond_types",
        ]

    _reset_props = [
        "structure",
        "local_basin_labels",
        "label_atom_map",
        "voronoi_planes",
        ] + _summary_props



    def __init__(
        self,
        charge_grid: Grid,
        reference_grid: Grid,
        total_charge_grid: Grid | None = None,
        include_nnas: bool = False,
        **kwargs,
    ):

        # create bader objects
        self._labeler = ElfLabeler(
            charge_grid=charge_grid,
            total_charge_grid=total_charge_grid,
            reference_grid=reference_grid,
            **kwargs,
        )
        
        self._include_nnas = include_nnas

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
    def include_nnas(self) -> float:
        return self._include_nnas

    @include_nnas.setter
    def include_nnas(self, value: float):
        self._include_nnas = value
        self._reset_properties()

    ###########################################################################
    # Helper Properties
    ###########################################################################
    @property
    def labeler(self) -> ElfLabeler:
        return self._labeler
    
    
    @property
    def structure(self) -> Structure:
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
        return self.labeler.elf_bader.maxima_basin_labels
    
    @property
    def label_atom_map(self) -> NDArray[int]:
        if self._label_atom_map is None:
            # get feature types
            basin_types = self.labeler.basin_types
            # get overlapping atoms
            atom_fracs = self.labeler.overlap.bond_fractions
            # get nna indices
            nna_indices = self.labeler.nna_indices
            
            num_basins = len(basin_types)
            
            label_map = np.empty(num_basins, dtype=np.int64)
            for idx,(basin_type, frac) in enumerate(zip(basin_types, atom_fracs)):
                if len(frac) == 1 or basin_type in FeatureType.unshared:
                    # get atoms index in structure
                    label_map[idx] = int(frac[0,0])
                elif self.include_nnas and basin_type == FeatureType.nna.value:
                    # get nna index in structure
                    label_map[idx] = np.searchsorted(nna_indices, idx) + len(self.reference_grid.structure)
                else:
                    # assign to value above possible structure lengths
                    label_map[idx] = len(self.structure)
            self._label_atom_map = label_map
                    
                    
        return self._label_atom_map


    ###########################################################################
    # Radii Properties
    ###########################################################################
    @property
    def bonding_pairs(self) -> (NDArray[int], NDArray[int]):
        if self._bonding_pairs is None:
            self._get_voronoi_radii()
        return self._bonding_pairs
    
    @property
    def all_radii(self) -> NDArray[float]:
        if self._all_radii is None:
            self._get_voronoi_radii()
        return self._all_radii
    
    @property
    def atom_radii(self) -> NDArray[float]:
        if self._atom_radii is None:
            all_radii = self.all_radii
            all_types = self.all_bond_types
            atom_radii = np.empty(len(self.structure), dtype=np.float64)
            atom_bond_types = []
            site_indices = self.bonding_pairs[0][:,0]
            for i in range(len(self.structure)):
                idx = np.searchsorted(site_indices, i)
                atom_radii[i] = all_radii[idx]
                atom_bond_types.append(all_types[idx])
            self._atom_radii = atom_radii
            self._atom_bond_types = np.array(atom_bond_types)
        return self._atom_radii
    
    @property
    def atom_bond_types(self) -> NDArray[str]:
        if self._atom_bond_types is None:
            self.atom_radii
        return self._atom_bond_types
            
    @property
    def all_bond_types(self) -> NDArray[str]:
        if self._all_bond_types is None:
            self._get_voronoi_radii()
        return np.where(self._all_bond_types, "covalent", "ionic")

    @property
    def voronoi_planes(self) -> (NDArray[float], NDArray[float]):
        if self._voronoi_planes is None:
            self._get_voronoi_radii()
        return self._voronoi_planes
    
    ###########################################################################
    # Methods
    ###########################################################################
    def _get_elf_radii_and_type(
        self,
        site_indices,
        neigh_indices,
        neigh_frac_coords,
        pair_dists,
        reversed_bonds,
    ):

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
            label_map = self.label_atom_map,
            equivalent_atoms=self.structure.equivalent_atoms
        )

    def _get_neigh_info_from_pymatgen(self, neigh_info: list):
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

    def _get_voronoi_radii(self):
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
        cnn = CrystalNN(
            weighted_cn=True,
            distance_cutoffs=None,
            x_diff_weight=0.0,
            porous_adjustment=False,
        )

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

        # get pure geometric neighbors
        neigh_info = cnn.get_all_nn_info(self.structure)

        # get neighbor info for unique atoms
        unique_neigh_info = [neigh_info[i] for i in unique_atoms]

        # get array representations
        neigh_indices, neigh_images, pair_dists = self._get_neigh_info_from_pymatgen(
            neigh_info=unique_neigh_info
        )
        neigh_coords = [
            self.structure.frac_coords[neigh_indices[i]] + neigh_images[i]
            for i in range(len(neigh_indices))
        ]

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

        # Now we want to calculate the ELF radii for each possible pair. To do
        # this efficiently, we first want to reduce to symmetrically equivalent
        # bonds. We do this by converting each bond into a canonical representation

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

        # Generate all bonds using symmetry operations
        all_bonds = generate_symmetric_bonds(
            site_indices=site_indices,
            neigh_indices=neigh_indices,
            neigh_coords=neigh_coords,
            all_bond_types=all_bond_types,
            all_frac_coords=self.structure.frac_coords,
            fracs=fracs,
            rotation_matrices=rotation_matrices,
            translation_vectors=translation_vectors,
            shape=self.reference_grid.shape,
            frac2cart=self.reference_grid.matrix,
            tol=1,
        )

        # reduce to unique
        all_bonds, indices = np.unique(all_bonds, return_index=True, axis=0)

        # Return partitioning information
        site_indices = all_bonds[:, 0].astype(int)
        neigh_indices = all_bonds[:, 1].astype(int)
        radii = all_bonds[:, 2]
        pair_dists = all_bonds[:, 3]
        all_bond_types = all_bonds[:, 4].astype(bool)
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
        
        
        self._bonding_pairs = np.column_stack((site_indices, neigh_indices)), (neigh_coords // 1).astype(int)
        self._pair_dists = pair_dists
        self._all_radii = radii
        self._all_bond_types = all_bond_types
        self._voronoi_planes = plane_points, plane_vectors
        
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
            **kwargs
            )
