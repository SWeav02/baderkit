# -*- coding: utf-8 -*-

import numpy as np
from numpy.typing import NDArray

from baderkit.core import Grid, Structure
from pymatgen.analysis.local_env import CrystalNN


from .elf_radii_numba import get_all_atom_elf_radii
from baderkit.core.utilities.voronoi import (
    get_planes_on_surface,
    get_canonical_bonds,
    generate_symmetric_bonds,
    )

from scipy.spatial import HalfspaceIntersection

class ElfRadiiTools:
    def __init__(
        self,
        grid: Grid,
        feature_labels: NDArray,
        feature_structure: Structure,
        covalent_symbols: list[str] = ["Z", "M"],
        override_structure: Structure | None = None,
    ):

        self.grid = grid
        self.cubic_coeffs = grid.cubic_spline_coeffs
        self.feature_structure = feature_structure
        self.feature_labels = feature_labels
        self.covalent_symbols = covalent_symbols

        if override_structure is None:
            self.structure = grid.structure
        else:
            self.structure = override_structure

        # NOTE: atom labels should correspond to labeled structure

    def _get_elf_radii_and_type(
        self,
        site_indices,
        neigh_indices,
        neigh_frac_coords,
        pair_dists,
        reversed_bonds,
    ):
        
        # create a map of covalent labels
        covalent_labels = np.zeros(len(self.feature_structure), dtype=np.bool_)
        for symbol in self.covalent_symbols:
            covalent_labels[np.array(self.feature_structure.indices_from_symbol(symbol), dtype=np.int64)] = True
            
        # calculate radii
        return get_all_atom_elf_radii(
            site_indices=site_indices,
            neigh_indices=neigh_indices,
            site_frac_coords=self.structure.frac_coords,
            neigh_frac_coords=neigh_frac_coords,
            neigh_dists=pair_dists,
            reversed_bonds=reversed_bonds,
            data=self.cubic_coeffs,
            feature_labels=self.feature_labels,
            covalent_labels=covalent_labels,
            equivalent_atoms=self.structure.equivalent_atoms,
        )
    
    def _get_neigh_info_from_pymatgen(
        self,
        neigh_info: list
            ):
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
        normals = neigh_coords - site_coord   # (N,3)
        
        # get point on plane
        if type(fracs) in (float, int):
            points = (normals*fracs) + site_coord
        else:
            points = (normals.T*fracs).T + site_coord
        
        # Normalize
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        
        # Get b (b = -n · neighbor)
        b = -np.einsum("ij,ij->i", normals, points)
        
        return normals, b
        

    def get_voronoi_radii(self):
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
        symm_ops = self.structure.spacegroup_analyzer.get_symmetry_operations(cartesian=False)
        rotation_matrices = [np.ascontiguousarray(i.rotation_matrix) for i in symm_ops]
        translation_vectors = [np.ascontiguousarray(i.translation_vector) for i in symm_ops]
        
        # get pure geometric neighbors
        neigh_info = cnn.get_all_nn_info(self.structure)
        
        # get neighbor info for unique atoms
        unique_neigh_info = [neigh_info[i] for i in unique_atoms]
        
        # get array representations
        neigh_indices, neigh_images, pair_dists = self._get_neigh_info_from_pymatgen(
            neigh_info=unique_neigh_info
            )
        neigh_coords = [self.structure.frac_coords[neigh_indices[i]] + neigh_images[i] for i in range(len(neigh_indices))]

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
                    dist*cutoff*2,
                    sites=[self.structure[site_idx]],
                    exclude_self=True
                    )
            # get neighbor frac coords
            max_neigh_coords = self.structure.frac_coords[max_neigh_indices] + max_neigh_images
            # get the neighbors that lie within our voronoi surface
            vals = A @ max_neigh_coords.T + b[:, None]
            important_mask = np.all(vals <= -1e-12, axis=0) & (max_pair_dists !=0)
            important_indices = np.where(important_mask)[0]
            # write our new neighbor info
            neigh_indices[unique_idx] = max_neigh_indices[important_indices]
            neigh_images[unique_idx] = max_neigh_images[important_indices]
            neigh_coords[unique_idx] = max_neigh_coords[important_indices]
            pair_dists[unique_idx] = max_pair_dists[important_indices]
        
        # Now we want to calculate the ELF radii for each possible pair. To do
        # this efficiently, we first want to reduce to symmetrically equivalent
        # bonds. We do this by converting each bond into a canonical representation
        
        # move all bond info to a single array
        site_indices = np.concatenate([[k for i in range(len(j))] for k, j in zip(unique_atoms,neigh_indices)],dtype=np.int32)
        neigh_indices = np.concatenate(neigh_indices)
        neigh_images = np.concatenate(neigh_images)
        neigh_coords = np.concatenate(neigh_coords)
        pair_dists = np.concatenate(pair_dists)

        canonical_bonds = get_canonical_bonds(
            site_indices = site_indices,
            neigh_indices = neigh_indices,
            neigh_coords=neigh_coords, 
            equivalent_atoms=equivalent_atoms,
            all_frac_coords=self.structure.frac_coords,
            rotation_matrices=rotation_matrices,
            translation_vectors=translation_vectors,
            pair_dists=pair_dists,
            tol=0.02,
            )

        # get a mask for reversed bonds
        reversed_mask = canonical_bonds[:,0] == 1
        
        # get unique bonds. Reverse bonds are counted as the same (i.e. Ca-N == N-Ca)
        unique_bonds, indices, inverse = np.unique(
            canonical_bonds[:,1:], 
            return_index=True, 
            return_inverse=True,
            axis=0
            )
        
        unique_site_indices = site_indices[indices]
        unique_neigh_indices = neigh_indices[indices]
        unique_neigh_images = neigh_images[indices]
        unique_neigh_coords = neigh_coords[indices]
        unique_pair_dists = pair_dists[indices]
        is_reverse = reversed_mask[indices]

        unique_neigh_coords = self.structure.frac_coords[unique_neigh_indices] + unique_neigh_images
        # get radii for each unique bond
        radii, fracs, bond_types = self._get_elf_radii_and_type(
            unique_site_indices, 
            unique_neigh_indices, 
            unique_neigh_coords, 
            unique_pair_dists,
            is_reverse,
            )

        # assign fractions back to each bond
        fracs = fracs[inverse]
        bond_types = bond_types[inverse]
        # reverse any that need it
        fracs[reversed_mask] = 1-fracs[reversed_mask]
        bond_types[reversed_mask] = pair_dists[reversed_mask] - bond_types[reversed_mask]

        # get ranges for each site
        site_ranges = np.where(site_indices[:-1] != site_indices[1:])[0]+1
        site_ranges = np.insert(site_ranges, [0, len(site_ranges)], [0, len(site_indices)])
        
        # calculate all of the plane equations
        halfspaces = []
        for unique_idx, site_idx in enumerate(unique_atoms):
            # get range of planes
            lower = site_ranges[unique_idx]
            upper = site_ranges[unique_idx+1]
            
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
            upper = site_ranges[unique_idx+1]
            
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
            unique_equiv_, unique_indices, unique_inverse = np.unique(unique_equiv, return_index=True, return_inverse=True)
            
            current_unique_halfspaces = halfspaces[lower:upper][unique_indices]

            important_planes = get_planes_on_surface(
                current_unique_halfspaces,
                vertices
                )
            # note important halfspaces
            important_plane_mask[lower:upper] = important_planes[unique_inverse]

        # expand important unique planes
        important_plane_mask = np.where(important_plane_mask)[0]
        site_indices = site_indices[important_plane_mask]
        neigh_indices = neigh_indices[important_plane_mask]
        neigh_images = neigh_images[important_plane_mask]
        neigh_coords = neigh_coords[important_plane_mask]
        fracs = fracs[important_plane_mask]
        bond_types = bond_types[important_plane_mask]

        # Generate all bonds using symmetry operations
        all_bonds = generate_symmetric_bonds(
                site_indices=site_indices,
                neigh_indices=neigh_indices,
                neigh_coords=neigh_coords,
                bond_types=bond_types,
                all_frac_coords=self.structure.frac_coords,
                fracs=fracs,
                rotation_matrices=rotation_matrices,
                translation_vectors=translation_vectors,
                shape=self.grid.shape,
                frac2cart=self.grid.matrix,
                tol=0.02,
                )

        # reduce to unique
        all_bonds, indices = np.unique(all_bonds, return_index=True, axis=0)

        # Return partitioning information
        site_indices = all_bonds[:,0].astype(int)
        neigh_indices = all_bonds[:,1].astype(int)
        radii = all_bonds[:,2]
        bond_types = all_bonds[:,3].astype(bool)
        plane_points = all_bonds[:, 4:7]
        plane_vectors = all_bonds[:, 7:10]
        neigh_coords = all_bonds[:, 10:]

        return (
            site_indices,
            neigh_indices,
            neigh_coords,
            radii,
            bond_types,
            plane_points,
            plane_vectors
            )



