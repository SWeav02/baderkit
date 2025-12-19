# -*- coding: utf-8 -*-
"""
This is a reimplementation of the ionic radius finder I created for BadELF in
[Simmate](https://github.com/jacksund/simmate/blob/main/src/simmate/apps/badelf/core/partitioning.py)
"""


import numpy as np
from numpy.typing import NDArray

from baderkit.core import Grid, Structure
from pymatgen.analysis.local_env import CrystalNN


from .elf_radii_numba import get_all_atom_elf_radii
from baderkit.core.utilities.voronoi import get_voronoi_planes_exact, reduce_voronoi_planes_conservative

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

    def get_all_neigh_elf_radii_and_type(
        self,
        site_indices,
        neigh_indices,
        neigh_frac_coords,
        neigh_dists,
    ):
        # first, get the unique bonds. We do this here because numba does not
        # currently allow np.unique axis argument
        # 1. get equivalent indices by symmetry for each atom/neighbor
        equivalent_atoms = self.structure.equivalent_atoms
        equiv_sites = equivalent_atoms[site_indices]
        equiv_neighs = equivalent_atoms[neigh_indices]

        # Create an array that is sorted such that the lower index of the bond
        # always come first. We will use this to avoid calculating the radius for
        # reverse bonds
        sorted_pair_array = np.column_stack((equiv_sites, equiv_neighs))
        sorted_indices = np.argsort(sorted_pair_array, axis=1)
        # get the indices that need to be flipped (we need these later)
        flipped_mask = sorted_indices[:,0] == 1
        # flip those indices
        flipped_indices = np.where(flipped_mask)[0]
        sorted_pair_array[flipped_indices]=np.fliplr(sorted_pair_array[flipped_indices])
        sorted_pair_array = np.column_stack((sorted_pair_array, neigh_dists))

        # get the unique bonds as a map pointing from a bond index to the equivalent
        # bond
        unique_bonds, index, inverse = np.unique(
            sorted_pair_array, return_inverse=True, return_index=True, axis=0
        )
        equivalent_bonds = index[inverse]
        
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
            neigh_dists=neigh_dists,
            equivalent_bonds=equivalent_bonds,
            reversed_bonds=flipped_mask,
            data=self.cubic_coeffs,
            feature_labels=self.feature_labels,
            covalent_labels=covalent_labels,
        )

    def get_voronoi_radii(self):
        # create crystalNN that will get all nearby atoms
        cnn = CrystalNN(
            weighted_cn=True,
            distance_cutoffs=None,
            x_diff_weight=0.0,
            porous_adjustment=False,
            )
        # get every sites neighbors
        neigh_info = cnn.get_all_nn_info(self.structure)
        
        # for each site, get all neighbors within a sphere of twice the largest
        # CrystalNN neighbor distance. Get relavent information
        site_indices = []
        neigh_indices = []
        neigh_coords = []
        pair_dists = []
        for i, site in enumerate(self.structure):
            nn = neigh_info[i]
            max_dist = max([i['site'].nn_distance for i in nn])
            # get neighbors in radius
            neighs = self.structure.get_sites_in_sphere(
                pt=site.coords,
                r=max_dist*2.1,
                include_index=True,
                include_image=True,
                )
            # collect information
            site_indices1 = [i for j in range(len(neighs))]
            neigh_indices1 = np.array([j.index for j in neighs], dtype=int)
            neigh_coords1 = np.array([j.frac_coords for j in neighs], dtype=float)
            pair_dists1 = np.array([j.nn_distance for j in neighs], dtype=float)
            
            # sort by neighbor distance
            sorted_indices = np.argsort(pair_dists1)
            
            # extend neighbor lists
            site_indices.extend(site_indices1)
            neigh_indices.extend(neigh_indices1[sorted_indices])
            neigh_coords.extend(neigh_coords1[sorted_indices])
            pair_dists.extend(pair_dists1[sorted_indices])

        # convert to arrays
        site_indices = np.array(site_indices, dtype=np.int64)
        neigh_indices = np.array(neigh_indices, dtype=np.int64)
        neigh_coords = np.array(neigh_coords, dtype=np.float64)
        pair_dists = np.array(pair_dists, dtype=np.float64).round(5)
        
        # remove site-site pairs
        site_site_mask = pair_dists != 0
        site_indices = site_indices[site_site_mask]
        neigh_indices = neigh_indices[site_site_mask]
        neigh_coords = neigh_coords[site_site_mask]
        pair_dists = pair_dists[site_site_mask]
        
        # get radii and bond types
        radii, bond_types = self.get_all_neigh_elf_radii_and_type(
            site_indices, neigh_indices, neigh_coords, pair_dists)

        # get cart coords 
        site_coords = self.structure.cart_coords[site_indices]
        neigh_coords = self.grid.frac_to_cart(neigh_coords)
        # calculate vectors from sites to neighs
        plane_vectors = neigh_coords - site_coords
        
        # calculate points on each plane
        fracs = np.array([radii / pair_dists]).T
        plane_points = site_coords + (plane_vectors * fracs)
        # normalize vectors
        magnitudes = np.linalg.norm(plane_vectors, axis=1, keepdims=True)
        plane_vectors /= magnitudes
        
        # remove planes that are nearly parallel with a closer plane
        important_planes = reduce_voronoi_planes_conservative(
            site_indices=site_indices,
            plane_points=plane_points,
            plane_vectors=plane_vectors
            )
        
        # remove planes that do not have a vertex on the voronoi surface of the
        # corresponding atom
        important_planes = get_voronoi_planes_exact(
            site_indices=site_indices,
            plane_points=plane_points,
            plane_vectors=plane_vectors,
            important=important_planes
            )

        # return voronoi information
        return (
            site_indices[important_planes],
            neigh_indices[important_planes],
            neigh_coords[important_planes],
            pair_dists[important_planes],
            radii[important_planes],
            bond_types[important_planes],
            plane_points[important_planes],
            plane_vectors[important_planes],
            )
