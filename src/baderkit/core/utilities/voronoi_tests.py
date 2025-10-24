# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 14:45:05 2025

@author: Sam
"""
import math

from numba import njit, prange
import numpy as np

from baderkit.core.utilities.coord_env import frac2cart_numba

@njit
def get_dist_to_plane(
    point,
    plane_point,
    plane_vector,
        ):
    
    x,y,z = plane_point
    a,b,c = plane_vector
    x1,y1,z1 = point
    return round((a * (x - x1) + b * (y - y1) + c * (z - z1)), 12)

@njit(parallel=True)
def get_tiled_atoms(
        lattice,
        frac_coords,
        min_size: float = 15,
        ):
    """gets the cartesian coordinates of a supercell of atoms"""
    
    # get the minimum number of transformations needed along each lattice vector.
    # To do this, we want to find the largest steps in our x/y/z directions our
    # lattice vectors allow us to make and set our limits based on that
    abs_lattice = np.abs(lattice)
    # get lattice direction that change x,y,z cart directions the most
    best_x = np.argmax(abs_lattice[:,0])
    best_y = np.argmax(abs_lattice[:,1])
    best_z = np.argmax(abs_lattice[:,2])
    # get the amount x,y,z cart directions change along these lattice directions
    max_x = abs_lattice[best_x,0]
    max_y = abs_lattice[best_y,1]
    max_z = abs_lattice[best_z,2]
    # get the minimum number of transformations along x,y,z directions that are
    # needed to reach our cutoff distance
    xn = np.ceil(min_size / max_x)
    yn = np.ceil(min_size / max_y)
    zn = np.ceil(min_size / max_z)
    # get the minimum number of transformations needed for each lattice vector
    min_trans = np.ones(3, dtype=np.int64)
    for latt_idx, n in zip((best_x, best_y, best_z), (xn,yn,zn)):
        if n > min_trans[latt_idx]:
            min_trans[latt_idx] = n
    ni, nj, nk = min_trans
    n_trans = (2*ni+1)*(2*nj+1)*(2*nk+1)
    # generate the possible transformations and order by the amount they increase
    # distance from the central cell
    transforms = np.empty((n_trans, 3), dtype=np.int64)     
    trans_dists = np.empty(n_trans, dtype=np.float64)
    required_transforms = np.zeros(n_trans, dtype=np.bool_)
    trans_idx = 0
    for i in range(-ni, ni+1):
        for j in range(-nj, nj+1):
            for k in range(-nk, nk+1):
                if abs(i) < 2 and abs(j) < 2 and abs(k) < 2:
                    required_transforms[trans_idx] = True
                transforms[trans_idx] = (i,j,k)
                # get the distance to this transformation
                dist = abs(i)*lattice[0] + abs(j)*lattice[1] + abs(k)*lattice[2]
                dist = np.linalg.norm(dist)
                trans_dists[trans_idx] = dist
                trans_idx += 1
    # get transforms within our cutoff (while ensuring at least one transform in
    # each direction)
    valid_trans_mask = (trans_dists <= min_size) | required_transforms
    valid_trans_indices = np.where(valid_trans_mask)[0]
    # cut down transforms
    transforms = transforms[valid_trans_indices]
    trans_dists = trans_dists[valid_trans_indices]
    # sort by distance
    sorted_trans_indices = np.argsort(trans_dists)
    transforms = transforms[sorted_trans_indices]
    
    # make an array to store the cart coords that will be generated
    n_atoms = len(frac_coords)
    cart_array = np.empty((len(transforms)*n_atoms, 3), dtype=np.float64)
    image_array = np.empty((len(transforms)*n_atoms, 3), dtype=np.int64)
    index_array = np.empty(len(transforms)*n_atoms, dtype=np.int64)
    for trans_idx in prange(len(transforms)):
        si, sj, sk = transforms[trans_idx]
        for atom_idx in range(n_atoms):
            fi, fj, fk = frac_coords[atom_idx]
            # get new coords
            ni = fi + si
            nj = fj + sj
            nk = fk + sk
            # convert to cartesian
            ci, cj, ck = frac2cart_numba(lattice, ni, nj, nk)
            # update array
            cart_idx = n_atoms * trans_idx + atom_idx
            cart_array[cart_idx] = (ci, cj, ck)
            image_array[cart_idx] = (si, sj, sk)
            index_array[cart_idx] = atom_idx
    return cart_array, image_array, index_array

@njit(parallel=True)
def get_neighs_in_radius(
    atom_idx,
    radius,
    tiled_cart_coords,
    tiled_images,
    tiled_indices
        ):
    # get the cartesian coords of this atom (first instance in array)
    atom_list_idx = np.searchsorted(tiled_indices, atom_idx)
    atom_cart_coords = tiled_cart_coords[atom_list_idx]
    # calculate distances
    n_coords = len(tiled_indices)
    dists = np.empty(n_coords, dtype=np.float64)
    neigh_trans = np.empty((n_coords, 3), dtype=np.float64)
    for idx in prange(n_coords):
        neigh_cart_coords = tiled_cart_coords[idx]
        diff = neigh_cart_coords-atom_cart_coords
        neigh_trans[idx] = diff
        dists[idx] = np.linalg.norm(diff)
    # get values within cutoff
    valid_indices = np.where(dists <= radius)[0]
    dists = dists[valid_indices]
    # sort
    sorted_indices = np.argsort(dists)
    # get sorted values to return
    neigh_indices = tiled_indices[valid_indices][sorted_indices]
    neigh_images = tiled_images[valid_indices][sorted_indices]
    neigh_trans = neigh_trans[valid_indices][sorted_indices]
    neigh_coords = tiled_cart_coords[valid_indices][sorted_indices]
    neigh_dists = dists[sorted_indices]
    # return without point at atom
    return neigh_indices[1:], neigh_images[1:], neigh_dists[1:], neigh_coords[1:], neigh_trans[1:]
    
        
def get_elf_voronoi_neighs():
    # Steps:
        # 1. Get atom neighbors that form closed region if a plane is placed
        # at each atom
        # 2. Set possible candidate neighbors to atoms sitting within planes if
        # they sit twice as far from the central atom
        # 3. For each candidate:
            # a. calculate plane position between atoms.
            # b. check if under all other planes
            # c. add or discard
        # 4. Repeat, check for all added planes
    pass    
