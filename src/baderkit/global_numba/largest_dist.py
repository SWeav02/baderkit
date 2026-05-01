# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import Voronoi, cKDTree

def generate_supercell(frac_coords, lattice):
    """Generate 3x3x3 periodic images."""
    shifts = np.array([[i, j, k] for i in [-1, 0, 1]
                                 for j in [-1, 0, 1]
                                 for k in [-1, 0, 1]])

    images = []
    for shift in shifts:
        shifted = frac_coords + shift
        images.append(shifted @ lattice)

    return np.vstack(images)


def wrap_to_unit_cell(cart_coords, lattice):
    """Map Cartesian coords back to unit cell (fractional in [0,1))."""
    inv_lat = np.linalg.inv(lattice)
    frac = cart_coords @ inv_lat
    frac_wrapped = frac % 1.0
    return frac_wrapped @ lattice


def largest_empty_sphere(lattice, frac_coords):
    """
    Find the point farthest from any lattice point (periodic).

    Returns:
        max_dist: radius of largest empty sphere
        best_point: Cartesian coordinates of that point
    """

    # Step 1: central + neighbors
    all_points = generate_supercell(frac_coords, lattice)

    # Step 2: Voronoi tessellation
    vor = Voronoi(all_points)

    # Step 3: KDTree for nearest neighbor queries
    tree = cKDTree(all_points)

    max_dist = -1
    best_point = None

    # Step 4: loop over Voronoi vertices
    for v in vor.vertices:
        # Wrap into unit cell
        v_wrapped = wrap_to_unit_cell(v, lattice)

        # Distance to nearest lattice point
        dist, _ = tree.query(v_wrapped)

        if dist > max_dist:
            max_dist = dist
            best_point = v_wrapped

    return max_dist, best_point