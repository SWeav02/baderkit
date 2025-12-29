# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

###############################################################################
# Plane related functions
###############################################################################


@njit(fastmath=True, cache=True)
def get_plane_dist(
    point: NDArray,
    plane_equation: NDArray,
):
    """
    Gets the signed distance of a point from a plane

    Args:
        point (ArrayLike):
            A point in cartesian coordinates to compare with a plane
        plane_equation (ArrayLike):
            The plane equation (A*x+b <= 0)

    Returns:
        The distance of the point to the plane.
    """

    a, b, c, d = plane_equation
    x, y, z = point
    value_of_plane_equation = round(a * x + b * y + c * z + d, 12)
    # negative value is "below" (opposite normal)
    return value_of_plane_equation


@njit(parallel=True, cache=True)
def get_planes_on_surface(
    plane_equations,
    vertices,
    tol=1e-12,
):
    important_planes = np.zeros(len(plane_equations), dtype=np.bool_)

    for plane_idx in prange(len(plane_equations)):
        a, b, c, d = plane_equations[plane_idx]
        onplane = 0
        for x, y, z in vertices:
            value = x * a + y * b + z * c + d
            if abs(value) > tol:
                continue
            onplane += 1
            if onplane == 3:
                important_planes[plane_idx] = True
                break

    return important_planes


@njit(parallel=True, cache=True)
def get_neighs_in_planes(
    site_indices,
    site_plane_points,
    site_plane_vectors,
    possible_site_indices,
    possible_neigh_coords,
):

    # create an array to track which points sit in the planes
    important = np.ones(len(possible_site_indices), dtype=np.bool_)

    # Get the range of indices for each site's planes
    plane_ranges = np.where(site_indices[1:] != site_indices[:-1])[0] + 1
    plane_ranges = np.insert(
        plane_ranges, np.array((0, len(plane_ranges))), np.array((0, len(site_indices)))
    )

    # get a map from site index to plane ranges
    plane_map = np.empty(site_indices.max(), dtype=np.int64)
    plane_map[site_indices] = np.arange(len(site_indices))

    # check each neighbor against the corresponding sites voronoi surface
    for pair_idx in prange(len(possible_site_indices)):
        site_index = possible_site_indices[pair_idx]
        neigh_coords = possible_neigh_coords[pair_idx]
        # iterate over planes in site
        low_bound = plane_ranges[plane_map[site_index]]
        upper_bound = plane_ranges[plane_map[site_index] + 1]
        for plane_idx in range(low_bound, upper_bound):
            plane_point = site_plane_points[plane_idx]
            plane_vector = site_plane_vectors[plane_idx]
            # calculate distance from plane
            dist = get_plane_dist(neigh_coords, plane_vector, plane_point)
            if dist < 0.0:
                important[pair_idx] = False
                break
    return important


@njit(cache=True)
def cube_fraction_under_plane(plane_vector, d):
    """
    Calculates the volume of the intersection of a unit cube [0,1]^3
    and the halfspace a*x + b*y + c*z + d <= 0.

    Args:
        plane_vector (tuple/list): Normal vector (a, b, c).
        d (float): Threshold value.
    """
    # flip d to transform convention: a*x <= -d
    d = -d

    # Normalize so all a_i >= 0 using cube symmetry
    # If a_i is negative, substitute x_i = (1 - x_i'), flipping the axis
    # also collect our nonzero coeffs and the denomenator

    # copy plane vector to avoid overwrite
    plane_vector = plane_vector.copy()

    coeffs = []
    denom = 1.0
    n = 0
    for i in range(3):
        if plane_vector[i] < 0:
            # flip sign and transform d
            plane_vector[i] = -plane_vector[i]
            d = d + plane_vector[i]

    # NOTE: Must be done after to ensure proper d sign
    for i in range(3):
        if plane_vector[i] == 0:
            if d < 0:
                return 0.0  # Plane is completely outside cube
        else:
            coeffs.append(plane_vector[i])
            n += 1
            denom *= n * plane_vector[i]

    if n == 0:
        return 1.0 if d <= 0 else 0.0

    # 2. Universal n-dimensional intersection formula
    # Formula: (1 / (n! * prod(a_i))) * sum over vertices v: (-1)^|v| * (t - a·v)_+^n

    volume = 0.0
    # Iterate through all 2^n vertices of the reduced-dimension cube
    # construct iteration for n coeffs
    out = np.empty((1 << n, n), dtype=np.int8)
    for i in range(1 << n):
        for j in range(n):
            out[i, j] = (i >> (n - 1 - j)) & 1

    for v in out:
        dot = 0
        for i, j in zip(v, coeffs):
            dot += i * j
        hamming_weight = np.sum(v)

        # Positive part operator (t - a·v)_+^n
        term_base = d - dot
        if term_base > 0:
            term = term_base**n
            if hamming_weight % 2 == 1:
                volume -= term
            else:
                volume += term

    volume /= denom

    # clamp for stability
    return min(1.0, max(0.0, abs(round(volume, 12))))


@njit(parallel=True, cache=True)
def get_cell_wrapped_voronoi(plane_points, plane_vectors, site_indices, tol=1e-12):
    num_sites = len(np.unique(site_indices))
    # get transformations to nearby unit cells. We put the self transform first
    # as this is generally the most likely to contain voxels
    transformations = np.empty((27, 3), dtype=np.int64)
    transformations[0] = (0, 0, 0)
    idx = 0
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                if i == 0 and j == 0 and k == 0:
                    continue
                idx += 1
                transformations[idx] = (i, j, k)

    # now, for each plane we transform it, check if it includes all or none of
    # the vertices, and record if we want to keep it or not
    all_site_indices = np.empty(len(site_indices) * 27, dtype=np.uint16)
    all_transforms = np.empty(len(site_indices) * 27, dtype=np.uint8)
    all_plane_points = np.empty((len(site_indices) * 27, 3), dtype=np.float64)
    all_plane_vectors = np.empty((len(site_indices) * 27, 3), dtype=np.float64)
    all_plane_volumes = np.empty(len(site_indices) * 27, dtype=np.float64)
    discard_planes = np.zeros(len(site_indices) * 27, dtype=np.bool_)
    discard_trans = np.zeros(num_sites * 27, dtype=np.bool_)
    for plane_idx in prange(len(plane_points)):
        site_idx = site_indices[plane_idx]
        point = plane_points[plane_idx]
        vector = plane_vectors[plane_idx]
        # round for stability
        vector = np.round(vector, 12)
        a, b, c = vector
        for trans_idx, transformation in enumerate(transformations):
            all_plane_idx = plane_idx * 27 + trans_idx
            trans_point = point + transformation
            # calculate the portion of the unit cell underneath this plane
            d = np.dot(-vector, trans_point)
            volume = cube_fraction_under_plane(
                vector,
                d,
            )

            if volume == 0:
                # if all_above:
                # no section of the grid is inside this site/transforms partitioning
                # and we want to remove all of them
                comb_idx = site_idx * 27 + trans_idx
                discard_trans[comb_idx] = True

            elif volume == 1.0:
                # elif all_under:
                # we can discard this plane as the entire unit cell is beneath it
                discard_planes[all_plane_idx] = True

            else:
                # this might be an important plane so we save it
                all_site_indices[all_plane_idx] = site_idx
                all_transforms[all_plane_idx] = trans_idx
                all_plane_points[all_plane_idx] = trans_point
                all_plane_vectors[all_plane_idx] = vector
                # we also go ahead and calculate the portion of the unit cell
                # that lies under this plane as we've already gone to the trouble
                # of calculating distances to each vertex.
                all_plane_volumes[all_plane_idx] = cube_fraction_under_plane(
                    vector,
                    d,
                )

    # Next we loop over again to remove any planes whose entire transformation is
    # lost
    for plane_idx in prange(len(plane_points)):
        site_idx = site_indices[plane_idx]
        for trans_idx, transformation in enumerate(transformations):
            comb_idx = site_idx * 27 + trans_idx
            if discard_trans[comb_idx]:
                all_plane_idx = plane_idx * 27 + trans_idx
                discard_planes[all_plane_idx] = True
    # remove planes we don't need
    plane_indices = np.where(~discard_planes)[0]
    all_site_indices = all_site_indices[plane_indices]
    all_transforms = all_transforms[plane_indices]
    all_plane_points = all_plane_points[plane_indices]
    all_plane_vectors = all_plane_vectors[plane_indices]
    all_plane_volumes = all_plane_volumes[plane_indices]

    return (
        all_site_indices,
        all_transforms,
        all_plane_points,
        all_plane_vectors,
        all_plane_volumes,
    )


###############################################################################
# Symmetry and bonding functions
###############################################################################


@njit(cache=True)
def find_site_in_tol(
    chunked_coords,
    site_coords,
    tol,
):
    chunked = np.round(site_coords / tol).astype(np.int64)
    # Match wrapped coord to site
    index = -1
    for j, chunked_coord in enumerate(chunked_coords):

        d = chunked - chunked_coord
        if np.sum(np.abs(d)) == 0:
            index = j
            break

    return index


@njit(cache=True)
def get_canonical_displacement(bond_displacement, tol):
    # wrap into cell
    bond_displacement -= np.round(bond_displacement)

    # quantize to tolerance
    v = np.round(bond_displacement / tol).astype(np.int64)

    # choose lexicographically positive representative
    if v[0] < 0 or (v[0] == 0 and v[1] < 0) or (v[0] == 0 and v[1] == 0 and v[2] < 0):
        v[0] = -v[0]
        v[1] = -v[1]
        v[2] = -v[2]

    return v


@njit(cache=True)
def get_canonical_bond(
    site_idx,
    neigh_idx,
    neigh_coords,
    all_frac_coords,
    equivalent_atoms,
    rotation_matrices,
    translation_vectors,
    included_ops,
    pair_dist,
    tol,
):

    # get frac coords for the site
    site_coords = all_frac_coords[site_idx]

    # create a placeholder for the best canonical bond
    best_rep = np.full(7, np.iinfo(np.int64).max, dtype=np.int64)

    # initially use the current site/neighbor pair
    if equivalent_atoms[site_idx] <= equivalent_atoms[neigh_idx]:
        best_rep[0] = 0  # bond is not inverted
        best_rep[1] = equivalent_atoms[site_idx]
        best_rep[2] = equivalent_atoms[neigh_idx]
    else:
        best_rep[0] = 1  # bond is inverted
        best_rep[2] = equivalent_atoms[site_idx]
        best_rep[1] = equivalent_atoms[neigh_idx]
    best_rep[3:6] = get_canonical_displacement(neigh_coords - site_coords, tol)
    # get quantized pair distance to retain information on bond length
    best_rep[6] = round(pair_dist / tol)

    # iterate over valid symmetry operations
    for trans_idx, valid in enumerate(included_ops):
        # skip operations that don't transform to the lowest index equivalent atom
        # (checked ahead of time for speed)
        if not valid:
            continue

        # get transformed site and neighbor
        matrix = rotation_matrices[trans_idx]
        vector = translation_vectors[trans_idx]

        trans_site_coords = matrix @ site_coords + vector
        trans_neigh_coords = matrix @ neigh_coords + vector

        # get the displacement vector
        displacement = trans_neigh_coords - trans_site_coords

        # canonize
        displacement = get_canonical_displacement(displacement, tol)

        # check if this is a better representation than the current one
        if displacement[0] > best_rep[3]:
            continue
        elif displacement[1] > best_rep[4]:
            continue
        elif displacement[2] > best_rep[5]:
            continue

        # if we're still here, this is equal or better than the best rep
        best_rep[3:6] = displacement

    # Choose unique canonical representative
    return best_rep


@njit(parallel=True, cache=True)
def get_canonical_bonds(
    site_indices,
    neigh_indices,
    neigh_coords,
    equivalent_atoms,
    all_frac_coords,
    rotation_matrices,
    translation_vectors,
    pair_dists,
    tol=0.02,
):

    chunked_coords = np.round(all_frac_coords / tol).astype(np.int64)

    # first, we narrow down the symmetry operations by including only those that
    # map to the lowest index equivalent atom
    unique_sites = np.unique(site_indices)
    unique_map = np.empty(unique_sites[-1] + 1, dtype=np.uint16)
    unique_map[unique_sites] = np.arange(len(unique_sites))

    symm_op_mask = np.zeros(
        (len(unique_sites), len(translation_vectors)), dtype=np.bool_
    )

    for unique_idx in prange(len(unique_sites)):
        site_idx = unique_sites[unique_idx]
        equiv_idx = equivalent_atoms[site_idx]
        site_coords = all_frac_coords[site_idx]
        for op_idx, (matrix, vector) in enumerate(
            zip(rotation_matrices, translation_vectors)
        ):
            trans_site_coords = matrix @ site_coords + vector
            # wrap into cell
            trans_site_coords %= 1

            # get index after operation
            new_idx = find_site_in_tol(
                chunked_coords,
                trans_site_coords,
                tol,
            )

            # if this is the lowest equivalent atom, we will include this operation
            if new_idx == equiv_idx:
                symm_op_mask[unique_idx, op_idx] = True

    # create an array to store canonical bonds
    canonical_bonds = np.empty((len(site_indices), 7), dtype=np.int64)

    # each row is in order of:
    # 1. whether or not the canonical rep is the reverse of the original bond
    # 2. The lower site index in the bond
    # 3. The higher site index in the bond
    # 4-6. integer representations of the lowest displacement vector

    for bond_idx in prange(len(site_indices)):
        site_idx = site_indices[bond_idx]
        neigh_idx = neigh_indices[bond_idx]
        neigh_coord = neigh_coords[bond_idx]
        pair_dist = pair_dists[bond_idx]
        included_ops = symm_op_mask[unique_map[site_idx]]

        canonical_bonds[bond_idx] = get_canonical_bond(
            site_idx=site_idx,
            neigh_idx=neigh_idx,
            neigh_coords=neigh_coord,
            all_frac_coords=all_frac_coords,
            equivalent_atoms=equivalent_atoms,
            rotation_matrices=rotation_matrices,
            translation_vectors=translation_vectors,
            included_ops=included_ops,
            pair_dist=pair_dist,
            tol=tol,
        )

    return canonical_bonds


@njit(parallel=True, cache=True)
def generate_symmetric_bonds(
    site_indices,
    neigh_indices,
    neigh_coords,
    bond_types,
    all_frac_coords,
    fracs,
    rotation_matrices,
    translation_vectors,
    shape,
    frac2cart,
    tol,
):

    # create an array to store all bond information
    # site, neighbor, plane equation
    n_transforms = len(translation_vectors)
    all_bonds = np.empty((n_transforms * len(site_indices), 13), dtype=np.float64)

    chunked_coords = np.round(all_frac_coords / tol).astype(np.int64)

    for pair_idx in prange(len(site_indices)):
        site_idx = site_indices[pair_idx]
        neigh_idx = neigh_indices[pair_idx]
        site_coord = all_frac_coords[site_idx]
        neigh_coord = neigh_coords[pair_idx]
        bond_type = bond_types[pair_idx]

        frac = fracs[pair_idx]

        # apply each transformation
        bond_idx = pair_idx * n_transforms
        for trans_idx, (matrix, vector) in enumerate(
            zip(rotation_matrices, translation_vectors)
        ):
            trans_site_coord = matrix @ site_coord + vector
            trans_neigh_coord = matrix @ neigh_coord + vector

            # get image of new site
            site_image = np.floor(trans_site_coord).astype(np.int64)
            # move site and get index
            test_trans_site_coord = trans_site_coord - site_image
            site_idx = find_site_in_tol(
                chunked_coords=chunked_coords,
                site_coords=test_trans_site_coord,
                tol=tol,
            )
            # if that didn't work, try again with tolerance before wrap
            if site_idx == -1:
                site_image = np.floor(trans_site_coord + tol).astype(np.int64)
                test_trans_site_coord = trans_site_coord - site_image
                site_idx = find_site_in_tol(
                    chunked_coords=chunked_coords,
                    site_coords=test_trans_site_coord,
                    tol=tol,
                )
            # do the same for the neighbor after shifting it
            trans_neigh_coord = trans_neigh_coord - site_image

            # get neigh image
            neigh_image = np.floor(trans_neigh_coord).astype(np.int64)
            # wrap neigh
            test_trans_neigh_coord = trans_neigh_coord - neigh_image
            neigh_idx = find_site_in_tol(
                chunked_coords=chunked_coords,
                site_coords=test_trans_neigh_coord,
                tol=tol,
            )
            if neigh_idx == -1:
                neigh_image = np.floor(trans_neigh_coord + tol).astype(np.int64)
                test_trans_neigh_coord = trans_neigh_coord - neigh_image
                neigh_idx = find_site_in_tol(
                    chunked_coords=chunked_coords,
                    site_coords=test_trans_neigh_coord,
                    tol=tol,
                )

            # get the exact site/neigh coords
            trans_site_coord = all_frac_coords[site_idx]
            trans_neigh_coord = all_frac_coords[neigh_idx] + neigh_image

            # calculate the exact plane vector and point for this neighbor. This
            # can be slightly different if atoms aren't at exact positions
            plane_vector = trans_neigh_coord - trans_site_coord

            # get fractional vector
            plane_vector *= frac
            # calculate radius
            cart_vector = plane_vector @ frac2cart
            radius = np.linalg.norm(cart_vector)

            # get plane equation
            plane_point = plane_vector + trans_site_coord
            plane_vector = plane_vector / np.linalg.norm(plane_vector)
            # set plane equation
            all_bonds[bond_idx, 0] = site_idx
            all_bonds[bond_idx, 1] = neigh_idx
            all_bonds[bond_idx, 2] = radius
            all_bonds[bond_idx, 3] = bond_type
            all_bonds[bond_idx, 4:7] = plane_point
            all_bonds[bond_idx, 7:10] = plane_vector
            all_bonds[bond_idx, 10:] = trans_neigh_coord
            bond_idx += 1

    return np.round(all_bonds, 12)
