# -*- coding: utf-8 -*-
import numpy as np
from numba import njit, prange

@njit(parallel=True, cache=True)
def get_atom_nearest_neighbors(
        atom_frac_coords,
        atom_cart_coords,
        frac2cart,
        ):
    
    # create arrays to store results
    best_dists = np.full(len(atom_frac_coords), 100.0, dtype=np.float64)
    best_neighs = np.empty(len(atom_frac_coords), dtype=np.int64)
    best_images = np.empty((len(atom_frac_coords), 3), dtype=np.int64)

    # loop over each fractional coordinate. Transform it to neighboring
    # cells. Check distance to each neighbor.
    for i in prange(len(atom_frac_coords)):
        fi, fj, fk = atom_frac_coords[i]
        # loop over transformations
        for si in (-1, 0, 1):
            for sj in (-1, 0, 1):
                for sk in (-1, 0, 1):
                    # transform frac coord
                    ti = fi + si
                    tj = fj + sj
                    tk = fk + sk
                    # convert to cartestian coord
                    ci = ti * frac2cart[0][0] + tj * frac2cart[1][0] + tk * frac2cart[2][0]
                    cj = ti * frac2cart[0][1] + tj * frac2cart[1][1] + tk * frac2cart[2][1]
                    ck = ti * frac2cart[0][2] + tj * frac2cart[1][2] + tk * frac2cart[2][2]
                    # compare distance to each neighbor
                    for j, (nci, ncj, nck) in enumerate(atom_cart_coords):
                        # skip if this is the current coord
                        if j == i and si==0 and sj==0 and sk==0:
                            continue
                        # otherwise, calculate the distance
                        dist = ((nci-ci)**2 + (ncj-cj)**2 + (nck-ck)**2) ** 0.5
                        # if its lower than previous calculated distances, update
                        # our entry
                        if dist < best_dists[i]:
                            best_dists[i] = dist
                            best_neighs[i] = j
                            best_images[i] = (-si, -sj, -sk)
    return best_neighs, best_dists, best_images

@njit(parallel=True, cache=True)
def get_dists_to_atoms(
    frac_coord,
    atom_frac_coords,
    atom_cart_coords,
    frac2cart,
        ):
    num_atoms = len(atom_cart_coords)

    atom_dists = np.full(num_atoms, 1e6, dtype=np.float64)
    atom_vecs = np.empty((num_atoms, 3), dtype=np.float64)
    
    # calculate cartesian coordinates at each transform for this point (avoids
    # repeat calc)
    trans_cart_coords = np.empty((27,3), dtype=np.float64)
    fi, fj, fk = frac_coord
    trans_idx = 0
    for si in (-1, 0, 1):
        for sj in (-1, 0, 1):
            for sk in (-1, 0, 1):
                ti = fi + si
                tj = fj + sj
                tk = fk + sk
                # convert to cartesian
                ci = ti * frac2cart[0][0] + tj * frac2cart[1][0] + tk * frac2cart[2][0]
                cj = ti * frac2cart[0][1] + tj * frac2cart[1][1] + tk * frac2cart[2][1]
                ck = ti * frac2cart[0][2] + tj * frac2cart[1][2] + tk * frac2cart[2][2]
                trans_cart_coords[trans_idx] = (ci, cj, ck)
                trans_idx += 1
    
    # calculate the distance to each each atom at each transform and record
    # the shortest for each
    for i in prange(len(atom_cart_coords)):
        (ai, aj, ak) = atom_cart_coords[i]
        for ci, cj, ck in trans_cart_coords:
            # calculate distance to each atom
            di = ai-ci
            dj = aj-cj
            dk = ak-ck
            dist = ((di)**2 + (dj)**2 + (dk)**2) ** 0.5
            # if its lower than previous calculated distances, update
            # our entry
            if dist <= atom_dists[i]:
                # update the nearest value
                atom_dists[i] = dist
                atom_vecs[i] = (di, dj, dk)

    return atom_dists

@njit(cache=True)
def check_covalent(
    feature_frac_coord,
    atom_frac_coords,
    atom_cart_coords,
    frac2cart,
    min_covalent_angle,
        ):
    
    # first we find the three closest neighbors to this point
    # create arrays to store distances and vectors
    first_dist = 1e6
    second_dist = 1e6
    third_dist = 1e6
    
    first_vec = np.empty(3, dtype=np.float64)
    second_vec = np.empty(3, dtype=np.float64)
    
    first_atom = -1
    second_atom = -1
    
    # transform the coord to each neighboring unit cell (and the current cell)
    fi, fj, fk = feature_frac_coord
    for si in (-1, 0, 1):
        for sj in (-1, 0, 1):
            for sk in (-1, 0, 1):
                ti = fi + si
                tj = fj + sj
                tk = fk + sk
                # convert to cartesian
                ci = ti * frac2cart[0][0] + tj * frac2cart[1][0] + tk * frac2cart[2][0]
                cj = ti * frac2cart[0][1] + tj * frac2cart[1][1] + tk * frac2cart[2][1]
                ck = ti * frac2cart[0][2] + tj * frac2cart[1][2] + tk * frac2cart[2][2]
                # calculate distance to each atom
                for i, (ai, aj, ak) in enumerate(atom_cart_coords):
                    di = ai-ci
                    dj = aj-cj
                    dk = ak-ck
                    dist = ((di)**2 + (dj)**2 + (dk)**2) ** 0.5
                    # if its lower than previous calculated distances, update
                    # our entry
                    if dist <= first_dist:
                        # move second to third
                        third_dist = second_dist
                        # move first to second
                        second_dist = first_dist
                        second_vec[:] = first_vec
                        second_atom = first_atom
                        # update first
                        first_dist = dist
                        first_vec[:] = (di, dj, dk)
                        first_atom = i
                    
                    elif dist <= second_dist:
                        # move second to third
                        third_dist = second_dist
                        # update second
                        second_dist = dist
                        second_vec[:] = (di, dj, dk)
                        second_atom = i
                    elif dist < third_dist:
                        # update third
                        third_dist = dist
    
    # check if third neighbor is within 1% of second
    if (third_dist - second_dist) / second_dist < 0.01:
        return False, first_atom, second_atom
    
    # First we check that we are reasonably close to being along this bond. We
    # do this by checking the angle between the neighboring atoms and our basin.
    # This is corresponds to:
        # θ = arccos((A ⋅ B) / (|A|*|B|))
    # where A and B are the vectors from the feature to each neighboring atom
    A = first_vec
    B = second_vec

    cos_theta = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
    # make sure our theta is within the bounds of arcos
    cos_theta = max(-1.0, min(1.0, cos_theta))
    # get theta
    theta = np.arccos(cos_theta)
    
    # If our angle is not above our tolerance, we return as not a covalent bond
    if theta < min_covalent_angle:
        return False, first_atom, second_atom
    else:
        return True, first_atom, second_atom

@njit(parallel=True, cache=True)
def check_all_covalent(
    feature_frac_coords,
    atom_frac_coords,
    atom_cart_coords,
    frac2cart,
    min_covalent_angle,
        ):
    # create an array to store if each feature is covalent
    covalent_features = np.zeros(len(feature_frac_coords), dtype=np.bool_)
    atom_neighs = np.empty((len(feature_frac_coords), 2), dtype=np.uint16)
    for i in prange(len(feature_frac_coords)):
        feature_frac_coord = feature_frac_coords[i]
        in_tolerance, atom0, atom1 = check_covalent(
            feature_frac_coord,
            atom_frac_coords,
            atom_cart_coords,
            frac2cart,
            min_covalent_angle,
            )
        covalent_features[i] = in_tolerance
        atom_neighs[i] = (atom0, atom1)
    return covalent_features, atom_neighs