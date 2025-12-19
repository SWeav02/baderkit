# -*- coding: utf-8 -*-

import numpy as np
from numpy.typing import NDArray
from numba import njit, prange


@njit(fastmath=True, cache=True)
def get_plane_dist(
    point: NDArray,
    plane_vector: NDArray,
    plane_point: NDArray,
):
    """
    Gets the sign associated with a point compared with a plane.

    Args:
        point (ArrayLike):
            A point in cartesian coordinates to compare with a plane
        plane_vector (ArrayLike):
            The vector normal to the plane of interest
        plane_point (ArrayLike):
            A point on the plane of interest

    Returns:
        The distance of the point to the plane.
    """
    # get all of the points in cartesian coordinates
    x, y, z = plane_point
    a, b, c = plane_vector
    x1, y1, z1 = point
    value_of_plane_equation = round(
        (a * (x - x1) + b * (y - y1) + c * (z - z1)), 12
    )
    # positive value is "below" (opposite normal)
    return value_of_plane_equation

@njit(parallel=True, cache=True)
def reduce_voronoi_planes_conservative(
    site_indices,
    plane_points,
    plane_vectors,
    angle_tol=1e-6,
    dist_tol=1e-12,
):
    n = len(site_indices)
    important = np.ones(n, dtype=np.bool_)

    for i in prange(n):
        si = site_indices[i]
        ni = plane_vectors[i]
        pi = plane_points[i]

        for j in range(n):
            if j == i or site_indices[j] != si:
                continue

            nj = plane_vectors[j]
            pj = plane_points[j]

            # check near-parallel
            dot = np.dot(ni, nj)
            if dot < 1 - angle_tol:
                continue

            # project pi onto plane j normal
            d = np.dot(pi - pj, nj)

            if d < -dist_tol:
                important[i] = False
                break

    return important

@njit(cache=True)
def intersect_three_planes(p1, n1, p2, n2, p3, n3, tol=1e-12):
    """
    Returns (success, point)
    """
    A = np.vstack((n1, n2, n3))
    det = np.linalg.det(A)
    if abs(det) < tol:
        return False, np.zeros(3)

    b = np.array([
        np.dot(n1, p1),
        np.dot(n2, p2),
        np.dot(n3, p3),
    ])

    x = np.linalg.solve(A, b)
    return True, x

@njit(cache=True)
def satisfies_all_planes(
        x, 
        i,
        j,
        k,
        si,
        site_indices,
        important_indices,
        important,
        plane_points, 
        plane_vectors, 
        ):
    for idx in important_indices:
        # skip points that don't belong to this site or are part of the group
        # of planes making up this intercept
        if site_indices[idx] != si:
            continue
        if idx == i or idx == j or idx == k:
            continue
        # if this is a previous plane that we've found to not be important, we
        # can skip it
        if idx < i and not important[idx]:
            continue
        # check if this intercept is outside this plane. If so, it is not important
        if get_plane_dist(
                point=x, 
                plane_point=plane_points[idx], 
                plane_vector=plane_vectors[idx]
                ) <= 0.0:
            return False
    return True

@njit(cache=True)
def get_voronoi_planes_exact(
    site_indices,
    plane_points,
    plane_vectors,
    important,
):
    # get the plane indices that are still considered important
    important_indices = np.where(important)[0]
    
    # create new important mask assuming none important
    important = np.zeros_like(important, dtype=np.bool_)

    for i in important_indices:
        # if we've already found this plane to be important, we can skip
        if important[i]:
            continue

        si = site_indices[i]
        intersect_found = False

        for jj in range(len(important_indices)):
            j = important_indices[jj]
            if site_indices[j] != si or j == i:
                continue
            
            # if this index is lower than i and doesn't correspond to an important
            # plane, we've already determined this plane to not be part of the
            # partitioning and can skip
            if j < i and not important[j]:
                continue

            for k in important_indices[jj:]:
                if site_indices[k] != si or k == i:
                    continue
                
                if k < i and not important[k]:
                    continue
                
                # find the intersection of these three planes
                ok, x = intersect_three_planes(
                    plane_points[i], plane_vectors[i],
                    plane_points[j], plane_vectors[j],
                    plane_points[k], plane_vectors[k],
                )
                
                # if no intersection was found (i.e. two parallel planes) continue
                if not ok:
                    continue
                
                intersect_found = True
                # check if this intersection lies below all other planes for
                # this site                
                if satisfies_all_planes(x, i, j, k, si, site_indices, important_indices, important, plane_points, plane_vectors):
                    # if it does, all three of these planes must be important
                    important[i] = True
                    important[j] = True
                    important[k] = True
                    break

            if important[i]:
                break
            
        # if we never found a valid intersect, we automatically accept this plane
        if not intersect_found:
            important[i] = True

    return important

