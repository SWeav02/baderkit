# -*- coding: utf-8 -*-
import math

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray


def get_lowest_uint(max_value):
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if np.iinfo(dtype).max > max_value:
            break
    return dtype


def get_lowest_int(max_value):
    for dtype in (np.int8, np.int16, np.int32, np.int64):
        if np.iinfo(dtype).max > max_value:
            break
    return dtype


@njit(cache=True)
def get_norm(i, j, k):
    return (i**2 + j**2 + k**2) ** (1 / 2)


@njit(cache=True, inline="always")
def dist(p1, p2):
    x, y, z = p1
    x1, y1, z1 = p2
    return get_norm(x1 - x, y1 - y, z1 - z)


@njit(cache=True)
def get_gradient_cart(i, j, k, data, dir2car):
    nx, ny, nz = data.shape

    c100 = data[(i + 1) % nx, j, k]
    c_100 = data[(i - 1) % nx, j, k]
    c010 = data[i, (j + 1) % ny, k]
    c0_10 = data[i, (j - 1) % ny, k]
    c001 = data[i, j, (k + 1) % nz]
    c00_1 = data[i, j, (k - 1) % nz]

    # central differences in fractional coordinates
    gi = (c100 - c_100) * 0.5
    gj = (c010 - c0_10) * 0.5
    gk = (c001 - c00_1) * 0.5

    # convert to Cartesian gradient
    gx = dir2car[0, 0] * gi + dir2car[0, 1] * gj + dir2car[0, 2] * gk
    gy = dir2car[1, 0] * gi + dir2car[1, 1] * gj + dir2car[1, 2] * gk
    gz = dir2car[2, 0] * gi + dir2car[2, 1] * gj + dir2car[2, 2] * gk

    return gx, gy, gz


@njit(cache=True, parallel=True, inline="always")
def mutiple_dists(p1s, p2s):
    dists = np.empty(len(p1s), dtype=np.float64)
    for idx in prange(len(p1s)):
        dists[idx] = dist(p1s[idx], p2s[idx])
    return dists


@njit(inline="always", cache=True)
def wrap_point(i, j, k, nx, ny, nz):

    if i < 0:
        i += nx
    elif i >= nx:
        i -= nx

    if j < 0:
        j += ny
    elif j >= ny:
        j -= ny

    if k < 0:
        k += nz
    elif k >= nz:
        k -= nz

    return i, j, k


@njit(inline="always", cache=True)
def wrap_point_w_shift(i, j, k, nx, ny, nz):

    si = 0
    sj = 0
    sk = 0

    if i < 0:
        i += nx
        si = -1
    elif i >= nx:
        i -= nx
        si = 1

    if j < 0:
        j += ny
        sj = -1
    elif j >= ny:
        j -= ny
        sj = 1

    if k < 0:
        k += nz
        sk = -1
    elif k >= nz:
        k -= nz
        sk = 1

    return i, j, k, si, sj, sk


@njit(fastmath=True, cache=True)
def flat_to_coords(idx, ny_nz, nz):
    i = idx // (ny_nz)
    j = (idx % (ny_nz)) // nz
    k = idx % nz
    return i, j, k


@njit(fastmath=True, cache=True)
def coords_to_flat(i, j, k, ny_nz, nz):
    return i * (ny_nz) + j * nz + k


@njit(fastmath=True, cache=True)
def merge_frac_coords(
    frac_coords,
):

    # We'll accumulate (unwrapped) coordinates into total
    total0 = 0.0
    total1 = 0.0
    total2 = 0.0
    count = 0

    # reference coord used for unwrapping
    ref0 = 0.0
    ref1 = 0.0
    ref2 = 0.0
    ref_set = False

    # scan all maxima and pick those that belong to this target_group
    for c0, c1, c2 in frac_coords:

        # first seen -> set reference for unwrapping
        if not ref_set:
            ref0, ref1, ref2 = c0, c1, c2
            ref_set = True

        # unwrap coordinate relative to reference: unwrapped = coord - round(coord - ref)
        # Using np.round via float -> use built-in round for numba compatibility
        # but call round(x) (returns float)
        un0 = c0 - round(c0 - ref0)
        un1 = c1 - round(c1 - ref1)
        un2 = c2 - round(c2 - ref2)

        # add to total
        total0 += un0
        total1 += un1
        total2 += un2
        count += 1

    if count == 1:
        # return original point wrapped to [0,1)
        return np.array((ref0 % 1.0, ref1 % 1.0, ref2 % 1.0), dtype=np.float64)

    else:
        # return average of points (round for floating point error)
        avg0 = round(total0 / count, 12) % 1.0
        avg1 = round(total1 / count, 12) % 1.0
        avg2 = round(total2 / count, 12) % 1.0
        return np.array((avg0, avg1, avg2), dtype=np.float64)


@njit(fastmath=True, cache=True)
def merge_frac_coords_weighted(
    frac_coords,
    values,
    ref_coord=None,
    wrap=True,
):
    # normalize values
    values /= values.sum()

    # We'll accumulate (unwrapped) coordinates into total
    total0 = 0.0
    total1 = 0.0
    total2 = 0.0

    # reference coord used for unwrapping
    if ref_coord is None:
        ref0 = 0.0
        ref1 = 0.0
        ref2 = 0.0
        ref_set = False
    else:
        ref0 = ref_coord[0]
        ref1 = ref_coord[1]
        ref2 = ref_coord[2]
        ref_set = True

    # scan all maxima and pick those that belong to this target_group
    for (c0, c1, c2), weight in zip(frac_coords, values):

        # first seen -> set reference for unwrapping
        if not ref_set:
            ref0, ref1, ref2 = c0, c1, c2
            ref_set = True

        # unwrap coordinate relative to reference: unwrapped = coord - round(coord - ref)
        # Using np.round via float -> use built-in round for numba compatibility
        # but call round(x) (returns float)
        un0 = c0 - round(c0 - ref0)
        un1 = c1 - round(c1 - ref1)
        un2 = c2 - round(c2 - ref2)

        # add to total
        total0 += un0 * weight
        total1 += un1 * weight
        total2 += un2 * weight

    if wrap:
        return np.array((total0 % 1.0, total1 % 1.0, total2 % 1.0), dtype=np.float64)
    else:
        return np.array((total0, total1, total2), dtype=np.float64)


@njit(parallel=True)
def get_transforms_in_radius(
    r: float,
    nx,
    ny,
    nz,
    lattice_matrix: NDArray,
):
    """
    Generates all transforms within a radius r for a given grid. Results are sorted
    by distance

    Parameters
    ----------
    r : float
        The radius to consider.
    nx : int
        The number of grid points along the x lattice direction
    ny : int
        The number of grid points along the y lattice direction
    nz : int
        The number of grid points along the z lattice direction
    lattice_matrix : NDArray
        The row matrix representing the lattice.

    Returns
    -------
    offsets : NDArray
        The offsets up to the requested radius.
    dists : NDArray
        The distances up to the requested radius.

    """
    shape = (nx, ny, nz)
    # row lattice vectors
    a1, a2, a3 = lattice_matrix

    # get fraction along each vector that matches radius
    f1 = r / np.linalg.norm(a1)
    f2 = r / np.linalg.norm(a2)
    f3 = r / np.linalg.norm(a3)
    # get the maximum corresponding number of grid points
    nmax = np.ceil(np.array((f1 * nx, f2 * ny, f3 * nz))).astype(np.int64)

    # get matrix map from grid points to cartesian
    grid2cart = np.empty((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            grid2cart[i, j] = lattice_matrix[i, j] / shape[i]

    offsets = []
    dists = []

    for i in range(-nmax[0], nmax[0] + 1):
        for j in range(-nmax[1], nmax[1] + 1):
            for k in range(-nmax[2], nmax[2] + 1):
                ci = i * grid2cart[0, 0] + j * grid2cart[1, 0] + k * grid2cart[2, 0]
                cj = i * grid2cart[0, 1] + j * grid2cart[1, 1] + k * grid2cart[2, 1]
                ck = i * grid2cart[0, 2] + j * grid2cart[1, 2] + k * grid2cart[2, 2]

                d = (ci * ci + cj * cj + ck * ck) ** 0.5
                # round for stability
                d = round(d, 12)

                if d <= r:
                    dists.append(d)
                    offsets.append((i, j, k))
    offsets = np.array(offsets, dtype=np.int32)
    dists = np.array(dists, dtype=np.float64)

    # sort by distance
    sorted_indices = np.argsort(dists)
    offsets = offsets[sorted_indices]
    dists = dists[sorted_indices]
    return offsets, dists

@njit(parallel=True)
def get_transforms_in_voxels(
    r: int,
    nx,
    ny,
    nz,
    lattice_matrix: NDArray,
):
    """
    Generates all transforms within the voxel neighborhood

    Parameters
    ----------
    r : int
        The number of voxels away
    nx : int
        The number of grid points along the x lattice direction
    ny : int
        The number of grid points along the y lattice direction
    nz : int
        The number of grid points along the z lattice direction
    lattice_matrix : NDArray
        The row matrix representing the lattice.

    Returns
    -------
    offsets : NDArray
        The offsets up to the requested radius.
    dists : NDArray
        The distances up to the requested radius.

    """
    r = int(r)
    shape = np.array((nx, ny, nz), dtype=np.int64)
    num_trans = (r*2+1)**3 - 1

    transforms = np.empty((num_trans, 3), dtype=np.int64)
    dists = np.empty(num_trans, dtype=np.float64)

    idx = 0
    for i in range(-r, r+1):
        for j in range(-r, r+1):
            for k in range(-r, r+1):
                if i == 0 and j == 0 and k == 0:
                    continue
                shift = np.array((i, j, k), dtype=np.int64)
                frac = shift / shape
                cart = frac @ lattice_matrix
                dist = np.linalg.norm(cart)

                transforms[idx] = shift
                dists[idx] = dist
                idx += 1
    return transforms, dists

@njit(cache=True)
def compute_wrap_offset(point1, point2):
    """
    Computes wrap from point1 to point2

    """
    best_d2 = np.inf
    best_i = 0
    best_j = 0
    best_k = 0

    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                dx = (point2[0] + i) - point1[0]
                dy = (point2[1] + j) - point1[1]
                dz = (point2[2] + k) - point1[2]
                d2 = dx * dx + dy * dy + dz * dz

                if d2 < best_d2:
                    best_d2 = d2
                    best_i = i
                    best_j = j
                    best_k = k

    return best_i, best_j, best_k


@njit(cache=True)
def get_ongrid_gradient_cart(i, j, k, data, dir2car):
    nx, ny, nz = data.shape

    c000 = data[i, j, k]
    c100 = data[(i + 1) % nx, j, k]
    c_100 = data[(i - 1) % nx, j, k]
    c010 = data[i, (j + 1) % ny, k]
    c0_10 = data[i, (j - 1) % ny, k]
    c001 = data[i, j, (k + 1) % nz]
    c00_1 = data[i, j, (k - 1) % nz]

    # central differences in voxel coordinates
    gi = (c100 - c_100) / (2.0)
    gj = (c010 - c0_10) / (2.0)
    gk = (c001 - c00_1) / (2.0)

    # optional extrema clamping
    if c100 <= c000 and c_100 <= c000:
        gi = 0.0
    if c010 <= c000 and c0_10 <= c000:
        gj = 0.0
    if c001 <= c000 and c00_1 <= c000:
        gk = 0.0

    # convert to fractional-coordinate gradient
    gi *= nx
    gj *= ny
    gk *= nz

    # convert to Cartesian gradient
    gx = dir2car[0, 0] * gi + dir2car[0, 1] * gj + dir2car[0, 2] * gk
    gy = dir2car[1, 0] * gi + dir2car[1, 1] * gj + dir2car[1, 2] * gk
    gz = dir2car[2, 0] * gi + dir2car[2, 1] * gj + dir2car[2, 2] * gk

    return gx, gy, gz