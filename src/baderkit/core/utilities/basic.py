# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, prange


@njit(cache=True, inline="always")
def dist(p1, p2):
    x, y, z = p1
    x1, y1, z1 = p2
    return ((x1 - x) ** 2 + (y1 - y) ** 2 + (z1 - z) ** 2) ** (0.5)


@njit(cache=True, parallel=True, inline="always")
def mutiple_dists(p1s, p2s):
    dists = np.empty(len(p1s), dtype=np.float64)
    for idx in prange(len(p1s)):
        dists[idx] = dist(p1s[idx], p2s[idx])
    return dists


@njit(cache=True, inline="always")
def wrap_point(
    i: np.int64, j: np.int64, k: np.int64, nx: np.int64, ny: np.int64, nz: np.int64
) -> tuple[np.int64, np.int64, np.int64]:
    """
    Wraps a 3D point (i, j, k) into the periodic bounds defined by the grid dimensions (nx, ny, nz).

    If any of the input coordinates are outside the bounds [0, nx), [0, ny), or [0, nz),
    they are wrapped around using periodic boundary conditions.

    Parameters
    ----------
    i : np.int64
        x-index of the point.
    j : np.int64
        y-index of the point.
    k : np.int64
        z-index of the point.
    nx : np.int64
        Number of grid points along x-direction.
    ny : np.int64
        Number of grid points along y-direction.
    nz : np.int64
        Number of grid points along z-direction.

    Returns
    -------
    tuple[np.int64, np.int64, np.int64]
        The wrapped (i, j, k) indices within the bounds.
    """
    if i >= nx:
        i -= nx
    elif i < 0:
        i += nx
    if j >= ny:
        j -= ny
    elif j < 0:
        j += ny
    if k >= nz:
        k -= nz
    elif k < 0:
        k += nz
    return i, j, k


@njit(inline="always", cache=True)
def wrap_point_w_shift(i, j, k, nx, ny, nz):

    si, sj, sk = (0, 0, 0)
    if i >= nx:
        i -= nx
        si = 1
    elif i < 0:
        i += nx
        si = -1
    if j >= ny:
        j -= ny
        sj = 1
    elif j < 0:
        j += ny
        sj = -1
    if k >= nz:
        k -= nz
        sk = 1
    elif k < 0:
        k += nz
        sk = -1
    return i, j, k, si, sj, sk


@njit(fastmath=True, cache=True, inline="always")
def flat_to_coords(idx, ny_nz, nz):
    i = idx // (ny_nz)
    j = (idx % (ny_nz)) // nz
    k = idx % nz
    return i, j, k


@njit(fastmath=True, cache=True, inline="always")
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
):
    # normalize values
    values /= values.sum()

    # We'll accumulate (unwrapped) coordinates into total
    total0 = 0.0
    total1 = 0.0
    total2 = 0.0

    # reference coord used for unwrapping
    ref0 = 0.0
    ref1 = 0.0
    ref2 = 0.0
    ref_set = False

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

    return np.array((total0 % 1.0, total1 % 1.0, total2 % 1.0), dtype=np.float64)
