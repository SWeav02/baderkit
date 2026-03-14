# -*- coding: utf-8 -*-
import math

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from baderkit.core.utilities.basic import wrap_point

###############################################################################
# Nearest point interpolation
###############################################################################


@njit(inline="always", cache=True, fastmath=True)
def interp_nearest(i, j, k, data, is_frac=True):
    nx, ny, nz = data.shape
    if is_frac:
        # convert to voxel coordinates
        i = i * nx
        j = j * ny
        k = k * nz

    # round and wrap
    ix = int(round(i)) % nx
    iy = int(round(j)) % ny
    iz = int(round(k)) % nz

    return data[ix, iy, iz]


###############################################################################
# Linear interpolation
###############################################################################
@njit(inline="always", cache=True, fastmath=True)
def interp_linear(i, j, k, data, is_frac=True):
    nx, ny, nz = data.shape

    if is_frac:
        # convert to voxel coordinates
        i = i * nx
        j = j * ny
        k = k * nz

    # get rounded down voxel coords
    ri = int(i // 1.0)
    rj = int(j // 1.0)
    rk = int(k // 1.0)

    # wrap coord
    ri, rj, rk = wrap_point(ri, rj, rk, nx, ny, nz)

    # get offset from rounded voxel coord
    di = i - ri
    dj = j - rj
    dk = k - rk

    # get data in 2x2x2 cube surrounding point
    v000 = data[ri, rj, rk]
    v100 = data[(ri + 1) % nx, rj, rk]
    v010 = data[ri, (rj + 1) % ny, rk]
    v001 = data[ri, rj, (rk + 1) % nz]
    v110 = data[(ri + 1) % nx, (rj + 1) % ny, rk]
    v101 = data[(ri + 1) % nx, rj, (rk + 1) % nz]
    v011 = data[ri, (rj + 1) % ny, (rk + 1) % nz]
    v111 = data[(ri + 1) % nx, (rj + 1) % ny, (rk + 1) % nz]

    # interpolate value from linear approximation
    return (
        (1 - di) * (1 - dj) * (1 - dk) * v000
        + di * (1 - dj) * (1 - dk) * v100
        + (1 - di) * dj * (1 - dk) * v010
        + (1 - di) * (1 - dj) * dk * v001
        + di * dj * (1 - dk) * v110
        + di * (1 - dj) * dk * v101
        + (1 - di) * dj * dk * v011
        + di * dj * dk * v111
    )


###############################################################################
# Cubic spline interpolation
###############################################################################


@njit(cache=True, inline="always", fastmath=True)
def cubic_bspline_weights(di, dj, dk):
    weights = np.empty((3, 4), dtype=np.float64)
    for d_idx, d in enumerate((di, dj, dk)):
        for i in range(4):
            x = abs((i - 1) - d)
            if x < 1.0:
                w = (4.0 - 6.0 * x * x + 3.0 * x * x * x) / 6.0
            elif x < 2.0:
                t = 2.0 - x
                w = (t * t * t) / 6.0
            else:
                w = 0.0
            weights[d_idx, i] = w
    return weights


@njit(cache=True, fastmath=True)
def interp_spline(i, j, k, data, is_frac=True):
    nx, ny, nz = data.shape

    # convert fractional to voxel coordinates
    if is_frac:
        i = i * nx
        j = j * ny
        k = k * nz

    # round down to get int value
    ri = int(math.floor(i))  # floor works with negative too
    rj = int(math.floor(j))
    rk = int(math.floor(k))

    # get fractional offsets in [0,1)
    di = i - ri
    dj = j - rj
    dk = k - rk

    # calculate weights
    weights = cubic_bspline_weights(di, dj, dk)

    # separable evaluation:
    # first convolve along x for the 4x4 neighborhood in y,z to produce tmp[4,4]
    tmp = np.zeros((4, 4), dtype=np.float64)  # tmp[j_index, k_index]
    for joff in range(4):
        yj = (rj - 1 + joff) % ny
        for koff in range(4):
            zk = (rk - 1 + koff) % nz
            # convolve along x for this (y,z)
            s = 0.0
            for ioff in range(4):
                xi = (ri - 1 + ioff) % nx
                s += weights[0, ioff] * data[xi, yj, zk]
            tmp[joff, koff] = s

    # now convolve tmp by wy along y and wz along z
    val = 0.0
    for joff in range(4):
        for koff in range(4):
            val += weights[1, joff] * weights[2, koff] * tmp[joff, koff]

    return val


###############################################################################
# Methods to interpolate points depending on requested method
###############################################################################


@njit(cache=True)
def interpolate_point(
    point,
    method,
    data,
    is_frac=True,
):
    i, j, k = point
    if method == "nearest":
        value = interp_nearest(i, j, k, data, is_frac)
    elif method == "linear":
        value = interp_linear(i, j, k, data, is_frac)
    elif method == "cubic":
        value = interp_spline(i, j, k, data, is_frac)

    return value


@njit(parallel=True, cache=True)
def interpolate_points(points, method, data, is_frac=True):
    out = np.empty(len(points))
    if method == "nearest":
        for point_idx in prange(len(points)):
            i, j, k = points[point_idx]
            out[point_idx] = interp_nearest(i, j, k, data, is_frac)
    elif method == "linear":
        for point_idx in prange(len(points)):
            i, j, k = points[point_idx]
            out[point_idx] = interp_linear(i, j, k, data, is_frac)
    elif method == "cubic":
        for point_idx in prange(len(points)):
            i, j, k = points[point_idx]
            out[point_idx] = interp_spline(i, j, k, data, is_frac)

    return out


@njit(cache=True)
def linear_slice(
    data,
    p1: NDArray[float],
    p2: NDArray[float],
    n: int = 100,
    is_frac=True,
    method="cubic",
):

    x_pts = np.linspace(p1[0], p2[0], num=n)
    y_pts = np.linspace(p1[1], p2[1], num=n)
    z_pts = np.linspace(p1[2], p2[2], num=n)
    coords = np.column_stack((x_pts, y_pts, z_pts))

    return interpolate_points(coords, method, data, is_frac)

###############################################################################
# Gradient and Hessian
###############################################################################

@njit(fastmath=True)
def spline_grad(i, j, k, data, h=0.25, is_frac=False):
    """
    Gradient of spline-interpolated scalar field
    with respect to grid coordinates (i, j, k).
    """
    nx, ny, nz = data.shape

    # convert fractional to voxel coordinates
    if is_frac:
        i = i * nx
        j = j * ny
        k = k * nz

    i = float(i)
    j = float(j)
    k = float(k)

    fxp = interp_spline(i + h, j, k, data, False)
    fxm = interp_spline(i - h, j, k, data, False)

    fyp = interp_spline(i, j + h, k, data, False)
    fym = interp_spline(i, j - h, k, data, False)

    fzp = interp_spline(i, j, k + h, data, False)
    fzm = interp_spline(i, j, k - h, data, False)

    gx = (fxp - fxm) / (2.0 * h)
    gy = (fyp - fym) / (2.0 * h)
    gz = (fzp - fzm) / (2.0 * h)

    return gx, gy, gz


@njit(fastmath=True)
def spline_hess(i, j, k, data, h=0.25, is_frac=False):
    """
    Hessian of spline-interpolated scalar field
    with respect to grid coordinates (i, j, k).
    """
    nx, ny, nz = data.shape
    if is_frac:
        i = i * nx
        j = j * ny
        k = k * nz

    i = float(i)
    j = float(j)
    k = float(k)

    hh = h * h
    hh4 = 4.0 * hh

    f0 = interp_spline(i, j, k, data, False)
    f02 = f0 * 2

    # Second derivatives
    f_xx = (
        interp_spline(i + h, j, k, data, False)
        - f02
        + interp_spline(i - h, j, k, data, False)
    ) / hh

    f_yy = (
        interp_spline(i, j + h, k, data, False)
        - f02
        + interp_spline(i, j - h, k, data, False)
    ) / hh

    f_zz = (
        interp_spline(i, j, k + h, data, False)
        - f02
        + interp_spline(i, j, k - h, data, False)
    ) / hh

    # Mixed partials
    f_xy = (
        interp_spline(i + h, j + h, k, data, False)
        - interp_spline(i + h, j - h, k, data, False)
        - interp_spline(i - h, j + h, k, data, False)
        + interp_spline(i - h, j - h, k, data, False)
    ) / hh4

    f_xz = (
        interp_spline(i + h, j, k + h, data, False)
        - interp_spline(i + h, j, k - h, data, False)
        - interp_spline(i - h, j, k + h, data, False)
        + interp_spline(i - h, j, k - h, data, False)
    ) / hh4

    f_yz = (
        interp_spline(i, j + h, k + h, data, False)
        - interp_spline(i, j + h, k - h, data, False)
        - interp_spline(i, j - h, k + h, data, False)
        + interp_spline(i, j - h, k - h, data, False)
    ) / hh4

    H = np.empty((3, 3))
    H[0, 0] = f_xx
    H[1, 1] = f_yy
    H[2, 2] = f_zz

    H[0, 1] = H[1, 0] = f_xy
    H[0, 2] = H[2, 0] = f_xz
    H[1, 2] = H[2, 1] = f_yz

    return H


@njit(fastmath=True)
def spline_grad_and_hess(coord, data, h=0.25):
    """
    Compute both gradient and Hessian of a spline-interpolated scalar field
    with respect to grid coordinates (i, j, k), minimizing redundant interpolations.
    """
    nx, ny, nz = data.shape

    i, j, k = coord

    h2 = 2.0 * h
    hh = h * h
    hh4 = 4.0 * hh

    # Central point
    f0 = interp_spline(i, j, k, data, False)
    f02 = f0 * 2

    # Neighbor points for gradient and Hessian
    fxp = interp_spline(i + h, j, k, data, False)
    fxm = interp_spline(i - h, j, k, data, False)
    fyp = interp_spline(i, j + h, k, data, False)
    fym = interp_spline(i, j - h, k, data, False)
    fzp = interp_spline(i, j, k + h, data, False)
    fzm = interp_spline(i, j, k - h, data, False)

    # Gradient
    gx = (fxp - fxm) / h2
    gy = (fyp - fym) / h2
    gz = (fzp - fzm) / h2

    # Second derivatives (Hessian diagonal)
    f_xx = (fxp - f02 + fxm) / hh
    f_yy = (fyp - f02 + fym) / hh
    f_zz = (fzp - f02 + fzm) / hh

    # Mixed partials
    f_xy = (
        interp_spline(i + h, j + h, k, data, False)
        - interp_spline(i + h, j - h, k, data, False)
        - interp_spline(i - h, j + h, k, data, False)
        + interp_spline(i - h, j - h, k, data, False)
    ) / hh4

    f_xz = (
        interp_spline(i + h, j, k + h, data, False)
        - interp_spline(i + h, j, k - h, data, False)
        - interp_spline(i - h, j, k + h, data, False)
        + interp_spline(i - h, j, k - h, data, False)
    ) / hh4

    f_yz = (
        interp_spline(i, j + h, k + h, data, False)
        - interp_spline(i, j + h, k - h, data, False)
        - interp_spline(i, j - h, k + h, data, False)
        + interp_spline(i, j - h, k - h, data, False)
    ) / hh4

    H = np.empty((3, 3))
    H[0, 0] = f_xx
    H[1, 1] = f_yy
    H[2, 2] = f_zz
    H[0, 1] = H[1, 0] = f_xy
    H[0, 2] = H[2, 0] = f_xz
    H[1, 2] = H[2, 1] = f_yz

    return np.array((gx, gy, gz), dtype=np.float64), H


@njit(fastmath=True)
def spline_grad_cart(i, j, k, data, dir2car, h=0.25, is_frac=False):
    nx, ny, nz = data.shape

    # get gradient in voxel coords
    gi, gj, gk = spline_grad(i, j, k, data, h, is_frac)

    # convert to fractional-coordinate gradient
    gi *= nx
    gj *= ny
    gk *= nz

    # convert to Cartesian gradient
    gx = dir2car[0, 0] * gi + dir2car[0, 1] * gj + dir2car[0, 2] * gk
    gy = dir2car[1, 0] * gi + dir2car[1, 1] * gj + dir2car[1, 2] * gk
    gz = dir2car[2, 0] * gi + dir2car[2, 1] * gj + dir2car[2, 2] * gk

    return gx, gy, gz


@njit(fastmath=True)
def spline_hess_cart(i, j, k, data, dir2car, h=0.25, is_frac=False):

    # get hessian in grid coords
    H = spline_hess(i, j, k, data, h, is_frac)

    # convert to cartesian and return
    return dir2car @ H @ dir2car.T