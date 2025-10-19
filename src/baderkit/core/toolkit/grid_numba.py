# -*- coding: utf-8 -*-
"""
Numba-based 3D RegularGridInterpolator for periodic fractional coordinates.
Supports: nearest, linear, cubic, quintic
"""
import math

import numpy as np
from numpy.typing import NDArray
from numba import njit, prange


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

###############################################################################
# Sig Fig Management
###############################################################################


# @njit(cache=True)
# def round_sig(value, num_sig_figs):
#     if value == 0:
#         return 0.0
#     return round(value, int(num_sig_figs - np.floor(np.log10(abs(value))) - 1))


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

    # wrap coord
    i, j, k = wrap_point(i, j, k, nx, ny, nz)

    # get rounded down voxel coords
    ri = int(i // 1.0)
    rj = int(j // 1.0)
    rk = int(k // 1.0)

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
    weights = np.empty((3,4), dtype=np.float64)
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
    # elif method == "quintic":
    #     value = interp_spline(i, j, k, data, order=5)

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
    # elif method == "quintic":
    #     for i in prange(len(points)):
    #         i, j, k = points[i]
    #         out[i] = interp_spline(i, j, k, data, order=5)

    return out

@njit(cache=True)
def linear_slice(data, p1: NDArray[float], p2: NDArray[float], n: int = 100, is_frac = True, method="cubic"):

    x_pts = np.linspace(p1[0], p2[0], num=n)
    y_pts = np.linspace(p1[1], p2[1], num=n)
    z_pts = np.linspace(p1[2], p2[2], num=n)
    coords = np.column_stack((x_pts, y_pts, z_pts))
    
    return interpolate_points(coords, method, data, is_frac)

###############################################################################
# Wrapper class for interpolation
###############################################################################


class Interpolator:
    def __init__(self, data, method="cubic"):
        self.data = np.asarray(data)
        self.method = method

    def __call__(self, points):
        # get points as a numpy array
        points = np.asarray(points, dtype=np.float64)
        # if 1D, convert to 2D
        if points.ndim == 1:
            points = points[None, :]

        return interpolate_points(
            points,
            self.method,
            self.data,
        )


###############################################################################
# Methods for finding offgrid maxima
###############################################################################

# @njit(cache=True)
# def get_valid_transforms(
#     i,j,k,
#     data,
#     neighbor_transforms,
#     value
#         ):
#     nx, ny, nz = data.shape
#     # If a transform has the same value as our central point on either side,
#     # we will get ringing in our cubic interpolation. As a quick fix, we remove
#     # any transforms that have the same value across our central point. There
#     # is surely a more sophisticated solution, but this is the best I have for now
#     valid_transforms = np.ones(len(neighbor_transforms), dtype=np.bool_)
#     for idx, (si, sj, sk) in enumerate(neighbor_transforms):
#         ui, uj, uk = wrap_point(i+si, j+sj, k+sk, nx, ny, nz)
#         bi, bj, bk = wrap_point(i-si, j-sj, k-sk, nx, ny, nz)
#         above_val = interp_spline(ui, uj, uk, data=data, is_frac=False)
#         below_val = interp_spline(ui, uj, uk, data=data, is_frac=False)
#         if (
#             abs(above_val - value) < 1e-12
#             and abs(below_val - value) < 1e-12
#                 ):
#             valid_transforms[idx] = False
#     return valid_transforms

@njit(parallel=True, fastmath=True, cache=True)
def refine_maxima(
    maxima_coords,
    data,
    neighbor_transforms,
    tol=1e-8,
    is_frac=True,
):
    nx, ny, nz = data.shape
    # copy initial maxima to avoid overwriting them
    maxima_coords = maxima_coords.copy()
    # copy transforms to avoid altering in place
    neighbor_transforms = neighbor_transforms.copy().astype(np.float64)
    
    # normalize in each direction to one
    for transform_idx, transform in enumerate(neighbor_transforms):
        neighbor_transforms[transform_idx] = transform / np.linalg.norm(transform)

    # if fractional, convert each coordinate to voxel coords
    if is_frac:
        for max_coord in maxima_coords:
            max_coord[0] *= nx
            max_coord[1] *= ny
            max_coord[2] *= nz

    # get the initial values
    current_values = interpolate_points(
        data=data,
        points=maxima_coords,
        method="cubic", 
        is_frac=False,
    )
    # loop over coords in parallel and optimize positions
    for coord_idx in prange(len(maxima_coords)):
        i, j, k = maxima_coords[coord_idx]

        frac_mult = 1
        # create initial delta magnitude
        delta_mag = 1.0
        loop_count = 0
        while delta_mag > tol and loop_count < 50:
            loop_count += 1
            # increase frac multiplier
            frac_mult *= 2
            # get smaller transform than last loop
            current_trans = neighbor_transforms / frac_mult
            # get current best position
            i, j, k = maxima_coords[coord_idx]
            # loop over transforms and check if they improve our value
            for si, sj, sk in current_trans:
                ti = i + si
                tj = j + sj
                tk = k + sk
                value = interp_spline(
                    ti, 
                    tj, 
                    tk, 
                    data=data,
                    is_frac=False,
                    )
                # if value is improved, update the best position/value
                if value > current_values[coord_idx]:
                    current_values[coord_idx] = value
                    maxima_coords[coord_idx] = (ti, tj, tk)
                    # calculate magnitude of delta in fractional coordinates
                    fsi = si / nx
                    fsj = sj / ny
                    fsk = sk / nz
                    delta_mag = (fsi * fsi + fsj * fsj + fsk * fsk) ** 0.5
    dec = -int(math.log10(tol))
    if is_frac:
        # convert to frac, round, and wrap
        for max_idx, (i, j, k) in enumerate(maxima_coords):
            i = round(i / nx, dec) % 1.0
            j = round(j / ny, dec) % 1.0
            k = round(k / nz, dec) % 1.0
            maxima_coords[max_idx] = (i, j, k)
    else:
        # round and wrap
        for max_idx, (i, j, k) in enumerate(maxima_coords):
            i = round(i, dec) % nx
            j = round(j, dec) % ny
            k = round(k, dec) % nz
            maxima_coords[max_idx] = (i, j, k)

    return maxima_coords, current_values


# @njit(inline='always', fastmath=True, cache=True)
# def get_gradient_and_hessian(i, j, k, data, d, is_frac=False):
#     nx, ny, nz = data.shape

#     # if coord is fractional, convert to voxel coords
#     if is_frac:
#         i = i*nx
#         j = j*ny
#         k = k*nz

#     # get squared shift to avoid repeat calcs
#     d2 = d*d
#     dx2 = 2*d
#     d2x4 = 4*d2

#     # Get values at shifts using cubic interpolation
#     v000 = interp_spline(i, j, k, data, False)
#     v100 = interp_spline(i+d, j, k, data, False)
#     v_100 = interp_spline(i-d, j, k, data, False)
#     v010 = interp_spline(i, j+d, k, data, False)
#     v0_10 = interp_spline(i, j-d, k, data, False)
#     v001 = interp_spline(i, j, k+d, data, False)
#     v00_1 = interp_spline(i, j, k-d, data, False)
#     v110 = interp_spline(i+d, j+d, k, data, False)
#     v_110 = interp_spline(i-d, j+d, k, data, False)
#     v1_10 = interp_spline(i+d, j-d, k, data, False)
#     v_1_10 = interp_spline(i-d, j-d, k, data, False)
#     v101 = interp_spline(i+d, j, k+d, data, False)
#     v_101 = interp_spline(i-d, j, k+d, data, False)
#     v10_1 = interp_spline(i+d, j, k-d, data, False)
#     v_10_1 = interp_spline(i-d, j, k-d, data, False)
#     v011 = interp_spline(i, j+d, k+d, data, False)
#     v0_11 = interp_spline(i, j-d, k+d, data, False)
#     v01_1 = interp_spline(i, j+d, k-d, data, False)
#     v0_1_1 = interp_spline(i, j-d, k-d, data, False)

#     # get gradient
#     fx = (v100 - v_100) / dx2
#     fy = (v010 - v0_10) / dx2
#     fz = (v001 - v00_1) / dx2
#     g = np.array((fx, fy, fz), dtype=np.float64)

#     # get hessian
#     v000x2 = v000 * 2
#     fxx = (v100 - v000x2 + v_100) / d2
#     fyy = (v010 - v000x2 + v0_10) / d2
#     fzz = (v001 - v000x2 + v00_1) / d2

#     fxy = (v110 - v1_10 - v_110 + v_1_10) / d2x4
#     fxz = (v101 - v10_1 - v_101 + v_10_1) / d2x4
#     fyz = (v011 - v01_1 - v0_11 + v0_1_1) / d2x4

#     H = np.array(
#         (
#         (fxx, fxy, fxz),
#         (fxy, fyy, fyz),
#         (fxz, fyz, fzz)
#         ), dtype=np.float64)

#     return g, H

# @njit(fastmath=True, cache=True, inline='always')
# def refine_maximum_newton(i, j, k, data, d=0.25, tol=1e-12, maxiter=50, is_frac=True):
#     nx, ny, nz = data.shape

#     # if coord is fractional, convert to voxel coords
#     if is_frac:
#         i = i*nx
#         j = j*ny
#         k = k*nz

#     # loop until convergence
#     for iter_num in range(maxiter):
#         # get gradient and hessian
#         g, H = get_gradient_and_hessian(i, j, k, data, d, is_frac=False)

#         # solve for delta
#         di, dj, dk = np.linalg.solve(H, -g)

#         # get new point
#         i = i+di
#         j = j+dj
#         k = k+dk

#         # check if magnitude of delta is below tolerance (in fractional coords)
#         fdi = di * nx
#         fdj = dj * ny
#         fdk = dk * nz
#         dmag = (fdi*fdi + fdj*fdj + fdk*fdk) ** 0.5
#         if dmag < tol:
#             break

#     # get value at final point
#     value = interp_spline(i, j, k, data, False)

#     if is_frac:
#         # convert back to fractional, round, and wrap
#         i = round(i/nx, 12) % 1.0
#         j = round(j/ny, 12) % 1.0
#         k = round(k/nz, 12) % 1.0
#     else:
#         # round and wrap final coord
#         i = round(i, 12) % nx
#         j = round(j, 12) % ny
#         k = round(k, 12) % nz

#     return np.array((i,j,k), dtype=np.float64), value

# @njit(parallel=True, cache=True)
# def refine_maxima_newton(maxima_coords, data, d=0.25, tol=1e-12, maxiter=50, is_frac=True):
#     # create array to store new coords/values
#     new_coords = np.empty_like(maxima_coords, dtype=np.float64)
#     new_values = np.empty(len(new_coords), dtype=np.float64)

#     for coord_idx in prange(len(maxima_coords)):
#         i, j, k = maxima_coords[coord_idx]
#         new_coord, new_value = refine_maximum_newton(
#             i,j,k,
#             data,
#             d=d,
#             tol=tol,
#             maxiter=maxiter,
#             is_frac=is_frac
#             )
#         new_coords[coord_idx] = new_coord
#         new_values[coord_idx] = new_value
#     return new_coords, new_values
