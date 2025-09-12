# -*- coding: utf-8 -*-
"""
Numba-based 3D RegularGridInterpolator for periodic fractional coordinates.
Supports: nearest, linear, cubic, quintic
"""
import math

import numpy as np
from numba import njit, prange

from baderkit.core.methods.shared_numba import wrap_point

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
# Spline interpolation
###############################################################################
@njit(inline="always", cache=True, fastmath=True)
def cubic_hermite_weights(t):
    """Return 4 cubic Hermite (Catmull-Rom) weights for fractional part t."""
    # get t mults to avoid repeat calcs.
    t2 = t * t
    t3 = t2 * t

    w_m1 = -0.5 * t3 + t2 - 0.5 * t
    w_0 = 1.5 * t3 - 2.5 * t2 + 1.0
    w_p1 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
    w_p2 = 0.5 * t3 - 0.5 * t2
    return np.array([w_m1, w_0, w_p1, w_p2])


# @njit(inline='always')
# def quintic_bspline_weights(t):
#     """
#     Return 6 quintic B-spline weights for the stencil offsets [-2,-1,0,1,2,3].
#     Valid for t in [0,1). Coefficients are the degree-5 polynomial pieces.
#     """
#     t2 = t * t
#     t3 = t2 * t
#     t4 = t3 * t
#     t5 = t4 * t

#     # coefficients (highest-order first) collapsed into Horner-like evaluation
#     # w0 -> offset -2, w1 -> offset -1, ..., w5 -> offset +3
#     w0 = (-1.0/120.0)*t5 + (1.0/24.0)*t4 + (-1.0/12.0)*t3 + (1.0/12.0)*t2 + (-1.0/24.0)*t + (1.0/120.0)
#     w1 = ( 1.0/24.0)*t5 + (-1.0/6.0)*t4 + (1.0/6.0)*t3 + (1.0/6.0)*t2 + (-5.0/12.0)*t + (13.0/60.0)
#     w2 = (-1.0/12.0)*t5 + (1.0/4.0)*t4 + 0.0*t3 + (-1.0/2.0)*t2 + 0.0*t + (11.0/20.0)
#     w3 = ( 1.0/12.0)*t5 + (-1.0/6.0)*t4 + (-1.0/6.0)*t3 + (1.0/6.0)*t2 + (5.0/12.0)*t + (13.0/60.0)
#     w4 = (-1.0/24.0)*t5 + (1.0/24.0)*t4 + (1.0/12.0)*t3 + (1.0/12.0)*t2 + (1.0/24.0)*t + (1.0/120.0)
#     w5 = ( 1.0/120.0)*t5  # remaining lower-order coefficients are zero

#     return np.array([w0, w1, w2, w3, w4, w5])


@njit(cache=True, fastmath=True, inline="always")
def interp_spline(i, j, k, data, is_frac=True):
    """
    3D Hermite cubic interpolation with periodic boundary conditions.
    """
    nx, ny, nz = data.shape

    if is_frac:
        # convert to voxel coordinates
        i = i * nx
        j = j * ny
        k = k * nz

    # No need to wrap as we do so later

    # get rounded down voxel coords
    ri = int(i // 1.0)
    rj = int(j // 1.0)
    rk = int(k // 1.0)

    # get offset from rounded voxel coord
    di = i - ri
    dj = j - rj
    dk = k - rk

    # get cubic weights
    wx = cubic_hermite_weights(di)
    wy = cubic_hermite_weights(dj)
    wz = cubic_hermite_weights(dk)
    offset = 1
    size = 4

    # if order == 3:
    #     wx = cubic_hermite_weights(dx)
    #     wy = cubic_hermite_weights(dy)
    #     wz = cubic_hermite_weights(dz)
    #     offset = 1
    #     size = 4
    # elif order == 5:
    #     wx = quintic_bspline_weights(dx)
    #     wy = quintic_bspline_weights(dy)
    #     wz = quintic_bspline_weights(dz)
    #     offset = 2
    #     size = 6
    # else:
    #     raise ValueError("Order must be 3 (cubic) or 5 (quintic)")

    val = 0.0
    for i in range(size):
        xi = (ri - offset + i) % nx
        for j in range(size):
            yj = (rj - offset + j) % ny
            for k in range(size):
                zk = (rk - offset + k) % nz
                val += wx[i] * wy[j] * wz[k] * data[xi, yj, zk]
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


# @njit(parallel=True, fastmath=True, cache=True)
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
    current_values = interpolate_points(maxima_coords, "cubic", data, False)
    # loop over coords in parallel and optimize positions
    for coord_idx in prange(len(maxima_coords)):
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
                value = interp_spline(ti, tj, tk, data, False)
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
