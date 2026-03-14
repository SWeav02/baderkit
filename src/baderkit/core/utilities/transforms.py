# -*- coding: utf-8 -*-
import itertools

import numpy as np
from numba import njit

from baderkit.core.utilities.basic import get_norm


IMAGE_TO_INT = np.empty([3, 3, 3], dtype=np.int64)
INT_TO_IMAGE = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
INT_TO_REV_INT = np.empty(len(INT_TO_IMAGE), dtype=np.int64)
for shift_idx, (i, j, k) in enumerate(INT_TO_IMAGE):
    IMAGE_TO_INT[i, j, k] = shift_idx
for shift_idx, (i, j, k) in enumerate(INT_TO_IMAGE):
    rev_idx = IMAGE_TO_INT[-i, -j, -k]
    INT_TO_REV_INT[shift_idx] = rev_idx

FACE_TRANSFORMS = np.array(
    [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ],
    dtype=np.int64,
)

EDGE_TRANSFORMS = np.array(
    [
        [1, 1, 0],
        [1, -1, 0],
        [-1, 1, 0],
        [-1, -1, 0],

        [1, 0, 1],
        [1, 0, -1],
        [-1, 0, 1],
        [-1, 0, -1],

        [0, 1, 1],
        [0, 1, -1],
        [0, -1, 1],
        [0, -1, -1],
    ],
    dtype=np.int64,
)

CORNER_TRANSFORMS = np.array(
    [
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [1, -1, -1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, 1],
        [-1, -1, -1],
    ],
    dtype=np.int64,
)

ALL_NEIGHBOR_TRANSFORMS = np.array(
    [
        # faces
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],

        # edges
        [1, 1, 0],
        [1, -1, 0],
        [-1, 1, 0],
        [-1, -1, 0],

        [1, 0, 1],
        [1, 0, -1],
        [-1, 0, 1],
        [-1, 0, -1],

        [0, 1, 1],
        [0, 1, -1],
        [0, -1, 1],
        [0, -1, -1],

        # corners
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [1, -1, -1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, 1],
        [-1, -1, -1],
    ],
    dtype=np.int64,
)

@njit(cache=True, inline="always")
def get_transform_dists(
    transforms,
    shape,
    matrix,
        ):
    # convert transforms to frac coords
    transforms = transforms / shape
    # convert to cart coords
    transforms = transforms @ matrix
    dists = np.empty(len(transforms), dtype=np.float64)
    for idx, (i,j,k) in enumerate(transforms):
        dists[idx] = get_norm(i,j,k)
    return dists