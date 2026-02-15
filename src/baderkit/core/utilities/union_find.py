# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, prange


###############################################################################
# Root Finding Methods
###############################################################################
@njit(cache=True, inline="always")
def find_root(parent, x):
    """Find root with partial path compression"""
    while x != parent[x]:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@njit(cache=True, inline="always")
def find_root_no_compression(parent, x):
    """Find root with no path compression. Parallel friendly."""
    while x != parent[x]:
        x = parent[x]
    return x


@njit(cache=True, inline="always")
def find_root_with_shift(parent, offset, x):
    """Find root with partial compression and accumulate offset for periodic cycle counting"""
    # local aliasing to avoid repeated global lookups
    y = x

    # Path-halving loop: compress path by setting parent[y] = parent[parent[y]]
    # and updating offset[y] to remain consistent.
    # This reduces the path length quickly with fewer writes than full compression.
    while parent[y] != y and parent[parent[y]] != parent[y]:
        p = parent[y]
        # add p's offset into y so that y points to p's parent consistently
        offset[y] += offset[p]
        # set y to point to grandparent
        parent[y] = parent[p]
        # advance y (we short-circuited one level)
        y = parent[y]

    # Final climb to accumulate the cumulative offset; path is now short.
    total_offset = np.zeros(3, dtype=np.int8)
    y = x
    while parent[y] != y:
        total_offset += offset[y]
        y = parent[y]

    return y, total_offset




@njit(cache=True, inline="always")
def find_root_with_shift_no_compression(parent, offset, x):
    """Find root and offset with no compression/accumulation. Parallel friendly"""
    cx = 0
    cy = 0
    cz = 0
    while parent[x] != x:
        cx += offset[x,0]
        cy += offset[x,0]
        cz += offset[x,0]
        x = parent[x]

    return x, cx, cy, cz


###############################################################################
# Union Methods
###############################################################################


@njit(cache=True, inline="always")
def union_w_roots(parents, x, y, root_mask):
    """Create union between two points and update a root mask"""
    rx = find_root(parents, x)
    ry = find_root(parents, y)

    parents[rx] = ry

    if root_mask[rx]:
        root_mask[rx] = False
    if not root_mask[ry]:
        root_mask[ry] = True


@njit(cache=True, inline="always")
def union(parents, x, y):
    """Create union between two points"""
    rx = find_root(parents, x)
    ry = find_root(parents, y)

    parents[rx] = ry


@njit(cache=True, inline="always")
def bulk_union(parent, xs, ys):
    """Create union between many points"""
    for i in range(len(xs)):
        parent = union(parent, xs[i], ys[i])
    return parent


@njit(cache=True, inline="always")
def union_with_shift(
    parent, offset, a, b, shift
):
    """Create union between two points and accumulate shift needed to calculate cycles around periodic boundaries"""
    ra, total_offset = find_root_with_shift(parent, offset, a)
    rb, total_offset1 = find_root_with_shift(parent, offset, b)

    if ra == rb:
        # no need to combine
        return
    
    new_offset = shift + total_offset1 - total_offset

    parent[ra] = rb
    offset[ra] = new_offset


###############################################################################
# Compression Methods
###############################################################################
@njit(cache=True, parallel=True, inline="always")
def compress_roots(parents):
    """Fully compress all paths to roots in parallel"""
    new_parents = np.empty_like(parents, dtype=parents.dtype)
    for i in prange(len(parents)):
        current_val = parents[i]
        if current_val == -1:
            # this basin hasn't been added yet. Note it and continue
            new_parents[i] = -1
            continue
        new_parents[i] = find_root(parents, current_val)
    return new_parents
