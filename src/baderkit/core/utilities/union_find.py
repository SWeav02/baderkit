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
def find_root_with_shift(parent, offset_x, offset_y, offset_z, x):
    """Find root with partial compression and accumulate offset for periodic cycle counting"""
    # local aliasing to avoid repeated global lookups
    y = x

    # Path-halving loop: compress path by setting parent[y] = parent[parent[y]]
    # and updating offset[y] to remain consistent.
    # This reduces the path length quickly with fewer writes than full compression.
    while parent[y] != y and parent[parent[y]] != parent[y]:
        p = parent[y]
        # add p's offset into y so that y points to p's parent consistently
        offset_x[y] += offset_x[p]
        offset_y[y] += offset_y[p]
        offset_z[y] += offset_z[p]
        # set y to point to grandparent
        parent[y] = parent[p]
        # advance y (we short-circuited one level)
        y = parent[y]

    # Final climb to accumulate the cumulative offset; path is now short.
    cx = 0
    cy = 0
    cz = 0
    y = x
    while parent[y] != y:
        cx += offset_x[y]
        cy += offset_y[y]
        cz += offset_z[y]
        y = parent[y]

    return y, cx, cy, cz


@njit(cache=True, inline="always")
def find_root_with_shift_no_compression(parent, offset_x, offset_y, offset_z, x):
    """Find root and offset with no compression/accumulation. Parallel friendly"""
    cx = 0
    cy = 0
    cz = 0
    while parent[x] != x:
        cx += offset_x[x]
        cy += offset_y[x]
        cz += offset_z[x]
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
    root_mask, parent, offset_x, offset_y, offset_z, size, a, b, si, sj, sk
):
    """Create union between two points and accumulate shift needed to calculate cycles around periodic boundaries"""
    ra, ox, oy, oz = find_root_with_shift(parent, offset_x, offset_y, offset_z, a)
    rb, ox1, oy1, oz1 = find_root_with_shift(parent, offset_x, offset_y, offset_z, b)

    if ra == rb:
        # no need to combine
        return

    cx = si + ox1 - ox
    cy = sj + oy1 - oy
    cz = sk + oz1 - oz

    # union-by-size: attach smaller under larger
    if size[ra] < size[rb]:
        # attach ra under rb. We must compute offset for ra => rb.
        # We currently have cx = pos(rb) - pos(ra) (by derivation above),
        # if we attach ra -> rb then off[ra] = cx (so pos(rb)=pos(ra)+off[ra])
        parent[ra] = rb
        offset_x[ra] = cx
        offset_y[ra] = cy
        offset_z[ra] = cz
        size[rb] += size[ra]
        if not root_mask[rb]:
            root_mask[rb] = True
        if root_mask[ra]:
            root_mask[ra] = False

    else:
        # attach rb under ra. Then we need off[rb] such that pos(ra) = pos(rb) + off[rb].
        # Since cx = pos(rb) - pos(ra), off[rb] must be -cx.
        parent[rb] = ra
        offset_x[rb] = -cx
        offset_y[rb] = -cy
        offset_z[rb] = -cz
        size[ra] += size[rb]
        if not root_mask[ra]:
            root_mask[ra] = True
        if root_mask[rb]:
            root_mask[rb] = False


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
