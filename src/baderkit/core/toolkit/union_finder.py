# -*- coding: utf-8 -*-
import numpy as np
from numba import njit

from baderkit.core.utilities.union_find import (
    find_root,
    bulk_union,
    compress_roots,
    )

@njit(cache=True)
def _union(parent, x, y):
    # if parent isn't long enough for x or y, extend it
    higher = max(x, y)
    while len(parent) <= higher:
        parent.append(len(parent))
        
    rx = find_root(parent, x)
    ry = find_root(parent, y)
    parent[rx] = ry
    return parent

class UnionFind:
    """
    A basic union finding class. Assumes entries are integers spanning 0 to N.
    """
    def __init__(self):
        self.parent = [0]
    
    def find_root(self, x):
        return find_root(self.parent, x)

    def union(self, x, y):
        self.parent = _union(self.parent, x, y)

    def bulk_union(self, xs, ys):
        """Union multiple pairs at once (xs[i], ys[i])"""
        self.parent = bulk_union(parent=self.parent, xs=xs, ys=ys)

    def groups(self):
        """Return groups as list of arrays"""
        roots = compress_roots(self.parent)
        unique_roots = np.unique(roots)
        # TODO: Replace the following for loop if possible
        return [np.where(roots == r)[0] for r in unique_roots]
    
    def groups_sets(self):
        return {frozenset(s) for s in self.groups()}