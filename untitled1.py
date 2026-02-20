# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 11:22:49 2026

@author: sammw
"""
import numpy as np

from baderkit.core import Bader
from baderkit.core.critical_points.hessian_based import find_saddle_points
from baderkit.core.critical_points import CriticalPoints

bader = Bader.from_vasp(reference_filename="ELFCAR", maxima_persistence_tol=0.03)

# maxima_groups, minima_groups = bader.get_persistence_groups()

critical_points = CriticalPoints(bader)
maxima_groups = critical_points.maxima_persistence_groups
minima_groups = critical_points.minima_persistence_groups
maxima_types = critical_points.maxima_group_types
minima_types = critical_points.minima_group_types
# maxima_groups = bader.maxima_voxel_groups
# minima_groups = bader.minima_voxel_groups

grid = bader.reference_grid.copy()
saddle_mask = np.full(
    grid.shape,
    np.iinfo(np.uint8).max,
    dtype=np.uint8
)
for group in maxima_groups:
    saddle_mask[group[:,0],group[:,1],group[:,2]] = 3
for group in minima_groups:
    saddle_mask[group[:,0],group[:,1],group[:,2]] = 0
    # get saddle coords
saddle_mask = find_saddle_points(
    data=grid.total,
    matrix=grid.matrix,
    saddle_mask=saddle_mask,
    max_iter=30,
    eig_rel_tol=0.001,
    )

test_grid = grid.copy()
for i in range(4):
    test_grid.total = saddle_mask == i
    test_grid.write_vasp(f"ELFCAR_test_{i}")
    
print(np.unique(saddle_mask, return_counts=True))
