# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 11:22:49 2026

@author: sammw
"""
import numpy as np

from baderkit.core import Bader, Grid
from baderkit.core.critical_points.hessian_based import find_saddle_points
from baderkit.core.critical_points import CriticalPoints
from baderkit.plotting.core import CriticalPointsPlotter

grid = Grid.from_vasp("CHGCAR")
elf = Grid.from_vasp("ELFCAR")

grid.total += abs(np.max(grid.total))

bader = Bader(
    charge_grid=grid,
    reference_grid=elf,
    maxima_persistence_tol=0.03, 
    minima_persistence_tol=0.001,
    vacuum_tol=0.01,
    nna_cutoff=1
    )
print(len(bader.maxima_frac))
bader.write_species_volume(
    "X",
    write_grid="reference_grid",
    filename="ELFCAR")

# maxima_groups, minima_groups = bader.get_persistence_groups()

critical_points = CriticalPoints(bader)

# maxima, minima = bader.get_persistence_groups()
# test_grid = bader.reference_grid.copy()
# mask = np.zeros(test_grid.shape, dtype=bool)
# for coords in maxima:
#     mask[coords[:,0],coords[:,1],coords[:,2]] = True
# test_grid.total=mask
# test_grid.write_vasp("ELFCAR_test")
# TODO:
    # make sure ring labeling works.
    # fix ring to use limited subset
    # fix issues with bad critical point connections by adding increasing
    # range of transformations


# plotter = CriticalPointsPlotter(critical_points)

manifold_labels = critical_points.manifold_labels


saddle1_conns = critical_points.saddle1_minima_connections
saddle2_conns = critical_points.saddle2_maxima_connections

saddle1_coords = critical_points.saddle1_vox
saddle2_coords = critical_points.saddle2_vox

saddle1_labels = manifold_labels[saddle1_coords[:,0],saddle1_coords[:,1],saddle1_coords[:,2]]
saddle2_labels = manifold_labels[saddle2_coords[:,0],saddle2_coords[:,1],saddle2_coords[:,2]]

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
    
test_grid.total = np.isin(manifold_labels, (6,1,2))
test_grid.write_vasp("ELFCAR_test_6")
    
print(np.unique(saddle_mask, return_counts=True))
