# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 11:22:49 2026

@author: sammw
"""
import numpy as np

from baderkit.core import Bader, Grid
from baderkit.core.critical_points import CriticalPoints
from baderkit.core.critical_points.hessian_based import find_saddle_points
from baderkit.plotting.core import CriticalPointsPlotter

bader = Bader.from_vasp(
    reference_filename="ELFCAR",
    # method="ongrid",
    maxima_persistence_tol=0.01,
    minima_persistence_tol=0.005,
)
print(len(bader.maxima_frac))
# bader.write_all_basin_volumes(write_grid="reference_grid")

critical_points = CriticalPoints(bader)
plotter = CriticalPointsPlotter(critical_points)

# TODO:
# determine best persistence score method
# update gui
# update docs

# New persistence scheme
# 1. Get basins
# 2. Get possible saddle points for each neighbor pair
#       a. Get exact ongrid connection value
#       b. Find other voxels that are likely to contain the same value
#          somewhere inside their bounds
# 3. Refine possible saddles and reject any that don't succeed
# 4. Perform gradient ascent to find connected maxima

# 5. If no saddle between maxima that share an edge, combine?
# 6. calculate persistence scores and combine?
