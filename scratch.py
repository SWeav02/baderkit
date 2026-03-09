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

bader = Bader.from_vasp(
    reference_filename="CHGCAR",
    # method="ongrid",
    maxima_persistence_tol=0.03, 
    minima_persistence_tol=0.0005,
    )
print(len(bader.maxima_frac))
# bader.write_all_basin_volumes(write_grid="reference_grid")

from baderkit.core import Badelf

badelf = Badelf.from_vasp()
badelf.charges
