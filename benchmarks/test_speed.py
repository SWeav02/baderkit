# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 13:53:47 2025

@author: sammw
"""

import time

from baderkit.core import Bader, Grid

grid = Grid.from_vasp("CHGCAR")

test_num = 100
results = {}
times = {}
for method in Bader.methods():
    t0 = time.time()
    for test_num in range(test_num):
        bader = Bader(
            charge_grid=grid,
            reference_grid=grid,
            method=method,
            refinement_method="recursive",
        )
        result = bader.results_summary
    t1 = time.time()
    results[method] = result
    times[method] = (t1 - t0) / test_num

for key, value in times.items():
    print(f"{key} average time: {value}")

for key, value in results.items():
    print(f"{key} first oxidation state: {value['atom_charges'][0]}")
