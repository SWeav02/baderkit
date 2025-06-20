#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from baderkit.core import Bader, Grid

methods = Bader.methods()
grid = Grid.from_vasp("CHGCAR")

test_num = 100

times = []
for method in methods:
    t0 = time.time()
    for i in range(test_num):
        bader = Bader(grid, grid, method=method)
        results = bader.results_summary
    t1 = time.time()
    times.append(f"{method}: {t1-t0}")
    