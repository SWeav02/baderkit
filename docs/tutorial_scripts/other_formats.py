# -*- coding: utf-8 -*-

from baderkit import Bader, Grid, Structure
import numpy as np

structure = Structure.from_file(filename = "mystructure.cif", fmt = "cif")

charge_data = np.array([
[[1,2,3],[3,4,5],[6,7,8]],
[[1,2,3],[3,4,5],[6,7,8]],
[[1,2,3],[3,4,5],[6,7,8]],
])

data = {"total": charge_data}
charge_grid = Grid(structure=structure, data=data)

bader = Bader(charge_grid=charge_grid)