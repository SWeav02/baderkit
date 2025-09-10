# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 21:33:53 2025

@author: sammw
"""

# Strategy:
    # 1. Get transforms 0.5 in each direction
    # 2. Interpolate all values at once
    # 3. Move each point to best value
    # 4. Divide transforms by 2 and repeat
    
from baderkit.core import Grid, Bader
import numpy as np
import time

# test_fracs = np.array([[0.675, 0.325, 0.75 ], [0.325, 0.675, 0.25 ]])

grid = Grid.from_vasp("CHGCAR", interpolation_method="quintic")
bader = Bader(grid)
test_fracs = bader.basin_maxima_frac

t0 = time.time()
transforms, _ = grid.point_neighbor_transforms
# normalize
transforms = (transforms.T / np.linalg.norm(transforms, axis=1)).T
# convert to frac
frac_trans = grid.grid_to_frac(transforms)
frac_mult = 1
current_values = grid.values_at(test_fracs)
current_coords = test_fracs.copy()
for i in range(15):
    # increase frac multiplier
    frac_mult *= 2
    current_trans = frac_trans / frac_mult
    # Add transforms to each frac coord
    # Broadcasting: (n, 1, 3) + (1, m, 3) -> (n, m, 3)
    sums = current_coords[:, None, :] + current_trans[None, :, :]
    
    # Flatten to (n*m, 3)
    transformed_coords = sums.reshape(-1, 3)

    # interpolate values
    values = grid.values_at(transformed_coords)
    
    # rearrange
    values = values.reshape((len(test_fracs), 26))
    # subtract current best values
    reduced_values = values.T-current_values
    
    # locate points that have found a higher neighbor
    has_higher = np.any(reduced_values > 0, axis=0)
    
    # for points that have found a higher neighbor, get the best higher neighbor
    max_vals = np.max(values[has_higher], axis=1)
    
    max_idx = 0
    for idx, coord in enumerate(current_coords):
        if not has_higher[idx]:
            continue
        # get highest index
        try:
            highest_idx = np.where(values[idx]==max_vals[max_idx])[0][0]
        except:
            breakpoint()
        # get corresponding highest coords
        new_coord = transformed_coords[26*idx + highest_idx]
        # update coord/vals at this point
        current_coords[idx] = new_coord
        current_values[idx] = max_vals[max_idx]
        max_idx += 1

t1 = time.time()
print(current_coords)
print(t1-t0)
    
    