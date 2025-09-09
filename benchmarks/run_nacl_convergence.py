# -*- coding: utf-8 -*-

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baderkit.core import Bader, Grid

###############################################################################
# Running Bader
###############################################################################

methods = Bader.all_methods()

directory = Path(".")
grid_nums = [60, 120, 180, 240, 300]
charges = {method: [] for method in methods}


for grid_num in grid_nums:
    folder = directory / str(grid_num)
    assert folder.exists()

    # load grids to avoid repeat slow file read times
    charge_grid = Grid.from_vasp(folder / "CHGCAR")
    reference_grid = Grid.from_vasp(folder / "CHGCAR_sum")

    # run each method and save Na oxidation state
    for method in methods:
        bader = Bader(charge_grid, reference_grid, method=method)
        charges[method].append(9 - bader.atom_charges[0])

###############################################################################
# DataFrames
###############################################################################
oxidation_df = pd.DataFrame(
    {
        "one_axis_grid_points": grid_nums,
        "ngrid_points": np.array(grid_nums) ** 3,
        **charges,
    }
)

oxidation_df.to_csv("oxidation_summary_baderkit.csv", index=False)

###############################################################################
# Plotting
###############################################################################
plt.style.use("seaborn-v0_8-darkgrid")

# --- Plot charges vs. grid points ---
fig, ax = plt.subplots()
for method in methods:
    ax.plot(
        oxidation_df["ngrid_points"] / 1000000,
        oxidation_df[method],
        marker="o",
        label=method,
    )
ax.set_xlabel("Grid points (millions)")
ax.set_ylabel("Na Oxidation State")
ax.set_title("BaderKit Na Charge vs Grid Density")
ax.legend()
plt.tight_layout()
plt.savefig("charges_vs_grid_baderkit.png", dpi=300)

plt.show()
