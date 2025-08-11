# -*- coding: utf-8 -*-

from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from baderkit import Bader

###############################################################################
# Running Bader
###############################################################################

directory = Path(".")
test_num = 10
grid_nums = [60, 120, 180, 240, 300]
all_charges = {}
all_time_avgs = {}
all_time_std = {}

methods = Bader.all_methods()

for method in methods:
    charges = []
    time_avg = []
    time_std = []
    for grid_num in grid_nums:
        folder = directory / str(grid_num)
        assert folder.exists()
        times = []
        for i in range(test_num):
            t0 = time.time()
            bader = Bader.from_vasp(folder/"CHGCAR", folder/"CHGCAR_sum", method=method)
            t1 = time.time()
            _ = bader.results_summary
            times.append(t1 - t0)
        charges.append(bader.atom_charges[0])
        time_avg.append(np.mean(times))
        time_std.append(np.std(times))
    all_charges[method] = charges
    all_time_avgs[method] = time_avg
    all_time_std[method] = time_std

###############################################################################
# DataFrames
###############################################################################
oxidation_df = pd.DataFrame({
    "one_axis_grid_points": grid_nums,
    "ngrid_points": np.array(grid_nums)**3,
    **all_charges
})
time_df = pd.DataFrame({
    "one_axis_grid_points": grid_nums,
    "ngrid_points": np.array(grid_nums)**3,
    **all_time_avgs
})
oxidation_df.to_csv("oxidation_summary.csv", index=False)
time_df.to_csv("time_summary.csv", index=False)

###############################################################################
# Plotting
###############################################################################
plt.style.use("seaborn-v0_8-darkgrid")

# --- Plot charges vs. grid points ---
fig, ax = plt.subplots()
for method in methods:
    ax.plot(
        oxidation_df["one_axis_grid_points"],
        oxidation_df[method],
        marker="o",
        label=method
    )
ax.set_xlabel("Grid points along one axis")
ax.set_ylabel("First atom charge")
ax.set_title("Bader Charges vs Grid Density")
ax.legend()
plt.tight_layout()
plt.savefig("charges_vs_grid.png", dpi=300)

# --- Plot timing vs. grid points with error bars ---
fig, ax = plt.subplots()
for method in methods:
    ax.errorbar(
        time_df["one_axis_grid_points"],
        time_df[method],
        yerr=all_time_std[method],
        marker="o",
        capsize=4,
        label=method
    )
ax.set_xlabel("Grid points along one axis")
ax.set_ylabel("Average runtime (s)")
ax.set_title("Bader Runtime vs Grid Density")
ax.legend()
plt.tight_layout()
plt.savefig("time_vs_grid.png", dpi=300)

plt.show()
