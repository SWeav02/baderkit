# -*- coding: utf-8 -*-

import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

###############################################################################
# Parameters
###############################################################################
directory = Path(".")
test_num = 10
grid_nums = [60, 120, 180, 240, 300]

# Replace this with the actual list of methods you want to test
methods = ["ongrid", "neargrid", "weight"]

all_time_avgs = {}
all_time_std = {}
charges = {method: [] for method in methods}

###############################################################################
# Timing bader CLI calls
###############################################################################
for method in methods:
    time_avg = []
    time_std = []
    for grid_num in grid_nums:
        folder = directory / str(grid_num)
        assert folder.exists()
        times = []
        for i in range(test_num):
            t0 = time.time()
            subprocess.run(
                ["bader", "CHGCAR", "-ref", "CHGCAR_sum", "-b", method],
                cwd=folder,
                stdout=None,
                stderr=None,
                check=True,
            )
            t1 = time.time()
            times.append(t1 - t0)
        time_avg.append(np.mean(times))
        time_std.append(np.std(times))
        # load results and get first atom oxidation state
        with open(folder / "ACF.dat", "r") as f:
            lines = f.readlines()
        charge = float(lines[2].split()[4])
        charges[method].append(9 - charge)
    all_time_avgs[method] = time_avg
    all_time_std[method] = time_std


###############################################################################
# DataFrame and CSV output
###############################################################################
time_df = pd.DataFrame(
    {
        "one_axis_grid_points": grid_nums,
        "ngrid_points": np.array(grid_nums) ** 3,
        **all_time_avgs,
    }
)
time_df.to_csv("time_summary_henk.csv", index=False)

oxidation_df = pd.DataFrame(
    {
        "one_axis_grid_points": grid_nums,
        "ngrid_points": np.array(grid_nums) ** 3,
        **charges,
    }
)

oxidation_df.to_csv("oxidation_summary_henk.csv", index=False)
###############################################################################
# Plotting
###############################################################################
plt.style.use("seaborn-v0_8-darkgrid")

fig, ax = plt.subplots()
for method in methods:
    ax.errorbar(
        time_df["ngrid_points"] / 1000000,
        time_df[method],
        yerr=all_time_std[method],
        marker="o",
        capsize=4,
        label=method,
    )
ax.set_xlabel("Grid points (millions)")
ax.set_ylabel("Average runtime (s)")
ax.set_title("Henkelman Runtime vs Grid Density")
ax.legend()
plt.tight_layout()
plt.savefig("time_vs_grid_henk.png", dpi=300)
plt.show()

# --- Plot charges vs. grid points ---
fig, ax = plt.subplots()
for method in methods:
    ax.plot(
        oxidation_df["ngrid_points"] / 1000000,
        oxidation_df[method],
        marker="o",
        label=method,
    )
ax.set_xlabel("Grid points (millions")
ax.set_ylabel("Na Oxidation State")
ax.set_title("BaderKit Na Charge vs Grid Density")
ax.legend()
plt.tight_layout()
plt.savefig("charges_vs_grid_henk.png", dpi=300)

plt.show()
