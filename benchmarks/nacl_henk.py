# -*- coding: utf-8 -*-

from pathlib import Path
import time
import numpy as np
import pandas as pd
import subprocess
import matplotlib.pyplot as plt

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
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            t1 = time.time()
            times.append(t1 - t0)
        time_avg.append(np.mean(times))
        time_std.append(np.std(times))
    all_time_avgs[method] = time_avg
    all_time_std[method] = time_std

###############################################################################
# DataFrame and CSV output
###############################################################################
time_df = pd.DataFrame({
    "one_axis_grid_points": grid_nums,
    "ngrid_points": np.array(grid_nums) ** 3,
    **all_time_avgs
})
time_df.to_csv("time_summary_cli.csv", index=False)

###############################################################################
# Plotting
###############################################################################
plt.style.use("seaborn-v0_8-darkgrid")

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
ax.set_title("Bader Runtime vs Grid Density (CLI)")
ax.legend()
plt.tight_layout()
plt.savefig("time_vs_grid_cli.png", dpi=300)
plt.show()


