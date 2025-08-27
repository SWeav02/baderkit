# -*- coding: utf-8 -*-

import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

from baderkit.core import Bader, Grid

###############################################################################
# Function for tracking memory
###############################################################################


def measure_peak_memory(
    func: Callable, *args, interval: float = 0.01, **kwargs
) -> Tuple[Any, float]:
    """
    Run a function and measure its peak memory usage (RSS).

    Parameters
    ----------
    func : Callable
        The function to run.
    *args, **kwargs :
        Arguments to pass to the function.
    interval : float, optional
        How often to sample memory usage in seconds (default: 0.01).

    Returns
    -------
    result : Any
        The return value of the function.
    peak_mb : float
        Peak resident set size (RSS) in MB.
    """
    process = psutil.Process(os.getpid())
    peak_rss = 0.0
    running = True

    def monitor():
        nonlocal peak_rss
        while running:
            rss = process.memory_info().rss
            if rss > peak_rss:
                peak_rss = rss
            time.sleep(interval)

    # Start monitor in background
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

    try:
        result = func(*args, **kwargs)
    finally:
        running = False
        thread.join()

    return result, peak_rss / 1024**2  # MB


# function for running bader
def run_bader(charge_grid, reference_grid, method, **kwargs):
    bader = Bader(charge_grid, reference_grid, method)
    return bader.results_summary


###############################################################################
# Running Bader
###############################################################################

methods = Bader.all_methods()

directory = Path(".")
# test_num = 10
grid_nums = [60, 120, 180, 240, 300, 400]
memory_used = {method: [] for method in methods}


for grid_num in grid_nums:
    folder = directory / str(grid_num)
    assert folder.exists()

    # load grids to avoid repeat slow file read times
    charge_grid = Grid.from_vasp(folder / "CHGCAR")
    reference_grid = Grid.from_vasp(folder / "CHGCAR_sum")

    # run each method and save peak memory usage
    for method in methods:
        result, memory = measure_peak_memory(
            run_bader,
            charge_grid=charge_grid,
            reference_grid=reference_grid,
            method=method,
        )
        memory_used[method].append(memory)

###############################################################################
# DataFrames
###############################################################################
memory_df = pd.DataFrame(
    {
        "one_axis_grid_points": grid_nums,
        "ngrid_points": np.array(grid_nums) ** 3,
        **memory_used,
    }
)

memory_df.to_csv("memory_summary_baderkit.csv", index=False)

###############################################################################
# Plotting
###############################################################################
plt.style.use("seaborn-v0_8-darkgrid")

# --- Plot charges vs. grid points ---
fig, ax = plt.subplots()
for method in methods:
    ax.plot(
        memory_df["ngrid_points"] / 1000000,
        memory_df[method],
        marker="o",
        label=method,
    )
ax.set_xlabel("Grid points (millions)")
ax.set_ylabel("Peak Memory Usage (MB)")
ax.set_title("BaderKit Memory Usage vs Grid Density")
ax.legend()
plt.tight_layout()
plt.savefig("memory_vs_grid_baderkit.png", dpi=300)

plt.show()
