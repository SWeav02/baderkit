# -*- coding: utf-8 -*-

import multiprocessing as mp
import os
import resource
import sys
import threading
import time
import traceback
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
    func: Callable, *args, ctx: mp.context = None, **kwargs
) -> Tuple[Any, float]:
    """
    Run `func(*args, **kwargs)` in a child process and return (result, peak_mb).
    Peak includes C allocations (reported by the OS for the child).
    Uses multiprocessing.Pipe to get (result, child_ru_maxrss) back from the child.

    Notes:
    - On Linux ru_maxrss is returned in kilobytes. On macOS it is bytes.
    - If your arguments (args/kwargs) include very large numpy arrays, prefer using
      a "fork" start method (default on Linux). On Windows or when spawn is used,
      arguments will be pickled and that duplicate may inflate memory usage.
    """
    # Choose context: prefer fork to avoid pickling big objects (only available on Unix)
    if ctx is None:
        try:
            ctx = mp.get_context("fork")
        except Exception:
            ctx = mp.get_context("spawn")

    parent_conn, child_conn = mp.Pipe(duplex=False)

    def _worker(conn, fn, a, kw):
        try:
            res = fn(*a, **kw)
            # ru_maxrss for this child (RUSAGE_SELF)
            r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            conn.send(("ok", res, r))
        except Exception:
            tb = traceback.format_exc()
            conn.send(("err", tb))
        finally:
            conn.close()

    p = ctx.Process(target=_worker, args=(child_conn, func, args, kwargs))
    p.start()

    # Wait for child to send result (this will block until the child sends)
    status, *payload = parent_conn.recv()
    p.join()

    if status == "err":
        raise RuntimeError(f"Child process raised an exception:\n{payload[0]}")

    result, ru_maxrss = payload  # (result, ru_maxrss)

    # Interpret ru_maxrss units: Linux -> kilobytes, macOS/BSD -> bytes.
    if sys.platform.startswith("linux"):
        peak_bytes = int(ru_maxrss) * 1024
    else:
        # macOS/bsd return bytes for ru_maxrss
        peak_bytes = int(ru_maxrss)

    peak_mb = peak_bytes / 1024**2
    return result, peak_mb


# function for running bader
def run_bader(charge_grid, reference_grid, method, **kwargs):
    bader = Bader(charge_grid=charge_grid, reference_grid=reference_grid, method=method)
    return bader.to_dict()


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
