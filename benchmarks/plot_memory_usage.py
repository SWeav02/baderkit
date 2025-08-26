# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd

from baderkit.core import Bader

# get all methods
methods = Bader.all_methods()

# load results
memory_df = pd.read_csv("memory_summary_baderkit.csv", index=False)

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

