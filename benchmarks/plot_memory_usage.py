# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd

from baderkit.core import Bader

# get all methods
methods = Bader.all_methods()

# load results
memory_df = pd.read_csv("memory_summary_baderkit.csv")

plt.style.use("seaborn-v0_8-darkgrid")

# different dash styles and markers for up to 4 methods
line_styles = ["-", "--", "-.", ":"]
markers = ["o", "s", "^", "D"]  # circle, square, triangle, diamond

# --- Plot memory vs. grid points ---
fig, ax = plt.subplots()
for i, method in enumerate(methods):
    ax.plot(
        memory_df["ngrid_points"] / 1_000_000,
        memory_df[method] / 1_000,
        marker=markers[i % len(markers)],
        linestyle=line_styles[i % len(line_styles)],
        label=method,
        alpha=0.6,  # add transparency
        linewidth=2,  # slightly thicker lines for clarity
        markersize=6,  # adjust marker size
    )

ax.set_xlabel("Grid points (millions)")
ax.set_ylabel("Peak Memory Usage (GB)")
ax.set_title("BaderKit Memory Usage vs Grid Density")
ax.legend()
plt.tight_layout()
plt.savefig("memory_vs_grid_baderkit.png", dpi=300)

plt.show()
