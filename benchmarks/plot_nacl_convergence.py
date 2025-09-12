# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd

from baderkit.core import Bader

# get all methods
methods = Bader.all_methods()

# get volume of structure (8 atom NaCl)
volume = 169.95413289447927

# load results
oxidation_df = pd.read_csv("oxidation_summary_baderkit.csv")

plt.style.use("seaborn-v0_8-darkgrid")

# different dash styles and markers for up to 4 methods
line_styles = ["-", "--", "-.", ":"]
markers = ["o", "s", "^", "D"]  # circle, square, triangle, diamond

# --- Plot charges vs. grid points ---
fig, ax = plt.subplots()
for i, method in enumerate(methods):
    ax.plot(
        oxidation_df["ngrid_points"] / volume,
        oxidation_df[method],
        marker=markers[i % len(markers)],
        linestyle=line_styles[i % len(line_styles)],
        label=method,
        alpha=0.6,  # add transparency
        linewidth=2,  # slightly thicker lines for clarity
        markersize=6,  # adjust marker size
    )
ax.set_xlabel("Grid Resolution (Pts / $Å^3$)")
ax.set_ylabel("Na Oxidation State")
ax.set_title("BaderKit Na Charge vs Grid Density")
ax.legend()
plt.tight_layout()
plt.savefig("charges_vs_grid_baderkit.png", dpi=300)

plt.show()
