# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd

from baderkit.core import Bader

methods = Bader.all_methods()

baderkit_time_df = pd.read_csv("time_summary_baderkit.csv")
henk_time_df = pd.read_csv("time_summary_henk.csv")

plt.style.use("seaborn-v0_8-darkgrid")

# different dash styles and markers for up to 4 methods
line_styles = ["-", "--", "-.", ":"]
markers = ["o", "s", "^", "D"]  # circle, square, triangle, diamond

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

# Left subplot: BaderKit
for i, method in enumerate(methods):
    axes[0].plot(
        baderkit_time_df["ngrid_points"] / 1_000_000,
        baderkit_time_df[method],
        marker=markers[i % len(markers)],
        linestyle=line_styles[i % len(line_styles)],
        label=method,
        alpha=0.6,  # add transparency
        linewidth=2,  # slightly thicker lines for clarity
        markersize=6,  # adjust marker size
    )

min_baderkit_time = baderkit_time_df[methods].min().min()
axes[0].axhline(min_baderkit_time, color="gray", linestyle="--", linewidth=1)
axes[0].set_title("BaderKit Runtime")
axes[0].set_xlabel("Grid points (millions)")
axes[0].set_ylabel("Average runtime (s)")
axes[0].set_ylim(-5, 50)
axes[0].legend()

# Right subplot: Henk
henk_methods = ["weight", "ongrid", "neargrid"]
for i, method in enumerate(henk_methods):
    axes[1].plot(
        henk_time_df["ngrid_points"] / 1_000_000,
        henk_time_df[method],
        marker=markers[i % len(markers)],
        linestyle=line_styles[i % len(line_styles)],
        label=method,
        alpha=0.6,  # add transparency
        linewidth=2,  # slightly thicker lines for clarity
        markersize=6,  # adjust marker size
    )
min_henk_time = henk_time_df[henk_methods].min().min()
axes[1].axhline(min_henk_time, color="gray", linestyle="--", linewidth=1)
axes[1].set_title("Henkelman Runtime")
axes[1].set_xlabel("Grid points (millions)")
axes[1].set_ylim(-5, 50)
axes[1].legend()

plt.tight_layout()
plt.savefig("time_vs_grid_baderkit_henk_subplots.png", dpi=300)
plt.show()
