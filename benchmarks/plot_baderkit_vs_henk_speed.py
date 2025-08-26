# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd

from baderkit.core import Bader

methods = Bader.all_methods()

baderkit_time_df = pd.read_csv("time_summary_baderkit.csv")
henk_time_df = pd.read_csv("time_summary_henk.csv")

plt.style.use("seaborn-v0_8-darkgrid")

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

# Left subplot: BaderKit
for method in methods:
    axes[0].plot(
        baderkit_time_df["ngrid_points"] / 1_000_000,
        baderkit_time_df[method],
        marker="o",
        label=method,
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
for method in henk_methods:
    try:
        axes[1].plot(
            henk_time_df["ngrid_points"] / 1_000_000,
            henk_time_df[method],
            marker="o",
            label=method,
        )
    except:
        pass
min_henk_time = henk_time_df[henk_methods].min().min()
axes[1].axhline(min_henk_time, color="gray", linestyle="--", linewidth=1)
axes[1].set_title("Henkelman Runtime")
axes[1].set_xlabel("Grid points (millions)")
axes[1].set_ylim(-5, 50)
axes[1].legend()

plt.tight_layout()
plt.savefig("time_vs_grid_baderkit_henk_subplots.png", dpi=300)
plt.show()
