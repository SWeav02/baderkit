import matplotlib.pyplot as plt
import pandas as pd

from baderkit.core import Bader

methods = Bader.all_methods()

baderkit_time_df = pd.read_csv("time_summary_baderkit.csv")
henk_time_df = pd.read_csv("time_summary_henk.csv")

plt.style.use("seaborn-v0_8-darkgrid")

fig, ax = plt.subplots()

# Plot both datasets with the same colors
for method in methods:
    # Solid line: BaderKit
    (line,) = ax.plot(
        baderkit_time_df["ngrid_points"] / 1_000_000,
        baderkit_time_df[method],
        marker="o",
        label=f"{method} (BaderKit)",
    )

    # Dashed line: Henk, using same color
    ax.plot(
        henk_time_df["ngrid_points"] / 1_000_000,
        henk_time_df[method],
        marker="s",
        linestyle="--",
        color=line.get_color(),
        label=f"{method} (Henk)",
    )

ax.set_xlabel("Grid points (millions)")
ax.set_ylabel("Average runtime (s)")
ax.set_title("BaderKit vs Henk Runtime vs Grid Density")
ax.legend()
plt.tight_layout()
plt.savefig("time_vs_grid_baderkit_henk.png", dpi=300)
plt.show()
