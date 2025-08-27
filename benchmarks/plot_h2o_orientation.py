# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd

from baderkit.core import Bader

# get all methods
methods = Bader.all_methods()

# load results
orientation_df = pd.read_csv("orientation_summary_baderkit.csv")

plt.style.use("seaborn-v0_8-darkgrid")

# different dash styles and markers for up to 4 methods
line_styles = ["-", "--", "-.", ":"]
markers = ["o", "s", "^", "D"]  # circle, square, triangle, diamond

fig, ax = plt.subplots()
for i, method in enumerate(methods):
    ax.plot(
        orientation_df["angles"],
        orientation_df[method],
        marker=markers[i % len(markers)],
        linestyle=line_styles[i % len(line_styles)],
        label=method,
        alpha=0.6,  # add transparency
        linewidth=2,  # slightly thicker lines for clarity
        markersize=6,  # adjust marker size
    )

ax.set_xlabel("Rotation angle (degrees)")
ax.set_ylabel("Oxygen charge (e)")
ax.set_title("Oxygen Charge vs. Rotation Angle")
ax.legend()
plt.tight_layout()
fig.savefig("oxygen_charge_vs_angle.png", dpi=300)
plt.show()

print("Plot saved as 'oxygen_charge_vs_angle.png'")
