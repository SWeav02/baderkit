# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd

from baderkit.core import Bader

# get all methods
methods = Bader.all_methods()

# load results
orientation_df = pd.read_csv("orientation_summary_baderkit.csv", index=False)

plt.style.use("seaborn-v0_8-darkgrid")

fig, ax = plt.subplots()
for method in methods:
    ax.plot(orientation_df["angles"], orientation_df[method], marker="o", label=method)

ax.set_xlabel("Rotation angle (degrees)")
ax.set_ylabel("Oxygen charge (e)")
ax.set_title("Oxygen Charge vs. Rotation Angle")
ax.legend()
plt.tight_layout()
fig.savefig("oxygen_charge_vs_angle.png", dpi=300)
plt.show()

print("Plot saved as 'oxygen_charge_vs_angle.png'")
