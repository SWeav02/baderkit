# -*- coding: utf-8 -*-

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from baderkit.core import Bader, Grid

base_dir = Path(".")

methods = ["weight", "ongrid", "neargrid", "neargrid-weight"]
angles = [
    "000",
    "015",
    "030",
    "045",
    "060",
    "075",
    "090",
    "105",
    "120",
    "135",
    "150",
    "165",
]

# Convert string angles to integers for plotting
angle_vals = [int(a) for a in angles]

# Results: results[method][angle] = oxygen charge
results = {method: [] for method in methods}

# Load and compute Bader charges
for angle in angles:
    # make sure angle result actually exists
    folder = base_dir / angle
    assert folder.exists()

    # load grids manually to avoid having to do for each method
    charge_grid = Grid.from_vasp(folder / "CHGCAR")
    reference_grid = Grid.from_vasp(folder / "CHGCAR_sum")

    # for each method, calculate the oxygen's oxidation state
    for method in methods:
        bader = Bader(
            method=method, charge_grid=charge_grid, reference_grid=reference_grid
        )
        # append the oxidation state by subtracting from PP electron count
        results[method].append(6 - bader.atom_charges[1])  # oxygen index

###############################################################################
# Save results
###############################################################################
orientation_df = pd.DataFrame(
    {
        "angles": angles,
        **results,
    }
)
orientation_df.to_csv("orientation_summary_baderkit.csv", index=False)

###############################################################################
# Combined Plot
###############################################################################
plt.style.use("seaborn-v0_8-darkgrid")

fig, ax = plt.subplots()
for method in methods:
    ax.plot(angle_vals, results[method], marker="o", label=method)

ax.set_xlabel("Rotation angle (degrees)")
ax.set_ylabel("Oxygen charge (e)")
ax.set_title("Oxygen Charge vs. Rotation Angle")
ax.legend()
plt.tight_layout()
fig.savefig("oxygen_charge_vs_angle.png", dpi=300)
plt.show()

print("Plot saved as 'oxygen_charge_vs_angle.png'")
