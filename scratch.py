#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from baderkit.core.elf_analysis.overlap.overlap import BasinOverlap
from baderkit.core.elf_analysis.elf_labeler1.elf_labeler import ElfLabeler
from baderkit.core.utilities.transforms import INT_TO_REV_INT, INT_TO_IMAGE
import numpy as np

from pathlib import Path
path = Path(".")

elements = []
best_vol_ratios = []
best_dist_ratios = []
best_dists = []
best_nna_fracs = []
core_populations = []
nna_populations = []
coulomb = []

for folder in path.iterdir():
    chgcar = folder / "CHGCAR"
    elf = folder / "ELFCAR"
    tot = folder / "CHGCAR_sum"
    pot = folder / "POTCAR"
    elements.append(folder.name)
    labeler = ElfLabeler.from_vasp(
        charge_filename=chgcar,
        reference_filename=elf,
        total_charge_filename=tot,
        persistence_tol=0.01,
        pseudopotential_filename = pot,
        )
    overlap: BasinOverlap = labeler.overlap
    bader = overlap.local_bader
    
    core_populations.append(overlap.atom_core_populations[0])
    
    types = labeler.basin_types
    nna_indices = np.array([i for i, j in enumerate(types) if j == "nna"])
    nna_charges = overlap.local_bader.basin_charges[nna_indices]
    
    volume_ratios = labeler.nna_core_volume_ratios
    dist_ratios = labeler.nna_core_distance_ratios
    dists = labeler.nna_core_distances
    fracs = labeler.nna_distance_ratios
    coul = labeler.nna_coulombic_potentials
    
    prominent_idx = np.argmax(nna_charges)
    
    best_vol_ratios.append(volume_ratios[prominent_idx])
    best_dist_ratios.append(dist_ratios[prominent_idx])
    best_dists.append(dists[prominent_idx])
    best_nna_fracs.append(fracs[prominent_idx])
    coulomb.append(coul[prominent_idx])
    
    nna_populations.append(nna_charges[prominent_idx])


best_vol_ratios = np.array(best_vol_ratios)
best_dist_ratios = np.array(best_dist_ratios)
best_dists = np.array(best_dists)
best_nna_fracs = np.array(best_nna_fracs)

core_populations = np.array(core_populations)
nna_populations = np.array(nna_populations)
coulomb = np.array(coulomb)
    
indices = np.argsort(elements)

print("Coulomb Potentials")
print("\n".join([str(i) for i in coulomb[indices]]))

print("Volume Ratios")
print("\n".join([str(i) for i in best_vol_ratios[indices]]))
print("Distance Ratios")
print("\n".join([str(i) for i in best_dist_ratios[indices]]))
print("Distance Beyond Atom")
print("\n".join([str(i) for i in best_dists[indices]]))
print("NNA Distance Fraction")
print("\n".join([str(i) for i in best_nna_fracs[indices]]))
print("Core Populations")
print("\n".join([str(i) for i in core_populations[indices]]))
print("NNA Populations")
print("\n".join([str(i) for i in nna_populations[indices]]))

test_grid = overlap.reference_grid.copy()
structure = labeler.reference_grid.structure.copy()
fracs = overlap.local_maxima_frac
for frac in fracs:
    structure.append("x", frac)
test_grid.structure = structure
test_grid.write_vasp("ELFCAR_test")



atom_frac = structure.frac_coords[0] + INT_TO_IMAGE[16]
point_frac = np.array((0.5,0.5,0.5))

line = overlap.reference_grid.linear_slice(atom_frac, point_frac)
matrix = overlap.reference_grid.matrix
dist = np.linalg.norm(atom_frac@matrix - point_frac@matrix)
x = np.linspace(0, dist, len(line))
import matplotlib.pyplot as plt
plt.plot(x, line)
# from baderkit.core import Structure
# structure = Structure.from_file("POSCAR")
# fracs = final_frac
# for frac in fracs:
#     structure.append("x", frac)
# structure.to("POSCAR_test")