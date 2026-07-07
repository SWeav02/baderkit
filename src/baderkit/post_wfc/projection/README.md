# INFO

This folder contains files to aid in reconstructing the non-interacting charge density for an atomic system. Each file is a numpy binary containing the atomic basis set, energies, and coefficients for each state. The basis sets have been compressed to a radial average form by assuming fractional occupancies of degenerate basis functions.

The coefficients were generated using PySCF v.2.13.0 using the r2SCAN functional with the [dyall-ae3z](10.5281/zenodo.7574629) all-electron basis set created by Dyall. The compressed representations were generated using the file 'generate_references.py' in this folder. Highly linearly dependent virtual orbitals are removed to avoid instability in the virtual states. The basis set is normalized prior to being compressed.

These files are used to calculate the non-bonding charge density for a given charge density range. This can be compared to the true charge density contributions to determine if a state contributes to bonding or antibonding interactions at a given point in space.