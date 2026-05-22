# Work in Progress

We have had difficulty reproducing the results in the VASP tutorials using Quantum ESPRESSO, largely due to the ELF output seemingly being significantly lower quality than VASP's (e.g. many small basins). For now we do not recommend using QE for ELF related analysis.

## Suggestions

If you have any suggestions for how to improve the ELF analysis in QE, we'd love to hear them! So far we have tried multiple sources of PAW potentials, increasing the charge grid to very fine densities, and even calculating the ELF ourselves using a method closer to that used by VASP and other codes. If you have ideas, please open a discussion or issue on our [github page](https://github.com/SWeav02/baderkit).