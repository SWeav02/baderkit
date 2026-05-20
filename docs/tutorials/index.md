# Tutorial Introduction

This module provides tutorials showing the complete process from start to finish using VASP or quantum espresso then moving to BaderKit. The files used in each tutorial can be downloaded [here](https://github.com/SWeav02/baderkit/releases/tag/0.10.0) or from the respective links in each tutorial. Each folder contains the input and output files required for the tutorial as well as the full baderkit python script used in the tutorial.

!!! Note
    We do not provide `POTCAR` files as VASP does not permit their distribution. If you use VASP you will have to provide these yourself.

## Useful Resources

### VASP

- [VASPKIT](https://vaspkit.com/): A tool for preparing input files and post-processing of VASP calculations.

- [VASP Wiki](https://vasp.at/wiki/Welcome): VASP's primary documentation.

### Quantum ESPRESSO

- [pslibrary](https://dalcorso.github.io/pslibrary/): High-quality pseudopotentials for a variety of functionals.

- [SSSP Database](https://legacy.materialscloud.org/discover/sssp/table/efficiency#sssp-license): A curacted collection of pseudopotentials. 

    !!! Warning
        Only PAW PPs can be used when reconstructing the total charge density in QE. Many of the SSSP PPs are ultrasoft PPs and should not be used for Bader analysis.

- [Quantum ESPRESSO Input Generator](https://qeinputgenerator.materialscloud.io/): A tool for generating input files for QE. The input files used in these tutorials were generated using the QE input generator on materialscloud.io using the `fast` protocol, then updated with desired PPs and paths. Please cite as described if used in publication.

- [Quantum ESPRESSO Documentation](https://www.quantum-espresso.org/documentation/): QE's primary repository of documentation.

## Suggestions

If you would like to see an example for some other task, or have an example you'd like to contribute, please open an [issue](https://github.com/SWeav02/baderkit/issues) on our github page.