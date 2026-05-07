# Tutorial Introduction

This module provides tutorials showing the complete process from start to finish using VASP or quantum espresso then moving to BaderKit. The files used in each tutorial can be downloaded [here](https://github.com/SWeav02/baderkit/releases/tag/0.10.0) or from the respective links in each tutorial. Each folder contains the input and output files required for the tutorial as well as the full baderkit python script used in the tutorial.

!!! Note
    We do not provide `POTCAR` files as VASP does not permit their distribution. If you use VASP you will have to provide these yourself.

## Useful Resources

### VASP

- [VASPKIT](https://vaspkit.com/): A useful tool for preparing input files and post-processing of VASP calculations.

- [VASP Wiki](https://vasp.at/wiki/Welcome): VASP's primary repository of documentation.

### Quantum Espresso

- [SSSP Database](https://legacy.materialscloud.org/discover/sssp/table/efficiency#sssp-license): A curacted collection of pseudopotentials. All QE tutorials in this module use the SSSP precision pbesol PPs. Please cite this tool and the original PP if used in publication.
    
    The full database of pbesol PPs (precision) can be also be downloaded and extracted as follows.
    ```
    wget https://archive.materialscloud.org/api/records/rcyfm-68h65/files/SSSP_1.3.0_PBEsol_precision.tar.gz/content -O sssp_efficiency.tar.gz
    wget https://archive.materialscloud.org/api/records/rcyfm-68h65/files/SSSP_1.3.0_PBE_precision.json/content -O SSSP_1.3.0_PBE_precision.json

    tar -xvf sssp_efficiency.tar.gz
    ```

    !!! Warning
        Only PAW PPs can be used when reconstructing the total charge density in QE. Many of the SSSP efficiency PPs and several SSSP precision PPs are ultrasoft PPs and should not be used for Bader analysis.

- [Quantum Espresso Input Generator](https://qeinputgenerator.materialscloud.io/): A tool for generating input files for QE. The input files used in these tutorials were generated using the QE input generator on materialscloud.io using the `fast` protocol, then updated with desired PPs and paths. Please cite as described if used in publication.

- [Quantum ESPRESSO Documentation](https://www.quantum-espresso.org/documentation/): QE's primary repository of documentation.

## Suggestions

If you would like to see an example for some other task, or have an example you'd like to contribute, please open an [issue](https://github.com/SWeav02/baderkit/issues) on our github page.