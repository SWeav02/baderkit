The `ElfLabeler` class can be used to automatically identify various chemical features in a system. This can be useful for a variety of automation task. Here we demonstrate its use for locating the covalent bonds in the the CO<sub>2</sub> molecules of dry ice. We also demonstrate how to calculate the exact and formal bond-order of the bond.

## VASP

1. Create the CO<sub>2</sub> POSCAR file.

    ```
    C4 O8
    1.0
    5.4970732799999986    0.0000000000000000    0.0000000000000003
    -0.0000000000000003    5.4970732799999986    0.0000000000000003
    0.0000000000000000    0.0000000000000000    5.4970732799999986
    C O
    4 8
    direct
    0.0000000000000000    0.0000000000000000    0.0000000000000000 C
    0.5000000000000000    0.0000000000000000    0.5000000000000000 C
    0.5000000000000000    0.5000000000000000    0.0000000000000000 C
    0.0000000000000000    0.5000000000000000    0.5000000000000000 C
    0.1225369799999990    0.1225369799999990    0.1225369799999990 O
    0.3774630200000000    0.8774630200000000    0.6225369800000000 O
    0.6225369800000000    0.3774630200000000    0.8774630200000000 O
    0.8774630200000000    0.6225369800000000    0.3774630200000000 O
    0.8774630200000000    0.8774630200000000    0.8774630200000000 O
    0.6225369800000000    0.1225369799999990    0.3774630200000000 O
    0.3774630200000000    0.6225369800000000    0.1225369799999990 O
    0.1225369799999990    0.3774630200000000    0.6225369800000000 O
    ```

2. Create your INCAR file. Below is a minimal example that writes the required CHGCAR, AECCAR, and ELFCAR files. In general, the grid density should be at least 10 pts/Å along each lattice vector for well converged Bader analysis.

    ```
    Global Parameters
    LAECHG = True         # Write AECCAR files
    LELF = True           # Write ELFCAR file
    EDIFF  = 1E-06        # SCF energy convergence, in eV
    ENCUT  = 520

    Grid Size             # Moderately grid density
    NGX    = 60
    NGY    = 60
    NGZ    = 60
    "Fine" Grid Size      # Must Match Standard Grid
    NGXF   = 60
    NGYF   = 60
    NGZF   = 60
    ```

3. Create your `POTCAR`. We cannot provide an example for this as the files are proprietary.

4. Run VASP. Depending on your system how you do this may vary. On our system we use the following command.

    ```
    mpirun -np 12 vasp_std
    ```

## BaderKit

=== "Python"

    1. If you would like to follow along, open your preferred IDE in an environment with BaderKit installed. Alternatively, the complete python script from this tutorial is available at the end of this page.

    2. Import the Badelf class

        ```Python
        import math
        from baderkit import Grid
        from baderkit.elf_analysis import ElfLabeler
        ```

    3. We recommend using the reconstructed total charge density as a reference for Bader partitioning when possible. In VASP we can construct this from the AECCAR files.

        ```Python
        core_grid = Grid.from_vasp("AECCAR0")
        val_grid = Grid.from_vasp("AECCAR2")
        total = core_grid.linear_add(val_grid)
        total.write_vasp("CHGCAR_sum")
        ```

    3. Now create the ElfLabeler class instance.

        ```Python
        labeler = ElfLabeler.from_vasp(
            charge_filename="CHGCAR",
            reference_filename="ELFCAR",
            total_charge_filename="CHGCAR_sum",
            pseudopotential_filename="POTCAR"
            )

        ```

    4. label the basins in the ELF and get the charge in each.
        ```Python
        features = labeler.basin_types
        charges = labeler.elf_bader.basin_charges
        ```

    5. Get the indices that correspond to covalent bonds, and calculate the bond-order at each
        ```Python
        covalent = [i for i, j in enumerate(features) if j == "covalent bond"]

        bond_orders = []
        for idx in covalent:
            bond_orders.append(charges[idx]/2)
        ```
    
    6. Finally, print the exact bond-orders and the formal bond-orders to the console.
        ```Python
        for idx, bo in zip(covalent, bond_orders):
            print(f"Basin {idx} Bond Order: {round(bo,2)} -> {math.ceil(charges[idx]/2)}")
        ```
    
        You should see logging information as BaderKit runs, then outputs similar to the following:
        
        ```
        Basin 8 Bond Order: 1.37 -> 2
        Basin 9 Bond Order: 1.37 -> 2
        Basin 10 Bond Order: 1.37 -> 2
        Basin 11 Bond Order: 1.37 -> 2
        Basin 12 Bond Order: 1.37 -> 2
        Basin 13 Bond Order: 1.37 -> 2
        Basin 14 Bond Order: 1.37 -> 2
        Basin 15 Bond Order: 1.37 -> 2
        ```

=== "Command Line"

    1. If you are using an environment manager, load your baderkit environment. For conda:

        ```Bash
        conda activate baderkit
        ```

    2. We recommend using the reconstructed total charge density as a reference for Bader partitioning when possible. In VASP we can construct this from the AECCAR files.

        ```Bash
        baderkit sum AECCAR0 AECCAR2
        ```

    3. Run the Badelf analysis.

        ```Bash
        baderkit badelf CHGCAR ELFCAR -tot CHGCAR_sum
        ```

        You should see logging information printed to the console and once complete a `labeler.json` file will be written which summarizes the results of the calculation.

And that's it! Try playing around with what else the `ElfLabeler` class offers.

## Download Resources

Tutorial Script: <a href="/tutorial_scripts/vasp/electrides_vasp.py" download>bond_order.py</a>

VASP Inputs/Outputs: <a href="https://github.com/SWeav02/baderkit/releases/download/0.10.0/CO2.zip" download>CO2.zip</a>

## Warnings for VASP

### Low Valence Pseudopotentials

VASP only includes the valence electrons in the ELFCAR. This means that for pseudopotentials with relatively few valence electrons, it is possible for the ELF to be zero at atom centers. We recommend using VASP's [GW potentials](https://www.vasp.at/wiki/Available_pseudopotentials), with additional valence electrons.

### Mismatched Grids

By default, VASP writes the CHGCAR and ELFCAR to different grid shapes (the "fine" and standard FFT meshes). The `Badelf` and `ElfLabeler` classes require the grid sizes match. This can be achieved by setting the `NGX(YZ)` and `NGX(YZ)F` tags in the INCAR to match. Alternatively, one can set the `PREC` tag to `single`, but this should be done with caution as it generally lowers the quality of the calculation unless the `ENCUT` and `NGX(YZ)` tags are set as well.

## Introduction

!!! Warning
    This module is still in development and may change rapidly. If you are interested in using it, be warned that there may be unexpected bugs.

The first step of the BadELF algorithm is to determine whether there are bare electrons in the system and, if so, where they are located. In the original [paper](https://pubs.acs.org/doi/10.1021/jacs.3c10876), this was done by using relatively simple distance and ELF value cutoffs. Since then, the `ElfLabeler` method has evolved to be more rigorous. Using exclusively the ELF, charge density, and crystal structure, the `ElfLabeler` class aims to automatically label not only bare electrons, but atom shells, covalent bonds, metallic features, and lone-pairs.

While it was originally conceived to support the BadELF algorithm, the current ElfLabeler class can be used as a general tool for analyzing the ELF, providing considerably more information on each ELF feature than the Badelf class.

Examples of `CHGCAR` and `ELFCAR` files for the Ca2N electride used in this tutorial can be found [here](https://github.com/SWeav02/baderkit/releases/tag/0.9.0 ).

## Basic Use

The `ElfLabeler` can be run through the command line interface or through Python script. Currently only VASP's CHGCAR and ELFCAR files are supported.


=== "Command Line"

    1. Activate your environment with BaderKit installed. If you are not using an environment manager, skip to step 2.

        ```bash
        conda activate my_env
        ```

    2. Navigate to the directory with your charge density and ELF files.

        ```bash
        cd /path/to/directory
        ```

    3. Run the `ElfLabeler` analysis. Replace 'chargefile' and 'elffile' with the names of your charge-density and ELF files.

        ```bash
        baderkit label chargefile elffile
        ```
    You should see an output similar to this:
        ```bash
        2026-03-10 13:45:05 INFO     Loading CHGCAR
                            INFO     Time: 0.11
                            INFO     Data type set as charge from data range
                            INFO     Loading ELFCAR
                            INFO     Time: 0.11
                            INFO     Data type set as elf from data range
                            INFO     Beginning Bader Algorithm Using 'weight' Method
        2026-03-10 13:45:06 INFO     Initializing Labels
        2026-03-10 13:45:07 INFO     Initialization Complete
                            INFO     Time: 0.36
                            INFO     Sorting Reference Data
                            INFO     Assigning Charges and Volumes
                            INFO     Combining Low-Persistence Basins
        2026-03-10 13:45:09 INFO     Refining Maxima
                            INFO     Bader Algorithm Complete
                            INFO     Time: 3.53
                            INFO     Assigning Atom Properties
                            INFO     Atom Assignment Finished
                            INFO     Time: 0.01
                            INFO     Beginning ELF Analysis
                            INFO     Locating Bifurcations
        2026-03-10 13:45:10 INFO     Time: 0.77
                            INFO     Finding contained atoms
        2026-03-10 13:45:11 INFO     Time: 0.8
                            INFO     Beginning Bader Algorithm Using 'weight' Method
                            INFO     Initializing Labels
                            INFO     Initialization Complete
                            INFO     Time: 0.12
                            INFO     Sorting Reference Data
                            INFO     Assigning Charges and Volumes
                            INFO     Combining Low-Persistence Basins
        2026-03-10 13:45:13 INFO     Refining Maxima
                            INFO     Bader Algorithm Complete
                            INFO     Time: 2.32
                            INFO     Marking atomic shells
                            INFO     Marking covalent features
                            INFO     Marking lone-pair features
                            INFO     Calculating atomic radii
        2026-03-10 13:45:14 INFO     Marking multi-centered and bare electron features
                            INFO     Finished labeling ELF
        ```

    A summary of all properties will be written to `elf_labeler.json`. See our [example files](https://github.com/SWeav02/baderkit/releases/tag/0.9.0 ). for an example.

=== "Python"

    1. Import the `ElfLabeler` class.

        ```python
        from baderkit.elf_analysis import ElfLabeler
        ```

    2. Use the `ElfLabeler` class' `from_vasp` method to read a `CHGCAR` and `ELFCAR` file.

        ```python
        # instantiate the class
        labeler = ElfLabeler.from_vasp("path/to/chargefile", "path/to/elffile")
        ```

    3. To run the analysis, we can call any class property. Try getting a complete summary in dictionary format.
        ```python
        results = labeler.to_dict()
        ```
    You should see an output similar to this:
        ```bash
        2026-03-10 13:47:41 INFO     Beginning Bader Algorithm Using 'weight' Method
                            INFO     Initializing Labels
                            INFO     Initialization Complete
                            INFO     Time: 0.09
                            INFO     Sorting Reference Data
                            INFO     Assigning Charges and Volumes
        2026-03-10 13:47:42 INFO     Combining Low-Persistence Basins
        2026-03-10 13:47:43 INFO     Refining Maxima
                            INFO     Bader Algorithm Complete
                            INFO     Time: 1.71
                            INFO     Assigning Atom Properties
                            INFO     Atom Assignment Finished
                            INFO     Time: 0.01
                            INFO     Beginning ELF Analysis
                            INFO     Locating Bifurcations
                            INFO     Time: 0.34
                            INFO     Finding contained atoms
        2026-03-10 13:47:44 INFO     Time: 0.55
                            INFO     Beginning Bader Algorithm Using 'weight' Method
                            INFO     Initializing Labels
                            INFO     Initialization Complete
                            INFO     Time: 0.08
                            INFO     Sorting Reference Data
                            INFO     Assigning Charges and Volumes
                            INFO     Combining Low-Persistence Basins
        2026-03-10 13:47:45 INFO     Refining Maxima
                            INFO     Bader Algorithm Complete
                            INFO     Time: 1.69
                            INFO     Marking atomic shells
                            INFO     Marking covalent features
                            INFO     Marking lone-pair features
                            INFO     Calculating atomic radii
        2026-03-10 13:47:46 INFO     Marking multi-centered and bare electron features
                            INFO     Finished labeling ELF
        ```

    4. Now try getting individual properties. For more details on each property, see the [API reference](../api_reference/core/labeler).
        ```python
        feature_types = labeler.feature_types
        feature_charges = labeler.feature_charges
        labeled_structure = labeler.labeled_structure
        ```

    5. BaderKit also provides convenience methods for writing results to file. Let's write a summary of the full analysis, as well as a plot of the bifurcation graph.

        ```python
        labeler.write_json()
        labeler.write_bifurcation_plot()
        ```

    !!! Note
        The `ElfLabeler` uses the `Bader` class for partitioning. Any extra keyword arguments, such as the `method` parameter will be passed to the `Bader` class.

=== "GUI App"

    !!! Warning
        Currently the GUI App only supports Bader analysis.

---

## Spin-Dependent Calculations

BaderKit provides a convenience class for using the `ElfLabeler` on the spin-up and spin-down ELF separately. The combined results are also calculated by taking either the average or sum of the respective property.

=== "Command Line"

    Run the command with the `--spin` tag.

    ```bash
    baderkit label chargefile elffile --spin
    ```

    The results for each spin system are then written to separate files.

=== "Python"

    1. Import the `SpinElfLabeler` class and read in your spin-dependent files.

        ```python
        from baderkit.elf_analysis import SpinElfLabeler
        # instantiate the class
        labeler = SpinElfLabeler.from_vasp("path/to/chargefile", "path/to/elffile")
        ```

    2. Get the separate results for the spin-up and spin-down systems.

        ```python
        spin_up = labeler.elf_labeler_up
        spin_down = labeler.elf_labeler_down
        ```

    3. View properties separately or combined.

        ```python
        up_charges = spin_up.feature_charges
        down_charges = spin_down.feature_charges
        total_charges = badelf.feature_charges
        ```

---


## Visualizing and Interpreting Results

### The Labeled Structure

A useful output from the `ElfLabeler` class is the labeled structure which is a [pymatgen](https://pymatgen.org/) `Structure` object with 'dummy' atoms representing the different types of ELF features. This can be obtained from the `labeled_structure` property.

Pymatgen limits what labels can be used for dummy atoms based on if they start with the labels of an existing atom. Currently, we have settled on the following "species" labels for dummy atoms.

| Feature | Label |
| --- | --- |
| Shell | "Xs" |
| Covalent | "Z" |
| Multi-centered | "Mc" |
| Lone-Pair | "Lp" |
| Metallic | "M" |
| Bare Electron | "E" |

This structure can be written to a cif or POSCAR format with the `Structure.to()` method.

### Feature Properties

The properties assigned to each feature in the labeled structure are available as class properties. They may also be written to file with the `ElfLabeler.to_json()` method, which is used when running through the command line. Properties are always in the same order as the dummy atoms in the `labeled_structure`.

Some properties have `_e` appended at the end. This indicates that the electride sites were treated as quasi-atoms for this property. The electrides are included in any CrystalNN related analysis and given their own charge and "oxidation state". The equivalent properties without `_e` are calculated treating electride sites as some form of multi-centered bond.

 ---

## Warnings for VASP

See the warning on the [Badelf](../../badelf/usage/#warnings-for-vasp) page.