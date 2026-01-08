## Introduction

The first step of the BadELF algorithm is to determine whether there are bare electrons in the system and, if so, where they are located. In the original [paper](https://pubs.acs.org/doi/10.1021/jacs.3c10876), this was done by using relatively simple distance and ELF value cutoffs. Since then, the `ElfLabeler` method has evolved to be more rigorous. Using exclusively the ELF, charge density, and crystal structure, the `ElfLabeler` class aims to automatically label not only bare electrons, but atom shells, covalent bonds, metallic features, and lone-pairs.

While it was originally conceived to support the BadELF algorithm, the current ElfLabeler class can be used as a general tool for analyzing the ELF, providing considerably more information on each ELF feature than the Badelf class.

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
    
    An output summary will automatically be written to an 'elf_labeler.json' file.

=== "Python"
    
    1. Import the `ElfLabeler` class.
    
        ```python
        from baderkit.core import ElfLabeler
        ```
    
    2. Use the `ElfLabeler` class' `from_vasp` method to read a `CHGCAR` and `ELFCAR` file.
    
        ```python
        # instantiate the class
        labeler = ElfLabeler.from_vasp("path/to/chargefile", "path/to/elffile")
        ```
    
    3. To run the analysis, we can call any class property. Try getting a complete summary in dictionary format.
        ```python
        results = labeler.to_json()
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
        from baderkit.core import SpinElfLabeler
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
