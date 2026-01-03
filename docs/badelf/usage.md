## Introduction

The `Badelf` class uses principles from Bader's Quantum Theory of Atoms in Molecules combined with the Electron Localization Function (ELF) to calculate atomic charges. It is primarily designed for charge analysis in electride systems, which was the motivation for the original [work](https://pubs.acs.org/doi/10.1021/jacs.3c10876). For tasks aiming to analyze other chemical features (e.g. covalent bonds, lone-pairs), the [ElfLabeler](../../elf_labeler/usage) class is likely more appropriate.

## Basic Use

BadELF can be run through the command line interface or through Python script. Currently only VASP's CHGCAR and ELFCAR files are supported.

By default, BadELF uses the 'badelf' algorithm which separates atoms with planes and electride sites with zero-flux surfaces (i.e. Bader's method). This works best when atoms are fairly spherical and ionic. In other cases, the [zero-flux](/methods) method may be more appropriate.

=== "Command Line"

    1. Activate your environment with BaderKit installed. If you are not using an
    environment manager, skip to step 2.
    
        ```bash
        conda activate my_env
        ```
        
    2. Navigate to the directory with your charge density and ELF files.
    
        ```bash
        cd /path/to/directory
        ```
    
    3. Run the BadELF analysis. Replace 'chargefile' and 'elffile' with the names of your charge-density and ELF files.
    
        ```bash
        baderkit badelf chargefile elffile
        ```
    
    Output files for atoms will be written automatically to a `badelf_atom_summary.tsv` which includes rows for each atom with columns for:

    - atom labels/assignments
    - coordinates (fractional) 
    - charges (*e*) 
    - volumes (Å<sup>3</sup>)
    - minimum surface distances (Å)
    
    Additional arguments and options such as those for printing output files or using different 
    algorithms can be viewed by running the help command.
    ```bash
    baderkit badelf --help
    ```

=== "Python"
    
    1. Import the `Badelf` class.
    
        ```python
        from baderkit.core import Badelf
        ```
    
    2. Use the `Badelf` class' `from_vasp` method to read a `CHGCAR` and `ELFCAR` file.
    
        ```python
        # instantiate the class
        badelf = Badelf.from_vasp("path/to/chargefile", "path/to/elffile")
        ```
    
    3. To run the analysis, we can call any class property. Try getting a complete
    summary in dictionary format.
        ```python
        results = badelf.to_json()
    
    4. Now try getting individual properties. For more details on each property,
    see the [API reference](../api_reference/core/badelf).
        ```python
        atom_charges = badelf.atom_charges # Total atom charges
        atom_labels = badelf.atom_labels # Atom assignments for each point in the grid
        maxima_coords = badelf.basin_maxima_frac # Frac coordinates of each attractor
        ```
    
    5. BaderKit also provides convenience methods for writing results to file. First,
    let's write a summary of the full analysis.
    
        ```python
        badelf.write_json()
        ```
    
    6. Now let's write the volume assigned to one of our atoms.
    
        ```python
        badelf.write_atom_volumes(atom_indices = [0])
        ```
    
    !!! Tip
        After creating a `Badelf` class object, it doesn't matter what order
        you call properties, summaries, or write methods in. BaderKit calculates
        properties/results only when they are needed and caches them.

=== "GUI App"

    !!! Warning
        Currently the GUI App only supports traditional Bader analysis.

---

## Spin-Dependent Calculations

BaderKit provides a convenience class for performing `BadELF` on the spin-up and spin-down ELF separately. The combined results are also calculated by taking either the average or sum of the respective property.

=== "Command Line"

    Run the command with the `--spin` tag.

    ```bash
    baderkit badelf chargefile elffile --spin
    ```

    The results for each spin system are then written to separate files.

=== "Python"
    
    1. Import the `SpinBadelf` class and read in your spin-dependent files.
    
        ```python
        from baderkit.core import SpinBadelf
        # instantiate the class
        badelf = SpinBadelf.from_vasp("path/to/chargefile", "path/to/elffile")
        ```
    
    2. Get the separate results for the spin-up and spin-down systems.

        ```python
        spin_up = badelf.badelf_up
        spin_down = badelf.badelf_down
        ```
    
    3. View properties separately or combined.

        ```python
        up_charges = spin_up.charges
        down_charges = spin_down.charges
        total_charges = badelf.charges
        ```

---

## Warnings for VASP

### Low Valence Pseudopotentials

VASP only includes the valence electrons in the ELFCAR. This means that for pseudopotentials with relatively few valence electrons, it is possible for the ELF to be zero at atom centers. To get around this, we recommend using VASP's [GW potentials](https://www.vasp.at/wiki/Available_pseudopotentials), with additional valence electrons.

### Atomic Position Precision

For BadELF methods involving planes (i.e. `badelf` and `voronelf`), results can change significantly with very small differences in atom position. VASP writes atomic positions in the CHGCAR and ELFCAR with limited precision, sometimes much lower than the values in the POSCAR. To help with this, we provide an option to override the crystal structure when reading in the CHGCAR/ELFCAR:

```python
from baderkit.core import Badelf

badelf = Badelf.from_vasp("path/to/chargefile", "path/to/elffile", poscar_file="path/to/poscar")
```
