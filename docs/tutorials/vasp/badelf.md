The `Badelf` class includes several methods for calculating oxidation states in materials combining principles from Bader's Quantum Theory of Atoms in Molecules with the Electron Localization Function (ELF). This is useful for calculating oxidation states in systems with localized electrons that sit far from atomic sites such as in [electride systems](https://pubs.acs.org/doi/10.1021/jacs.3c10876). This tutorial walks through this process for the common Ca<sub>2</sub>N electride.

## VASP

1. Create your Ca<sub>2</sub>N POSCAR file.

    ```
    Ca2 N1
    1.0
    3.537074 0.051133 5.740763
    1.665193 3.120999 5.740763
    0.083858 0.051133 6.742420
    Ca N
    2 1
    direct
    0.731317 0.731317 0.731317 Ca
    0.268683 0.268683 0.268683 Ca
    0.000000 0.000000 -0.000000 N
    ```

2. Create your INCAR file. Below is a minimal example that writes the required CHGCAR and ELFCAR files. In general, the grid density should be at least 10 pts/Å along each lattice vector for well converged Bader analysis.

    ```
    Global Parameters
    LELF = True         # Write ELFCAR file
    EDIFF  = 1E-06        # SCF energy convergence, in eV
    ENCUT  = 520

    Grid Size             # Moderately grid density
    NGX    = 70
    NGY    = 70
    NGZ    = 70
    "Fine" Grid Size      # Must Match Standard Grid
    NGXF   = 70
    NGYF   = 70
    NGZF   = 70
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
        from baderkit.elf_analysis import Badelf
        ```

    3. Now create the Badelf class instance.

        ```Python
        badelf = Badelf.from_vasp(
        charge_filename="CHGCAR",
        reference_filename="ELFCAR",
        pseudopotential_filename="POTCAR"
        )
        ```

    4. Finally, print some useful information to the console.
        ```Python
        # structure including electride site
        print(f"Electride Structure: {badelf.nna_structure}")

        # print electron counts
        print(f"Electron Count: {badelf.nnas_per_reduced_formula}")

        # print dimensionality
        print(f"Electride Dimensionality: {badelf.nna_dimensionality}")
        ```
    
    You should see logging information as BaderKit runs, then outputs similar to the following:
        `array([ 0.87331308, -0.8732974 ])`

=== "Command Line"

    1. If you are using an environment manager, load your baderkit environment. For conda:

        ```Bash
        conda activate baderkit
        ```

    2. We recommend using the reconstructed total charge density as a reference for Bader partitioning when possible. In VASP we can construct this from the AECCAR files.

        ```Bash
        baderkit sum AECCAR0 AECCAR2
        ```

    3. Run the Bader analysis.

        ```Bash
        baderkit bader CHGCAR -ref CHGCAR_sum
        ```

        You should see logging information printed to the console and once complete a `bader.json` file will be written which summarizes the results of the calculation.

And that's it! Try playing around with what else the `Bader` class offers.

## Download Resources

Tutorial Script: <a href="/tutorial_scripts/vasp/oxidation_states_vasp.py" download>oxidation_states_vasp.py</a>

VASP Inputs/Outputs: <a href="https://github.com/SWeav02/baderkit/releases/download/0.9.0/NaCl.zip" download>NaCl.zip</a>


## Basic Use

BadELF can be run through the command line interface or through Python script. Currently only VASP's CHGCAR and ELFCAR files are supported.

By default, BadELF uses the 'zero-flux' algorithm which separates all ELF basins at zero-flux surfaces, following traditional ELF charge analysis. The original [badelf](../methods/#__tabbed_1_3) algorithm also incorporates voronoi-like planes and may be more appropriate in some systems.

Examples of `CHGCAR` and `ELFCAR` files for the Ca2N electride used in this tutorial can be found [here](https://github.com/SWeav02/baderkit/releases/tag/0.9.0 ).

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

    You should see an output similar to this:

    ```bash
    2026-03-10 13:38:33 INFO     Loading ELFCAR
                        INFO     Time: 0.1
                        INFO     Data type set as elf from data range
                        INFO     Loading CHGCAR
                        INFO     Time: 0.1
                        INFO     Data type set as charge from data range
                        WARNING  No POTCAR file found in the requested directory. Oxidation states cannot be calculated
                        INFO     Beginning voxel assignment
                        INFO     Beginning Bader Algorithm Using 'weight' Method
    2026-03-10 13:38:34 INFO     Initializing Labels
                        INFO     Initialization Complete
                        INFO     Time: 0.4
                        INFO     Sorting Reference Data
                        INFO     Assigning Charges and Volumes
                        INFO     Combining Low-Persistence Basins
    2026-03-10 13:38:36 INFO     Refining Maxima
                        INFO     Bader Algorithm Complete
                        INFO     Time: 3.45
                        INFO     Assigning Atom Properties
                        INFO     Atom Assignment Finished
                        INFO     Time: 0.01
                        INFO     Beginning ELF Analysis
                        INFO     Locating Bifurcations
    2026-03-10 13:38:37 INFO     Time: 0.69
                        INFO     Finding contained atoms
    2026-03-10 13:38:38 INFO     Time: 0.68
                        INFO     Beginning Bader Algorithm Using 'weight' Method
                        INFO     Initializing Labels
                        INFO     Initialization Complete
                        INFO     Time: 0.08
                        INFO     Sorting Reference Data
                        INFO     Assigning Charges and Volumes
                        INFO     Combining Low-Persistence Basins
    2026-03-10 13:38:40 INFO     Refining Maxima
                        INFO     Bader Algorithm Complete
                        INFO     Time: 1.91
                        INFO     Marking atomic shells
                        INFO     Marking covalent features
                        INFO     Marking lone-pair features
                        INFO     Calculating atomic radii
                        INFO     Marking multi-centered and bare electron features
                        INFO     Finished labeling ELF
                        INFO     Finished voxel assignment
                        INFO     Finding electride dimensionality cutoffs
                        INFO     Calculating dimensionality at 0 ELF
    2026-03-10 13:38:41 INFO     Max electride dimensionality: 2
                        INFO     Refining cutoff for dimension 1
    100%|███████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 53.60it/s]
                        INFO     Refining cutoff for dimension 0
    100%|███████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 54.06it/s]
    ```

    A summary of all properties will be written to `badelf.json`. See our [example files](https://github.com/SWeav02/baderkit/releases/tag/0.9.0 ). for an example.

    Additional arguments and options such as those for printing output files or using different
    algorithms can be viewed by running the help command.
    ```bash
    baderkit badelf --help
    ```

=== "Python"

    1. Import the `Badelf` class.

        ```python
        from baderkit.elf_analysis.badelf import Badelf
        ```

    2. Use the `Badelf` class' `from_vasp` method to read a `CHGCAR` and `ELFCAR` file.

        ```python
        # instantiate the class
        badelf = Badelf.from_vasp("path/to/chargefile", "path/to/elffile")
        ```

    3. To run the analysis, we can call any class property. Try getting a complete summary in dictionary format.
        ```python
        results = badelf.to_dict()
        ```
    You should see an output similar to this:
        ```bash
                            INFO     Beginning voxel assignment
                            INFO     Beginning Bader Algorithm Using 'weight' Method
        2026-03-10 13:42:36 INFO     Initializing Labels
                            INFO     Initialization Complete
                            INFO     Time: 0.38
                            INFO     Sorting Reference Data
                            INFO     Assigning Charges and Volumes
                            INFO     Combining Low-Persistence Basins
        2026-03-10 13:42:38 INFO     Refining Maxima
                            INFO     Bader Algorithm Complete
                            INFO     Time: 3.08
                            INFO     Assigning Atom Properties
                            INFO     Atom Assignment Finished
                            INFO     Time: 0.0
                            INFO     Beginning ELF Analysis
                            INFO     Locating Bifurcations
        2026-03-10 13:42:39 INFO     Time: 0.68
                            INFO     Finding contained atoms
                            INFO     Time: 0.7
                            INFO     Beginning Bader Algorithm Using 'weight' Method
        2026-03-10 13:42:40 INFO     Initializing Labels
                            INFO     Initialization Complete
                            INFO     Time: 0.08
                            INFO     Sorting Reference Data
                            INFO     Assigning Charges and Volumes
                            INFO     Combining Low-Persistence Basins
        2026-03-10 13:42:41 INFO     Refining Maxima
                            INFO     Bader Algorithm Complete
                            INFO     Time: 1.97
        2026-03-10 13:42:42 INFO     Marking atomic shells
                            INFO     Marking covalent features
                            INFO     Marking lone-pair features
                            INFO     Calculating atomic radii
                            INFO     Marking multi-centered and bare electron features
                            INFO     Finished labeling ELF
                            INFO     Finished voxel assignment
                            INFO     Finding electride dimensionality cutoffs
                            INFO     Calculating dimensionality at 0 ELF
                            INFO     Max electride dimensionality: 2
                            INFO     Refining cutoff for dimension 1
        100%|██████████| 15/15 [00:00<00:00, 55.09it/s]
        2026-03-10 13:42:43 INFO     Refining cutoff for dimension 0
        100%|██████████| 15/15 [00:00<00:00, 48.17it/s]
        ```

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
        Currently the GUI App only supports Bader analysis.

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
        from baderkit.elf_analysis import SpinBadelf
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

## Labeling the ELF

In Bader analysis there are typically maxima only at the center of each atom making it quite easy to assign each basin. In the ELF this becomes much more complicated as a single basin may belong to multiple atoms (e.g. covalent bonds) or even no atoms (e.g. electrides). Because of this, it is necessary to label each basin prior to assigning atomic charges to know how each basin should be split. This process is handled by the [ElfLabeler](../../elf_labeler/usage) class. Keyword arguments can be fed to the `ElfLabeler` when running BadELF through python by providing them in dictionary format to the `elf_labeler` argument.

---

## Warnings for VASP

### Low Valence Pseudopotentials

VASP only includes the valence electrons in the ELFCAR. This means that for pseudopotentials with relatively few valence electrons, it is possible for the ELF to be zero at atom centers. We recommend using VASP's [GW potentials](https://www.vasp.at/wiki/Available_pseudopotentials), with additional valence electrons.

### Mismatched Grids

By default, VASP writes the CHGCAR and ELFCAR to different grid shapes (the "fine" and standard FFT meshes). The `Badelf` and `ElfLabeler` classes require the grid sizes match. This can be achieved by setting the `NGX(YZ)` and `NGX(YZ)F` tags in the INCAR to match. Alternatively, one can set the `PREC` tag to `single`, but this should be done with caution as it generally lowers the quality of the calculation unless the `ENCUT` and `NGX(YZ)` tags are set as well.

### Atomic Position Precision

For BadELF alorithms involving planes (i.e. `badelf` and `voronelf`), results can change significantly with very small differences in atom position. VASP writes atomic positions in the CHGCAR and ELFCAR with limited precision, sometimes much lower than the values in the POSCAR/CONTCAR. To help with this, we provide an option to override the crystal structure when reading in the CHGCAR/ELFCAR:

```python
from baderkit.elf_analysis import Badelf

badelf = Badelf.from_vasp("path/to/chargefile", "path/to/elffile", poscar_file="path/to/poscar")
```