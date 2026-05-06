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
        electride_structure = badelf.nna_structure
        electrides_per_formula = badelf.nnas_per_reduced_formula
        electride_dimensionality = badelf.nna_dimensionality

        # structure including electride site
        print(f"Electride Structure: {electride_structure}")

        # print electron counts
        print(f"Electron Count: {electrides_per_formula}")

        # print dimensionality
        print(f"Electride Dimensionality: {electride_dimensionality}")
        ```
    
        You should see logging information as BaderKit runs, then outputs similar to the following:
            ```
            Electride Structure: Full Formula (Xmc1 Ca2 N1)
            Reduced Formula: XmcCa2N
            abc   :   6.743135   6.743134   6.743135
            angles:  30.925110  30.925113  30.925113
            pbc   :       True       True       True
            Sites (4)
            #  SP            a         b         c  label
            ---  -----  --------  --------  --------  -------
            0  Ca     0.731317  0.731317  0.731317  Ca
            1  Ca     0.268683  0.268683  0.268683  Ca
            2  N      0         0         0         N
            3  Xmc0+  0.5       0.5       0.5       Xmc
            Electron Count: 1.0358152597
            Electride Dimensionality: 2

            ```

=== "Command Line"

    1. If you are using an environment manager, load your baderkit environment. For conda:

        ```Bash
        conda activate baderkit
        ```

    3. Run the Badelf analysis.

        ```Bash
        baderkit badelf CHGCAR ELFCAR
        ```

        You should see logging information printed to the console and once complete a `badelf.json` file will be written which summarizes the results of the calculation.

And that's it! Try playing around with what else the `Badelf` class offers.

## Download Resources

Tutorial Script: <a href="/tutorial_scripts/vasp/electride_charge.py" download>electride_charge.py</a>

VASP Inputs/Outputs: <a href="https://github.com/SWeav02/baderkit/releases/download/0.10.0/Ca2N.zip" download>Ca2N.zip</a>

## Warnings for VASP

### Low Valence Pseudopotentials

VASP only includes the valence electrons in the ELFCAR. This means that for pseudopotentials with relatively few valence electrons, it is possible for the ELF to be zero at atom centers. We recommend using VASP's [GW potentials](https://www.vasp.at/wiki/Available_pseudopotentials), with additional valence electrons.

### Mismatched Grids

By default, VASP writes the CHGCAR and ELFCAR to different grid shapes (the "fine" and standard FFT meshes). The `Badelf` and `ElfLabeler` classes require the grid sizes match. This can be achieved by setting the `NGX(YZ)` and `NGX(YZ)F` tags in the INCAR to match. Alternatively, one can set the `PREC` tag to `single`, but this should be done with caution as it generally lowers the quality of the calculation unless the `ENCUT` and `NGX(YZ)` tags are set as well.