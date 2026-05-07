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

