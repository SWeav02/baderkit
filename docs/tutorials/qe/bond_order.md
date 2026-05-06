The `ElfLabeler` class can be used to automatically identify various chemical features in a system. This can be useful for a variety of automation task. Here we demonstrate its use for locating the covalent bonds in the the CO<sub>2</sub> molecule. We also demonstrate how to calculate the exact and formal bond-order of the bond.

## VASP

1. Create the CO<sub>2</sub> POSCAR file.

    ```
    CO2 in 10x10x10 cubic box
    1.0
    10.000000 0.000000 0.000000
    0.000000 10.000000 0.000000
    0.000000 0.000000 10.000000
    C O
    1 2
    Direct
    0.500000 0.500000 0.500000
    0.500000 0.500000 0.616000
    0.500000 0.500000 0.384000

    ```

2. Create your INCAR file. Below is a minimal example that writes the required CHGCAR, AECCAR, and ELFCAR files. In general, the grid density should be at least 10 pts/Å along each lattice vector for well converged Bader analysis.

    ```
    Global Parameters
    LAECHG = True         # Write AECCAR files
    LELF = True           # Write ELFCAR file
    EDIFF  = 1E-06        # SCF energy convergence, in eV
    ENCUT  = 520

    Grid Size             # Moderately grid density
    NGX    = 100
    NGY    = 100
    NGZ    = 100
    "Fine" Grid Size      # Must Match Standard Grid
    NGXF   = 100
    NGYF   = 100
    NGZF   = 100
    ```

3. Create your `POTCAR`. We cannot provide an example for this as the files are proprietary.

4. Run VASP. Depending on your system how you do this may vary. On our system we use the following command.

    ```
    mpirun -np 12 vasp_std
    ```

## BaderKit

=== "Python"

    1. If you would like to follow along, open your preferred IDE in an environment with BaderKit installed. Alternatively, the complete python script from this tutorial is available at the end of this page.

    2. Import the ElfLabeler class

        ```Python
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

    5. Get the index corresponding to the first covalent bond between C and O and print the bond-order to the console.
        ```Python
        covalent_idx = features.index("covalent bond")
        print(f"Bond Order: {charges[covalent_idx]/2}")
        ```
    
        You should see logging information as BaderKit runs, then outputs similar to the following:
        
        ```
        Bond Order: 1.3742746666
        ```
        
        Note that the BO is lower than the formal value of 2. This is because the CO bond is partially ionic in nature, with Oxygen taking the dominant share.

=== "Command Line"

    1. If you are using an environment manager, load your baderkit environment. For conda:

        ```Bash
        conda activate baderkit
        ```

    2. We recommend using the reconstructed total charge density as a reference for Bader partitioning when possible. In VASP we can construct this from the AECCAR files.

        ```Bash
        baderkit sum AECCAR0 AECCAR2
        ```

    3. Run the labeler analysis.

        ```Bash
        baderkit label CHGCAR ELFCAR -tot CHGCAR_sum
        ```

        You should see logging information printed to the console and once complete a `labeler.json` file will be written which summarizes the results of the calculation.

And that's it! Try playing around with what else the `ElfLabeler` class offers.

## Download Resources

Tutorial Script: <a href="/tutorial_scripts/vasp/electrides_vasp.py" download>bond_order.py</a>

VASP Inputs/Outputs: <a href="https://github.com/SWeav02/baderkit/releases/download/0.10.0/CO2.zip" download>CO2.zip</a>
