It is common for systems to have differing ELF topologies in the spin-up and spin-down electron systems. In these cases, it is useful to perform separate analyses on each spin system. Here we use the classic magnetic system of Fe as an example.

## VASP

1. Create your Fe POSCAR file.

    ```
    Fe1
    1.0
    -1.4315177494749578    1.4315177494749580    1.4315177494749580
    1.4315177494749578   -1.4315177494749580    1.4315177494749580
    1.4315177494749582    1.4315177494749580   -1.4315177494749578
    Fe
    1
    direct
    0.0000000000000000    0.0000000000000000    0.0000000000000000 Fe
    ```

2. Create your INCAR file. Below is a minimal example that writes the required CHGCAR and ELFCAR files. In general, the grid density should be at least 10 pts/Å along each lattice vector for well converged Bader analysis.

    ```
    Global Parameters
    ISPIN = 2             # Spin-polarized
    LELF = True           # Write ELFCAR file
    EDIFF  = 1E-06        # SCF energy convergence, in eV
    ENCUT  = 520

    Grid Size             # Moderately grid density
    NGX    = 30
    NGY    = 30
    NGZ    = 30
    "Fine" Grid Size      # Must Match Standard Grid
    NGXF   = 30
    NGYF   = 30
    NGZF   = 30
    ```

3. Create your `POTCAR`. We cannot provide an example for this as the files are proprietary. We recommend using a POTCAR with extra valence electrons such as 'Fe_sv' to ensure the ELF contains some core electrons.

4. Run VASP. Depending on your system how you do this may vary. On our system we use the following command.

    ```
    mpirun -np 12 vasp_std
    ```

## BaderKit

=== "Python"

    1. If you would like to follow along, open your preferred IDE in an environment with BaderKit installed. Alternatively, the complete python script from this tutorial is available at the end of this page.

    2. Import the Grid and Badelf class

        ```Python
        from baderkit import Grid
        from baderkit.elf_analysis import Badelf
        ```

    3. Load the spin polarized grids

        ```python
        polarized_charge = Grid.from_vasp("CHGCAR", total_only=False)
        polarized_elf = Grid.from_vasp("ELFCAR", total_only=False)
        ```
    
    4. Split the polarized grids into their spin-up and spin-down components

        ```python
        charge_up, charge_down = polarized_charge.split_to_spin()
        elf_up, elf_down = polarized_elf.split_to_spin()
        ```

    5. Create the polarized BadELF objects.

        ```python
        badelf_up = Badelf(
            charge_grid=charge_up,
            reference_grid=elf_up,
            valence_counts={
                "Fe": 16
                }
            )
        badelf_down = Badelf(
            charge_grid=charge_down,
            reference_grid=elf_down,
            valence_counts={
                "Fe": 16
                }
            )
        ```

    5. Finally, print some useful information to the console.
        ```Python
        metal_bonds_up = badelf_up.nnas_per_reduced_formula
        metal_bonds_down = badelf_down.nnas_per_reduced_formula

        print(f"Spin-up metal bond population: {metal_bonds_up}")
        print(f"Spin-down metal bond population: {metal_bonds_down}")
        ```
    
        You should see logging information as BaderKit runs, then outputs similar to the following:
            ```
            Spin-up metal bond population: 1.336115618
            Spin-down metal bond population: 0.9357502685
            ```

=== "Command Line"

    1. If you are using an environment manager, load your baderkit environment. For conda:

        ```Bash
        conda activate baderkit
        ```

    2. Split the charge density and ELF into spin-up and spin-down systems
        ```bash
        baderkit split CHGCAR
        baderkit split ELFCAR
        ```

    3. Run the Badelf analysis on each system separately. Make sure to change the name of the output .json file to avoid overwriting it.

        ```bash
        baderkit badelf CHGCAR_up ELFCAR_up
        mv badelf.json badelf_up.json
        baderkit badelf CHGCAR_down ELFCAR_down
        mv badelf.json badelf_down.json
        ```


And that's it! Try playing around with what else the `Badelf` class offers.

## Download Resources

Tutorial Script: <a href="/tutorial_scripts/vasp/spin_badelf_vasp.py" download>spin_badelf_vasp.py</a>

VASP Inputs/Outputs: <a href="https://github.com/SWeav02/baderkit/releases/download/0.10.0/Fe.zip" download>Fe.zip</a>

