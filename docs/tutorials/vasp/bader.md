This tutorial provides a basic example of calculating oxidation states using VASP and BaderKit.

## VASP

1. Create your POSCAR file. As an example, we used the following NaCl structure.

    ```
    Na1 Cl1
    1.0
    3.4220145991671784    0.0000000000000000    1.9757010500000005
    1.1406715330557264    3.2263063045206364    1.9757010500000005
    0.0000000000000000    0.0000000000000000    3.9514021000000001
    Na Cl
    1 1
    direct
    0.0000000000000000    0.0000000000000000    0.0000000000000000 Na+
    0.5000000000000000    0.5000000000000000    0.5000000000000000 Cl-
    ```
2. Create your INCAR file. Below is a minimal example that writes a reasonable charge density for the above NaCl structure. In general, the grid density should be at least 10 pts/Å along each lattice vector for well converged Bader analysis.

    ```
    Global Parameters
    ISTART =  1            (Read existing wavefunction, if there)
    ISPIN  =  1            (Non-Spin polarised DFT)
    LREAL  = AUTO          (Projection operators: automatic)
    LWAVE  = .TRUE.        (Write WAVECAR or not)
    LCHARG = .TRUE.        (Write CHGCAR or not)

    Static Calculation
    ISMEAR =  0            (gaussian smearing method)
    SIGMA  =  0.05         (please check the width of the smearing)
    NELM   =  60           (Max electronic SCF steps)
    EDIFF  =  1E-06        (SCF energy convergence, in eV)

    LAECHG = .TRUE.        (Activate AECCAR files)
    LELF   = .TRUE.        (Activate ELF)

    Grid Size              (Must set fine grid to be the same)
    NGX    = 30
    NGY    = 30
    NGZ    = 30
    NGXF    = 30
    NGYF    = 30
    NGZF    = 30
    ```

3. Create your `POTCAR`. We cannot provide an example for this as the files are proprietary.

4. Run VASP. Depending on your system how you do this may vary. On our system we use the following command.

    ```
    mpirun -np 12 vasp_std
    ```

## BaderKit

=== "Python"

    1. If you would like to follow along, open your preferred IDE in an environment with BaderKit installed. Alternatively, the complete python script from this tutorial is available at the end of this page.

    2. Import the Grid utility class and main Bader class.

        ```Python
        from baderkit import Grid, Bader
        ```

    3. We recommend using the reconstructed total charge density as a reference for Bader partitioning when possible. In VASP we can construct this from the AECCAR files.

        ```Python
        core_grid = Grid.from_vasp("AECCAR0")
        val_grid = Grid.from_vasp("AECCAR2")
        total = core_grid.linear_add(val_grid)
        total.write_vasp("CHGCAR_sum")
        ```

    3. Now create the Bader class instance.

        ```Python
        charge = Grid.from_vasp("CHGCAR")

        bader = Bader(
            charge_filename="CHGCAR", 
            total_charge_filename="CHGCAR_sum",
            pseudopotential_filename="POTCAR"
        )
        ```

    4. Finally, we can print the oxidation states to console.
        ```Python
        print(bader.oxidation_states)
        ```
    
        You should see logging information as BaderKit runs, then the oxidation states of each atom in the structure:
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

Tutorial Script: <a href="/tutorial_scripts/vasp/oxidation_states.py" download>oxidation_states.py</a>

VASP Inputs/Outputs: <a href="https://github.com/SWeav02/baderkit/releases/download/0.10.0/NaCl.zip" download>NaCl.zip</a>
