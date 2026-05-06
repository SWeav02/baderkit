This tutorial provides a basic example of calculating oxidation states using Quantum Espresso and BaderKit.

## Quantum Espresso

1. Create an input file for an electronic relaxation. Here we provide an example for the NaCl structure which we named `scf.in` on our system.
    ```
    &CONTROL
    calculation = 'scf',
    prefix = 'nacl',
    outdir = './scf',
    pseudo_dir = '.'
    /

    &SYSTEM
    ibrav = 0,
    nat = 3,
    ntyp = 2,
    ecutwfc = 40.0,
    ecutrho = 320.0,
    /

    &ELECTRONS
    conv_thr = 1.d-6,
    mixing_beta = 0.7
    /

    ATOMIC_SPECIES
    Na  22.990  na_pbe_v1.5.uspp.F.UPF
    Cl  35.453  cl_pbe_v1.4.uspp.F.UPF

    CELL_PARAMETERS angstrom
    3.4220145991671784    0.0000000000000000    1.9757010500000005
    1.1406715330557264    3.2263063045206364    1.9757010500000005
    0.0000000000000000    0.0000000000000000    3.9514021000000001

    ATOMIC_POSITIONS crystal
    Na 0.000000 0.000000 0.000000
    Cl 0.500000 0.500000 0.500000

    K_POINTS automatic
    2 2 2 0 0 0

    ```
    Make sure you have appropriate pseudopotentials in the folder, or point to a directory containing them. We use the standard solid-state pseudopotentials database [SSSP](https://legacy.materialscloud.org/discover/sssp/table/efficiency#sssp-license).

2. Run the scf calculation. On our system we use the following command.

    `mpirun -np 12 pw.x -in scf.in`

3. We need the valence charge density produced by the calculation as well as the 'all-electron' (valence and core) density. To generate these, we must run the post-processing package, `pp.x`, once for each file. Here we provide reasonable inputs for both.
    === "Valence"
        ```
        &INPUTPP
        prefix = 'nacl',
        outdir = './scf/',
        plot_num = 0
        /

        &PLOT
        nfile = 1
        iflag = 3
        output_format = 6
        fileout = 'chg.cube'
        nx = 30
        ny = 30
        nz = 30
        /
        ```

    === "Total"
        ```
        &INPUTPP
        prefix = 'nacl',
        outdir = './scf/',
        plot_num = 21
        /

        &PLOT
        nfile = 1
        iflag = 3
        output_format = 6
        fileout = 'tot_chg.cube'
        nx = 30
        ny = 30
        nz = 30
        /
        ```

4. Run the post-processing on each file to produce the required cube files.
    ```
    mpirun -np 12 pp.x -in chg.in
    mpirun -np 12 pp.x -in tot_chg.in
    ```

    You should now be ready to move over to BaderKit

## BaderKit

=== "Python"

    1. If you would like to follow along, open your preferred IDE in an environment with BaderKit installed. Alternatively, the complete python script from this tutorial is available at the end of this page.

    2. Import the main Bader class.

        ```Python
        from baderkit import Bader
        ```

    3. Create the Bader class instance.

        ```Python
        bader = Bader.from_cube(
            charge_filename="chg.cube",
            total_charge_filename="tot_chg.cube",
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

Tutorial Script: <a href="/tutorial_scripts/vasp/oxidation_states_vasp.py" download>oxidation_states_vasp.py</a>

VASP Inputs/Outputs: <a href="https://github.com/SWeav02/baderkit/releases/download/0.10.0/NaCl.zip" download>NaCl.zip</a>
