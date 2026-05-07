This tutorial provides a basic example of calculating oxidation states using Quantum Espresso and BaderKit.

## Quantum Espresso

1. Create an input file for an electronic relaxation. Here we provide an example for the NaCl structure which we named `scf.in` on our system. While you can use any name, we like to use the `.in` suffix for clarity.
    ```
    &CONTROL
    calculation = 'scf'
    etot_conv_thr =   2.0000000000d-04
    forc_conv_thr =   1.0000000000d-03
    outdir = './scf'
    prefix = 'nacl'
    pseudo_dir = '.'
    tprnfor = .true.
    tstress = .true.
    verbosity = 'high'
    /
    &SYSTEM
    degauss =   2.7500000000d-02
    ecutrho =   320.0
    ecutwfc =   70.0
    ibrav = 0
    nat = 2
    nosym = .false.
    ntyp = 2
    occupations = 'fixed'
    /
    &ELECTRONS
    conv_thr =   8.0000000000d-10
    electron_maxstep = 80
    mixing_beta =   4.0000000000d-01
    /
    ATOMIC_SPECIES
    Cl     35.453 Cl.pbesol-n-kjpaw_psl.1.0.0.UPF
    Na     22.98977 Na.pbesol-spn-kjpaw_psl.1.0.0.UPF
    ATOMIC_POSITIONS crystal
    Na           0.0000000000       0.0000000000       0.0000000000
    Cl           0.5000000000       0.5000000000       0.5000000000
    K_POINTS automatic
    7 7 7 0 0 0
    CELL_PARAMETERS angstrom
        3.4220145992       0.0000000000       1.9757010500
        1.1406715331       3.2263063045       1.9757010500
        0.0000000000       0.0000000000       3.9514021000
    ```

    Make sure you have appropriate pseudopotentials and point `pseudo_dir` to their location. We copy the pseudopotentials into the active directory so that BaderKit can automatically parse them. For this tutorial, we used PPs generated from [pslibrary v1.0.0](https://dalcorso.github.io/pslibrary/).

2. Run the scf calculation. On our system we use the following command.

    ```
    mpirun -np 12 pw.x -in scf.in
    ```

3. We need the valence charge density produced by the calculation as well as the 'all-electron' (valence and core) density. To generate these, we must run the post-processing package, `pp.x`, once for each file. Here we provide the inputs for both which we named 'chg.in' and 'tot_chg.in' respectively.

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
        /
        ```

4. Run the post-processing on each file to produce the required cube files.
    ```
    mpirun -np 12 pp.x -in chg.in
    mpirun -np 12 pp.x -in tot_chg.in
    ```
    This should print `.cube` files for the valence and total charge densities.

!!! Tip
    We have also added functionality for XCrySDen's `.xsf` format if you prefer. Note that we currently only parse the first density grid in the file.


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
            `array([ 0.86285438 -0.86280059])`
    
    !!! Tip
        If you get `None` BaderKit likely can't find your pseudopotentials. Made sure they are in the active directory or you point to them using the `pseudopotential_filename` tag.

=== "Command Line"

    1. If you are using an environment manager, load your baderkit environment. For conda:

        ```Bash
        conda activate baderkit
        ```

    2. Run the Bader analysis.

        ```Bash
        baderkit bader chg.cube -ref tot_chg.cube
        ```

        You should see logging information printed to the console and once complete a `bader.json` file will be written which summarizes the results of the calculation.

And that's it! Try playing around with what else the `Bader` class offers.

## Download Resources

Tutorial Script: <a href="/tutorial_scripts/qe/oxidation_states.py" download>oxidation_states.py</a>

VASP Inputs/Outputs: <a href="https://github.com/SWeav02/baderkit/releases/download/0.10.0/NaCl_qe.zip" download>NaCl_qe.zip</a>
