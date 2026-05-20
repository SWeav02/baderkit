The `Badelf` class includes several methods for calculating oxidation states in materials combining principles from Bader's Quantum Theory of Atoms in Molecules with the Electron Localization Function (ELF). This is useful for calculating oxidation states in systems with localized electrons that sit far from atomic sites such as in [electride systems](https://pubs.acs.org/doi/10.1021/jacs.3c10876). This tutorial walks through this process for the common Ca<sub>2</sub>N electride using Quantum Espresso.

## Quantum Espresso

1. Create an input file for the Ca<sub>2</sub>N electronic relaxation. Here we provide an example which we named `scf.in` on our system. While you can use any name, we like to use the `.in` suffix for clarity.

    ```
    &CONTROL
    calculation = 'scf'
    etot_conv_thr =   3.0000000000d-04
    forc_conv_thr =   1.0000000000d-03
    outdir = './scf/'
    prefix = 'ca2n'
    pseudo_dir = '.'
    tprnfor = .true.
    tstress = .true.
    verbosity = 'high'
    /
    &SYSTEM
    degauss =   2.7500000000d-02
    ecutrho =   370
    ecutwfc =   50
    ibrav = 0
    nat = 3
    nosym = .false.
    ntyp = 2
    occupations = 'smearing'
    smearing = 'cold'
    /
    &ELECTRONS
    conv_thr =   1.2000000000d-09
    electron_maxstep = 80
    mixing_beta =   4.0000000000d-01
    /
    ATOMIC_SPECIES
    Ca     40.078 Ca.pbesol-spn-kjpaw_psl.1.0.0.UPF
    N      14.0067 N.pbesol-n-kjpaw_psl.1.0.0.UPF
    ATOMIC_POSITIONS crystal
    Ca           0.7313170000       0.7313170000       0.7313170000
    Ca           0.2686830000       0.2686830000       0.2686830000
    N            0.0000000000       0.0000000000       0.0000000000
    K_POINTS automatic
    7 7 7 0 0 0
    CELL_PARAMETERS angstrom
        3.5370740000       0.0511330000       5.7407630000
        1.6651930000       3.1209990000       5.7407630000
        0.0838580000       0.0511330000       6.7424200000
    ```

    Make sure you have appropriate pseudopotentials and point `pseudo_dir` to their location. We copy the pseudopotentials into the active directory so that BaderKit can automatically parse them. For this tutorial, we used PPs generated from [pslibrary v1.0.0](https://dalcorso.github.io/pslibrary/).

2. Run the scf calculation. On our system we use the following command.

    ```
    mpirun -np 12 pw.x -in scf.in
    ```

3. We need the valence charge density,'all-electron' (valence and core) density, and electron localization function (ELF). To generate these, we must run the post-processing package, `pp.x`, once for each file. Here we provide the inputs for both which we named `chg.in`, `tot_chg.in`, and `elf.in` respectively.

    === "Valence"
        ```
        &INPUTPP
        prefix = 'ca2n',
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
        prefix = 'ca2n',
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

    === "ELF"
        ```
        &INPUTPP
        prefix = 'ca2n',
        outdir = './scf/',
        plot_num = 8
        /

        &PLOT
        nfile = 1
        iflag = 3
        output_format = 6
        fileout = 'elf.cube'
        /
        ```


4. Run the post-processing on each file to produce the required cube files.
    ```
    mpirun -np 12 pp.x -in chg.in
    mpirun -np 12 pp.x -in tot_chg.in
    mpirun -np 12 pp.x -in elf.in
    ```
    This should print the required `.cube` files.

!!! Tip
    We have also added functionality for XCrySDen's `.xsf` format if you prefer. Note that we currently only parse the first density grid in the file.

## BaderKit

=== "Python"

    1. If you would like to follow along, open your preferred IDE in an environment with BaderKit installed. Alternatively, the complete python script from this tutorial is available at the end of this page.

    2. Import the Badelf class

        ```Python
        from baderkit.elf_analysis import Badelf
        ```

    3. Now create the Badelf class instance.

        ```Python
        badelf = Badelf.from_cube(
            charge_filename="chg.cube",
            total_charge_filename="tot_chg.cube",
            reference_filename="elf.cube",
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
            abc   :   6.743120   6.743122   6.743139
            angles:  30.924969  30.924958  30.925132
            pbc   :       True       True       True
            Sites (4)
            #  SP            a         b         c  label
            ---  -----  --------  --------  --------  -------
            0  Ca     0.731324  0.731313  0.731311  Ca
            1  Ca     0.268686  0.268682  0.268681  Ca
            2  N      1e-05     0.999995  0.999992  N
            3  Xmc0+  0.5       0.5       0.5       Xmc
            Electron Count: 0.9834565223
            Electride Dimensionality: 2
            ```

=== "Command Line"

    1. If you are using an environment manager, load your baderkit environment. For conda:

        ```Bash
        conda activate baderkit
        ```

    3. Run the Badelf analysis.

        ```Bash
        baderkit badelf chg.cube elf.cube -tot tot_chg.cube
        ```

        You should see logging information printed to the console and once complete a `badelf.json` file will be written which summarizes the results of the calculation.

And that's it! Try playing around with what else the `Badelf` class offers.

## Download Resources

Tutorial Script: <a href="/tutorial_scripts/qe/electride_charge.py" download>electride_charge.py</a>

VASP Inputs/Outputs: <a href="https://github.com/SWeav02/baderkit/releases/download/0.10.0/Ca2N_qe.zip" download>Ca2N_qe.zip</a>
