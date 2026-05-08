The `ElfLabeler` class can be used to automatically identify various chemical features in a system. This can be useful for a variety of automation task. Here we demonstrate its use for locating the covalent bonds in the the CO<sub>2</sub> molecules of dry ice. We also demonstrate how to calculate the exact and formal bond-order of CO<sub>2</sub> covalent bonds.

## Quantum Espresso

1. Create an input file for the CO<sub>2</sub> electronic relaxation. Here we provide an example which we named `scf.in` on our system. While you can use any name, we like to use the `.in` suffix for clarity.

    ```
    &CONTROL
    calculation = 'scf'
    etot_conv_thr =   3.0000000000d-04
    forc_conv_thr =   1.0000000000d-03
    outdir = './scf/'
    prefix = 'co2'
    pseudo_dir = '.'
    tprnfor = .true.
    tstress = .true.
    verbosity = 'high'
    /
    &SYSTEM
    degauss =   2.7500000000d-02
    ecutrho =   4.0000000000d+02
    ecutwfc =   5.0000000000d+01
    ibrav = 0
    nat = 3
    nosym = .false.
    ntyp = 2
    occupations = 'fixed'
    /
    &ELECTRONS
    conv_thr =   1.2000000000d-09
    electron_maxstep = 80
    mixing_beta =   4.0000000000d-01
    /
    ATOMIC_SPECIES
    C      12.0107 C.pbesol-n-kjpaw_psl.1.0.0.UPF
    O      15.9994 O.pbesol-n-kjpaw_psl.1.0.0.UPF
    ATOMIC_POSITIONS crystal
    C            0.5000000000       0.5000000000       0.5000000000
    O            0.5000000000       0.5000000000       0.6160000000
    O            0.5000000000       0.5000000000       0.3840000000
    K_POINTS automatic
    3 3 3 0 0 0
    CELL_PARAMETERS angstrom
        10.0000000000       0.0000000000       0.0000000000
        0.0000000000      10.0000000000       0.0000000000
        0.0000000000       0.0000000000      10.0000000000
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
        prefix = 'co2',
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
        prefix = 'co2',
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
        prefix = 'co2',
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
        from baderkit.elf_analysis import ElfLabeler
        ```

    3. Now create the ElfLabeler class instance.

        ```Python
        labeler = ElfLabeler.from_vasp(
            charge_filename="chg.cube",
            reference_filename="elf.cube",
            total_charge_filename="tot_chg.cube",
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

    3. Run the ElfLabeler analysis.

        ```Bash
        baderkit label chg.cube elf.cube -tot tot_chg.cube
        ```

        You should see logging information printed to the console and once complete a `labeler.json` file will be written which summarizes the results of the calculation.

And that's it! Try playing around with what else the `ElfLabeler` class offers.

