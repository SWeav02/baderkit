It is common for systems to have differing ELF topologies in the spin-up and spin-down electron systems. In these cases, it is useful to perform separate analyses on each spin system. Here we use the classic magnetic system of Fe as an example.

## Quantum Espresso

1. Create an input file for the Fe electronic relaxation. Here we provide an example which we named `scf.in` on our system. While you can use any name, we like to use the `.in` suffix for clarity.

    ```
    &CONTROL
    calculation = 'scf'
    etot_conv_thr =   1.0000000000d-04
    forc_conv_thr =   1.0000000000d-03
    outdir = './scf/'
    prefix = 'fe'
    pseudo_dir = '.'
    tprnfor = .true.
    tstress = .true.
    verbosity = 'high'
    /
    &SYSTEM
    degauss =   2.7500000000d-02
    ecutrho =   1.0800000000d+03
    ecutwfc =   9.0000000000d+01
    ibrav = 0
    nat = 1
    nosym = .false.
    nspin = 2
    ntyp = 1
    occupations = 'smearing'
    smearing = 'cold'
    starting_magnetization(1) =   3.1250000000d-01
    /
    &ELECTRONS
    conv_thr =   4.0000000000d-10
    electron_maxstep = 80
    mixing_beta =   4.0000000000d-01
    /
    ATOMIC_SPECIES
    Fe     55.845 Fe.pbesol-spn-kjpaw_psl.0.2.1.UPF
    ATOMIC_POSITIONS crystal
    Fe           0.0000000000       0.0000000000       0.0000000000
    K_POINTS automatic
    11 11 11 0 0 0
    CELL_PARAMETERS angstrom
        -1.4315177495       1.4315177495       1.4315177495
        1.4315177495      -1.4315177495       1.4315177495
        1.4315177495       1.4315177495      -1.4315177495
    ```

    Make sure you have appropriate pseudopotentials and point `pseudo_dir` to their location. We copy the pseudopotentials into the active directory so that BaderKit can automatically parse them. For this tutorial, we pulled from the [Standard Solid-State Pseudopotentials Table](https://legacy.materialscloud.org/discover/sssp/table/precision#sssp-license).

2. Run the scf calculation. On our system we use the following command.

    ```
    mpirun -np 12 pw.x -in scf.in
    ```

3. We need the valence charge density,'all-electron' (valence and core) density, and electron localization function (ELF). To generate these, we must run the post-processing package, `pp.x`, once for each file. Here we provide the inputs for both which we named `chg.in`, `tot_chg.in`, and `elf.in` respectively.

    === "Valence"
        ```
        &INPUTPP
        prefix = 'fe',
        outdir = './scf/',
        plot_num = 0
        /

        &PLOT
        nfile = 1
        iflag = 3
        output_format = 6
        fileout = 'fe.cube'
        /
        ```

    === "Total"
        ```
        &INPUTPP
        prefix = 'fe',
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
        prefix = 'fe',
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

