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
3. We need the valence charge density and electron localization function for each spin. Unfortunately, Quantum ESPRESSO does not provide a method for spin-polarized ELF. Instead, BaderKit provides a convenience method, based on QE's `elf.f90` module, that calculates the ELF from the charge density and kinetic energy density for each spin. To generate these we must run the post-processing package, `pp.x` for each file. Here we provide the inputs for each required file which we name `chg_up.in`, `chg_down.in`, `kin_up.in` and `kin_down.in`.

    === "Charge Up"
        ```
        &INPUTPP
        prefix = 'fe',
        outdir = './scf/',
        plot_num = 0,
        spin_component = 1
        /

        &PLOT
        nfile = 1
        iflag = 3
        output_format = 5
        fileout = 'chg_up.xsf'
        /
        ```

    === "Charge Down"
        ```
        &INPUTPP
        prefix = 'fe',
        outdir = './scf/',
        plot_num = 0,
        spin_component = 2
        /

        &PLOT
        nfile = 1
        iflag = 3
        output_format = 5
        fileout = 'chg_down.xsf'
        /
        ```

    === "KED Up"
        ```
        &INPUTPP
        prefix = 'fe',
        outdir = './scf/',
        plot_num = 22,
        spin_component = 1
        /

        &PLOT
        nfile = 1
        iflag = 3
        output_format = 5
        fileout = 'kin_up.xsf'
        /
        ```

    === "KED Down"
        ```
        &INPUTPP
        prefix = 'fe',
        outdir = './scf/',
        plot_num = 0,
        spin_component = 2
        /

        &PLOT
        nfile = 1
        iflag = 3
        output_format = 5
        fileout = 'kin_down.xsf'
        /
        ```

4. Run the post-processing on each file to produce the required cube files.
    ```
    mpirun -np 12 pp.x -in chg_up.in
    mpirun -np 12 pp.x -in chg_down.in
    mpirun -np 12 pp.x -in kin_up.in
    mpirun -np 12 pp.x -in kin_down.in
    ```
    This should write the required `.xsf` files.

!!! Note
    We use the XCrySDen .xsf file format in this tutorial which writes on the exact FFT grid used by Quantum ESPRESSO without interpolation. We have found that ELF calculations using this format more closely match the ELF calculated by QE for total electron calculations. You can still use the .cube format if you prefer, but the exact ELF value may slightly differ.

## BaderKit

=== "Python"

    1. If you would like to follow along, open your preferred IDE in an environment with BaderKit installed. Alternatively, the complete python script from this tutorial is available at the end of this page.

    2. Import the Grid and Badelf class as well as the ELF helper function.

        ```Python
        from baderkit import Grid
        from baderkit.elf_analysis import Badelf
        from baderkit.global_numba.elf_construction import compute_elf_from_grid

        ```

    3. Load the spin polarized grids

        ```python
        charge_up = Grid.from_xsf("chg_up.xsf")
        charge_down = Grid.from_xsf("chg_down.xsf")
        ked_up = Grid.from_xsf("kin_up.xsf")
        ked_down = Grid.from_xsf("kin_down.xsf")

        ```
    
    4. Calculate the ELF.

        ```python
        elf_up = compute_elf_from_grid(
            charge_grid=charge_up,
            ked_grid=ked_up,
            spin=True,
            )
        elf_down = compute_elf_from_grid(
            charge_grid=charge_down,
            ked_grid=ked_down,
            spin=True,
            )
        ```

    5. Create the polarized BadELF objects.

        ```python
        badelf_up = Badelf(
            charge_grid=charge_up,
            reference_grid=elf_up,
            )
        badelf_down = Badelf(
            charge_grid=charge_down,
            reference_grid=elf_down,
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
            Spin-up metal bond population: 0.946546023
            Spin-down metal bond population: 1.2712379158
            ```

=== "Command Line"

    1. If you are using an environment manager, load your baderkit environment. For conda:

        ```Bash
        conda activate baderkit
        ```

    2. Calculate the polarized ELF
        ```bash
        baderkit make-elf chg_up.xsf kin_up.xsf -s -o elf_up.xsf
        baderkit make-elf chg_down.xsf kin_down.xsf -s -o elf_down.xsf
        ```

    3. Run the Badelf analysis on each system separately. Make sure to change the name of the output .json file to avoid overwriting it.

        ```bash
        baderkit badelf chg_up.xsf elf_up.xsf
        mv badelf.json badelf_up.json
        baderkit badelf chg_down.xsf elf_down.xsf
        mv badelf.json badelf_down.json
        ```


And that's it! Try playing around with what else the `Badelf` class offers.

## Download Resources

Tutorial Script: <a href="/tutorial_scripts/qe/spin_badelf.py" download>spin_badelf.py</a>

VASP Inputs/Outputs: <a href="https://github.com/SWeav02/baderkit/releases/download/0.10.0/Fe_qe.zip" download>Fe_qe.zip</a>

