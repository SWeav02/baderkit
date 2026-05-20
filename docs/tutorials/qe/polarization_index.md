The `BasinOverlap` class is the basis for most other ELF analysis methods. A key tool is the ability to calculate the degree of polarization in each local basin. To demonstrate this, we will compare the bonds in P3<sub>1</sub>21 (α-quartz) SiO<sub>2</sub> and SiSe<sub>2</sub>. We will also demonstrate how this directly corresponds to how the `ElfLabeler` class decides the type of bonding interaction.

## VASP

1. Create folders for SiO<sub>2</sub> and SiSe<sub>2</sub> then add the relavent input files.

    === "SiO<sub>2</sub>"

        ```
        &CONTROL
        calculation = 'scf'
        etot_conv_thr =   9.0000000000d-04
        forc_conv_thr =   1.0000000000d-03
        outdir = './scf/'
        prefix = 'sio2'
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
        nat = 9
        nosym = .false.
        ntyp = 2
        occupations = 'fixed'
        /
        &ELECTRONS
        conv_thr =   3.6000000000d-09
        electron_maxstep = 80
        mixing_beta =   4.0000000000d-01
        /
        ATOMIC_SPECIES
        O      15.9994 O.pbesol-n-kjpaw_psl.1.0.0.UPF
        Si     28.0855 Si.pbesol-n-kjpaw_psl.1.0.0.UPF
        ATOMIC_POSITIONS crystal
        Si           0.5392609666       0.5392609666       0.0000000000
        Si           1.0000000000       0.4607390334       0.6666666667
        Si           0.4607390334      -0.0000000000       0.3333333333
        O            0.2823785520       0.4073820558       0.7734433510
        O            0.5926179442       0.8749964962       0.1067766843
        O            0.1250035038       0.7176214480       0.4401100176
        O            0.4073820558       0.2823785520       0.2265566490
        O            0.7176214480       0.1250035038       0.5598899824
        O            0.8749964962       0.5926179442       0.8932233157
        K_POINTS automatic
        5 5 4 0 0 0
        CELL_PARAMETERS angstrom
            2.4258960288      -4.2017751758       0.0000000000
            2.4258960288       4.2017751758      -0.0000000000
            0.0000000000       0.0000000000       5.3743607716
        ```

    === "SiSe<sub>2</sub>"
        ```
        &CONTROL
        calculation = 'scf'
        etot_conv_thr =   9.0000000000d-04
        forc_conv_thr =   1.0000000000d-03
        outdir = './scf/'
        prefix = 'sise2'
        pseudo_dir = '.'
        tprnfor = .true.
        tstress = .true.
        verbosity = 'high'
        /
        &SYSTEM
        degauss =   2.7500000000d-02
        ecutrho =   2.4000000000d+02
        ecutwfc =   3.0000000000d+01
        ibrav = 0
        nat = 9
        nosym = .false.
        ntyp = 2
        occupations = 'fixed'
        /
        &ELECTRONS
        conv_thr =   3.6000000000d-09
        electron_maxstep = 80
        mixing_beta =   4.0000000000d-01
        /
        ATOMIC_SPECIES
        Se     78.96 Se.pbesol-dn-kjpaw_psl.1.0.0.UPF
        Si     28.0855 Si.pbesol-n-kjpaw_psl.1.0.0.UPF
        ATOMIC_POSITIONS crystal
        Si           0.5425008932       0.5425008932       0.0000000000
        Si          -0.0000000000       0.4574991068       0.6666666667
        Si           0.4574991068      -0.0000000000       0.3333333333
        Se           0.3146965009       0.3670459293       0.7360315685
        Se           0.6329540707       0.9476505716       0.0693649019
        Se           0.0523494284       0.6853034991       0.4026982352
        Se           0.3670459293       0.3146965009       0.2639684315
        Se           0.6853034991       0.0523494284       0.5973017648
        Se           0.9476505716       0.6329540707       0.9306350981
        K_POINTS automatic
        4 4 3 0 0 0
        CELL_PARAMETERS angstrom
            3.0736185369      -5.3236634689       0.0000000000
            3.0736185369       5.3236634689      -0.0000000000
            0.0000000000       0.0000000000       7.3581003118
        ```
    Make sure you have appropriate pseudopotentials and point `pseudo_dir` to their location. We copy the pseudopotentials into the active directory so that BaderKit can automatically parse them. For this tutorial, we used PPs generated from [pslibrary v1.0.0](https://dalcorso.github.io/pslibrary/).

2. Run the scf calculation. On our system we use the following command.

    ```
    mpirun -np 12 pw.x -in scf.in
    ```

3. We need the valence charge density,'all-electron' (valence and core) density, and electron localization function (ELF). To generate these, we must run the post-processing package, `pp.x`, once for each file. Here we provide the inputs for both which we named `chg.in`, `tot_chg.in`, and `elf.in` respectively. Add these to the corresponding system directories.

    === "Valence - SiO<sub>2</sub>"
        ```
        &INPUTPP
        prefix = 'sio2',
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

    === "Total - SiO<sub>2</sub>"
        ```
        &INPUTPP
        prefix = 'sio2',
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

    === "ELF - SiO<sub>2</sub>"
        ```
        &INPUTPP
        prefix = 'sio2',
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

    === "Valence - SiSe<sub>2</sub>"
        ```
        &INPUTPP
        prefix = 'sise2',
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

    === "Total - SiSe<sub>2</sub>"
        ```
        &INPUTPP
        prefix = 'sise2',
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

    === "ELF - SiSe<sub>2</sub>"
        ```
        &INPUTPP
        prefix = 'sise2',
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

    2. Import the ElfLabeler class.

        ```Python
        from pathlib import Path
        from baderkit.elf_analysis import ElfLabeler
        ```
    
    3. Note the folders where your QE results are stored and create lists to store the results.
        ```Python
        folders = [
        Path("SiO2"),
        Path("SiSe2")
        ]

        # lists to hold bond information
        bond_types = []
        bond_polarities = []
        ```

    4. Loop over each folder. For each we will generate the total charge density file, find the first basin that is labeled as part of a bond, then extract its polarity index from the BasinOverlap class.

        ```Python
        for folder in folders:

            # load labeler
            labeler = ElfLabeler.from_cube(
                charge_filename=folder / "chg.cube",
                reference_filename=folder / "elf.cube",
                total_charge_filename=folder / "tot_chg.cube",
                )

            # get the first basin corresponding to a bond
            basin_type = None
            basin_idx = None
            for idx, i in enumerate(labeler.basin_types):
                if "bond" in i:
                    basin_type = i
                    basin_idx = idx
                    break

            # get the bond polarity
            bond_polarity = labeler.overlap.polarization_indexes[basin_idx]

            # add bond types and polarities to our lists
            bond_types.append(basin_type)
            bond_polarities.append(bond_polarity)
        ```

    5. Finally, print the results to console.

        ```Python
        for system, bond, polarity in zip(folders, bond_types, bond_polarities):
            # print the polarity and basin type
            print(f"{system.name} Bond Polarity: {polarity} -> {bond}")
        ```
    
        You should see logging information as BaderKit runs, then outputs similar to the following:
        
        ```
        SiO2 Bond Polarity: 0.8609 -> ionic bond
        SiSe2 Bond Polarity: 0.4514 -> covalent bond
        ```
        
        A value of 1.0 is fully polar, while a value of 0.0 is fully non-polar. Our results match the chemical intuition that a larger electronegativity difference results in more ionic character. 
        
        BaderKit by default considers a polarization index > 0.5 to be highly polarized. "Polarized" here refers to belonging to one atom significantly more than any others. Highly polarized features include ionic bonds/shells, atomic cores/shells and lone-pairs. Features with low polarization indices include covalent, metallic, and multi-centered bonds.

=== "Command Line"

    1. If you are using an environment manager, load your baderkit environment. For conda:

        ```Bash
        conda activate baderkit
        ```

    2. In each folder, combine the AECCAR files to generate the total charge density.

        ```Bash
        baderkit sum AECCAR0 AECCAR2
        ```

    3. In each folder, run the overlap and labeler methods.

        ```Bash
        baderkit label CHGCAR ELFCAR -tot CHGCAR_sum
        baderkit overlap CHGCAR ELFCAR -tot CHGCAR_sum
        ```

        You should see logging information printed to the console and once complete a `labeler.json` file and `overlap.json` file will be written which summarizes the results of the calculation.

And that's it! Try playing around with what else the `ElfLabeler` and `BasinOverlap` classes offer.

## Download Resources

Tutorial Script: <a href="/tutorial_scripts/vasp/electrides_vasp.py" download>polarization_index.py</a>

VASP Inputs/Outputs: <a href="https://github.com/SWeav02/baderkit/releases/download/0.10.0/polarization_index.zip" download>polarization_index.zip</a>
