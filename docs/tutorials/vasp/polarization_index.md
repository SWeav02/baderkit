The `BasinOverlap` class is the basis for most other ELF analysis methods. A key tool is the ability to calculate the degree of polarization in each local basin. To demonstrate this, we will compare the bonds in P3<sub>1</sub>21 (α-quartz) SiO<sub>2</sub> and SiSe<sub>2</sub>. We will also demonstrate how this directly corresponds to how the `ElfLabeler` class decides the type of bonding interaction.

## VASP

1. Create folders for SiO<sub>2</sub> and SiSe<sub>2</sub> then add the relavent POSCARs.

    === "SiO<sub>2</sub>"

        ```
        Si3 O6
        1.0000000000000000
            2.9065788033759707   -5.0343421636499297    0.0000000000000000
            2.9065788033759707    5.0343421636499297   -0.0000000000000000
            0.0000000000000000   -0.0000000000000000    6.8388368079686197
        Si   O
            3     6
        Direct
        0.5426351758707367  0.5426351758707367  0.0000000000000000
        0.0000000000000000  0.4573648241292635  0.6666666666666643
        0.4573648241292635  0.0000000000000000  0.3333333333333357
        0.3105338987464629  0.3715918000134620  0.7397430676385232
        0.6284081999865376  0.9389420987330006  0.0730764009718589
        0.0610579012669993  0.6894661012535368  0.4064097343051874
        0.3715918000134621  0.3105338987464629  0.2602569323614768
        0.6894661012535368  0.0610579012669993  0.5935902656948125
        0.9389420987330006  0.6284081999865376  0.9269235990281411
        ```

    === "SiSe<sub>2</sub>"
        ```
        Si3 Se6
        1.0000000000000000
            3.0736185368658857   -5.3236634689372266    0.0000000000000000
            3.0736185368658857    5.3236634689372266   -0.0000000000000000
            0.0000000000000000    0.0000000000000000    7.3581003117850452
        Si   Se
            3     6
        Direct
        0.5425008931935045  0.5425008931935045  0.0000000000000000
        0.0000000000000000  0.4574991068064957  0.6666666666666643
        0.4574991068064957  0.0000000000000000  0.3333333333333357
        0.3146965008777560  0.3670459293023581  0.7360315685352555
        0.6329540706976418  0.9476505715753980  0.0693649018685914
        0.0523494284246021  0.6853034991222440  0.4026982352019201
        0.3670459293023581  0.3146965008777560  0.2639684314647442
        0.6853034991222440  0.0523494284246021  0.5973017647980802
        0.9476505715753980  0.6329540706976418  0.9306350981314088
        ```

2. Create your INCAR files. Below is a minimal example that writes the required CHGCAR, AECCAR, and ELFCAR files. In general, the grid density should be at least 10 pts/Å along each lattice vector for well converged Bader analysis. Since both systems are a similar size, we use the same grid.

    ```
    Global Parameters
    LAECHG = True         # Write AECCAR files
    LELF = True           # Write ELFCAR file
    EDIFF  = 1E-06        # SCF energy convergence, in eV
    ENCUT  = 520

    Grid Size             # Moderately grid density
    NGX    = 60
    NGY    = 60
    NGZ    = 70
    "Fine" Grid Size      # Must Match Standard Grid
    NGXF   = 60
    NGYF   = 60
    NGZF   = 70
    ```

3. Create your `POTCAR`. We cannot provide an example for this as the files are proprietary. We recommend using pseudopotentials with some core electrons such as 'Si_sv' and 'Se_sv' to properly separate the core region in the ELF.

4. Run VASP. Depending on your system how you do this may vary. On our system we use the following command.

    ```
    mpirun -np 12 vasp_std
    ```

## BaderKit

=== "Python"

    1. If you would like to follow along, open your preferred IDE in an environment with BaderKit installed. Alternatively, the complete python script from this tutorial is available at the end of this page.

    2. Import the ElfLabeler class.

        ```Python
        from pathlib import Path
        from baderkit import Grid
        from baderkit.elf_analysis import ElfLabeler
        ```
    
    3. Note the folders where your VASP results are stored and create lists to store the results.
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

            # generate CHGCAR_sum file
            core_grid = Grid.from_vasp(folder / "AECCAR0")
            val_grid = Grid.from_vasp(folder / "AECCAR2")
            total = core_grid.linear_add(val_grid)
            total.write_vasp(folder / "CHGCAR_sum")
            
            # load labeler
            labeler = ElfLabeler.from_vasp(
                charge_filename=folder / "CHGCAR",
                reference_filename=folder / "ELFCAR",
                total_charge_filename=folder / "CHGCAR_sum",
                pseudopotential_filename=folder / "POTCAR"
                )
            
            # get the first basin corresponding to a bond
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
