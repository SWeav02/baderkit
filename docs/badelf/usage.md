## Introduction

The `Badelf` class uses principles from Bader's Quantum Theory of Atoms in Molecules combined with the Electron Localization Function (ELF) to calculate atomic charges. It is primarily designed for calculating oxidation states in electride systems, which was the motivation for the original [work](https://pubs.acs.org/doi/10.1021/jacs.3c10876). For more in-depth analysis of the ELF, particularly in systems with non-nuclear chemical features (e.g. covalent bonds, lone-pairs), the [ElfLabeler](../../elf_labeler/usage) class is more appropriate.

## Basic Use

BadELF can be run through the command line interface or through Python script. Currently only VASP's CHGCAR and ELFCAR files are supported.

By default, BadELF uses the 'zero-flux' algorithm which separates all ELF basins at zero-flux surfaces, following traditional ELF charge analysis. The original [badelf](../methods/#__tabbed_1_3) algorithm also incorporates voronoi-like planes and may be more appropriate in some systems.

Examples of `CHGCAR` and `ELFCAR` files for the Ca2N electride used in this tutorial can be found [here](https://github.com/SWeav02/baderkit/releases/tag/0.9.0 ).

=== "Command Line"

    1. Activate your environment with BaderKit installed. If you are not using an
    environment manager, skip to step 2.
    
        ```bash
        conda activate my_env
        ```
        
    2. Navigate to the directory with your charge density and ELF files.
    
        ```bash
        cd /path/to/directory
        ```
    
    3. Run the BadELF analysis. Replace 'chargefile' and 'elffile' with the names of your charge-density and ELF files.
    
        ```bash
        baderkit badelf chargefile elffile
        ```
    
    You should see an output similar to this:

    ```bash
    2026-03-10 13:38:33 INFO     Loading ELFCAR
                        INFO     Time: 0.1
                        INFO     Data type set as elf from data range
                        INFO     Loading CHGCAR
                        INFO     Time: 0.1
                        INFO     Data type set as charge from data range
                        WARNING  No POTCAR file found in the requested directory. Oxidation states cannot be calculated
                        INFO     Beginning voxel assignment
                        INFO     Beginning Bader Algorithm Using 'weight' Method
    2026-03-10 13:38:34 INFO     Initializing Labels
                        INFO     Initialization Complete
                        INFO     Time: 0.4
                        INFO     Sorting Reference Data
                        INFO     Assigning Charges and Volumes
                        INFO     Combining Low-Persistence Basins
    2026-03-10 13:38:36 INFO     Refining Maxima
                        INFO     Bader Algorithm Complete
                        INFO     Time: 3.45
                        INFO     Assigning Atom Properties
                        INFO     Atom Assignment Finished
                        INFO     Time: 0.01
                        INFO     Beginning ELF Analysis
                        INFO     Locating Bifurcations
    2026-03-10 13:38:37 INFO     Time: 0.69
                        INFO     Finding contained atoms
    2026-03-10 13:38:38 INFO     Time: 0.68
                        INFO     Beginning Bader Algorithm Using 'weight' Method
                        INFO     Initializing Labels
                        INFO     Initialization Complete
                        INFO     Time: 0.08
                        INFO     Sorting Reference Data
                        INFO     Assigning Charges and Volumes
                        INFO     Combining Low-Persistence Basins
    2026-03-10 13:38:40 INFO     Refining Maxima
                        INFO     Bader Algorithm Complete
                        INFO     Time: 1.91
                        INFO     Marking atomic shells
                        INFO     Marking covalent features
                        INFO     Marking lone-pair features
                        INFO     Calculating atomic radii
                        INFO     Marking multi-centered and bare electron features
                        INFO     Finished labeling ELF
                        INFO     Finished voxel assignment
                        INFO     Finding electride dimensionality cutoffs
                        INFO     Calculating dimensionality at 0 ELF
    2026-03-10 13:38:41 INFO     Max electride dimensionality: 2
                        INFO     Refining cutoff for dimension 1
    100%|███████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 53.60it/s]
                        INFO     Refining cutoff for dimension 0
    100%|███████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 54.06it/s]
    ```
    
    A summary of all properties will be written to `badelf.json`. See our [example files](https://github.com/SWeav02/baderkit/releases/tag/0.9.0 ). for an example.
    
    Additional arguments and options such as those for printing output files or using different 
    algorithms can be viewed by running the help command.
    ```bash
    baderkit badelf --help
    ```

=== "Python"
    
    1. Import the `Badelf` class.
    
        ```python
        from baderkit.core import Badelf
        ```
    
    2. Use the `Badelf` class' `from_vasp` method to read a `CHGCAR` and `ELFCAR` file.
    
        ```python
        # instantiate the class
        badelf = Badelf.from_vasp("path/to/chargefile", "path/to/elffile")
        ```
    
    3. To run the analysis, we can call any class property. Try getting a complete summary in dictionary format.
        ```python
        results = badelf.to_dict()
        ```
    You should see an output similar to this:
        ```bash   
                            INFO     Beginning voxel assignment                        
                            INFO     Beginning Bader Algorithm Using 'weight' Method   
        2026-03-10 13:42:36 INFO     Initializing Labels                               
                            INFO     Initialization Complete                           
                            INFO     Time: 0.38                                        
                            INFO     Sorting Reference Data                            
                            INFO     Assigning Charges and Volumes                     
                            INFO     Combining Low-Persistence Basins                  
        2026-03-10 13:42:38 INFO     Refining Maxima                                   
                            INFO     Bader Algorithm Complete                          
                            INFO     Time: 3.08                                        
                            INFO     Assigning Atom Properties                         
                            INFO     Atom Assignment Finished                          
                            INFO     Time: 0.0                                         
                            INFO     Beginning ELF Analysis                            
                            INFO     Locating Bifurcations                             
        2026-03-10 13:42:39 INFO     Time: 0.68                                        
                            INFO     Finding contained atoms                           
                            INFO     Time: 0.7                                         
                            INFO     Beginning Bader Algorithm Using 'weight' Method   
        2026-03-10 13:42:40 INFO     Initializing Labels                               
                            INFO     Initialization Complete                           
                            INFO     Time: 0.08                                        
                            INFO     Sorting Reference Data                            
                            INFO     Assigning Charges and Volumes                     
                            INFO     Combining Low-Persistence Basins                  
        2026-03-10 13:42:41 INFO     Refining Maxima                                   
                            INFO     Bader Algorithm Complete                          
                            INFO     Time: 1.97                                        
        2026-03-10 13:42:42 INFO     Marking atomic shells                             
                            INFO     Marking covalent features                         
                            INFO     Marking lone-pair features                        
                            INFO     Calculating atomic radii                          
                            INFO     Marking multi-centered and bare electron features 
                            INFO     Finished labeling ELF                             
                            INFO     Finished voxel assignment                         
                            INFO     Finding electride dimensionality cutoffs          
                            INFO     Calculating dimensionality at 0 ELF               
                            INFO     Max electride dimensionality: 2                   
                            INFO     Refining cutoff for dimension 1                   
        100%|██████████| 15/15 [00:00<00:00, 55.09it/s]
        2026-03-10 13:42:43 INFO     Refining cutoff for dimension 0                   
        100%|██████████| 15/15 [00:00<00:00, 48.17it/s]
        ```
    
    4. Now try getting individual properties. For more details on each property,
    see the [API reference](../api_reference/core/badelf).
        ```python
        atom_charges = badelf.atom_charges # Total atom charges
        atom_labels = badelf.atom_labels # Atom assignments for each point in the grid
        maxima_coords = badelf.basin_maxima_frac # Frac coordinates of each attractor
        ```
    
    5. BaderKit also provides convenience methods for writing results to file. First,
    let's write a summary of the full analysis.
    
        ```python
        badelf.write_json()
        ```
    
    6. Now let's write the volume assigned to one of our atoms.
    
        ```python
        badelf.write_atom_volumes(atom_indices = [0])
        ```
    
    !!! Tip
        After creating a `Badelf` class object, it doesn't matter what order
        you call properties, summaries, or write methods in. BaderKit calculates
        properties/results only when they are needed and caches them.

=== "GUI App"

    !!! Warning
        Currently the GUI App only supports Bader analysis.

---

## Spin-Dependent Calculations

BaderKit provides a convenience class for performing `BadELF` on the spin-up and spin-down ELF separately. The combined results are also calculated by taking either the average or sum of the respective property.

=== "Command Line"

    Run the command with the `--spin` tag.

    ```bash
    baderkit badelf chargefile elffile --spin
    ```

    The results for each spin system are then written to separate files.

=== "Python"
    
    1. Import the `SpinBadelf` class and read in your spin-dependent files.
    
        ```python
        from baderkit.core import SpinBadelf
        # instantiate the class
        badelf = SpinBadelf.from_vasp("path/to/chargefile", "path/to/elffile")
        ```
    
    2. Get the separate results for the spin-up and spin-down systems.

        ```python
        spin_up = badelf.badelf_up
        spin_down = badelf.badelf_down
        ```
    
    3. View properties separately or combined.

        ```python
        up_charges = spin_up.charges
        down_charges = spin_down.charges
        total_charges = badelf.charges
        ```

---

## Labeling the ELF

In Bader analysis there are typically maxima only at the center of each atom making it quite easy to assign each basin. In the ELF this becomes much more complicated as a single basin may belong to multiple atoms (e.g. covalent bonds) or even no atoms (e.g. electrides). Because of this, it is necessary to label each basin prior to assigning atomic charges to know how each basin should be split. This process is handled by the [ElfLabeler](../../elf_labeler/usage) class. Keyword arguments can be fed to the `ElfLabeler` when running BadELF through python by providing them in dictionary format to the `elf_labeler` argument.

---

## Warnings for VASP

### Low Valence Pseudopotentials

VASP only includes the valence electrons in the ELFCAR. This means that for pseudopotentials with relatively few valence electrons, it is possible for the ELF to be zero at atom centers. We recommend using VASP's [GW potentials](https://www.vasp.at/wiki/Available_pseudopotentials), with additional valence electrons.

### Mismatched Grids

By default, VASP writes the CHGCAR and ELFCAR to different grid shapes (the "fine" and standard FFT meshes). The `Badelf` and `ElfLabeler` classes require the grid sizes match. This can be achieved by setting the `NGX(YZ)` and `NGX(YZ)F` tags in the INCAR to match. Alternatively, one can set the `PREC` tag to `single`, but this should be done with caution as it generally lowers the quality of the calculation unless the `ENCUT` and `NGX(YZ)` tags are set as well.

### Atomic Position Precision

For BadELF alorithms involving planes (i.e. `badelf` and `voronelf`), results can change significantly with very small differences in atom position. VASP writes atomic positions in the CHGCAR and ELFCAR with limited precision, sometimes much lower than the values in the POSCAR/CONTCAR. To help with this, we provide an option to override the crystal structure when reading in the CHGCAR/ELFCAR:

```python
from baderkit.core import Badelf

badelf = Badelf.from_vasp("path/to/chargefile", "path/to/elffile", poscar_file="path/to/poscar")
```
