## Introduction

The `Bader` class reproduces the methods for Bader charge analysis available in the 
[Henkelman group's](https://theory.cm.utexas.edu/henkelman/code/bader/) excellent 
Fortran code within the Python ecosystem. It is built on
top of the [PyMatGen](https://pymatgen.org/) package, allowing for
easy extension to other projects. Under the hood, `Bader` runs on [Numba](https://numba.pydata.org/numba-doc/dev/index.html) 
and [NumPy](https://numpy.org/doc/stable/index.html) to parallelize and vectorize
calculations. This allows `Bader` to reach speeds [comparable or faster](../methods/#__tabbed_2_1)
than the original code.

## Basic Use

BaderKit can be used through the command line interface or through
Python script. This page covers only the most simple use case of running Bader charge analysis on a VASP `CHGCAR` or Gaussian `cube` file. For more advance usage, see our [API reference](/baderkit/api_reference/core/bader) and [Examples](/baderkit/examples) pages.

The files used in this tutorial can be downloaded [here](https://github.com/SWeav02/baderkit/releases/tag/0.9.0).


=== "Command Line"

    1. Activate your environment with BaderKit installed. If you are not using an
    environment manager, skip to step 2.
    
        ```bash
        conda activate my_env
        ```
        
    2. Navigate to the directory with your charge density file.
    
        ```bash
        cd /path/to/directory
        ```
    
    3. Run the bader analysis. Replace 'chargefile' with the name of your file.
    
        ```bash
        baderkit run chargefile
        ```
    You should see an output similar to this:

    ```bash
    2026-03-10 11:24:50 INFO     Loading CHGCAR
                        INFO     Time: 0.06
                        INFO     Data type set as charge from data range
                        INFO     Beginning Bader Algorithm Using 'weight' Method
    2026-03-10 11:24:51 INFO     Initializing Labels
                        INFO     Initialization Complete
                        INFO     Time: 0.14
                        INFO     Sorting Reference Data
                        INFO     Assigning Charges and Volumes
                        INFO     Combining Low-Persistence Basins
    2026-03-10 11:24:53 INFO     Refining Maxima
                        INFO     Bader Algorithm Complete
                        INFO     Time: 2.75
                        INFO     Assigning Atom Properties
                        INFO     Atom Assignment Finished
                        INFO     Time: 0.01
    ```
    
    A summary of all properties will be written to `bader.json`. See the Ag system in our [example files](https://github.com/SWeav02/baderkit/releases/tag/0.9.0 ).
    
    Additional arguments and options such as those for printing output files or using different 
    algorithms can be viewed by running the help command.
    ```bash
    baderkit run --help
    ```

=== "Python"
    
    1. Import the `Bader` class.
    
        ```python
        from baderkit.core import Bader
        ```
    
    2. Use the `Bader` class' `from_dynamic` method to read a `CHGCAR` or `cube` file.
    
        ```python
        # instantiate the class
        bader = Bader.from_dynamic("path/to/charge_file")
        ```
    
    3. To run the analysis, we can call any class property. Try getting a complete summary in dictionary format.
        ```python
        results = bader.to_dict()
        ```
    You should see an output similar to this:
        ```bash
        2026-03-10 11:33:36 INFO     Beginning Bader Algorithm Using 'weight' Method   
        2026-03-10 11:33:37 INFO     Initializing Labels                               
                            INFO     Initialization Complete                           
                            INFO     Time: 0.29                                        
                            INFO     Sorting Reference Data                            
                            INFO     Assigning Charges and Volumes                     
                            INFO     Combining Low-Persistence Basins                  
                            INFO     Refining Maxima                                   
        2026-03-10 11:33:38 INFO     Bader Algorithm Complete                          
                            INFO     Time: 1.38                                        
                            INFO     Assigning Atom Properties                         
                            INFO     Atom Assignment Finished                          
                            INFO     Time: 0.0 
        ```
    
    4. Now try getting an individual property. 
        ```python
        atom_charges = bader.atom_charges # Total atom charges
        basin_volumes = bader.basin_volumes # The volumes of each 

        print(atom_charges)
        print(basin_volumes)
        ```
    This should show something like the following:
        ```python
        [18.99905303 18.99905303 19.00088048 19.00088048]
        [17.36053278 17.36053278 17.36794633 17.36794633]
        ```
    For more details on each property, see the [API reference](../api_reference/core/bader/#src.baderkit.core.bader.Bader).
    
    5. BaderKit also provides convenience methods for writing results to file. First,
    let's write a summary of the full analysis.
    
        ```python
        bader.write_json("bader.json")
        ```
    
    6. Now let's write the volume assigned to one of our atoms.
    
        ```python
        bader.write_atom_volumes(atom_indices = [0])
        ```
    
    !!! Tip
        After creating a `Bader` class object, it doesn't matter what order
        you call properties, summaries, or write methods in. BaderKit calculates
        properties/results only when they are needed and caches them.

=== "GUI App"

    1. Activate your environment with BaderKit installed. If you are not using an
    environment manager, skip to step 2.
    
        ```bash
        conda activate my_env
        ```
    
    2. Run the BaderKit GUI.
        ```bash
        baderkit gui
        ```
        
        This will launch a new window:
        ![pyqt_app](/images/pyqt_screenshot.png)

    3. Browse to find you charge density file, select your method, and run!

!!! Note
    To automatically calculate oxidation states, the path to a POTCAR file must be specified. If none is provided, oxidation states will not be included in any output summaries, and only charges (which depend on the pseudopotential used) will be provided. If you would like us to add functionality for other codes, please open an issue on our [Issues page](https://github.com/SWeav02/baderkit/issues)
---

## Warning for VASP (And other pseudopotential codes)

VASP and other pseudopotential codes only include valence electrons in their charge density outputs. Most allow you to write out the core electron density separately. For example, in VASP this can be done by adding the tag `LAECHG=.TRUE.` to your `INCAR` file. This will write the core charge density to an `AECCAR0` file and the valence to `AECCAR2` which can be summed together to get a total charge density that is much more accurate for partitioning and locating vacuum regions. **We highly recommend doing this**.

=== "Command Line"
    1. Sum the files using BaderKit's convenience method.
    ```bash
    baderkit sum AECCAR0 AECCAR2
    ```
    2. Run the analysis using this total charge density as the reference for
    partitioning.
    ```bash
    baderkit run CHGCAR -tot CHGCAR_sum
    ```
    
=== "Python"
    1. Import the Bader and Grid classes.
    ```python
    from baderkit.core import Bader, Grid
    ```
    2. Load the CHGCAR, AECCAR0 and AECCAR2
    ``` python
    charge_grid = Grid.from_vasp("CHGCAR")
    aeccar0_grid = Grid.from_vasp("AECCAR0")
    aeccar2_grid = Grid.from_vasp("AECCAR2")
    ```
    3. Sum the AECCAR files
    ```python
    reference_grid = aeccar0.linear_add(aeccar2_grid)
    ```
    4. Create the bader object
    ```python
    bader = Bader(
        charge_grid = charge_grid,
        reference_grid = reference_grid
        )
    ```
    From here, the `Bader` class object can be used as described in the [Basic Use](/baderkit/#__tabbed_2_2)
    section.

---

## Suggested DFT Parameters

By experience, Bader charge analysis typically requires only moderate calculation quality. As a starting point, our lab typically uses the PBE or PBEsol functional, a Monkhorst–Pack k-point mesh with a spacing of 0.25 Å<sup>-1</sup>, a plane wave kinetic energy cutoff that is 1.3 times the highest energy cutoff in the pseudopotential file, and a energy difference cutoff of 1 x 10<sup>-6</sup> eV. Depending on the system these may need to be adjusted. For example, metallic systems require a much finer k-point mesh.

For grid based software such as this, the most important parameter is the grid density. The exact required grid density will depend on the system and the selected algorithm (see [Methods](/baderkit/bader/methods/#__tabbed_2_2)), but we have found that a grid density of ~22 pts/Å along each lattice vector (~10000 pts / Å<sup>3</sup>) is fine enough in most cases.

As always, when extremely accurate results are required, we recommend increasing the quality of each parameter until convergence is reached for your system.