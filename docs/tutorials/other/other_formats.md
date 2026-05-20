# Custom Data Formats

BaderKit only provides convenience functions for loading VASP `CHGCAR`-like files
or Gaussian `cube`-like files. However, as long as you can read in your charge
density into a NumPy array, you can still use BaderKit by constructing the `Structure`,
`Grid`, and `Bader` classes manually. This tutorial provides the outline for how to do this.

1. Import the required classes from BaderKit as well as numpy
    ```Python
    from baderkit import Bader, Grid, Structure
    import numpy as np
    ```

2. Create a PyMatGen Structure object. This is usually easiest to do from a file, but can also be made manually.
    ```Python
    structure = Structure.from_file(filename = "mystructure.cif", fmt = "cif")
    ```

3. Load your data, however you can, into a numpy array. Here we manually construct a fake grid as the actual method will be specific to your use case.

    ```Python
    charge_data = np.array([
    [[1,2,3],[3,4,5],[6,7,8]],
    [[1,2,3],[3,4,5],[6,7,8]],
    [[1,2,3],[3,4,5],[6,7,8]],
    ])
    ```

4. Construct a data dictionary and then the Grid object. 
    ```Python
    data = {"total": charge_data}
    charge_grid = Grid(structure=structure, data=data)
    ```

    !!! Note
        Charge density data is assumed to be in VASP's default format i.e. it should be
        stored as

        data(*r*) = n(*r*) x V<sub>grid</sub> x V<sub>cell</sub>

        where

        - n(*r*) = charge density in 1/Ang at point *r*
        - V<sub>grid</sub> = the total number of grid points
        - V<sub>cell</sub> = the volume of the simulation cell

        See [VASP's wiki](https://www.vasp.at/wiki/index.php/CHGCAR) for more details.

5. Construct a Bader class object.

    ```Python
    bader = Bader(charge_grid=charge_grid)
    ```

    From here you can use the Bader class as you wish.

## Download Resources

Tutorial Script: <a href="/tutorial_scripts/other_formats.py" download>other_formats.py</a>
