# Spin Separated Calculations

BaderKit runs on whatever set of data is stored in the `total` property of the
`Grid` class. By default, this data is read from the first set of
data in a `CHGCAR` or `cube` format file. This typically represents the total
charge density of both the spin-up and spin-down electrons.

In VASP, spin polarized calculations will also write an additional set of data
representing the difference between the spin-up and spin-down charge densities.
When read from file, this data is stored in the `diff` property of the `Grid`
class.

Assuming we have run a spin-polarized calculation with VASP, we can run the
Bader analysis on the separate spin-up and spin-down systems by creating two
`Grid` and `Bader` class objects with the appropriate charge densities in the
`total` property.

---

=== "Command Line"
    1. Split your file using the provided helper command. Replace 'chargefile'
    with your actual file path.
    
        ```bash
        baderkit split chargefile
        ```
    
    2. Run the bader analysis on the spin up system and copy it to avoid overwriting.
    
        ```bash
        baderkit run chargefile_up
        mv bader_atom_summary.tsv bader_atom_summary_up.tsv
        mv bader_basin_summary.tsv bader_basin_summary_up.tsv
        ```
    
    3. Run the bader analysis on the spin down system.
    
        ```bash
        baderkit run chargefile_down
        mv bader_atom_summary.tsv bader_atom_summary_down.tsv
        mv bader_basin_summary.tsv bader_basin_summary_down.tsv
        ```

=== "Python"
    ```python
    # import 
    from baderkit.core import Bader, Grid
    
    # load the spin polarized charge grid. Make sure the `total_only` tag is set to
    # false to indicate that we want to load all sets of data.
    polarized_grid = Grid.from_vasp("CHGCAR", total_only=False)
    
    # split the polarized grid to the spin up and spin down components
    grid_up, grid_down = polarized_grid.split_to_spin()
    
    # create our Bader classes
    bader_up = Bader(grid_up)
    bader_down = Bader(grid_down)
    
    # get results
    results_up = bader_up.results_summary
    results_down = bader_down.results_summary
    ```
    

!!! Tip
    This analysis can be run on results from softwares other than VASP as well.
    Either set up the `polarized_grid` so that it follows VASP's `total` and `diff`
    format, or load/create the `grid_up` and `grid_down` from already split
    data.

!!! Warning
    This example does not use the core + valence charge density as we suggest in
    our [warning](/baderkit/#warning-for-vasp-and-other-pseudopotential-codes).
    This is because VASP does not write spin separated `AECCAR0` or `AECCAR2` files.
    You can still use the `CHGCAR_sum` as a reference, but the bader basins will
    be defined by the total charge density, not the spin-polarized ones.