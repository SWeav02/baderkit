# Basic Usage

Classes in BaderKit are constructed to follow the same usage pattern. Though we only show an example for using the `Bader` method here, other methods such as the `Badelf` or `ElfRadii` methods can be used in much the same way. For more examples see our [Tutorials](../tutorials).

=== "Python"
     1. Import the `Bader` class.

        ```python
        from baderkit import Bader
        ```

    2. Use the `Bader` class' `from_dynamic` method to read a `CHGCAR` or `cube` file.

        ```python
        # instantiate the class
        bader = Bader.from_dynamic("path/to/charge_file")
        ```

    3. Call your desired property
        ```python
        charges = bader.atom_charges
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

    4. Try printing some information.
        ```python
        print(atom_charges)
        ```
    This should show something like the following:
        ```python
        [18.99905303 18.99905303 19.00088048 19.00088048]
        ```
    For details on available properties, see the [API reference](../api_reference/core/bader/#src.baderkit.bader.Bader).

    5. We can also write a summary of results to file.

        ```python
        bader.write_json("bader.json")
        ```

    !!! Tip
        After creating a `Bader` class object, it doesn't matter what order
        you call properties, summaries, or write methods in. BaderKit calculates
        properties/results only when they are needed and caches them.

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
        baderkit bader chargefile
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

    A summary of all properties will be written to `bader.json`.

    Additional arguments and options can be viewed by running the help command.
    ```bash
    baderkit --help
    ```

    !!! Tip
        Other methods (e.g. BadELF, ElfRadii, etc.) can be used from the CLI using the same pattern
        ```bash
        baderkit methodname chargefile referencefile
        ```

=== "GUI"

    1. Activate your environment with BaderKit installed. If you are not using an
    environment manager, skip to step 2.

        ```bash
        conda activate my_env
        ```

    2. Launch the GUI Applitcation

        ```bash
        baderkit gui
        ```