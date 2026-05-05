This tutorial provides a basic example of calculating oxidation states using BaderKit. We will use the results from a VASP calculation on NaCl. Note that we provide a fake `POTCAR` file as the true `POTCAR` file is proprietary and cannot be distributed.

1. Import the Bader class
    ```Python
    from baderkit import Bader
    ```

2. Create a Bader class instance. To automatically calculate oxidation states, you must include a path to a pseudopotential file, a dictionary mapping species to the number of valence electrons, or specify `False` to indicate pseudopotentials were not used.
    ```Python
    bader = Bader.from_vasp(
    charge_filename="CHGCAR", 
    total_charge_filename="CHGCAR_sum",
    pseudopotential_filename="POTCAR"
    )
    ```

3. Print the oxidation states
    ```Python
    print(bader.oxidation_states)
    ```
    You should see a set of logging information as BaderKit runs, then the oxidation states of each atom in the structure:
    `array([ 0.87331308, -0.8732974 ])`

And that's it! Try playing around with what else the `Bader` class offers.