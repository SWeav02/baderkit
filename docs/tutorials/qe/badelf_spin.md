# Spin Separated ELF Calculations

It is common for systems to have differing ELF topologies in the spin-up and spin-down electron systems. In these cases, it is useful to perform separate analyses on each spin system. 

---

=== "Command Line"
    1. If you have VASP CHGCAR-like files, split your file using the provided helper command. If you are using the `.cube` format, the spin-up/down systems should already be written separately.

        ```bash
        baderkit split chargefile
        baderkit split elffile
        ```

    2. Run your analysis on each system separately. This example uses BadELF.

        ```bash
        baderkit badelf chargefile_up elffile_up
        mv badelf.json badelf_up.json
        baderkit badelf chargefile_down elffile_down
        mv badelf.json badelf_down.json
        ```

=== "Python - VASP"
    ```python
    # import the grid class and analysis of interest
    from baderkit import Grid
    from baderkit.elf_analysis import Badelf

    # load the spin polarized grids with the `total_only` tag set to False. 
    polarized_charge = Grid.from_vasp("CHGCAR", total_only=False)
    polarized_elf = Grid.from_vasp("ELFCAR, total_only=False)

    # split the polarized grids to the spin up and spin down components
    charge_up, charge_down = polarized_charge.split_to_spin()
    elf_up, elf_down = polarized_elf.split_to_spin()

    # create our Badelf classes
    badelf_up = Badelf(charge_up, elf_up)
    badelf_down = Badelf(charge_down, elf_down)

    # get desired results
    print(badelf_up.atom_charges)
    print(badelf_down.atom_charges)
    ```

=== "Python - cube"
    ```python
    # import the grid class and analysis of interest
    from baderkit import Grid
    from baderkit.elf_analysis import Badelf

    # load the spin polarized grids separately.
    charge_up = Grid.from_cube("charge_up.cube")
    charge_down = Grid.from_cube("charge_down.cube")
    elf_up = Grid.from_cube("elf_up.cube")
    elf_down = Grid.from_cube("elf_down.cube")

    # create our Badelf classes
    badelf_up = Badelf(charge_up, elf_up)
    badelf_down = Badelf(charge_down, elf_down)

    # get desired results
    print(badelf_up.atom_charges)
    print(badelf_down.atom_charges)
    ```


!!! Warning
    Oxidation states calculated from spin-polarized calculations will not make sense unless you combine the charges prior to comparing to the neutral atom states.
