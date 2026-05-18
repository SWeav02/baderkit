For some Classes, BaderKit provides a basic plotting API. This is also what the GUI is built on top of. This tutorial introduces the basics of how to use a BaderKit Plotter class. We will use the `BaderPlotter` class to reproduce an image from the original BaderKit paper.


## Download Resources

For this tutorial, we recommend downloading the resources directly. This includes the input files needed to reproduce the calculation with VASP as well as the outputs of running `bader-v1.05` (the original Henkelman group's code) and `BaderKit`.

VASP Inputs/Outputs: <a href="https://github.com/SWeav02/baderkit/releases/download/0.10.0/NaCl.zip" download>NaCl.zip</a>

## BaderKit

1. First, import the `Grid` class. We will use this to build our plots.

    ```python
    from baderkit import Grid
    ```

Tutorial Script: <a href="/tutorial_scripts/vasp/oxidation_states.py" download>oxidation_states.py</a>