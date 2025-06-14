# BaderKit

## About

BaderKit is a python implementation of Bader's Atomic Theory of Atoms in Molecules. It is largely based on the algorithms of [Henkelman et. al.](https://theory.cm.utexas.edu/henkelman/code/bader/) at UT Austin. The app is loosely part of my PhD at UNC Chapel Hill in the [Warren Lab](https://materials-lab.io/) with funding from the [NSF's GRFP](https://nsfgrfp.org/), but is largely my own passion project aimed at making my life easier when developing other packages.

BaderKit aims to reproduce the algorithms available in the Henkelman excellent Fortran code, but utilizes Python's class oriented system to allow for easy extension to other projects. I have reimplemented the code primarily using [Numba](https://numba.pydata.org/numba-doc/dev/index.html) and [NumPy](https://numpy.org/doc/stable/index.html) to maintain speed where I can. [Pymatgen](https://pymatgen.org/) is used under the hood to build out several core classes to improve ease of use.

This project is currently an early work in progress. So far, I have implemented a simple Bader class and CLI. I hope to add the following algorithms:
 

 - [x]  On Grid [Comput. Mater. Sci. 36, 354-360 (2006)](https://www.sciencedirect.com/science/article/abs/pii/S0927025605001849)
 - [ ] Near Grid [J. Phys.: Condens. Matter 21, 084204 (2009)](https://iopscience.iop.org/article/10.1088/0953-8984/21/8/084204)
 - [x] Weighted [J. Chem. Phys. 134, 064111 (2011)](https://pubs.aip.org/aip/jcp/article-abstract/134/6/064111/645588/Accurate-and-efficient-algorithm-for-Bader-charge?redirectedFrom=fulltext)
 
## Installation
 
Once all methods are successfully implemented I will make the package available on Pip and Conda. For now, it can be installed with the following instructions.
 
 1. Create a new Conda env and activate it
 2. Clone this repo to a local folder
 3. Navigate to the folder
 4. Run `conda env --file dev_environment.yml`
 5. Run `pip install -e .`

You can also install the package with other python environment managers simply using the final command, though be warry that I haven't setup proper dependencies yet.
 
## Use
The Bader class can be easily called through python in a directory with VASP output files.

```python
from baderkit.core import Bader
from pathlib import Path

# instantiate the class
bader = Bader.from_dynamic(
    charge_filename = "path/to/CHGCARorcube",
    reference_filename = "path/to/CHGCAR_sumorcube", # Optional. Defaults to charge_file data if empty
    method="neargrid", # Optional. Defaults to neargrid
    directory = Path("path/to/dir") # Optional. The directory to write to.
    )

# run bader and get a summary of results
results = bader.results_summary

# Or access results as class properties. For example:
atom_charges = bader.atom_charges # The total charge assigned to each atom
labels = bader.atom_labels # An array assigning each point in the charge grid to an atom

# The charge density of basins or atoms can also be printed
bader.write_basin_volumes([0])
bader.write_atom_volumes_sum([0,1,2])

```
For now, the `Bader` class only has a convenience function for loading VASP or .cube files. However, the Bader class can also be created from a Path object and Baderkit's custom Grid class. 

Behind the scenes, the Grid class inherits from Pymatgen's [VolumetricData class](https://pymatgen.org/pymatgen.io.vasp.html#pymatgen.io.vasp.outputs.VolumetricData) allowing for creation from a variety of formats. The `Grid` class has convenience functions for loading from `CHGCAR/ELFCAR`, `.cube` and `hdf5` files, but can also be created directly from a Pymatgen Structure and a dictionary of containing Array's representing the Charge Density.

```python
from baderkit.core import Bader
from baderkit.utilities import Grid
from pathlib import Path

# Create a Grid object from a .cube or any other source you can load into NumPy arrays
charge_grid = Grid.from_cube("path/to/charge-density.cube")
# Optionally indicate the path you would like to write any results to
path = Path("path/to/dir")
# Create the Bader object
bader = Bader(
    charge_grid = charge_grid,
    reference_grid = charge_grid,
    method="neargrid", # Optional
    directory = path,
    )

# run bader and get a summary of results
results = bader.results_summary
```

In addition to the Python interface, BaderKit can be run from the command line.
1. Activate your environment with BaderKit installed
2. Navigate to the directory with your charge density and reference file
3. Run `baderkit run CHGCAR -ref CHGCAR_sum`

Additional arguments and options can be viewed by running `baderkit run --help`. The command will automatically try and detect vasp and .cube formats based on the file names.

## Plotting and Web GUI

Writing bader basins to file without knowing what they are can be annoying. To help with this, I have implemented a `BaderPlotter` class that uses [pyvista](https://pyvista.org/) under the hood. This can be interacted with in python directly, or through a relatively basic web app I created using [Streamlit](https://streamlit.io/). The app can be started from the command line with:
1. Activate your environment with BaderKit installed
2. Navigate to the directory with your charge density and reference file
3. Run `baderkit tools webapp CHGCAR -ref CHGCAR_sum`

This will open a window in your browser similar to this:
![streamlit_app](docs/streamlit_screenshot.png)

The basins can then be selected using the `Bader` tab on the left. Simple settings for the isosurface and aotms are available under the `Grid` and `Atoms` tab. Some settings for the viewport are available under the `view` tab. The selected basins can be exported to vasp-like files in the folder you started the webapp from under the `Export` tab. The viewport can also be exported to a variety of image formats.

> **ℹ️ Note:** Currently the viewport is made by exporting the pyvista Plotter object to html and embedding it directly. Changes made by interacting with the view port directly (rotation) will not show up in exported images.

## Contributing

If you are interested in this project and have suggestions, please use this repositories Issues or Discussions tab. Any suggestions or discussion would be deeply appreciated!