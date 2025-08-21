# Custom Data Formats

BaderKit only provides convenience functions for loading VASP `CHGCAR`-like files
or Gaussian `cube`-like files. However, as long as you can read in your charge
density into a NumPy array, you can still use BaderKit by constructing the `Structure`, 
`Grid`, and `Bader` classes manually.

---

```python
# import 
from baderkit.core import Bader, Grid, Structure
import numpy as np

# Create a PyMatGen Structure object. This is usually easiest to do from a
# file, but can also be made manually.
structure = Structure.from_file(filename = "mystructure.cif", fmt = "cif")

# Load your data, however you can, into a numpy array.
charge_data = np.array([
[[1,2,3],[3,4,5],[6,7,8]],
[[1,2,3],[3,4,5],[6,7,8]],
[[1,2,3],[3,4,5],[6,7,8]],
])

# Create a data dictionary
data = {"total": charge_data}

# create Grid objects for the charge-density (and reference file if needed)
charge_grid = Grid(structure=structure, data=data)

# Create the Bader object
bader = Bader(charge_grid = charge_grid)
```

!!! Warning
    The charge density data must follow VASP's conventions, i.e. it should be
    stored as 
    
    data(*r*) = n(*r*) x V<sub>grid</sub> x V<sub>cell</sub>
    
    where
    
    - n(*r*) = charge density in 1/Ang at point *r*
    - V<sub>grid</sub> = the total number of grid points
    - V<sub>cell</sub> = the volume of the simulation cell
    
    See [VASP's wiki](https://www.vasp.at/wiki/index.php/CHGCAR) for more details.
