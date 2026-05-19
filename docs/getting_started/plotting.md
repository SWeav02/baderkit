For some classes we provide simple plotting built using [PyVista](https://pyvista.org/). This is also used under the hood by our GUI. Plotting is currently limited to only a small set of BaderKit classes. For those that have been implemented, you can quickly access the plotter with the `to_plotter()` method. For example, with the `Bader` class you may do the following.

```Bash
from baderkit import Bader
bader = Bader.from_vasp()
plotter = bader.to_plotter()
```

This will run the Bader analysis and open a simple plotting window displaying a VTK plot of your object similar to below. The plotter can then be updated directly by changing its class properties. For more details see our [plotting tutorial](/baderkit/tutorials/other/plotting) and [Plotting API](/baderkit/api_reference/plotting/structure)

## Implemented Plotters

| Class Name | Implemented |
|------------|-------------|
| Structure  |:material-checkbox-marked-circle:{.green-check}|
| Grid  |:material-checkbox-marked-circle:{.green-check}|
| Bader  |:material-checkbox-marked-circle:{.green-check}|
| Badelf  |:material-checkbox-marked-circle:{.green-check}|
| ElfLabeler       |:material-close-box:{.red-x}|
| ElfRadii       |:material-close-box:{.red-x}|
| BasinOverlap       |:material-close-box:{.red-x}|

