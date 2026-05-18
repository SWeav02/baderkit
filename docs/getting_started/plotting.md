For some classes we provide simple plotting built using [PyVista](https://pyvista.org/). This is also used under the hood by our GUI. Plotting is currently limited to only a small set of BaderKit classes. For those that have been implemented, you can quickly access the plotter with the `to_plotter()` method. For example, with the `Bader` class you may do the following.

```Bash
from baderkit import Bader
bader = Bader.from_vasp()
plotter = bader.to_plotter()
```

This will run the Bader analysis and open a simple plotting window displaying a VTK plot of your object similar to below. The plotter can then be updated directly by changing its class properties. For more details see our [plotting tutorial](/baderkit/tutorials/other/plotting) and [Plotting API](/baderkit/api_reference/plotting/structure)

```Bash
plotter.visible_bader_basins = []
plotter.visible_atom_basins = [0]
```

Similarly, you can change the isosurface value and color.

```Bash
# display full volume of atom
plotter.iso_val = 0.00001
# use a solid color instead of a colormap
plotter.use_solid_cap_color = True
plotter.use_solid_surface_color = True
# set the color of the isosurface and caps
plotter.cap_color = "yellow"
plotter.surface_color = "yellow"
# Make the surface appear solid
plotter.cap_opacity = 1.0
plotter.surface_opacity = 1.0
```

