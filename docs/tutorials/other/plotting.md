For some Classes, BaderKit provides a basic plotting API. This is also what the GUI is built on. This tutorial introduces the basics of how to use a BaderKit Plotter class. We will use the `BadelfPlotter` class to plot the electride electron in Ca<sub>2</sub>N.

!!! Note
    We will not show the VASP calculation used in this tutorial. The resources can be downloaded at the bottom of this page or you can follow the VASP portion of [this tutorial](../../vasp/badelf).

## BaderKit

1. First, import the `Badelf` class.

    ```python
    from baderkit.elf_analysis import Badelf
    ```

2. Create a `Badelf` instance from file.
    ```python
    badelf = Badelf.from_vasp()
    ```

3. Create the plotter.
    ```python
    plotter = badelf.to_plotter()
    ```
    This should open a window similar to the following.
    ![baderkit_plotter1](/images/plotter1.png)

4. Now we want to update the settings so that our plot looks a bit nicer. First, we turn on physically based rendering which improves the visual quality of our atoms at the cost of some graphical resources. We also make the atoms slightly metallic in appearance.
    ```python
    plotter.pbr = True
    plotter.atom_metallicness = 0.5
    ```
    ![baderkit_plotter2](/images/plotter2.png)

5. Currently, we aren't showing any ELF basins. Let's set our plot to show the electride basin (the 4th "atom" in our system, i.e. index 3) and hide the electride dummy atom.
    ```python
    plotter.visible_atom_basins = [3]
    plotter.atom_opacity = [1,1,1,0]
    ```
    ![baderkit_plotter3](/images/plotter3.png)

6. By default, BaderKit uses a colormap for the isosurface and caps. Let's change this to a solid color.
    ```python
    plotter.use_solid_cap_color = True
    plotter.use_solid_surface_color = True
    plotter.cap_opacity = 1.0
    plotter.surface_opacity = 1.0
    ```
    ![baderkit_plotter4](/images/plotter4.png)

7. The current camera angle isn't ideal. While we could drag the viewer around to try and get it to a good place, it's much easier and more reproducible to do this programattically.
    ```python
    plotter.set_camera_to_hkl(1,1,0)
    ```
    ![baderkit_plotter5](/images/plotter5.png)

8. Finally, we should save our plot. You can do this from the viewer or through python.
    ```python
    plotter.get_plot_screenshot(
        filename="Ca2N_electride.png",
        transparent_background=True
        )
    ```
    Note that the python method allows us to script out plot generation. For those cases we recommend setting `off_screen=True` when first generating the plotter to avoid starting the gui.

And that's it! The plotter can do much more, so try messing around with it on your own.


## Download Resources

VASP Inputs/Outputs: <a href="https://github.com/SWeav02/baderkit/releases/download/0.10.0/Ca2N.zip" download>Ca2N.zip</a>

Tutorial Script: <a href="/tutorial_scripts/other/plotting.py" download>oxidation_states.py</a>