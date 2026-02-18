# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 01:12:33 2026

@author: sammw
"""
import numpy as np
import pyvista as pv
from baderkit.core import Grid
import numpy as np
import pyvista as pv
import vtk
from ttk import ttkMorseSmaleComplex



def compute_morse_smale(grid, scalar_name="values"):
    """
    Compute the Morse-Smale complex using TTK.
    
    Parameters
    ----------
    grid : vtkStructuredGrid
    scalar_name : str
    
    Returns
    -------
    vtkStructuredGrid
        TTK output with critical points and separatrices.
    """
    msc = ttkMorseSmaleComplex()
    msc.SetInputData(grid)
    msc.SetScalarField(scalar_name)
    msc.Update()
    
    return msc.GetOutput()

# ----------------------
# Example usage
# ----------------------

# Example lattice (row vectors)
lattice = np.array([[1.0, 0.2, 0.1],
                    [0.0, 1.0, 0.1],
                    [0.0, 0.0, 1.0]])

# Example scalar field
nx, ny, nz = 30, 30, 30
scalar_field = np.random.rand(nx, ny, nz)  # replace with your data

grid = Grid.from_vasp("ELFCAR")
data = grid.total

values = np.pad(grid.total, pad_width=((0, 1), (0, 1), (0, 1)), mode="wrap")
shape = values.shape
indices = np.indices(shape).reshape(3, -1, order="F").T
points = grid.grid_to_cart(indices)
structured_grid = pv.StructuredGrid()
structured_grid.points = points
structured_grid.dimensions = shape
structured_grid["values"] = values.ravel(order="F")

# Compute Morse-Smale complex
msc_output = compute_morse_smale(structured_grid.vtkStructuredGrid)

points = msc_output.GetPoints()
for i in range(points.GetNumberOfPoints()):
    pt = np.array(points.GetPoint(i))
    frac = np.linalg.solve(lattice.T, pt)  # real -> fractional
    frac = frac % 1.0                        # wrap
    points.SetPoint(i, *(frac @ lattice))    # back to Cartesian


# Optional: convert to PyVista for visualization
pv_grid = pv.wrap(msc_output)
pv_grid.plot(scalars="values")
