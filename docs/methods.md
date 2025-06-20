# Methods

The goal of Bader analysis is to separate the charge density into regions that 
"belong" to a given atom. In BaderKit this is achieved by climbing from each point
in the grid to a local maximum, and assigning all of the points along the path
to this maximum.

BaderKit is based on the [Henkelman group](https://theory.cm.utexas.edu/henkelman/code/bader/)'s excellent Fortran code.
The same methods available in their code are also available here. We have 
additionally added some slight variants that we find to be useful.

=== "ongrid"
    
    This is the original algorithm proposed by Henkelman et. al. For each point
    in the grid, the gradient is calculated for the 26 nearest neighbors, and
    the neighbor with the steepest gradient is selected as the next point in the
    path.
    
    **This algorithm is extroardinarily fast, but is prone to error. For example, it gives slightly different oxidation states for different orientations of a molecule or material. We recommend using it only if speed is a major concern.**
    
    In the original code, this path is followed until a maximum is reached or
    a previous path is hit. All of the points in the path are then assigned to
    this maximum (or to the previous path's assignment in the later case).
    
    In our implementation, for each point only the first step of the path is taken
    to establish a pointer to the best neighbor. This allows us to parallelize
    the operation using Numba. The assignments are then made using a forest of
    trees algorithm.
    
    **Reference**
    
    G. Henkelman, A. Arnaldsson, and H. Jónsson, A fast and robust algorithm for Bader decomposition of charge density, [Comput. Mater. Sci. 36, 354-360 (2006)](https://theory.cm.utexas.edu/henkelman/code/bader/download/henkelman06_354.pdf)
    
=== "neargrid"

    In the neargrid method, a gradient vector is calculated at each point using 
    the three nearest neighbors. A step is then made to the neighbor that is 
    closest to this gradient vector. A correction vector is stored that points 
    from this "ongrid" step to the original point that would have been assigned
    by the more accurate gradient vector. At each step, the difference between
    the gradient and the ongrid step is added to this correction vector. If any
    component of the vector is ever closer to a neighboring point than the current
    step, a correction is made to keep the path closer to the true gradient.
    
    After this step, a refinement must be made to the points on the edge.
    
    **This algorithm is slower than the ongrid method, but is less prone to errors related to orientation. It is generally less accurate than the weight method, but is useful in situations where it is desirable to have only one assignment per point.**
    
    !!! Note
        Comparing our implementation to the Henkelman groups, we found that while 
        atom charges/volumes tend to be identical, individual basin sometimes
        are not. Additionally, though the original paper and code suggests only 
        one refinement of the edges is needed, we found that several  refinements
        are needed.
        In our test case (an Ag4 structure) the Henkelman groups code assigned
        asymmetrical charges/volumes to symmetrical basins. Our code reached
        symmetry after several iterations.
    
    **Reference**
    
    W. Tang, E. Sanville, and G. Henkelman, A grid-based Bader analysis algorithm without lattice bias, [J. Phys.: Condens. Matter 21, 084204 (2009)](https://theory.cm.utexas.edu/henkelman/code/bader/download/tang09_084204.pdf)
    
=== "weight"
    
    **Reference**
    
    M. Yu and D. R. Trinkle, Accurate and efficient algorithm for Bader charge integration, [J. Chem. Phys. 134, 064111 (2011)](https://theory.cm.utexas.edu/henkelman/code/bader/download/yu11_064111.pdf)

=== "hybrid-neargrid (Default)"

    Pass

=== "hybrid-weight"

    Pass