# Methods

BaderKit is based on the [Henkelman group's](https://theory.cm.utexas.edu/henkelman/code/bader/) excellent Fortran code.
The same methods available in their code are available here in addition to some
variants that we find useful. For a more in depth look at the basics of Bader analysis, 
see our [Background](./background.md) page.

| Method        | Speed   | Accuracy | Single Basin Assignments |
|---------------|---------|----------|--------------------------|
|ongrid         |Very Fast|Low       |:material-check:          |
|neargrid       |Medium   |High      |:material-check:          |
|hybrid-neargrid|Medium   |High      |:material-check:          |
|weight         |Fast     |Very High |:material-close:          |
|hybrid-weight  |Fast     |Very High |:material-close:          |

=== "ongrid"
    
    This is the original algorithm proposed by Henkelman et. al. It is extremely
    fast, but prone to error. For example, it gives slightly different oxidation 
    states for different orientations of a molecule or material. We recommend 
    using it only if speed is a major concern.
    
    For each point on the grid, the gradient is calculated for the 26 nearest 
    neighbors, and the neighbor with the steepest gradient is selected as the 
    next point in the path.
    
    In the original code, this path is followed until a maximum is reached or
    a previous path is hit. In the former case, all of the points in the path
    are assigned to the maximum, and in the latter they are assigned to the same
    maximum as the colliding path.
    
    In our implementation, the steepest gradient is calculated once at each point
    to establish a pointer to the best neighbor. The points are then assigned to 
    maxima using a pointer doubling algorithm. This gives the same results as the 
    original algorithm, while allowing us to parallelize the operation.
    
    **Reference**
    
    G. Henkelman, A. Arnaldsson, and H. Jónsson, A fast and robust algorithm for Bader decomposition of charge density, [Comput. Mater. Sci. 36, 354-360 (2006)](https://theory.cm.utexas.edu/henkelman/code/bader/download/henkelman06_354.pdf)
    
=== "neargrid"
    
    This algorithm was developed by Henkelman et. al. several years after the
    ongrid method to fix orientation errors. It is generally less accurate than 
    the weight method, but is useful in situations where it is desirable to have 
    only one assignment per point.
    
    A gradient vector is calculated at each point using the three nearest neighbors. 
    A step is then made to the neighboring point that is closest to this gradient vector. 
    To preserve information about the true (offgrid) gradient, a correction vector 
    is stored that points from the new point to the original gradient vector.
    
    At each step, the difference between the gradient and the ongrid step is added 
    to this correction vector. If any component of the vector is ever closer to 
    a neighboring point than the current one, a correction is made to keep 
    the path closer to the true gradient.
    
    After all of the points are assigned, a refinement must be made to the 
    points on the edge, as the accumulation of the gradient is technically only
    correct for the first point in the path.
    
    
    !!! Note
        In BaderKit, the atomic charges/volumes tend to be identical to the original
        Henkelman groups algorithm, but individual basin's sometimes are not. 
        Additionally, though the original paper and code suggests only 
        one refinement of the edges is needed, we found that several are sometimes
        required to reach convergence.
        
        For example, in our test case ([a Ag structure](https://next-gen.materialsproject.org/materials/mp-8566?formula=Ag)) the Henkelman groups code assigns
        asymmetrical charges/volumes to symmetrical basins while our code reaches
        symmetry after several iterations.
        
        By default we use iterative refinement, but this can be changed to the
        original single refinement by setting `refinement_method="single"` or
        `--refinement-method single` in the command-line.
    
    **Reference**
    
    W. Tang, E. Sanville, and G. Henkelman, A grid-based Bader analysis algorithm without lattice bias, [J. Phys.: Condens. Matter 21, 084204 (2009)](https://theory.cm.utexas.edu/henkelman/code/bader/download/tang09_084204.pdf)
    
=== "weight (Default)"

    This method reduces errors due to orientation by allowing each point to be
    partially assigned to multiple basins. This method tends to provide the most accurate 
    charges, but may not be the best for workflows that use each points assignment 
    after the Bader algorithm is run.
    
    A voronoi cell is generated at each point on the grid. A "flux" is calculated 
    from this point to each neighbor sharing a voronoi facet using the difference 
    in charge density modified by the distance to the neighbor and area of the 
    shared voronoi facet. This flux is then normalized to create "weights" which
    indicate the fraction of the volume that flows to each neighbor.
    
    Moving from highest to lowest, each point is assigned to basins by assigning
    the weight going to each neighbor to that neighbors own fractional assignments.
    The ordering from highest to lowest ensures that the higher neighbors have
    already received their assignment.
    
    In the original implementation, the flux is calculated as the algorithm steps
    down the points in order. In our implementation, the flux is calculated in
    a separate first step, allowing for parallelization, then the flux is assigned
    as normal.
    
    **Reference**
    
    M. Yu and D. R. Trinkle, Accurate and efficient algorithm for Bader charge integration, [J. Chem. Phys. 134, 064111 (2011)](https://theory.cm.utexas.edu/henkelman/code/bader/download/yu11_064111.pdf)

=== "hybrid-neargrid"
    
    Because we found that iterative edge refinement is required for the neargrid method anyways (see the [Note](http://127.0.0.1:8000/baderkit/methods/#__tabbed_1_2)),
    we created a variation where the faster ongrid method is performed first, 
    then refined using the neargrid method. To avoid repeat calculations, each 
    point in the grid will only ever be refined once.
    
=== "hybrid-weight"

    In the original weight method, the use of a voronoi cell restricts neighbors
    to exclusively those that share a voronoi facet. This is different from the
    other methods and can results in "local maxima" that have lower values than
    one of their 26 neighbors. This also tends to result in more basins being
    found.
    
    This method performs the weight method, but consolidates basins such that
    their maxima are higher than all 26 neighbors.