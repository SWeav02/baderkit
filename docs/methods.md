# Methods

BaderKit is based on the [Henkelman group's](https://theory.cm.utexas.edu/henkelman/code/bader/) excellent Fortran code.
In addition to the algorithms available in their code, we have created several
variations. This page provides descriptions and recommendations for when to use
each method.

## Summary

| Method        | Speed   | Accuracy | Single Basin Assignments |
|---------------|---------|----------|--------------------------|
|ongrid         |Very Fast|Low       |:material-check:          |
|neargrid       |Medium   |High      |:material-check:          |
|reverse-neargrid|Fast    |High      |:material-check:          |
|weight         |Fast     |Very High |:material-close:          |
|hybrid-weight  |Fast     |Very High |:material-close:          |

## Descriptions

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
    only one basin assignment per point on the grid.
    
    A gradient vector is calculated at each point using the three nearest neighbors. 
    A step is then made to the neighboring point that is closest to this gradient vector. 
    To preserve information about the true (offgrid) gradient, a correction vector 
    is stored that points from the new point to the original gradient vector.
    
    At each step, the difference between the gradient and the ongrid step is added 
    to this correction vector. If any component of the correction vector is ever closer to 
    a neighboring point than the current one, a correction is made to keep 
    the path closer to the true gradient.
    
    After all of the points are assigned, a refinement must be made to the 
    points on the edge, as the accumulation of the gradient is technically only
    correct for the first point in the path.
    
    
    !!! Note
        Although the original paper and code suggests only 
        one refinement of the edges is needed, we found that several are usually
        required. In our test case ([a Ag structure](https://next-gen.materialsproject.org/materials/mp-8566?formula=Ag)) 
        on a course grid, the Henkelman groups code assigns asymmetrical charges/volumes to 
        symmetrical basins while our code reaches symmetry after several iterations.
        
        By default we use iterative refinement, which results in this method being
        ~4-10x slower than the other methods. This can be changed to the
        original single refinement by setting `refinement_method="single"` in python or
        `--refinement-method single` in the command-line.
    
    **Reference**
    
    W. Tang, E. Sanville, and G. Henkelman, A grid-based Bader analysis algorithm without lattice bias, [J. Phys.: Condens. Matter 21, 084204 (2009)](https://theory.cm.utexas.edu/henkelman/code/bader/download/tang09_084204.pdf)
 
=== "weight"

    This method reduces errors due to orientation by allowing each point to be
    partially assigned to multiple basins. This method tends to provide the most accurate 
    charges, but may not be the best for workflows that rely on each point being
    assigned to a single basin.
    
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
    
    !!! Note
        The `bader.basin_labels` and `bader.atom_labels` properties 
        for this method are generated by assigning the points to the basin with 
        the highest weight, or the first basin index in the case of a tie.
    
    **Reference**
    
    M. Yu and D. R. Trinkle, Accurate and efficient algorithm for Bader charge integration, [J. Chem. Phys. 134, 064111 (2011)](https://theory.cm.utexas.edu/henkelman/code/bader/download/yu11_064111.pdf)   

=== "reverse-neargrid (Default)"
    
    This method is our own improvement to the original neargrid method, inspired
    by the weight method. It is at least as accurate as the neargrid method while
    avoiding any edge refinements.
    
    The method is largely similar to the [original](/baderkit/methods/#__tabbed_1_2). However,
    instead of following an ascending path, we first sort the grid points from
    highest to lowest, then descend starting form the highest. At each point, the 
    gradient is calculated, and used to construct a pointer to the steepest neighbor.
    A correction vector, the difference between this pointer and the true gradient, 
    is then calculated and stored.
    
    For each point other than the local maxima, the steepest neighbor will have
    already gone through this process. For these points, the steepest neighbor's
    correction vector is added to the points own correction vector. This provides
    some memory of variation from the true gradient as the path is descended.
    If the correction vector is ever large enough that one of its components is
    closer to a neighbor than the current point, a correction is made. This moves
    moves the current points steepest neighbor to the nearest point in the direction
    of the correction. Whether or not a correction is made, the current point is
    assigned the same label as its steepest neighbor.
    
    In this way, the gradient correction history is preserved without need for
    iterative refinement of the edges. In our testing we found this method to be 
    just as or more accurate than the original neargrid method.
    
=== "hybrid-weight"

    In the original weight method, the use of a voronoi cell restricts neighbors
    to exclusively those that share a voronoi facet. This is different from the
    other methods and can results in "local maxima" that have lower values than
    one of their 26 neighbors. This also tends to result in more basins being
    found.
    
    This method performs the weight method, but merges basins that are not
    maxima relative to all 26 nearest neighbors into basins that are.
    
    !!! Note
        The `bader.basin_labels` and `bader.atom_labels` properties 
        for this method are generated by assigning the points to the basin with 
        the highest weight, or the first basin index in the case of a tie.
    