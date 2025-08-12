## Gradient Weight Method

The `gradient-weight` method was in testing, but I've determined it doesn't work
better than the weight or neargrid methods and in its current state it is slower
anyways. Therefore I'm removing it from the list of methods and removing the
__init__.py for now.

### Doc Description

| Method        | Speed   | Accuracy | Single Basin Assignments |
|---------------|---------|----------|--------------------------|
|gradient-weight|Fast     |High      |:material-check:          |


**Key Takeaways:** An experimental method with the goal of reaching similar
accuracy to the `neargrid` method without an edge refinment.

This method combines aspects of the `weight` and `ongrid` methods to
achieve high accuracy and speed while maintaining low memory usage.
**It is a work in progress and has not been tested thoroughly.**

Similar to the weight method, a voronoi cell is generated for each point on the
grid. The vectors pointing to each voronoi neighbor are normalized and scaled
based on facet area and distance. For each point on the grid, the scaled
vectors for each higher voronoi neighbor are multiplied by the difference between
the neighbor and central point, then summed together to get a total weighted
gradient. The points are then assigned to the neighbor closest to this vector.