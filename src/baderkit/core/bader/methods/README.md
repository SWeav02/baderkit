This module houses the core code for each Bader partitioning method. 
Each method is organized in a specific format. For an example method with the 
name `example-name`:

1. Must be a class inheriting from the `MethodBase` class in `base.py`
2. Must have a `_run_bader` method that returns all of the required basin charge,
volume, and label information.
3. Must follow a specific naming convention: `ExampleNameMethod`
4. Must be importable from a submodule with the name `example_method`