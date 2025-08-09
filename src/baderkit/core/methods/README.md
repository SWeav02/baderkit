This module houses the core code for each method. Each method is organized in
a specific format. For an example method with the name `example-name`:

1. Must be a class inheriting from the `MethodBase` class in `base.py`
2. Must have a `run` method that returns all of the required basin and vacuum
information as a string.
3. Must follow a specific naming convention: `ExampleNameMethod`
4. Must be importable from a submodule with the name `example_method`