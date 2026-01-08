![title](/images/full_title.svg)

BaderKit is a python package providing tools for calculating atomic charges using principles from Bader's Quantum Theory of Atoms in Molecules. The project has expanded to include methods for analyzing the Electron Localization Function. For more information on the core classes of the package, see their quick-start pages:

 - [Bader](/bader/usage.md)
 - [BadELF](/badelf/usage.md)
 - [ElfLabeler](/elf_labeler/usage.md)

---

## Installation

=== "conda (recommended)"
    If you haven't already, install a conda environment manager such as [Anaconda](https://www.anaconda.com/download).
    
    1. Create a conda environment using the following command in your terminal.
    Replace `my_env` with whatever name you prefer.
    ```bash
    conda create -n my_env
    ```
    2. Activate your environment.
    ```bash
    conda activate my_env
    ```
    3. Install BaderKit.
    ```bash
    conda install -c conda-forge baderkit
    ```
    4. Confirm the install by running the help command.
    ```bash
    baderkit --help
    ```

=== "pip"
    We generally recommend using a virtual environment manager such as
    [Anaconda](https://www.anaconda.com/download) or [venv](https://docs.python.org/3/library/venv.html)
    to keep your Python work environments isolated. If you don't want to,
    you can still use pip so long as Python is installed.
    
    1. Install BaderKit with the following command in your terminal.
    ```bash
    pip install baderkit
    ```
    2. Confirm the install by running the help command
    ```bash
    baderkit --help
    ```
        
=== "GUI App (Optional)"

    In addition to the core package, there is an optional GUI feature which allows
    for easy viewing and plotting of results. This requires extra dependencies which
    can be installed through pip.
    
    ```bash
    pip install baderkit[gui]
    ```
    
    !!! Note
        This is kept as optional as the GUI requires significantly more dependencies
        than the base app. Unfortunately, this means conda cannot
        be used, as it does not allow for optional dependencies.