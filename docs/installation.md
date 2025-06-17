# Installation

BaderKit can be installed with either pip or conda. We recommend using conda as it is generally more stable.

1. Download and install [anaconda](https://www.anaconda.com/download)
2. Within a conda shell, create a conda environment and install BaderKit
   ```bash
   conda create -n {your_env_name} -c conda-forge baderkit
   conda activate {your_env_name}
   ```
3. Confirm the install by running the help command
   ```bash
   baderkit --help
   ```

Alternatively, if you don't mind the potential struggles, install using pip.
```bash
pip install baderkit
```

!!! Note
    Much of this package runs on [Numba](https://numba.pydata.org/) which compiles python code to machine code at runtime. The compiled code is cached after the first time it runs. As such, the first time you run a Bader algorithm it will likely be much slower than subsequent runs. 