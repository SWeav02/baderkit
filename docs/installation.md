# Installation

## Pip

To install BaderKit with pip, install python and run the following command in
your terminal.

```bash
pip install baderkit
```

## Conda

Alternatively, it is often more stable to install through Conda. Note that while 
pip is updated immediately on each release, there is some downtime before 
conda-forge is updated with the most recent version.

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

## Optional Webapp

In addition to the core package, there is an optional webapp feature which allows
for easy viewing and plotting of results. This requires extra dependencies which
can be installed through pip with the following command:
```bash
pip install baderkit[webapp]
```

