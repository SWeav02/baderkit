# About

BaderKit aims to reproduce the algorithms available in the [Henkelman group's](https://theory.cm.utexas.edu/henkelman/code/bader/) excellent Fortran code, while utilizing Python's class oriented system to allow for easy extension to other projects. I have reimplemented the code primarily using [Numba](https://numba.pydata.org/numba-doc/dev/index.html) and [NumPy](https://numpy.org/doc/stable/index.html) to maintain speed where I can. [Pymatgen](https://pymatgen.org/) is used under the hood to build out several core classes to improve ease of use.

This project is currently an early work in progress. So far, I have implemented a simple Bader class and CLI. I hope to add the following algorithms:
 

 - [x]  On Grid [Comput. Mater. Sci. 36, 354-360 (2006)](https://www.sciencedirect.com/science/article/abs/pii/S0927025605001849)
 - [ ] Near Grid [J. Phys.: Condens. Matter 21, 084204 (2009)](https://iopscience.iop.org/article/10.1088/0953-8984/21/8/084204)
 - [x] Weighted [J. Chem. Phys. 134, 064111 (2011)](https://pubs.aip.org/aip/jcp/article-abstract/134/6/064111/645588/Accurate-and-efficient-algorithm-for-Bader-charge?redirectedFrom=fulltext)

The near grid method is working, but gives slightly different results from the Henkelman groups code.