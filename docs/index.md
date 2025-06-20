# About

BaderKit aims to reproduce the algorithms available in the 
[Henkelman group's](https://theory.cm.utexas.edu/henkelman/code/bader/) excellent 
Fortran code, while utilizing Python's class oriented system to allow for easy 
extension to other projects. The code primarily uses [Numba](https://numba.pydata.org/numba-doc/dev/index.html) 
and [NumPy](https://numpy.org/doc/stable/index.html) to maintain comparable speed. 
[Pymatgen](https://pymatgen.org/) is used under the hood to build out several core 
classes to improve ease of use.

So far, the following algorithms have been implemented.
 

 - [x] On Grid [Comput. Mater. Sci. 36, 354-360 (2006)](https://www.sciencedirect.com/science/article/abs/pii/S0927025605001849)
 - [x] Near Grid [J. Phys.: Condens. Matter 21, 084204 (2009)](https://iopscience.iop.org/article/10.1088/0953-8984/21/8/084204)
 - [x] Weighted [J. Chem. Phys. 134, 064111 (2011)](https://pubs.aip.org/aip/jcp/article-abstract/134/6/064111/645588/Accurate-and-efficient-algorithm-for-Bader-charge?redirectedFrom=fulltext)


!!! Note
    Much of this package runs on [Numba](https://numba.pydata.org/) which compiles 
    python code to machine code at runtime. The compiled code is cached after the 
    first time it runs. As such, the first time you run a Bader algorithm it will 
    be much slower than subsequent runs. 