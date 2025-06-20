# Background

In chemistry and materials science, we often find ourselves talking about the 
oxidation state of a given atom. We discuss the atom as if it has taken or given 
exactly one electron. However, in real molecules and materials, the charge density 
is a smooth continuous function, and the electrons do not "belong" to any given atom.

Indeed, the concept of oxidation states is not founded in quantum mechanics, and 
is effectively just a tool invented by chemists through the years. However, it 
is an exceptionally useful tool that allows for qualitative understanding of a 
vast array of concepts from bonding in organic molecules, to doping in metal alloys, 
to the charging of lithium ion batteries.

Because oxidation states are not a quantum phenomenon, many different methods 
have been proposed to derive them in convenient and appealing ways. Many, such 
as Mülliken analysis, derive oxidation states by approximating molecular orbitals 
as a linear combination of atomic orbitals (the famous LCAO concept). However, 
the choice to approximate molecular orbitals this way is entirely arbitrary, and 
slight differences can result in vastly different oxidation states.

An alternative method, proposed by Bader in his Quantum Theory of Atoms in Molecules 
(QTAIM), relies solely on the charge density. The charge density is in general less 
sensitive to the choice of basis set used to approximate the wavefunction of the system, 
and is observable through experiment. In the QTAIM method, the charge is separated 
into regions by a surface located at local minima throughout the charge density. 
This is often called a zero-flux surface and, more technically, is made up of points 
where the gradient normal to the surface is zero.

We describe the regions defined by these surfaces as *basins*, though they are 
also commonly called *Bader regions*. Each basin has exactly one local maximum, 
sometimes termed an *attractor*. Each attractor/basin typically (though not always) 
correspond to an atom, and the charge and oxidation state of the atom can be determined 
by integrating the charge density within this region.

In practice, it is often difficult and computationally expense to thoroughly 
sample the true zero-flux surface defining basins. To avoid this problem, 
Henkelman et. al. proposed a brilliantly simple alternative utilizing the 
common representation of the charge density as points on a regular grid. 
Each point on the grid is assigned to a basin by climbing a path along the 
steepest gradient until a local maximum is reached. Repeat gradient calculations 
can be avoided by stopping the ascent when a previously assigned path is reached. 
The end result is a robust and efficient method for dividing the charge density 
into basins, without ever needing to calculate the exact location of the zero-flux 
surface.

Through the years, several methods for performing this steepest ascent have been
developed. We have implemented the same methods that exist in the Henkelman group's
excellent Fortran code, as well as several slightly altered versions of them. We
highly select reading about each on our [Methods](./methods.md) page to determine
which is the best for your use case.

