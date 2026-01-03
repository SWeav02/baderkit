# Background

In chemistry and materials science, we often find ourselves talking about the 
oxidation state of a given atom. We discuss the atom as if it has taken or given 
exactly one electron. However, in real molecules and materials, the charge density 
is a smooth continuous function, and the electrons do not "belong" to any given atom. 
Indeed, the concept of oxidation states is not founded in quantum mechanics, and 
is effectively a tool invented by chemists through the years. Still, it 
is an exceptionally useful tool that allows for qualitative understanding of a 
vast array of concepts from bonding in organic molecules, to doping in metal alloys, 
to the charging of lithium ion batteries.

There have been many methods for calculating oxidation states proposed through
the years. One of the most popular was proposed by Bader in his Quantum Theory 
of Atoms in Molecules (QTAIM). This method relies solely on the charge density,
which is relatively insensitive to the choise of basis set used to approximate 
the wavefunction of the system, and is observable through experiment. 
In the QTAIM method, the charge is separated 
into regions by a surface located at local minima throughout the charge density. 
More technically, this "zero-flux" surface is made up of points where the gradient 
normal to the surface is zero.

We describe the regions defined by these surfaces as *basins*, though they are 
also commonly called *Bader regions*. Each basin has exactly one local maximum, 
sometimes termed an *attractor*. Each attractor typically (though not always) 
correspond to an atom, and the charge and oxidation state of the atom can be determined 
by integrating the charge density within this region.

![bader_separation](/images/bader_separation_wb.png)

In practice, it is often difficult and computationally expense to thoroughly 
sample the zero-flux surface defining basins. To avoid this problem, 
Henkelman et. al. proposed a brilliantly simple alternative utilizing the 
common representation of the charge density as points on a regular grid. 
Each point on the grid is assigned to a basin by climbing the 
steepest gradient until a local maximum is reached. Repeat gradient calculations 
can be avoided by stopping the ascent when a previously assigned path is reached. 
The end result is a robust and efficient method for dividing the charge density 
into basins, without ever needing to calculate the exact location of the zero-flux 
surface.

Since the development of their original algorithm the Henkelman group and others have
developed several improved methods for performing this steepest ascent. Each has
its own advantages and disadvantages. We recommend reading through our [Methods and Benchmarks](/baderkit/bader/methods)
page to determine the best one for your use case.
    