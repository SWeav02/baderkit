# Methods and Benchmarks

The available BadELF methods differ from the [Bader](../../bader/methods) methods in that they refer to different partitioning schemes rather than methods of approximating the smooth zero-flux surface.

## Methods

=== "zero-flux (default)"

    **Key Takeaway:** *Fastest method with some potential interstitial bias.*
    
    This is the primary method used historically. It is conceptually very similar to Bader's method, dividing the ELF into regions (called basins) separated by a zero-flux surface. The charge density is then integrated to assign charge to each basin. This method essentially wraps the `Bader` class and retains all of its speed and rigor. 
    
    There are two potential downsides of this method. The first is that in highly ionic systems, nearly all of the interstitial region is assigned to the atom with a higher ELF value (typically the anion). This leads to near formal charges, and it is difficult to pull out information about any degree of covalency. The second is that chemical features that are shared by multiple atomes (i.e. covalent/metallic bonds) are not rigorously divided, requiring an arbitrary choice of how to split them up.
    
    Despite these downsides, we have chosen to use this method as the default as it scales significantly better than the alternative methods and has been prefered in the historical literature.
    
=== "voronelf"

    **Key Takeaway:** *Tries to alleviate issues in the zero-flux method. Works well for systems with entirely concave basins*
    
    This method is designed to alleviate the issues described in the zero-flux method. Rather than relying on a full zero-flux surface, planes are placed perpendicular to the vector between each atom and its neighbors. The placement of the planes depends on the type of interaction. For ionic bonds, the plane is placed at the minimum between the atoms which corresponds to a point on the zero-flux surface. For covalent and metallic bonds, a separate basin appears in the ELF between the atoms, and the plane is placed at the maximum along the bond which falls in this region. 
    
    The use of planes means the interstitial space is more evenly divided between the atoms and allows covalent/metallic features to be separated rigorously. However, this method has several issues of its own. The most important is that it assumes the ELF basin's to be concave. This is usually the case for atoms, but is not true for many other features such as metal bonds and electride electrons. The second is that it scales rather poorly, depending on both the number of grid points and the number of planes that must be constructed. Therefore, we tend to prefer the other methods.

=== "badelf"

    **Key Takeaways:** *A hybrid of the zero-flux and voronelf methods. Often best for electride systems.*
    
    This is the original 'BadELF' method created by the [Warren Lab](https://pubs.acs.org/doi/10.1021/jacs.3c10876) at UNC. It is a hybrid of the other two methods, using a zero-flux surface to separate bare electrons in electrides and planes to separate atoms. For electride systems, this often helps alleviate the issues of the voronelf method. The downsides are that it retains the potential bias of the zero-flux method around electride sites and will still behave poorly for systems where atoms or metallic bonds are not concave and fairly spherical.
    
    We recommend this method for most electrides, but suggest a more in-depth analysis for complex systems that may have heavy polarization or mixed metal/electride features.

