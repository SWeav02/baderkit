# Methods and Benchmarks

The available BadELF methods differ from the [Bader](../../bader/methods) methods in that they refer to different partitioning schemes rather than methods of approximating the smooth zero-flux surface.

---

## Methods
    
=== "badelf (default)"

    **Key Takeaways:** *A hybrid method that mixes voronoi-like planes between atoms and a zero-flux surface surrounding electrides.*
    
    This method is designed to alleviate issues when the ELF value between adjacent atoms differs heavily. When using zero-flux partitioning, atoms with greater ELF values tend to fully dominate the interstitial area even when the difference in ELF values is not so great. To help reduce this bias, the BadELF algorithm uses a weighted voronoi scheme with planes placed at minima along atomic bonds to separate atoms. However, this concept relies on the idea that atoms are relatively spherical. While this is often the case for atoms, there are many cases for electrides where the bare electrons are not spherical. Thus we continue to use the zero-flux separation for bare electrons, sacrificing some bias in the interstitial region.

    This method is the original 'BadELF' method, and is thus the default here. However, it is worth noting that the reliance on planes rather than the zero-flux surface makes this method take significantly longer and adds a reliance on the assumption that each atom is relatively spherical. In general, the original zero-flux method may be more consistent.
    
=== "zero-flux"

    **Key Takeaway:** *Faster than BadELF with some potential interstitial bias.*
    
    The zero-flux method really predates the BadELF method, having been used since the ELF's conception. It is essentially identical to Bader's method with the exception that the system is divided into regions using the ELF rather than the charge-density. As such, this method relies fully on the `Bader` class under the hood and retains all of its speed and rigor. Its main downside is a potential biasing of charge towards those atoms with higher ELf values. A secondary downside is that covalent and metallic bonds must be empirically divided to their neighboring atoms rather than using a dividing surface like in `badelf` or `voronelf`.
    

=== "vornelf"

    **Key Takeaway:** *Similar to BadELF but potentially better for spherical electrides.*

    This method foregoes the zero-flux method entirely, instead opting to use voronoi-like planes for separating both atoms and bare electrons. This may work well if all atom/electride species are fairly spherical, but has the potential to produce nonsense for non-spherical electrides.