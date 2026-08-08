from baderkit.post_wfc.projection.projection_environment import AtomicProjectionEnvironment
from baderkit.post_wfc.base_env import PostWFC

wfc = PostWFC()
proj = wfc.get_iao_projection()
proj.get_real_space_crystal_orbital_population(
    return_plot=True,
    plot_range=[-6,10],
)