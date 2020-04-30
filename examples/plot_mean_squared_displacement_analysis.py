"""
Mean Squared Displacement Analysis
==================================

This example will guide you through the process of simulating some data and analyzing it using *Mean Squared Displacemet Analysis* (MSD).
"""

# %%
# The ``simulators`` module from ``iscat_lib`` can be used to simulate hopping diffusion.

from iscat_lib import simulators

s = simulators.HoppingDiffusion(Tmax=2.5, dt=0.5e-4, HL=1e-6, seed=42)
s.run()
s.display_trajectory()
s.print_parameters()

# %%
# We can the analyze the track using the ``Track`` class from the ``analysis`` module.

from iscat_lib.analysis import Track

# %%
# We create a new ``Track`` from the simulated trajectory.

track = Track.from_dict(s.trajectory)

# %%
# A ``Track`` instance contains not only information about the trajectory but can also hold the MSD data and analysis results.

track

# %%
# Applying the MSD analysis is simple:

track.msd_analysis()

# %%
# The analysis results are returned as a dictionary. We can also access them at any time using ``Track.get_adc_analysis_results``.

# %%
# The ``Track`` instance now holds updated information.

track

# %%
# The results can also be plotted:

track.plot_msd_analysis_results()