"""
Analyze Imported Data
======================

This example will guide you through the process of importing and analyzing data.
"""

# %%
# To import data and analyse it, we only need the ``Track`` class.

from iscat_lib.analysis import Track

# %%
# We can now import data from a ``.csv`` file.
track = Track.from_file("track1.csv", unit_length='micrometres')

# %%
# The track is now imported and can be used for analysis.
# The representation of the track instance holds some information about it.
track

# %%
# To view the trajectory, we can use `Track.plot_trajectory()`.

track.plot_trajectory()

# %%
# We can choose the range of data used for the fits with the keyword argument `fractionFitPoints`.

track.msd_analysis(fraction_fit_points = 0.15)
track.plot_msd_analysis_results()