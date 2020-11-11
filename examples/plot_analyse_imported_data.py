"""
Analyze Imported Data
======================

This example will guide you through the process of importing and analyzing data.
"""

# %%
# To import data and analyse it, we only need the ``Track`` class.

from trait2d.analysis import Track

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

# %% In order to run an analysis, we need to load models first.

from trait2d.analysis.models import ModelBrownian, ModelConfined, ModelHop
from trait2d.analysis import ModelDB

ModelDB().add_model(ModelBrownian)
ModelDB().add_model(ModelConfined)
ModelDB().add_model(ModelHop)

# %%
# We can choose the range of data used for the fits with the keyword argument `fractionFitPoints`.

track.adc_analysis(fraction_fit_points = 0.15)
track.plot_adc_analysis_results()

# %%
# It is a good idea to use `ModelDB().cleanup()` at the end of your notebooks to remove all models again. Otherwise they may carry over into other open notebooks.

ModelDB().cleanup()