"""
Bulk processing with ``ListOfTracks``
=====================================

This example will guide you through the process of importing and analyzing multiple tracks from a single ``.csv`` file in bulk.
"""

# %%
# The ``ListOfTracks`` class can load and store multiple tracks in bulk.

from trait2d.analysis import ListOfTracks

# %%
# We can now import data from a ``.csv`` file. In order to separate tracks, the file has to contain the column ``id`` which assigns a unique integer track identifier to each row. You can have a look at :ref:`csv_files` for a complete documentation. Alternative column names can also be supplied using the ``col_name_*`` keyword arguments. (See :meth:`trait2d.analysis.ListOfTracks.from_file` and :meth:`trait2d.analysis.Track.from_file` for this.)
tracks = ListOfTracks.from_file("three_tracks.csv", unit_length='micrometres')

# %%
# The tracks are now imported and can be used for analysis.
# The representation of the ``ListOfTracks`` instance holds some information about it.
tracks

# %%
# To get an overview, we can use :meth:`trait2d.analysis.ListOfTracks.plot_trajectories`

tracks.plot_trajectories()

# %% In order to run an analysis, we need to load models first.

from trait2d.analysis.models import ModelBrownian, ModelConfined, ModelHop
from trait2d.analysis import ModelDB

ModelDB().add_model(ModelBrownian)
ModelDB().add_model(ModelConfined)
ModelDB().add_model(ModelHop)

# %%
# For analysis, :class:`trait2d.analysis.ListOfTracks` uses the same semantics as :class:`Track` for analysis. You can get a summary of the analysis results using :meth:`trait2d.analysis.ListOfTracks.adc_summary`.

tracks.adc_analysis(fraction_fit_points = 0.15)
tracks.adc_summary(plot_dapp=True, plot_pie_chart=True)

# %%
# Single tracks can also be retrieved and viewed easily. It is also possible to run individual analysis on these tracks.

tracks.get_track(0).plot_adc_analysis_results()

# %%
# :meth:`trait2d.analysis.ListOfTracks.adc_summary`'s way of averaging the parameters as well as `Dapp` for each model after analysing each track can be inaccurate. It may thus be desirable to get an average of the categorised curves first and then run an additional analysis. :meth:`trait2d.analysis.ListOfTracks.average` can help with this.
# It returns a special :class:`trait2d.analysis.MSDTrack` object which does not store trajectory data but instead the average MSD values. It can be used for analysis like an ordinary :class:`trait2d.analysis.Track` object.
# Here, we analyse the average of all tracks classified as Confined Diffusion.

confined_tracks = tracks.get_sublist(ModelConfined).average()
confined_tracks.adc_analysis()
confined_tracks.plot_adc_analysis_results()

# %%
# It is a good idea to use `ModelDB().cleanup()` at the end of your notebooks to remove all models again. Otherwise they may carry over into other open notebooks.

ModelDB().cleanup()