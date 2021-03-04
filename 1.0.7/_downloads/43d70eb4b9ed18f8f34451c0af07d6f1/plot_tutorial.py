"""
TRAIT Tutorial
==============

This example demonstrates some of the library's core features, namely simulation and analysis of 2D diffusion tracks. You will learn about:

Simulation
~~~~~~~~~~

- Simulating multiple tracks with different diffusion models

Analysis
~~~~~~~~

- Apparent Diffusion Coefficient (ADC) analysis of a single track
- ADC analysis of multiple tracks in bulk
- Retreiving an analysis summary of multiple tracks
- Filtering tracks by diffusion category
"""

# %%
# Simulate tracks
# ---------------

# %%
# First, import the required simulators:

from trait2d.simulators import BrownianDiffusion
from trait2d.simulators import HoppingDiffusion

# %%
# A simulator can be initialised with different parameters. For Brownian diffusion, we choose the following:

params = dict()
params["Tmax"] = 0.5 # Maximum simulation time (s)
params["dt"] = 1e-4 # Simulation time resolution (s)
params["dL"] = 1e-12 # Simulation spatial resolution (m)
params["d"] = 1e-12 # Diffusion coefficient (m^2/s)
params["L"] = 1e-5 # Simulation domain size (m)
params["seed"] = 42 # Seed to initialize the random generator (for reproducibility)
params["quantize"] = False # Quantize the position to the simulation spatial resolution grid.

simulator_brownian = BrownianDiffusion(**params)

# %%
# Parameters differ between simulators.

params = dict()
params["Tmax"] = 0.5 # Maximum simulation time (s)
params["dt"] = 1e-4 # Simulation time resolution (s)
params["dL"] = 1e-8 # Simulation spatial resolution (m)
params["Df"] = 8e-13 # Free diffusion coefficient [m^2/s]
params["L"] = 1e-5 # Simulation domain size (m)
params["HP"] = 0.01 # Hopping probability [0-1]
params["HL"] = 1e-6 # Average compartment diameter/length [m]
params["seed"] = 42 # Seed to initialize the random generator (for reproducibility)
params["quantize"] = False # Quantize the position to the simulation spatial resolution grid.

simulator_hop = HoppingDiffusion(**params)

# %%
# After initialisation the simulations can be run. The results will be stored in the simulator object.

simulator_brownian.run();
simulator_hop.run();

# %%
# The simulated trajectoies can be plotted:

simulator_brownian.display_trajectory()
simulator_hop.display_trajectory()

# %%
# It is also possible to export the simulated tracks as videos using the ``trait2d.simulators.iscat_movie`` class. Currently, the tracks need first to be saved e.g. as a ``.csv`` using ``BrownianDiffusion.save()`` (or any other Diffusion model) and then load them again using ``iscat_movie.load_tracks()``. You also need to load a PSF with ``iscat_movie.load_psf()``.

# %%
# Analyse tracks
# --------------

# %%
# Before we start fitting our data, we need to add some models. ``trait2d.analysis.models`` contains a few models that we can add to ``ModelDB``. All models added this way will be used during analysis.

from trait2d.analysis import ModelDB
from trait2d.analysis.models import ModelBrownian, ModelConfined, ModelHop

ModelDB().add_model(ModelBrownian)
ModelDB().add_model(ModelConfined)
ModelDB().add_model(ModelHop)

# %%
# Single tracks are stored in a ``Track`` object.

from trait2d.analysis import Track

# %%
# We can create a single track from our last simulation:

single_track = Track.from_dict(simulator_brownian.trajectory)

# %%
# We can now do ADC analysis on the track:

results = single_track.adc_analysis(fit_max_time=0.5e-1)

# %%
# Analysis results like the calculated values for :math:`D_{app}`, fit parameters and much more are returned in a dictionary. We can also retreive the dictionary of the last analysis at any time with ``get_adc_analysis_results``.

fit_results = results["fit_results"]
best_model = results["best_model"]
print(fit_results)
print(best_model)

single_track.plot_adc_analysis_results()

# %%
# Multiple tracks are then stored in a ``ListOfTracks`` object.

from trait2d.analysis import ListOfTracks

# %%
# For now, we just simulate some more tracks and create a single ``ListOfTracks`` from these tracks. Multiple tracks can also be loaded from a single file using ``ListOfTracks.from_file()``.

import random
tracks = []
for i in range(10):
    simulator_brownian.run()
    simulator_hop.run()
    tracks.append(Track.from_dict(simulator_brownian.trajectory))
    tracks.append(Track.from_dict(simulator_hop.trajectory))
    
tracks = ListOfTracks(tracks)

# %%
# In order to set initial parameters or bounds for the fits, we need to modify the models inside ``ModelDB``. These will then be applied during all analysis from this point on.

ModelDB().get_model(ModelBrownian).initial = fit_results["ModelBrownian"]["params"]
ModelDB().get_model(ModelConfined).initial = fit_results["ModelConfined"]["params"]
ModelDB().get_model(ModelHop).initial = fit_results["ModelHop"]["params"]

# %%
# Here, we set all initial parameters to the results of our single fit from before.

# %%
# Now that we set our initial guesses, let's analyse the remaining tracks at once.
#
# Enabling logarithmic sampling is a good idea since the time axis will be scaled logarithmically by default. We can also set the maximum time on the time for which to fit.
#
# ``adc_analysis`` will return a list containing the indices of all tracks for which a fit has failed. These can then be retreived with ``get_track`` and analysed further.

tracks.adc_analysis(fit_max_time=50e-3, enable_log_sampling=True)

# %%
# ``adc_summary`` gives an overview of the analysis results including optional plots, the averaged parameters for each model, the averaged MSD for each model and the averaged $D_{app}$ for each model.
#
# (We need to set ``interpolation = True`` since some of the time differences in the simulated tracks deviate *slightly* from the expected value.)

tracks.adc_summary(plot_dapp=True, plot_pie_chart=True, interpolation=True)

# %%
# Now that analysis is done we can also retrieve all tracks that fit a certain diffusion category best:

tracks_brownian = tracks.get_sublist(model=ModelBrownian)
tracks_brownian.adc_summary(plot_dapp=True, interpolation=True)

# %%
# As mentioned before, we can retreive the analysis results for any track, at any time. Single tracks can be received with ``ListOfTracks.get_track``.

tracks_brownian.get_track(0).get_adc_analysis_results()

# %%
# We can also plot them:

tracks_brownian.get_track(0).plot_adc_analysis_results()

# %%
# It is a good idea to use `ModelDB().cleanup()` at the end of your notebooks to remove all models again. Otherwise they may carry over into other open notebooks.

ModelDB().cleanup()