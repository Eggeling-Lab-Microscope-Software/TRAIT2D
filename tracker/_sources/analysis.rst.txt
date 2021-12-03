.. _analysis:

Analysis Library
================

Analysis Methods
~~~~~~~~~~~~~~~~

The analysis library supports MSD analysis as well as ADC analysis. ADC analysis generalises MSD analysis by introducing a time dependant *apparent diffusion coefficient* :math:`D_\mathrm{app}(t)` and takes into account the localisation imprecision as well as the microsope's illumination mode.

MSD analysis includes a linear model as well as a generic power law model. ADC analysis includes predefined models for *brownian*, *confined*, *hop*, and *immobile diffusion* as well as the possibility to add custom models.

Single Track
~~~~~~~~~~~~

Simple single-track import and analysis can be done using the ``Track`` class.

.. autoclass:: trait2d.analysis.Track
    :members:

.. autoclass:: trait2d.analysis.MSDTrack

Multiple Tracks
~~~~~~~~~~~~~~~

If multiple tracks have to be imported and analysed in bulk, the ``ListOfTracks`` class can be used.

.. autoclass:: trait2d.analysis.ListOfTracks
    :members:

ModelDB
~~~~~~~

``ModelDB`` allows managing which models are used for ADC analysis. You can define your own models or use predefined models (see the :ref:`Models` section).

.. autoclass:: trait2d.analysis.ModelDB
    :members:

Models
~~~~~~

There are some predefined models available for analysis. Some of these are only used for MSD analysis internally and should not be used for ADC analysis. For more information on adding or removing models to the analysis, see the :ref:`ModelDB` section.

In the model formulas, :math:`\delta` refers to the time-invariable localisation imprecision and :math:`dt` is the dicretisation time step of the track.

Be aware that :math:`R` is *not* a fit parameter but a constant value which describes the point scanning across the field of view depending on the microscope's illumination mode. It can be set in the analysis GUI or supplied as a keyword parameter to the :meth:`trait2d.analysis.Track.adc_analysis` method.

.. automodule:: trait2d.analysis.models
    :members: