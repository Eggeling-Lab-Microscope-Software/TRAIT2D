.. _analysis:

Analysis Library
================

Single Track
~~~~~~~~~~~~

Simple single-track import and analysis can be done using the ``Track`` class.

.. autoclass:: trait2d.analysis.Track
    :members:

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

.. automodule:: trait2d.analysis.models
    :members: