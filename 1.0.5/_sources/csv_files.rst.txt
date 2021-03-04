.. _csv_files:

CSV Files
=========

``.csv`` is currently the only supported file format for data input using :meth:`trait2d.analysis.Track.from_file` or :meth:`trait2d.analysis.ListOfTracks.from_file`. Other file formats will have to be converted to a ``pandas.DataFrame`` first which can then be imported using :meth:`trait2d.analysis.Track.from_dataframe`.

.. note:: *Rudimentary* import form native Python dictionaries is supported through :meth:`trait2d.analysis.Track.from_dict`. It is the least flexible though, and thus *not* recommended.

Below, the expected strucure of the ``.csv`` files is described. The same structure also applies to ``DataFrame`` objects used for data import.

Data Structure
--------------

The ``.csv`` file has to contain the following columns:

* ``x``: x-component of the particle localisations
* ``y``: y-component of the particle localisations
* ``t``: time component of the particle localisations

In case there is more than one track stored inside the ``.csv`` file, the following column has to be present as well:

* ``id``: unique ID of the track for *each* row

The default column names can also be changed using the ``col_name_*`` keyword arguments. See :meth:`trait2d.analysis.Track.from_file` and :meth:`trait2d.analysis.Track.from_dataframe` for detailed information.

.. note:: :meth:`trait2d.analysis.Track.from_dict` does *not* allow custom column names.

Units
-----

SI units are assumed, that is, metres for ``x`` and ``y`` and seconds for ``t``. However, units can also be specified using the ``unit_*`` keyword arguments. See :meth:`trait2d.analysis.Track.from_file` and :meth:`trait2d.analysis.Track.from_dataframe` for detailed information.

.. note:: :meth:`trait2d.analysis.Track.from_dict` does *not* allow custom units.