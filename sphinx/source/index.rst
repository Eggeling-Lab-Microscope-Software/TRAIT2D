.. ISCAT_Tracking documentation master file, created by
   sphinx-quickstart on Sun Apr 26 16:19:42 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TRAIT-2D's documentation!
==========================================

***This is taken directly from the preprint as of yet.***

TRAIT-2D is a cross-platform Python software package with compilable graphical user interfaces (GUIs) to support Single Particle Tracking experiments.  The software can be divided, for simplicity, in three main sections:  the tracker, the simulator and the data analyzer.

Features
~~~~~~~~

* user-friendly GUIs for simple tasks
* customisable libraries for more advanced users
* open source under the GNU General Public License

Installation
~~~~~~~~~~~~

This method of installation has been tested on Linux and Windows.

* install a conda distribution (e.g. `Anaconda <https://www.anaconda.com/>`_ or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_)
* Launch a terminal (or the Anaconda Prompt on Windows)
* Go to the source directory, and create a conda environment with: ``conda env create -f environment.yml``
* Activate your virtual environment with ``conda activate iscat``
* Install the ``iscat_lib`` package with ``pip install -e``.

.. toctree::
   :maxdepth: 2
   :caption: Libraries

   analysis
   simulators
   exceptions

.. toctree::
   :maxdepth: 2
   :caption: Graphical Applications
   
   analysis_gui
   simulator_gui
   tracker_gui

.. toctree::
   :maxdepth: 2
   :caption: File Formats

   csv_files

.. toctree::
   :maxdepth: 2
   :caption: Tutorials and Examples
   
   auto_examples/index

Indices and tables
~~~~~~~~~~~~~~~~~~

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Dependencies
~~~~~~~~~~~~

The ``trait-2d`` package depends on external libraries. If you followed the instructions in the :ref:`Installation` section, these should be installed automatically.

They are nevertheless listed here for completeness:

================= =========================================================================
Package Name      License
================= =========================================================================
``imageio``       BSD License (BSD-2-Clause)
``matplotlib``    Python Software Foundation License (PSF)
``numpy``         OSI Approved (BSD)
``opencv-python`` MIT License (MIT)
``pandas``        BSD
``pyqtgraph``     MIT License (MIT)
``PyQt5``         GPL v3
``scikit-image``  BSD License (Modified BSD)
``scipy``         BSD License (BSD)
``tk``            Apache Software License (Apache 2.0 Licence)
``tqdm``          MIT License, Mozilla Public License 2.0 (MPL 2.0) (MPLv2.0, MIT Licences)
================= =========================================================================