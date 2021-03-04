.. ISCAT_Tracking documentation master file, created by
   sphinx-quickstart on Sun Apr 26 16:19:42 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TRAIT2D's documentation!
====================================

TRAIT2D (available as ``trait2d``) is a cross-platform Python software package with compilable graphical user interfaces (GUIs) to support Single Particle Tracking (SPT) experiments.

Features
--------

* SPT simulation, tracking and analysis
* user-friendly GUIs for simple tasks
* customisable libraries for more advanced users
* open source under the GNU General Public License

Installation
------------

Installation methods have been tested on Linux and Windows.

Install from PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prerequisites:

* A Python installation (version >= 3.7 is required)

Installation:

* run ``pip install trait2d``

Install from Source (Not Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prerequisites:

* A Python installation (version >= 3.7 is required)
* *Optional*: A Git installation

Installation:

* clone the GitHub repository

   * run ``git clone https://github.com/Eggeling-Lab-Microscope-Software/TRAIT2D``
   * **OR**

      * visit `https://github.com/Eggeling-Lab-Microscope-Software/TRAIT2D`
      * click the green *Code* button and then *Download ZIP*
      * extract the downloaded ``.zip`` file anywhere on your computer

* change to the directory that was just created (should contain a ``setup.py`` file)
* run ``pip install -e .``

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
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Dependencies
------------

The ``trait2d`` package depends on external libraries. If you followed the instructions in the :ref:`Installation` section, these should be installed automatically.

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