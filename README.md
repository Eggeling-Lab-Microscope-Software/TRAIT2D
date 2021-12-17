# TRAIT2D

[![PyPI](https://img.shields.io/pypi/v/trait2d)](https://pypi.org/project/trait2d/)
[![PyPI - Wheel](https://img.shields.io/pypi/wheel/trait2d)](https://pypi.org/project/trait2d/#files)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/trait2d)](https://pypi.org/project/trait2d)
[![PyPI - License](https://img.shields.io/pypi/l/trait2d)](https://pypi.org/project/trait2d)
[![DOI](https://zenodo.org/badge/175222492.svg)](https://zenodo.org/badge/latestdoi/175222492)

[![Tests](https://github.com/Eggeling-Lab-Microscope-Software/TRAIT2D/actions/workflows/run-tests.yml/badge.svg)](https://github.com/Eggeling-Lab-Microscope-Software/TRAIT2D/actions/workflows/run-tests.yml)

TRAIT2D (available as `trait2d`) is a cross-platform Python software package with compilable graphical user interfaces (GUIs) to support Single Particle Tracking experiments.  The software can be divided, in three main sections:  the tracker, the simulator and the data analyzer.

The documentation is available at [GitHub Pages](https://eggeling-lab-microscope-software.github.io/TRAIT2D/).

Further information on the tool, together with more extensive theoretical foundations, is available on the related [F1000Research Article](https://f1000research.com/articles/10-838).

## Citing Information 

Reina F, Wigg JMA, Dmitrieva M et al. TRAIT2D: a Software for Quantitative Analysis of Single Particle Diffusion Data \[version 1; peer review: 1 approved, 1 approved with reservations\]. F1000Research 2021, 10:838 \(https://doi.org/10.12688/f1000research.54788.1)

## Features

* SPT simulation, tracking and analysis
* user-friendly GUIs for simple tasks
* customisable libraries for more advanced users
* open source under the GNU General Public License

## Contributing

Guidelines for contributing to the project can be found [here](CONTRIBUTING.md).

## Installation

Installation methods have been tested on Linux and Windows.

### Install from PyPI (Recommended)

Prerequisites:

* A Python installation (version >= 3.7 is required)

Installation:

* run `pip install trait2d`

### Install from Source (Not Recommended)

Prerequisites:

* A Python installation (version >= 3.7 is required)
* *Optional*: A Git installation

Installation:

* clone the GitHub repository

   * run `git clone https://github.com/Eggeling-Lab-Microscope-Software/TRAIT2D`
   * **OR**

      * visit https://github.com/Eggeling-Lab-Microscope-Software/TRAIT2D
      * click the green *Code* button and then *Download ZIP*
      * extract the downloaded `.zip` file anywhere on your computer

* change to the directory that was just created (should contain a `setup.py` file)
* run `pip install -e .`

## Quickstart

### GUIs

There are GUIs available for simple simulation, tracking and analysis tasks.

To start using them follow these steps:

* open a terminal
* if you installed the package with conda to a separate environment, activate it (e.g. `conda activate trait2d`)
* type `trait2d_analysis_gui`, `trait2d_simulator_gui` or `trait2d_tracker_gui` and hit enter

### Library Modules

To use the `trait2d` modules, you can import them in your Python scripts or notebooks.

The simulator module is available as `trait2d.simulators` and the analysis module as `trait2d.analysis`. 

For more information, check the documentation on the [simulation](https://Eggeling-Lab-Microscope-Software.github.io/iSCAT_analysis/release/analysis.html) and [analysis](https://Eggeling-Lab-Microscope-Software.github.io/iSCAT_analysis/release/simulators.html) libraries.

Examples are available in the [gallery](https://Eggeling-Lab-Microscope-Software.github.io/iSCAT_analysis/release/auto_examples/index.html).

## Further GUI Usage

You can find more information and GUI descriptions in the documentation on the [analysis](https://Eggeling-Lab-Microscope-Software.github.io/iSCAT_analysis/release/analysis_gui.html), [simulator](https://Eggeling-Lab-Microscope-Software.github.io/iSCAT_analysis/release/simulator_gui.html), and [tracker](https://Eggeling-Lab-Microscope-Software.github.io/iSCAT_analysis/release/tracker_gui.html) GUIs.

### Detection and Tracking
* Launch a terminal (or the `Anaconda Prompt` on Windows)
* if neccessary, activate the environment (e.g. `conda activate trait2d`)
* run `trait2d_tracker_gui`

#### Setting parameters: 

Use “Preview” button to evaluate performance of the detector. It shows detections for the current frame. 

Parameters:  
* SEF: sigma – parameters of the Laplacian-of-Gaussian(LoG) kernel. It relates to the spot size (increase to detect larger particles) 
* SEF: threshold – parameter for the image threshold after LoG. It relates to the intensity of the spots (decrease to detect particles with lower intensity) 
* SEF: min peak value – parameter of the detected peak size. It relates to the intensity of the spots (decrease to detect particles with lower intensity) 
* patch size – size of the region of interest with the particle (should be bigger than expected particle size). It can influence the particle localisation accuracy.  
* Linking: max distance – maximum possible distance (in pixels) between detections to be linked (increase to link detections located further away from each other) 
* Linking: frame gap – maximum possible number of frames between detections to be linked (increase if the final trajectory is broken into parts) 
* Minimum track length – helps to eliminate short tracks created by false detections 
* resolution - set the value to get trajectories in mircons 
* frame rate - set the value to get timescale of the trajectories in seconds

Proposed workflow:  

1) choose timelapse tiff sequence and run pre-processing step if necessary
2) choose between dark or light spots
3) tune detection parameters to detect all the particles. It is recommended to test the results for a few different frames using "Preview" button 
4) set resolution and frame rate (optional) 
5) set linking parameters
6) run linking by pressing "Run tracking" button. It will run linking algorithm and offer to save tiff file with plotted trajectories. Check the trajectories and change the linking parameters if needed.  Use minimum track length parameter to eliminate short tracks 
7) when the tracks provided by the tracker is good enough - save csv file with the particle trajectories (button “Save data”) 

##### Advice: 

* If the final trajectory is broken into parts - it means, that the detection is failing in a sequence of frames. Firstly, the detection settings can be tuned to detect particles in the sequence, secondly the frame gap can be increased to connect the detection after the sequence of lost detections.  

### Movie simulation: command line
* Launch a terminal (or the `Anaconda Prompt` on Windows)
* Activate the environment with: `conda activate iscat`
* Go to the source directory
* Run the simulation
`python scripts/simulate_iscat_movie.py /path/to/track.csv /path/to/output.tif --psf /path/to/psf.tif --gaussian_noise --poisson_noise`
  * `tracks.csv` is a file containing the tracks to reconstruct
  * The optional `psf.tif` file is a 3D PSF stack were the middle slice is in focus.
  * The PSF can be generated with the ImageJ plugin [DeconvolutionLab2](http://bigwww.epfl.ch/deconvolution/deconvolutionlab2/)
  * Other simulation options can be listed with: `python scripts/simulate_iscat_movie.py --help`
  
### Movie Simulation: GUI
* Launch a terminal (or the `Anaconda Prompt` on Windows)
* If neccessary activate the environment (e.g. `conda activate trait2d`)
* run `trait2d_simulator_gui`
* Generate/load trajectory first, then generate the image sequence and save it

 **Simulated track and iscat movie example. (Left) Raw image, (Right) convolved with a synthetic PSF.**
 <p align="center">
  <img width="608" height="304" src="https://github.com/Eggeling-Lab-Microscope-Software/TRAIT2D/raw/master/examples/simulated_hopping_diffusion_with_and_without_psf.gif">
</p>
