# iSCAT_tracking
particle tracker for iSCAT data (collaboration with Eggeling group)

* data_processing.py - detection and tracking of a movie specified inside the script
* gui_iscat.py - GUI with code for tracking inside
* iscat_lib - classes and functions for detection and tracking

## Dependencies
* [Anaconda 3](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
* Python 3
* A git client to clone the repository, and pull the latest version.
  * Linux and Mac user can use the command line git client
  * Recommended clients for Windows are [Github Desktop](https://desktop.github.com/) or [Sourcetree](https://www.sourcetreeapp.com/)

## Installation
* Install [Anaconda](https://www.anaconda.com/distribution/)  (Python 3.7 version)
* Launch a terminal (or the `Anaconda Prompt` on Windows)
* Go to the source directory, and create a conda environment with:
```conda env create -f environment.yml```
* Activate your virtual environment with: `conda activate iscat`
* Install the `iscat_lib` package with `pip install -e .`

## Simplified installation on Windows (tested on Windows 10 Pro)
* Install [Anaconda](https://www.anaconda.com/distribution/)  (Python 3.7 version) - don't change the default installation path
* run (double click) run_environmentSetup.bat to install all necessary packages 
* run (double click) run_gui.bat to activate the environment and run the software (if necessary change the "CODE_FOLDER" to the location of the folder with code on your PC)

## Update
* On Linux or Mac, Open a terminal. On Windows, open the application `Anaconda Prompt`
* Go to the repository directory
* Update the environment with: `conda env update -f environment.yml`

## Usage
### Detection and Tracking
* Launch a terminal (or the `Anaconda Prompt` on Windows)
* Activate the environment with: `conda activate iscat`
* Go to the source directory
* Run the GUI with: `python gui_iscat.py`
* After the analysis, to close the environment use the command `conda deactivate`

### iScat Movie simulation
* Launch a terminal (or the `Anaconda Prompt` on Windows)
* Activate the environment with: `conda activate iscat`
* Go to the source directory
* Run the simulation
`python scripts/simulate_iscat_movie.py /path/to/track.csv /path/to/output.tif --psf /path/to/psf.tif --gaussian_noise --poisson_noise`
  * `tracks.csv` is a file containing the tracks to reconstruct
  * The optional `psf.tif` file is a 3D PSF stack were the middle slice is in focus.
  * The PSF can be generated with the ImageJ plugin [DeconvolutionLab2](http://bigwww.epfl.ch/deconvolution/deconvolutionlab2/)
  * Other simulation options can be listed with: `python scripts/simulate_iscat_movie.py --help`
