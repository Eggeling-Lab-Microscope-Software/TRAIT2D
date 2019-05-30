# iSCAT_tracking
particle tracker for iSCAT data (collaboration with Eggeling group)

* data_processing.py - detection and tracking of a movie specified inside the script
* gui_iscat.py - GUI with code for tracking inside
* iscat_lib - classes and functions for detection and tracking

## Installation
* Go to the source directory, and create a conda environment with:
```conda env create -f environment.yml```
* Activate your virtual environment with: `source activate iscat`
* Install the `iscat_lib` package with `pip install -e .`

## Update
* Go to the repository directory
* Update the environment with: `conda env update -f environment.yml`

## Usage
### Detection and Tracking
* Activate the environment with: `conda activate iscat`
* Go to the source directory
* Run the GUI with: `python gui_iscat.py`
* After the analysis, to close the environment use the command `conda deactivate`

### iScat Movie simulation
* Activate the environment with: `conda activate iscat`
* Go to the source directory
* Run the simulation
`python scripts/simulate_iscat_movie.py /path/to/track.csv /path/to/output.tif --psf /path/to/psf.tif --gaussian_noise --poisson_noise`
  * `tracks.csv` is a file containing the tracks to reconstruct
  * The optional `psf.tif` file is a 3D PSF stack were the middle slice is in focus.
  * The PSF can be generated with the ImageJ pluging [DeconvolutionLab2](http://bigwww.epfl.ch/deconvolution/deconvolutionlab2/)
  * Other simulation options can be listed with: `python scripts/simulate_iscat_movie.py --help`
