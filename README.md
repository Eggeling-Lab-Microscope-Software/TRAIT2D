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
* Activate the environment with: `conda activate iscat`
* Go to the source directory
* Run the GUI with: `python gui_iscat.py`
* After the analysis, to close the environment use the command `conda deactivate`
