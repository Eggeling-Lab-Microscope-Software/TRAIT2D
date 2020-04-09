# iSCAT_analysis
particle tracker for iSCAT data (collaboration with Eggeling group)


* data_processing.py - detection and tracking of a movie specified inside the script
* gui_iscat.py - GUI with code for tracking inside
* gui_simulator.py - GUI with code for a simulator (in development)
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

#### Setting parameters: 

Use “preview” button to evaluate detection of the particles.  It shows detection for a random frame with a given parameters. 

Parameters:  
* SEF: sigma – relates to the spot size (increase to detect bigger particles) 
* SEF: threshold – relates to the intensity of the spots (decrease to detect particles with less intensity) 
* SEF: min peak value – relates to the intensity of the spots (decrease to detect particles with less intensity) 
* patch size – size of the region of interest with the particle (bigger than expected particle size). It can influence the particle localisation accuracy.  
* Linking: max distance – maximum possible distance between detections to be linked (decrease to eliminate wrong linking, increase if the right detections are not linked) 
* Linking: frame gap – maximum possible number of frames between detections to be linked (increase if the final trajectory is broken into parts) 
* Minimum track length – helps to eliminate short tracks created by false detections 

Proposed strategy:  

1) choose movie sequence and run pre-processing step if necessary; 
2) choose between dark or light spots; 
3)  tune the setting to detect all the particles. It is better to have false positive detections rather than miss some particles; 
4) run linking. It offers to save tiff file with plotted trajectories. Check the trajectories and set the linking parameters if needed.  Use minimum track length parameter to eliminate short tracks; 
5) when the tracks provided by the tracker is good enough - save csv file with the particle trajectories (button “save data”) 

##### Advice: 

* If the final trajectory is broken into parts - it means, that the detection is failing in a sequence of frames. Firstly, detection settings can be tuned to detect particles in the sequence, secondly the frame gap can be increased to connect the detection after the sequence.  

### iScat Movie simulation: command line
* Launch a terminal (or the `Anaconda Prompt` on Windows)
* Activate the environment with: `conda activate iscat`
* Go to the source directory
* Run the simulation
`python scripts/simulate_iscat_movie.py /path/to/track.csv /path/to/output.tif --psf /path/to/psf.tif --gaussian_noise --poisson_noise`
  * `tracks.csv` is a file containing the tracks to reconstruct
  * The optional `psf.tif` file is a 3D PSF stack were the middle slice is in focus.
  * The PSF can be generated with the ImageJ plugin [DeconvolutionLab2](http://bigwww.epfl.ch/deconvolution/deconvolutionlab2/)
  * Other simulation options can be listed with: `python scripts/simulate_iscat_movie.py --help`
  
### iScat Movie simulation: GUI
* Launch a terminal (or the `Anaconda Prompt` on Windows)
* Activate the environment with: `conda activate iscat`
* Go to the source directory
* Run the simulator `python gui_simulator.py`
* Generate/load trajectory first, then generate the image sequence and save it

 **Simulated track and iscat movie example. (Left) Raw image, (Right) convolved with a synthetic PSF.**
 <p align="center">
  <img width="608" height="304" src="examples/simulated_hopping_diffusion_with_and_without_psf.gif">
</p>
