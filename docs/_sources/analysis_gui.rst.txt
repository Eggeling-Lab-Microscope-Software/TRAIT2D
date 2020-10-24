.. _analysis_gui:

Analysis GUI
============

The analysis GUI allows single-track analysis without the need for writing Python code or notebooks yourself. It allows to import single tracks from ``.csv`` files and then apply MSD or ADC analysis to them.

Starting the GUI
----------------

The GUI can be started from command line by entering ``trait_analysis_gui``. The GUI and its components are described below.

Description of the GUI
----------------------

TODO: Insert screenshot.

General
~~~~~~~
Button "Load Track":
  * Load a single track from a CSV file. The CSV file can contain multiple tracks, each identified by a unique track ID, from which a single track can be selected.

Tabs "MSD Analysis" & "ADC Analysis":
	Switch between the different analysis interfaces.

Both Tabs "MSD Analysis" & "ADC Analysis"
-----------------------------------------
Group box with model name:
	Displays results, errors, relative likelihoods and Bayesian Information Critetion for each analysis model.

Input "R (Motion blur correction)":
[Note: I will add this to the MSD analysis tab as well.]
	Input the constant R for all models.

Button "Results to Clipboard":
	Copy a dictionary containing all analysis results to the clipboard.

Input "Max. fit iterations":
	Maximum iterations until the least squares fit will abort.

Checkbox "Use parameters as initial guesses":
	If checked, the results of the previous fit attempt will be used as initial guesses for the new fit.

Button "Analyze":
	Start model fitting and analysis.

Graph display:
	Displays the fit results and range. Fit range can be modified by dragging the vertical lines.

Button and SpinBox "Set Range":
	Manual input of fit range.

Checkbox "Log Scale":
	If checked, the time axis will be scaled logarithmically.
