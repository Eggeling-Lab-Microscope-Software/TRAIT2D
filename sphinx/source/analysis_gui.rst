.. _analysis_gui:

Analysis GUI
============

The analysis GUI allows single-track analysis without the need for writing Python code or notebooks yourself. It allows to import single tracks from ``.csv`` files and then apply MSD or ADC analysis to them.

Starting the GUI
----------------

The GUI can be started from command line by entering ``trait2d_analysis_gui``. The GUI and its components are described below.

Description of the GUI
----------------------

Data Import Dialog
~~~~~~~~~~~~~~~~~~

.. image:: images/data_import_dialog.png

1. **Select track ID**: Select the ID of the track you want to import. Leave blank for files containing only a single track.

2. **Select x position column**: Select the name of the column containing the x positions.

3. **Select y position column**: Select the name of the column containing the y positions.

4. **Select time column**: Select the name of the column containing the time values.

5. **Select unit of length**: Select the unit of the x and y positions.

6. **Select unit of time**: Select the unit of the time values.

MSD Analysis Tab
~~~~~~~~~~~~~~~~

.. image:: images/msd_analysis_tab.png

1. **Load new track**: Opens the :ref:`Data Import Dialog` to load a new track.

.. warning:: Loading a new track will clear the current track and analysis results.

2. **Linear model results**: Displays the latest analysis results for the linear model.

3. **Power model results**: Displays the latest analysis results for the power model.

4. **Input R**: Input value for the point scanning across the field of view that will be used for all models.

5. **Copy results to clipboard**: Copy all MSD analysis results to the clipboard as a Python dictionary.

6. **Select maximum fit iterations**: Select the maximum number of iterations used internally by ``scipy.optmize`` until the fit aborts.

7. **Use previous results as initial guess**: If checked, the parameter values from the result fields (2. and 3.) will be used as initial values for the next fit.

.. note:: You can enter your own values in the result boxes to choose initial values for the fits yourself.

8. **Run analysis**: Run MSD analysis on the currently loaded track with the current settings.

9. **Plot window**: Will display the fitted curves. Will only display after an analysis has been run.

10. **Set fit range**: Set a maximum fit time ``tmax``. Models will only be fitted in the time interval ``[0, tmax]``.

.. note:: The maximum fit time ``tmax`` can also be set by dragging the white vertical line in the plot window.

11. **Scale time values logarithmically**: If checked, the time axis is scaled logarithmically in the plot window.


ADC Analysis Tab
~~~~~~~~~~~~~~~~

.. image:: images/adc_analysis_tab.png

1. **Load new track**: Opens the :ref:`Data Import Dialog` to load a new track.

.. warning:: Loading a new track will clear the current track and analysis results.

2. **Brownian diff. results**: Displays the latest analysis results for the Brownian diffusion model.

3. **Confined diff. results**: Displays the latest analysis results for the Confined diffusion model.

4. **Hopping diff. results**: Displays the latest analysis results for the Hopping diffusion model.

5. **Input R**: Input value for the point scanning across the field of view that will be used for all models.

6. **Copy results to clipboard**: Copy all ADC analysis results to the clipboard as a Python dictionary.

7. **Select maximum fit iterations**: Select the maximum number of iterations used internally by ``scipy.optmize`` until the fit aborts.

8. **Use previous results as initial guess**: If checked, the parameter values from the result fields (2., 3., and 4.) will be used as initial values for the next fit.

.. note:: You can enter your own values in the result boxes to choose initial values for the fits yourself.

9. **Run analysis**: Run ADC analysis on the currently loaded track with the current settings.

10. **Plot window**: Will display the fitted curves. Will only display after an analysis has been run.

11. **Set fit range**: Set a maximum fit time ``tmax``. Models will only be fitted in the time interval ``[0, tmax]``.

.. note:: The maximum fit time ``tmax`` can also be set by dragging the white vertical line in the plot window.

12. **Scale time values logarithmically**: If checked, the time axis is scaled logarithmically in the plot window.