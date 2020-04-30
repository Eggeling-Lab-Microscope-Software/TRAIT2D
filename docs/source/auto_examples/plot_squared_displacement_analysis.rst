.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_plot_squared_displacement_analysis.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_plot_squared_displacement_analysis.py:


Squared Displacement Analysis
=============================

This example will guide you through the process of simulating some data and analyzing it using *Squared Displacemet Analysis* (SD).

The ``simulators`` module from ``iscat_lib`` can be used to simulate hopping diffusion.


.. code-block:: default


    from iscat_lib import simulators

    s = simulators.HoppingDiffusion(Tmax=2.5, dt=0.5e-4, HL=1e-6, seed=42)
    s.run()
    s.display_trajectory()
    s.print_parameters()




.. image:: /auto_examples/images/sphx_glr_plot_squared_displacement_analysis_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Simulation:   0%|                                                                                                                                                                                                                    | 0/50000 [00:00<?, ?it/s]    Simulation:  27%|#####################################################                                                                                                                                               | 13543/50000 [00:00<00:00, 135313.67it/s]    Simulation:  54%|##########################################################################################################2                                                                                         | 27099/50000 [00:00<00:00, 135352.70it/s]    Simulation:  82%|###############################################################################################################################################################9                                    | 40791/50000 [00:00<00:00, 135784.28it/s]    Simulation: 100%|####################################################################################################################################################################################################| 50000/50000 [00:00<00:00, 136122.75it/s]
    C:\Users\John\Projekte\iSCAT_tracking\iscat_lib\simulators.py:71: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      plt.show()
    {'Df': 8e-13,
     'HL': 1e-06,
     'HP': 0.01,
     'L': 1e-05,
     'Tmax': 2.5,
     'dL': 2e-08,
     'dt': 5e-05,
     'quantize': True,
     'seed': 42}




We can the analyze the track using the ``Track`` class from the ``analysis`` module.


.. code-block:: default


    from iscat_lib.analysis import Track








We create a new ``Track`` from the simulated trajectory.


.. code-block:: default


    track = Track.from_dict(s.trajectory)








A ``Track`` instance contains not only information about the trajectory but can also hold the MSD data and analysis results.


.. code-block:: default


    track





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    <Track instance at 1573023090560>
    ------------------------
    Track length:      50000
    ------------------------
    MSD calculated:    False
    MSD analysis done: False
    SD analysis done:  False
    ADC analysis done: False




Applying the SD analysis is simple:


.. code-block:: default


    track.sd_analysis()





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    SD analysis for single track:   0%|                                                                                                                                                                                                     | 0/23 [00:00<?, ?it/s]    SD analysis for single track: 100%|###########################################################################################################################################################################################| 23/23 [00:00<00:00, 280.25it/s]
    c:\users\john\miniconda3\envs\iscat\lib\site-packages\scipy\optimize\minpack.py:808: OptimizeWarning: Covariance of the parameters could not be estimated
      category=OptimizeWarning)

    {'analyzed': True, 'model': 'brownian', 'Dapp': array([3.01642024e-12, 1.91500963e-12, 1.46227526e-12, 1.17459978e-12,
           1.20665170e-12, 1.11013515e-12, 7.77450271e-13, 1.00169532e-12,
           8.88695274e-13, 8.56762955e-13, 7.81359254e-13, 8.42912262e-13,
           8.41427696e-13, 8.43006367e-13, 8.11776552e-13, 7.97155661e-13,
           7.95857603e-13, 8.06512831e-13, 8.08971360e-13, 7.96377478e-13,
           7.85209871e-13, 7.86418084e-13, 7.75739694e-13]), 'J': array([  1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  15,  20,  25,
            30,  35,  40,  45,  50,  60,  70,  80,  90, 100]), 'results': {'brownian': {'params': array([7.62615639e-13, 1.25296663e-08]), 'errors': array([4.07902413e-14, 3.40300211e-10]), 'bic': -57.16448412113177, 'rel_likelihood': 1.0}, 'confined': {'params': array([0.00000000e+00, 1.59484431e-08, 0.00000000e+00]), 'errors': array([inf, inf, inf]), 'bic': -53.67477205600419, 'rel_likelihood': 0.17467013396810624}, 'hop': {'params': array([7.62615639e-13, 0.00000000e+00, 1.25296663e-08, 0.00000000e+00]), 'errors': array([inf, inf, inf, inf]), 'bic': -57.16448412113177, 'rel_likelihood': 1.0}, 'n_points': 13, 'R': 0.16666666666666666}}



The analysis results are returned as a dictionary. We can also access them at any time using ``Track.get_adc_analysis_results``.

The ``Track`` instance now holds updated information.


.. code-block:: default


    track





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    <Track instance at 1573023090560>
    ------------------------
    Track length:      50000
    ------------------------
    MSD calculated:    False
    MSD analysis done: False
    SD analysis done:   True
    ADC analysis done: False




The results can also be plotted:


.. code-block:: default


    track.plot_sd_analysis_results()


.. image:: /auto_examples/images/sphx_glr_plot_squared_displacement_analysis_002.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    C:\Users\John\Projekte\iSCAT_tracking\iscat_lib\analysis.py:590: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      plt.show()





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  1.284 seconds)


.. _sphx_glr_download_auto_examples_plot_squared_displacement_analysis.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_squared_displacement_analysis.py <plot_squared_displacement_analysis.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_squared_displacement_analysis.ipynb <plot_squared_displacement_analysis.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
