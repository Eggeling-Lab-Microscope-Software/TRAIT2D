.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_plot_mean_squared_displacement_analysis.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_plot_mean_squared_displacement_analysis.py:


Mean Squared Displacement Analysis
==================================

This example will guide you through the process of simulating some data and analyzing it using *Mean Squared Displacemet Analysis* (MSD).

The ``simulators`` module from ``iscat_lib`` can be used to simulate hopping diffusion.


.. code-block:: default


    from iscat_lib import simulators

    s = simulators.HoppingDiffusion(Tmax=2.5, dt=0.5e-4, HL=1e-6, seed=42)
    s.run()
    s.display_trajectory()
    s.print_parameters()




.. image:: /auto_examples/images/sphx_glr_plot_mean_squared_displacement_analysis_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Simulation:   0%|                                                                                                                                                                                                                    | 0/50000 [00:00<?, ?it/s]    Simulation:  27%|#####################################################3                                                                                                                                              | 13598/50000 [00:00<00:00, 135863.19it/s]    Simulation:  54%|##########################################################################################################3                                                                                         | 27139/50000 [00:00<00:00, 135691.84it/s]    Simulation:  82%|################################################################################################################################################################8                                   | 41039/50000 [00:00<00:00, 136632.89it/s]    Simulation: 100%|####################################################################################################################################################################################################| 50000/50000 [00:00<00:00, 137244.57it/s]
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


    <Track instance at 2170267376832>
    ------------------------
    Track length:      50000
    ------------------------
    MSD calculated:    False
    MSD analysis done: False
    SD analysis done:  False
    ADC analysis done: False




Applying the MSD analysis is simple:


.. code-block:: default


    track.msd_analysis()





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    MSD calculation (workers: 16):   0%|                                                                                                                                                                                                 | 0/49997 [00:00<?, ?it/s]    MSD calculation (workers: 16):   0%|                                                                                                                                                                                      | 1/49997 [00:01<24:09:52,  1.74s/it]    MSD calculation (workers: 16):   2%|##8                                                                                                                                                                                 | 801/49997 [00:01<16:38:44,  1.22s/it]    MSD calculation (workers: 16):   4%|#######1                                                                                                                                                                           | 2001/49997 [00:02<11:22:04,  1.17it/s]    MSD calculation (workers: 16):   5%|########8                                                                                                                                                                           | 2448/49997 [00:02<7:53:05,  1.68it/s]    MSD calculation (workers: 16):   6%|##########4                                                                                                                                                                         | 2901/49997 [00:02<5:28:04,  2.39it/s]    MSD calculation (workers: 16):   7%|###########8                                                                                                                                                                        | 3294/49997 [00:02<3:47:53,  3.42it/s]    MSD calculation (workers: 16):   9%|################9                                                                                                                                                                   | 4701/49997 [00:02<2:34:44,  4.88it/s]    MSD calculation (workers: 16):  11%|###################2                                                                                                                                                                | 5344/49997 [00:02<1:46:49,  6.97it/s]    MSD calculation (workers: 16):  12%|#####################4                                                                                                                                                              | 5953/49997 [00:02<1:13:48,  9.95it/s]    MSD calculation (workers: 16):  13%|#######################6                                                                                                                                                              | 6501/49997 [00:03<51:05, 14.19it/s]    MSD calculation (workers: 16):  15%|############################                                                                                                                                                          | 7701/49997 [00:03<34:48, 20.25it/s]    MSD calculation (workers: 16):  17%|##############################                                                                                                                                                        | 8258/49997 [00:03<24:07, 28.83it/s]    MSD calculation (workers: 16):  20%|####################################                                                                                                                                                  | 9901/49997 [00:03<16:15, 41.12it/s]    MSD calculation (workers: 16):  22%|########################################1                                                                                                                                            | 11101/49997 [00:03<11:03, 58.60it/s]    MSD calculation (workers: 16):  24%|###########################################                                                                                                                                          | 11901/49997 [00:04<07:37, 83.30it/s]    MSD calculation (workers: 16):  25%|#############################################3                                                                                                                                      | 12601/49997 [00:04<05:15, 118.35it/s]    MSD calculation (workers: 16):  27%|################################################6                                                                                                                                   | 13501/49997 [00:04<03:37, 168.06it/s]    MSD calculation (workers: 16):  28%|###################################################                                                                                                                                 | 14175/49997 [00:04<02:31, 235.99it/s]    MSD calculation (workers: 16):  30%|######################################################                                                                                                                              | 15001/49997 [00:04<01:46, 329.56it/s]    MSD calculation (workers: 16):  33%|##########################################################6                                                                                                                         | 16301/49997 [00:04<01:12, 464.16it/s]    MSD calculation (workers: 16):  35%|##############################################################2                                                                                                                     | 17301/49997 [00:04<00:50, 646.92it/s]    MSD calculation (workers: 16):  38%|###################################################################6                                                                                                                | 18801/49997 [00:05<00:34, 895.51it/s]    MSD calculation (workers: 16):  40%|######################################################################8                                                                                                            | 19801/49997 [00:05<00:24, 1229.70it/s]    MSD calculation (workers: 16):  44%|###############################################################################1                                                                                                   | 22101/49997 [00:05<00:16, 1711.19it/s]    MSD calculation (workers: 16):  50%|#########################################################################################8                                                                                         | 25101/49997 [00:05<00:10, 2374.27it/s]    MSD calculation (workers: 16):  56%|###################################################################################################5                                                                               | 27801/49997 [00:05<00:06, 3256.10it/s]    MSD calculation (workers: 16):  60%|###########################################################################################################5                                                                       | 30047/49997 [00:05<00:04, 4379.25it/s]    MSD calculation (workers: 16):  66%|######################################################################################################################1                                                            | 33001/49997 [00:05<00:02, 5843.61it/s]    MSD calculation (workers: 16):  74%|#####################################################################################################################################1                                             | 37201/49997 [00:05<00:01, 7849.12it/s]    MSD calculation (workers: 16):  84%|#####################################################################################################################################################2                            | 41924/49997 [00:05<00:00, 10466.90it/s]    MSD calculation (workers: 16):  95%|#########################################################################################################################################################################1        | 47501/49997 [00:06<00:00, 13833.49it/s]    MSD calculation (workers: 16): 100%|###################################################################################################################################################################################| 49997/49997 [00:06<00:00, 8302.14it/s]

    {'analyzed': True, 'results': {'model1': {'params': array([5.50212239e-13, 3.37196711e-16]), 'errors': array([7.00587039e-16, 1.41889154e-17]), 'bic': -50.678457239139235, 'rel_likelihood': 0.44585652752829624}, 'model2': {'params': array([ 4.69218746e-13, -7.26080258e-17,  8.63938192e-01]), 'errors': array([5.13750089e-16, 7.78899031e-18, 6.64370407e-04]), 'bic': -52.29397337088078, 'rel_likelihood': 1.0}, 'n_points': 12499}}



The analysis results are returned as a dictionary. We can also access them at any time using ``Track.get_adc_analysis_results``.

The ``Track`` instance now holds updated information.


.. code-block:: default


    track





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    <Track instance at 2170267376832>
    ------------------------
    Track length:      50000
    ------------------------
    MSD calculated:     True
    MSD analysis done:  True
    SD analysis done:  False
    ADC analysis done: False




The results can also be plotted:


.. code-block:: default


    track.plot_msd_analysis_results()


.. image:: /auto_examples/images/sphx_glr_plot_mean_squared_displacement_analysis_002.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    C:\Users\John\Projekte\iSCAT_tracking\iscat_lib\analysis.py:467: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      plt.show()





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  7.805 seconds)


.. _sphx_glr_download_auto_examples_plot_mean_squared_displacement_analysis.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_mean_squared_displacement_analysis.py <plot_mean_squared_displacement_analysis.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_mean_squared_displacement_analysis.ipynb <plot_mean_squared_displacement_analysis.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
