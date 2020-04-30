.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_plot_apparent_diffusion_coefficient_analysis.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_plot_apparent_diffusion_coefficient_analysis.py:


Apparent Diffusion Coefficient Analysis
=======================================

This example will guide you through the process of simulating some data and analyzing it using *Apparent Diffusion Coefficient Analysis* (ADC).

The ``simulators`` module from ``iscat_lib`` can be used to simulate hopping diffusion.


.. code-block:: default


    from iscat_lib import simulators

    s = simulators.HoppingDiffusion(Tmax=2.5, dt=0.5e-4, HL=1e-6, seed=42)
    s.run()
    s.display_trajectory()
    s.print_parameters()




.. image:: /auto_examples/images/sphx_glr_plot_apparent_diffusion_coefficient_analysis_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Simulation:   0%|                                                                                                                                                                                                                    | 0/50000 [00:00<?, ?it/s]    Simulation:  27%|####################################################9                                                                                                                                               | 13496/50000 [00:00<00:00, 134844.07it/s]    Simulation:  54%|#########################################################################################################5                                                                                          | 26930/50000 [00:00<00:00, 134657.73it/s]    Simulation:  81%|##############################################################################################################################################################8                                     | 40514/50000 [00:00<00:00, 134975.55it/s]    Simulation: 100%|####################################################################################################################################################################################################| 50000/50000 [00:00<00:00, 135201.65it/s]
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


    <Track instance at 2950231095336>
    ------------------------
    Track length:      50000
    ------------------------
    MSD calculated:    False
    MSD analysis done: False
    SD analysis done:  False
    ADC analysis done: False




Applying the ADC analysis is simple:


.. code-block:: default


    track.adc_analysis()





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    MSD calculation (workers: 16):   0%|                                                                                                                                                                                                 | 0/49997 [00:00<?, ?it/s]    MSD calculation (workers: 16):   0%|                                                                                                                                                                                      | 1/49997 [00:01<23:55:42,  1.72s/it]    MSD calculation (workers: 16):   3%|#####7                                                                                                                                                                             | 1601/49997 [00:01<16:12:50,  1.21s/it]    MSD calculation (workers: 16):   4%|#######1                                                                                                                                                                           | 1990/49997 [00:02<11:15:38,  1.18it/s]    MSD calculation (workers: 16):   6%|##########                                                                                                                                                                          | 2801/49997 [00:02<7:44:59,  1.69it/s]    MSD calculation (workers: 16):   7%|###########8                                                                                                                                                                        | 3301/49997 [00:02<5:22:05,  2.42it/s]    MSD calculation (workers: 16):   8%|#############5                                                                                                                                                                      | 3765/49997 [00:02<3:43:17,  3.45it/s]    MSD calculation (workers: 16):  10%|#################6                                                                                                                                                                  | 4901/49997 [00:02<2:32:29,  4.93it/s]    MSD calculation (workers: 16):  11%|###################8                                                                                                                                                                | 5501/49997 [00:02<1:45:22,  7.04it/s]    MSD calculation (workers: 16):  12%|######################3                                                                                                                                                             | 6201/49997 [00:02<1:12:40, 10.04it/s]    MSD calculation (workers: 16):  15%|##########################9                                                                                                                                                           | 7401/49997 [00:03<49:30, 14.34it/s]    MSD calculation (workers: 16):  16%|#############################1                                                                                                                                                        | 8001/49997 [00:03<34:12, 20.46it/s]    MSD calculation (workers: 16):  18%|################################7                                                                                                                                                     | 9001/49997 [00:03<23:24, 29.19it/s]    MSD calculation (workers: 16):  19%|##################################8                                                                                                                                                   | 9586/49997 [00:03<16:11, 41.61it/s]    MSD calculation (workers: 16):  20%|####################################7                                                                                                                                                | 10149/49997 [00:03<11:12, 59.22it/s]    MSD calculation (workers: 16):  22%|#######################################8                                                                                                                                             | 11001/49997 [00:03<07:43, 84.14it/s]    MSD calculation (workers: 16):  23%|#########################################4                                                                                                                                          | 11519/49997 [00:03<05:22, 119.35it/s]    MSD calculation (workers: 16):  25%|#############################################7                                                                                                                                      | 12701/49997 [00:03<03:40, 169.29it/s]    MSD calculation (workers: 16):  27%|###############################################8                                                                                                                                    | 13301/49997 [00:04<02:33, 238.60it/s]    MSD calculation (workers: 16):  29%|###################################################8                                                                                                                                | 14401/49997 [00:04<01:45, 337.12it/s]    MSD calculation (workers: 16):  30%|######################################################3                                                                                                                             | 15101/49997 [00:04<01:14, 468.39it/s]    MSD calculation (workers: 16):  33%|##########################################################6                                                                                                                         | 16301/49997 [00:04<00:51, 655.47it/s]    MSD calculation (workers: 16):  34%|#############################################################3                                                                                                                      | 17027/49997 [00:04<00:37, 885.30it/s]    MSD calculation (workers: 16):  37%|##################################################################5                                                                                                                | 18601/49997 [00:04<00:25, 1231.73it/s]    MSD calculation (workers: 16):  39%|#####################################################################8                                                                                                             | 19501/49997 [00:04<00:18, 1650.60it/s]    MSD calculation (workers: 16):  43%|############################################################################6                                                                                                      | 21401/49997 [00:04<00:12, 2260.30it/s]    MSD calculation (workers: 16):  47%|####################################################################################4                                                                                              | 23601/49997 [00:05<00:08, 3016.93it/s]    MSD calculation (workers: 16):  56%|###################################################################################################5                                                                               | 27801/49997 [00:05<00:05, 4176.09it/s]    MSD calculation (workers: 16):  62%|##############################################################################################################2                                                                    | 30801/49997 [00:05<00:03, 5623.36it/s]    MSD calculation (workers: 16):  68%|#########################################################################################################################7                                                         | 34001/49997 [00:05<00:02, 7415.70it/s]    MSD calculation (workers: 16):  76%|########################################################################################################################################                                           | 38001/49997 [00:05<00:01, 9809.85it/s]    MSD calculation (workers: 16):  87%|##########################################################################################################################################################1                       | 43301/49997 [00:05<00:00, 12926.24it/s]    MSD calculation (workers: 16): 100%|###################################################################################################################################################################################| 49997/49997 [00:05<00:00, 8737.80it/s]
    c:\users\john\miniconda3\envs\iscat\lib\site-packages\scipy\optimize\minpack.py:808: OptimizeWarning: Covariance of the parameters could not be estimated
      category=OptimizeWarning)

    {'analyzed': True, 'model': 'hop', 'Dapp': array([2.18780751e-12, 1.35533732e-12, 1.14315145e-12, ...,
           4.05181783e-13, 4.03886926e-13, 4.00018668e-13]), 'results': {'brownian': {'params': array([7.85781122e-13, 9.67714666e-09]), 'errors': array([1.59537685e-16, 9.52620267e-13]), 'bic': -48.84040571259864, 'rel_likelihood': 0.9999999999063398}, 'confined': {'params': array([0., 0., 0.]), 'errors': array([inf, inf, inf]), 'bic': -46.870403132753964, 'rel_likelihood': 0.3734387451943948}, 'hop': {'params': array([7.85781122e-13, 0.00000000e+00, 9.67714666e-09, 0.00000000e+00]), 'errors': array([inf, inf, inf, inf]), 'bic': -48.84040571278596, 'rel_likelihood': 1.0}, 'n_points': 12499, 'R': 0.16666666666666666}}



The analysis results are returned as a dictionary. We can also access them at any time using ``Track.get_adc_analysis_results``.

The ``Track`` instance now holds updated information.


.. code-block:: default


    track





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    <Track instance at 2950231095336>
    ------------------------
    Track length:      50000
    ------------------------
    MSD calculated:     True
    MSD analysis done: False
    SD analysis done:  False
    ADC analysis done:  True




The results can also be plotted:


.. code-block:: default


    track.plot_adc_analysis_results()


.. image:: /auto_examples/images/sphx_glr_plot_apparent_diffusion_coefficient_analysis_002.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    C:\Users\John\Projekte\iSCAT_tracking\iscat_lib\analysis.py:528: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      plt.show()





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  7.163 seconds)


.. _sphx_glr_download_auto_examples_plot_apparent_diffusion_coefficient_analysis.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_apparent_diffusion_coefficient_analysis.py <plot_apparent_diffusion_coefficient_analysis.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_apparent_diffusion_coefficient_analysis.ipynb <plot_apparent_diffusion_coefficient_analysis.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
