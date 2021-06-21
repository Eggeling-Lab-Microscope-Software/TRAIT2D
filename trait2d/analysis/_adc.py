from trait2d.analysis import ModelDB

import numpy as np

def delete_adc_analysis_results(self):
    """ Delete the ADC analysis results."""
    self._adc_analysis_results = None

def get_adc_analysis_results(self):
    """Returns the ADC analysis results."""
    return self._adc_analysis_results

def adc_analysis(self, R: float = 1/6, fraction_fit_points: float=0.25, fit_max_time: float = None, maxfev = 1000, enable_log_sampling = False, log_sampling_dist = 0.2, weighting = 'error'):
    """Revised analysis using the apparent diffusion coefficient

    Parameters
    ----------
    R: float
        Point scanning across the field of view.
    fraction_fit_points: float
        Fraction of points to use for fitting. Defaults to 25 %.
    fit_max_time: float
        Maximum time in fit range. Will override fraction_fit_points.
    maxfev: int
        Maximum function evaluations by scipy.optimize.curve_fit. The fit will fail if this number is exceeded.
    enable_log_sampling: bool
        Only sample logarithmically spaced time points for analysis.
    log_sampling_dist: float
        Exponent of logarithmic sampling (base 10).
    weighting: str
        Weighting of the datapoints used in the fit residual calculation. Can be `error` (weight by inverse standard
        deviation), `inverse_variance` (weight by inverse variance), `variance` (weight by variance)
        or `disabled` (no weighting). Default is `error`.

    Returns
    -------
    adc_analysis_results: dict
        Dictionary containing all analysis results.
        Can also be retreived using `Track.get_adc_analysis_results()`.
    """
    # Calculate MSD if this has not been done yet.
    if self._msd is None:
        self.calculate_msd()

    dt = self._t[1] - self._t[0]

    N = self._msd.size

    # Time coordinates
    # This is the time array, as the fits will be MSD vs T
    T = np.linspace(dt, dt*N, N, endpoint=True)

    # Compute  the time-dependent apparent diffusion coefficient.
    Dapp = self._msd / (4 * T * (1 - 2*R*dt / T))
    Dapp_err = self._msd_error / (4 * T * (1 - 2*R*dt / T))

    model, fit_indices, fit_results = self._categorize(np.array(Dapp), np.arange(
        1, N+1), Dapp_err = Dapp_err, R=R, fraction_fit_points=fraction_fit_points, fit_max_time=fit_max_time, maxfev=maxfev, enable_log_sampling=enable_log_sampling, log_sampling_dist=log_sampling_dist, weighting = weighting)

    self._adc_analysis_results = {}
    self._adc_analysis_results["Dapp"] = np.array(Dapp)
    self._adc_analysis_results["Dapp_err"] = np.array(Dapp_err)
    self._adc_analysis_results["fit_indices"] = fit_indices
    self._adc_analysis_results["fit_results"] = fit_results
    self._adc_analysis_results["best_model"] = model

    return self._adc_analysis_results

    
def plot_adc_analysis_results(self):
    """Plot the ADC analysis results.

    Raises
    ------
    ValueError
        Track has not been analyzed using ADC analysis yet.
    """
    import matplotlib.pyplot as plt
    if self.get_adc_analysis_results() is None:
        raise ValueError(
            "Track has not been analyzed using adc_analysis yet!")

    dt = self._t[1] - self._t[0]
    N = self._t.size
    T = np.linspace(1, N, N-3,  endpoint=True) * dt

    fit_results = self.get_adc_analysis_results()["fit_results"]

    Dapp = self.get_adc_analysis_results()["Dapp"]
    Dapp_err = self.get_adc_analysis_results()["Dapp_err"]
    idxs = self.get_adc_analysis_results()["fit_indices"]

    n_points = idxs[-1]

    plt.figure(figsize=(8, 4))
    plt.grid(linestyle='dashed', color='grey')
    plt.semilogx(T, Dapp, label="Data", color='black')
    plt.fill_between(T, Dapp-Dapp_err, Dapp+Dapp_err, color='black', alpha=0.5)
    plt.semilogx(T[idxs], Dapp[idxs], label="Sampled Points", linestyle="", marker="|", markersize=15.0, color='red', zorder=-1)
    for model in fit_results:
        r = fit_results[model]["params"]
        rel_likelihood = fit_results[model]["rel_likelihood"]
        ks_p_value = fit_results[model]["KStestPValue"]
        m = None
        for c in ModelDB().models:
            if c.__class__.__name__ == model:
                m = c
                break
        if m is None:
            raise ValueError("Can't plot results for model {}; make sure the model is loaded in ModelDB()".format(model))
        pred = m(T, *r)
        plt.semilogx(T[0:n_points], pred[0:n_points],
                    label=f"{model}\nrel. likelihood={rel_likelihood:.2e}\nKS p-value={ks_p_value:.2e}")
    model = self.get_adc_analysis_results()["best_model"]

    plt.axvspan(T[0], T[n_points], alpha=0.25,
                color='gray', label="Fit region")
    plt.xlabel("Time in s")
    plt.ylabel("Apparent Diffusion Coefficient")
    plt.title("Diffusion Category: {}".format(model))
    plt.xlim(T[0], T[-1])
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.subplots_adjust(right=0.7)
    plt.show()