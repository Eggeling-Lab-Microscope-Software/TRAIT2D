from iscat_lib.analysis import ModelDB

import numpy as np

def delete_adc_analysis_results(self):
    """ Delete the ADC analysis results."""
    self._adc_analysis_results = {
        "analyzed": False, "model": "unknown", "Dapp": None, "J": None, "results": None}

def get_adc_analysis_results(self):
    """Returns the ADC analysis results."""
    return self._adc_analysis_results

def adc_analysis(self, R: float = 1/6, fraction_fit_points: float=0.25, fit_max_time: float = None, num_workers=None, chunksize=100, initial_guesses = {}, maxfev = 1000, enable_log_sampling = False, log_sampling_dist = 0.2):
    """Revised analysis using the apparent diffusion coefficient

    Parameters
    ----------
    R: float
        Point scanning across the field of view.
    fraction_fit_points: float
        Fraction of points to use for fitting. Defaults to 25 %.
    fit_max_time: float
        Maximum time in fit range. Will override fraction_fit_points.
    num_workers: int
        Number or processes used for calculation. Defaults to number of system cores.
    chunksize: int
        Chunksize for process pool mapping. Small numbers might have negative performance impacts.
    initial_guesses: dict
        Dictionary containing initial guesses for the parameters. Keys can be "brownian", "confined" and "hop".
        All values default to 1.
    maxfev: int
        Maximum function evaluations by scipy.optimize.curve_fit. The fit will fail if this number is exceeded.
    """
    # Calculate MSD if this has not been done yet.
    if self._msd is None:
        self.calculate_msd(num_workers=num_workers, chunksize=chunksize)

    dt = self._t[1] - self._t[0]

    N = self._msd.size

    # Time coordinates
    # This is the time array, as the fits will be MSD vs T
    T = np.linspace(dt, dt*N, N, endpoint=True)

    # Compute  the time-dependent apparent diffusion coefficient.
    Dapp = self._msd / (4 * T * (1 - 2*R*dt / T))
    Dapp_err = self._msd_error / (4 * T * (1 - 2*R*dt / T))

    model, results = self._categorize(np.array(Dapp), np.arange(
        1, N+1), Dapp_err = Dapp_err, fraction_fit_points=fraction_fit_points, fit_max_time=fit_max_time, initial_guesses = initial_guesses, maxfev=maxfev, enable_log_sampling=enable_log_sampling, log_sampling_dist=log_sampling_dist)

    self._adc_analysis_results["analyzed"] = True
    self._adc_analysis_results["Dapp"] = np.array(Dapp)
    self._adc_analysis_results["model"] = model
    self._adc_analysis_results["results"] = results

    return self._adc_analysis_results

    
def plot_adc_analysis_results(self):
    """Plot the ADC analysis results.

    Raises
    ------
    ValueError
        Track has not been analyzed using ADC analysis yet.
    """
    import matplotlib.pyplot as plt
    if self.get_adc_analysis_results()["analyzed"] == False:
        raise ValueError(
            "Track has not been analyzed using adc_analysis yet!")

    dt = self._t[1] - self._t[0]
    N = self._t.size
    T = np.linspace(1, N, N-3,  endpoint=True) * dt

    results = self.get_adc_analysis_results()["results"]

    Dapp = self.get_adc_analysis_results()["Dapp"]
    idxs = results["indexes"]
    n_points = idxs[-1]
    R = results["R"]
    plt.semilogx(T, Dapp, label="Data", marker="o")
    plt.semilogx(T[idxs], Dapp[idxs], label="Sampled Points", linestyle="", marker="o")
    for model in results["models"]:
        r = results["models"][model]["params"]
        rel_likelihood = results["models"][model]["rel_likelihood"]
        for c in ModelDB().models:
            if c.__class__.__name__ == model:
                m = c
        pred = m(T, *r)
        plt.semilogx(T[0:n_points], pred[0:n_points],
                    label=f"{model}, Rel_Likelihood={rel_likelihood:.2e}")
    model = self.get_adc_analysis_results()["model"]

    plt.axvspan(T[0], T[n_points], alpha=0.25,
                color='gray', label="Fitted data")
    plt.xlabel("Time [step]")
    plt.ylabel("Normalized ADC")
    plt.title("Diffusion Category: {}".format(model))
    plt.legend()
    plt.show()