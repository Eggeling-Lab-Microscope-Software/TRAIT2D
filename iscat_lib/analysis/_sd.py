from iscat_lib.analysis import ModelDB

import tqdm
import numpy as np
from scipy import optimize

import itertools
import os
from concurrent.futures import ProcessPoolExecutor

def get_sd_analysis_results(self):
    """Returns the SD analysis results."""
    return self._sd_analysis_results

def delete_sd_analysis_results(self):
    """Delete the SD analyis results."""
    self._sd_analysis_results = None

def sd_analysis(self, display_fit: bool = False, R=1/6, binsize_nm: float = 10.0,
                J: list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100],
                fraction_fit_points: float = 0.25, fit_max_time: float = None, initial_guesses = {}, maxfev=1000,
                enable_log_sampling = False, log_sampling_dist = 0.2, num_workers: int = None, verbose: bool = False):
    """Squared Displacement Analysis strategy to obtain apparent diffusion coefficient.
    
    Parameters
    ----------
    display_fit: bool
        display fit for every timepoint
    binsize_nm: float
        binsize in nm
    J: list
        list of timepoints to consider
    fraction_fit_points: float
        Fraction of track to use for fitting. Defaults to 25 %.
    fit_max_time: float
        Maximum time in fit range. Will override fraction_fit_points.
    initial_guesses: dict
        Dictionary containing initial guesses for the parameters. Keys can be "brownian", "confined" and "hop".
        All values default to 1.
    maxfev: int
        Maximum function evaluations by scipy.optimize.curve_fit. The fit will fail if this number is exceeded.
    enable_log_sampling: bool
        Only sample logarithmically spaced time points for analysis.
    log_sampling_dist: float
        Exponent of logarithmic sampling (base 10).
    num_workers: int
        Number or processes used for calculation. Defaults to number of system cores.
    verbose: bool
        If `True`, a progress bar will be printed in the console.
    """
    # Convert binsize to m
    binsize = binsize_nm * 1e-9

    dt = self._t[1] - self._t[0]

    # We define a list of timepoints at which to calculate the distribution
    # can be more, I don't think less.

    # Perform the analysis for a single track
    dapp_list = [None] * len(J)
    err_list = [None] * len(J)

    if num_workers == None:
        workers = os.cpu_count()
    else:
        workers = num_workers

    if verbose:
        tqdm_wrapper = tqdm.tqdm
    else:
        tqdm_wrapper = lambda x, **kwargs: x

    pos_x = np.array(self._x)
    pos_y = np.array(self._y)
    n = len(self._x)

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            j = J

            results = list(tqdm_wrapper(executor.map(calculate_dapp_at, j,
                                                    itertools.repeat(pos_x),
                                                    itertools.repeat(pos_y),
                                                    itertools.repeat(n),
                                                    itertools.repeat(dt),
                                                    itertools.repeat(binsize),
                                                    chunksize=1),
                                        total=len(j),
                                        desc="SD analysis for single track (workers: {})".format(workers)))
        i = 0
        for (dapp_i, error_i) in results:
            dapp_list[i] = dapp_i
            err_list[i] = error_i
            i += 1

    else:
        i = 0
        for j in tqdm_wrapper(J, desc="SD analysis for single track"):
            dapp_list[i], err_list[i] = calculate_dapp_at(j, pos_x, pos_y, n, dt, binsize)
            i += 1

    model, indices, results = self._categorize(np.array(dapp_list), np.array(
        J), Dapp_err=np.array(err_list), R=R, fraction_fit_points=fraction_fit_points, fit_max_time=fit_max_time, initial_guesses=initial_guesses, maxfev=maxfev, enable_log_sampling=enable_log_sampling, log_sampling_dist=log_sampling_dist)

    self._sd_analysis_results = {}
    self._sd_analysis_results["Dapp"] = np.array(dapp_list)
    self._sd_analysis_results["J"] = np.array(J)
    self._sd_analysis_results["best_model"] = model
    self._sd_analysis_results["fit_results"] = results
    self._sd_analysis_results["fit_indices"] = indices

    return self._sd_analysis_results

def plot_sd_analysis_results(self):
    """Plot the SD analysis results.

    Raises
    ------
    ValueError
        Track has not been analyzed using SD analyis yet.
    """
    import matplotlib.pyplot as plt
    if self.get_sd_analysis_results() is None:
        raise ValueError(
            "Track has not been analyzed using sd_analysis yet!")

    J = self.get_sd_analysis_results()["J"]

    dt = self._t[1] - self._t[0]
    T = J * dt

    fit_results = self.get_sd_analysis_results()["fit_results"]

    Dapp = self.get_sd_analysis_results()["Dapp"]
    idxs = self.get_sd_analysis_results()["fit_indexes"]

    n_points = idxs[-1]

    plt.semilogx(T, Dapp, label="Data", marker="o")
    for model in fit_results:
        r = fit_results[model]["params"]
        rel_likelihood = fit_results[model]["rel_likelihood"]
        m = None
        for c in ModelDB().models:
            if c.__class__.__name__ == model:
                m = c
                break
        if m is None:
            raise ValueError("Can't plot results for model {}; make sure the model is loaded in ModelDB()".format(model))
        pred = m(T, *r)
        plt.semilogx(T[0:n_points], pred[0:n_points],
                    label=f"{model}, Rel_Likelihood={rel_likelihood:.2e}")

    model = self.get_sd_analysis_results()["best_model"]

    plt.axvspan(T[0], T[n_points], alpha=0.25,
                color='gray', label="Fitted data")
    plt.xlabel("Time [step]")
    plt.ylabel("Normalized ADC")
    plt.title("Diffusion Category: {}".format(model))
    plt.legend()
    plt.show()

def _calculate_sd_at(j: int, pos_x, pos_y, n):
    """Squared displacement calculation for single time point
    Parameters
    ----------
    j: int
        Index of timepoint in 2D track

    Returns
    -------
    SD: ndarray
        Squared displacements at timepoint j sorted
        from smallest to largest value
    """
    #length_array = self._x.size  # Length of the track

    #pos_x = np.array(self._x)
    #pos_y = np.array(self._y)

    length_array = n

    idx_0 = np.arange(0, length_array-j-1, 1)
    idx_t = idx_0 + j

    SD = (pos_x[idx_t] - pos_x[idx_0])**2 + \
        (pos_y[idx_t] - pos_y[idx_0])**2

    SD.sort()

    return SD

def calculate_dapp_at(j: int, pos_x, pos_y, n, dt, binsize):
            # Calculate the SD
        sd = _calculate_sd_at(j, pos_x, pos_y, n)

        t_lag = j * dt

        x_fit = np.sqrt(sd)
        # Calculate bins, x_fit is already sorted
        max_x = x_fit[-1]
        min_x = x_fit[0]
        num_bins = int(np.ceil((max_x - min_x) / binsize))
        hist_SD, bins = np.histogram(x_fit, bins=num_bins, density=True)
        bin_mids = (bins[1:] + bins[:-1]) / 2.0
        # Fit Rayleigh PDF to histogram. The bin size gives a good order-of-maginutde approximation
        # for the initial guess of sigma
        popt, pcov = optimize.curve_fit(
            rayleighPDF, bin_mids, hist_SD, p0=(max_x-min_x))
        #if display_fit:
        #    import matplotlib.pyplot as plt
        #    # Plot binned data
        #    plt.bar(bins[:-1], hist_SD, width=(bins[1] - bins[0]),
        #            align='edge', alpha=0.5, label="Data")
        #    plt.gca().set_xlim(0, 4.0e-7)
        #   # Plot the fit
        #    eval_x = np.linspace(bins[0], bins[-1], 100)
        #    plt.plot(eval_x, rayleighPDF(
        #        eval_x, popt[0]), label="Rayleigh Fit")
        #    plt.legend()
        #    plt.title("$n = {}$".format(j))
        #    plt.show()

        sigma = popt[0]
        error = np.sqrt(np.diag(pcov))[0]
        error = error**2 / (2 * t_lag)
        dapp = sigma**2 / (2 * t_lag)
        return dapp, error

def rayleighPDF(x, sigma):
    return x / sigma**2 * np.exp(- x**2 / (2 * sigma**2))