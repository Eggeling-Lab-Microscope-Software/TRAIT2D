from scipy import optimize
import numpy as np
import warnings
import tqdm

from trait2d.analysis.models import ModelLinear, ModelPower

def delete_msd_analysis_results(self):
    """Delete the MSD analysis results."""
    self._msd_analysis_results = None

def get_msd_analysis_results(self):
    """Returns the MSD analysis results."""
    return self._msd_analysis_results

def msd_analysis(self, fraction_fit_points: float = 0.25, n_fit_points: int = None, fit_max_time: float = None, initial_guesses = { }, maxfev = 1000, R : float = 0):
    """ Classical Mean Squared Displacement Analysis for single track

    Parameters
    ----------
    fraction_fit_points: float
        Fraction of points to use for fitting if n_fit_points is not specified.
    n_fit_points: int
        Number of points to user for fitting. Will override fraction_fit_points.
    fit_max_time: float
        Maximum time in fit range. Will override fraction_fit_points and n_fit_points.
    initial_guesses: dict
        Dictionary containing initial guesses for the parameters. Keys can be "model1" and "model2".
        All values default to 1.
    maxfev: int
        Maximum function evaluations by scipy.optimize.curve_fit. The fit will fail if this number is exceeded.
    R: float
        Point scanning across the field of view.
    Returns
    -------
    msd_analysis_results: dict
        Dictionary containing all MSD analysis results. Can also be retreived using `Track.get_msd_analysis_results()`.
    """
    p0 = {"model1" : 2 * [None], "model2" : 3*[None]}
    p0.update(initial_guesses)

    # Calculate MSD if this has not been done yet.
    if self._msd is None:
        self.calculate_msd()

    # Number time frames for this track
    N = self._msd.size

    # This is the time array, as the fits will be MSD vs T
    T = self._t[0:-3]

    dt = self._t[1] - self._t[0]

    # Define the number of points to use for fitting
    if fit_max_time is not None:
        n_points = int(np.argwhere(T < fit_max_time)[-1])
    elif n_fit_points is not None:
        n_points = int(n_fit_points)
    else:
        n_points = int(fraction_fit_points * N)

    # Asserting that the n_fit_points is valid
    assert n_points >= 2, f"n_fit_points={n_points} is not enough"
    if n_points > int(0.25 * N):
        warnings.warn(
            "Using too many points for the fit means including points which have higher measurment errors.")
        # Selecting more points than 25% should be possible, but not advised

    from trait2d.analysis.models import ModelLinear, ModelPower
    model1 = ModelLinear()
    model2 = ModelPower()

    model1.R = R
    model2.R = R
    model1.dt = dt
    model2.dt = dt

    p0_model1 = [0.0, 0.0]
    for i in range(len(p0_model1)):
        if not p0["model1"][i] is None:
            p0_model1[i] = p0["model1"][i]

    reg1 = optimize.curve_fit(
        model1, T[0:n_points], self._msd[0:n_points], p0 = p0_model1, sigma=self._msd_error[0:n_points], maxfev=maxfev, method='trf', bounds=(0.0, np.inf))

    p0_model2 = [0.0, 0.0, 0.0]
    for i in range(len(p0_model2)):
        if not p0["model2"][i] is None:
            p0_model2[i] = p0["model2"][i]
    reg2 = optimize.curve_fit(model2, T[0:n_points], self._msd[0:n_points], p0 = p0_model2, sigma=self._msd_error[0:n_points], maxfev=maxfev, method='trf', bounds=(0.0, np.inf))


    # Compute standard deviation of parameters
    perr_m1 = np.sqrt(np.diag(reg1[1]))
    perr_m2 = np.sqrt(np.diag(reg2[1]))

    # Compute BIC for both models
    m1 = model1(T, *reg1[0])
    m2 = model2(T, *reg2[0])
    bic1 = BIC(m1[0:n_points], self._msd[0:n_points], 2, 1)
    bic2 = BIC(m2[0:n_points], self._msd[0:n_points], 2, 1)
    # FIXME: numerical instabilities due to low position values. should normalize before analysis, and then report those adimentional values.

    # Relative Likelihood for each model
    rel_likelihood_1 = np.exp((-bic1 + min([bic1, bic2])) * 0.5)
    rel_likelihood_2 = np.exp((-bic2 + min([bic1, bic2])) * 0.5)

    self._msd_analysis_results = {}
    self._msd_analysis_results["fit_results"] = {"model1": {"params": reg1[0], "errors" : perr_m1, "bic": bic1, "rel_likelihood": rel_likelihood_1},
                                                "model2": {"params": reg2[0], "errors" : perr_m2, "bic": bic2, "rel_likelihood": rel_likelihood_2}}
    self._msd_analysis_results["n_points"] = n_points

    return self._msd_analysis_results

def plot_msd_analysis_results(self, scale: str = 'log'):
    """Plot the MSD analysis results.

    Parameters
    ----------
    scale: str
        How to scale the plot over time. Possible values: 'log', 'linear'.

    Raises
    ------
    ValueError
        Track has not been analyzed using MSD analysis yet.
    """
    import matplotlib.pyplot as plt
    if self.get_msd_analysis_results() is None:
        raise ValueError(
            "Track as not been analyzed using msd_analysis yet!")

    # Definining the models used for the fit
    model1 = ModelLinear()
    model2 = ModelPower()

    results = self.get_msd_analysis_results()["fit_results"]
    N = self._x.size
    T = self._t[0:-3]
    n_points = self.get_msd_analysis_results()["n_points"]
    reg1 = results["model1"]["params"]
    reg2 = results["model2"]["params"]
    m1 = model1(T, *reg1)
    m2 = model2(T, *reg2)
    rel_likelihood_1 = results["model1"]["rel_likelihood"]
    rel_likelihood_2 = results["model2"]["rel_likelihood"]
    # Plot the results
    plt.figure()
    plt.grid(linestyle='dashed', color='grey')
    if scale == 'linear':
        plt.plot(T, self._msd, color='black', label="Data")
    elif scale == 'log':
        plt.semilogx(T, self._msd, color='black', label="Data")
    else:
        raise ValueError("scale must be 'log' or 'linear'.")
    plt.fill_between(T, self._msd-self._msd_error, self._msd+self._msd_error, color='black', alpha=0.5)
    plt.plot(T[0:n_points], m1[0:n_points],
                label=f"Model1, Rel_Likelihood={rel_likelihood_1:.2e}")
    plt.plot(T[0:n_points], m2[0:n_points],
                label=f"Model2, Rel_Likelihood={rel_likelihood_2:.2e}")
    plt.axvspan(T[0], T[n_points], alpha=0.5,
                color='gray', label="Fitted data")
    plt.xlabel("Time")
    plt.ylabel("MSD")
    plt.legend()
    plt.show()

def MSD_loop(i, pos_x, pos_y, N):
    idx_0 = np.arange(1, N-i-1, 1)
    idx_t = idx_0 + i
    this_msd = (pos_x[idx_t] - pos_x[idx_0])**2 + \
        (pos_y[idx_t] - pos_y[idx_0])**2

    MSD = np.mean(this_msd)
    MSD_error = np.std(this_msd) / np.sqrt(len(this_msd))

    return MSD, MSD_error


def rayleighPDF(x, sigma):
    return x / sigma**2 * np.exp(- x**2 / (2 * sigma**2))


def BIC(pred: list, target: list, k: int, n: int):
    """Bayesian Information Criterion
    Parameters
    ----------
    pred: list
        Model prediction
    target: list
        Model targe

    k: int
        Number of free parameters in the fit
    n: int
        Number of data points used to fit the model
    Returns
    -------
    bic : float
        Bayesian Information Criterion
    """
    # Compute RSS
    RSS = np.sum((np.array(pred) - np.array(target)) ** 2)
    bic = k * np.log(n) + n * np.log(RSS / n)
    return bic