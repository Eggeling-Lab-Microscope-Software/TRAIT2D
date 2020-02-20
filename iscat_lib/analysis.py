#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tqdm
import warnings
from scipy import optimize
from scipy.stats import rayleigh
import matplotlib.pyplot as plt


def normalize(track):
    x = np.array(track["x"])
    y = np.array(track["y"])
    t = np.array(track["t"])

    # Getting the span of the diffusion.
    xmin = x.min(); xmax = x.max()
    ymin = y.min(); ymax = y.max()
    tmin = t.min(); tmax = t.max()

    # Normalizing the coordinates
    xy_min = min([xmin, ymin])
    xy_max = min([xmax, ymax])
    x = (x - xy_min) / (xy_max - xy_min)
    y = (y - xy_min) / (xy_max - xy_min)
    t = (t - tmin) / (tmax - tmin)

    # Update the track
    track["x"] = list(x)
    track["y"] = list(y)
    track["t"] = list(t)
    track["xy_min"] = xy_min
    track["xy_max"] = xy_max
    track["tmin"] = tmin
    track["tmax"] = tmax


    # # Smallest time interval
    # dt = np.abs(np.diff(t)).min()
    #
    # # Smallest x and y distances
    # dx = np.abs(np.diff(x)).min()
    # dy = np.abs(np.diff(y)).min()
    # dxy = min([dx, dy])
    #
    # track["dt"] = dt
    # track["dxy"] = dxy
    #
    # # Normalize the track
    # track["x"] = list(np.array(x) / dx)
    # track["y"] = list(np.array(y) / dy)
    # track["t"] = list(np.array(t) / dt)


    return track

def SD(x, y, j):
    """Squared displacement calculation for single time point
    Parameters
    ----------
    x: list or ndarray
        X coordinates of a 2D track
    y: list or ndarray
        Y coordinates of a 2D track
    j: int
        Index of timepoint in 2D track

    Returns
    -------
    SD: ndarray
        Squared displacements at timepoint j sorted
        from smallest to largest value
    """

    length_array = len(x) # Length of the track

    pos_x = np.array(x)
    pos_y = np.array(y)

    idx_0 = np.arange(0, length_array-j-1, 1)
    idx_t = idx_0 + j

    SD = (pos_x[idx_t] - pos_x[idx_0])**2 + (pos_y[idx_t] - pos_y[idx_0])**2

    SD.sort()

    return SD


def MSD(x, y, N: int=None):
    """Mean squared displacement calculation
    Parameters
    ----------
    x: (N,) list or ndarray
        X coordinates of a 2D track (of length N)
    y: (N,) list or ndarray
        Y coordinates of a 2D track (of length N)
    N: int
        Maximum MSD length to consider (if none, will be computed from the track length)

    Returns
    -------
    MSD: (N-3,) ndarray
        Mean squared displacements
    MSD_error: (N-3, ) ndarray
        Standard error of the mean of the MSD values
    """
    if N is None:
        N = len(x)  # Length of the track

    MSD = np.zeros((N-3,))
    MSD_error = np.zeros((N-3,))
    pos_x = np.array(x)
    pos_y = np.array(y)
    for i in tqdm.tqdm(range(1, N-2), desc="MSD calculation"):
        idx_0 = np.arange(1, N-i-1, 1)
        idx_t = idx_0 + i
        this_msd = (pos_x[idx_t] - pos_x[idx_0])**2 + (pos_y[idx_t] - pos_y[idx_0])**2

        MSD[i-1] = np.mean(this_msd)
        MSD_error[i-1] = np.std(this_msd) / np.sqrt(len(this_msd))

    return MSD, MSD_error

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

def classicalMSDAnalysis(tracks: list, fractionFitPoints: float=0.25, nFitPoints: int=None, dt: float=1.0, useNormalization=True, linearPlot=False):
    n_tracks = len(tracks)

    # Calculate MSD for each track
    msd_list = []
    for track in tracks:
        if useNormalization:
            track = normalize(track)
        msd_list.append(MSD(track["x"], track["y"]))

    # Loop over the MSDs, and perform fits.
    for this_msd, this_msd_error in msd_list:
        # Number time frames for this track
        N = len(this_msd)

        # Define the number of points to use for fitting
        if nFitPoints is None:
            n_points = int(fractionFitPoints * N)
        else:
            n_points = int(nFitPoints)

        # Asserting that the nFitPoints is valid
        assert n_points >= 2, f"nFitPoints={n_points} is not enough"
        if n_points > int(0.25 * N):
            warnings.warn("Using too many points for the fit means including points which have higher measurment errors.")
            # Selecting more points than 25% should be possible, but not advised

        T = np.linspace(1, N, N,  endpoint=True) # This is the time array, as the fits will be MSD vs T

        # Definining the models used for the fit
        model1 = lambda t, D, delta2: 4 * D * t + 2 * delta2
        model2 = lambda t, D, delta2, alpha: 4 * D * t**alpha + 2 * delta2

        # Fit the data to these 2 models using weighted least-squares fit
        # TODO: normalize the curves to make the fit easier to perform.
        reg1 = optimize.curve_fit(model1, T[0:n_points], this_msd[0:n_points], sigma=this_msd_error[0:n_points])
        # print(f"reg1 parameters: {reg1[0]}") # Debug
        reg2 = optimize.curve_fit(model2, T[0:n_points], this_msd[0:n_points], [*reg1[0][0:2], 1.0], sigma=this_msd_error[0:n_points])
        # reg2 = optimize.curve_fit(model2, T[0:n_points], this_msd[0:n_points], sigma=this_msd_error[0:n_points])
        # print(f"reg2 parameters: {reg2[0]}") #Debug

        # Compute BIC for both models
        m1 = model1(T, *reg1[0])
        m2 = model2(T, *reg2[0])
        bic1 = BIC(m1[0:n_points], this_msd[0:n_points], 2, 1)
        bic2 = BIC(m2[0:n_points], this_msd[0:n_points], 2, 1)
        print(bic1, bic2) # FIXME: numerical instabilities due to low position values. should normalize before analysis, and then report those adimentional values.

        # Relative Likelihood for each model
        rel_likelihood_1 = np.exp((bic1 - min([bic1, bic2])) * 0.5)
        rel_likelihood_2 = np.exp((bic2 - min([bic1, bic2])) * 0.5)

        # Plot the results
        if linearPlot:
            plt.plot(T, this_msd, label="Data")
        else:
            plt.semilogx(T, this_msd, label="Data")
        plt.plot(T[0:n_points], m1[0:n_points], label=f"Model1, Rel_Likelihood={rel_likelihood_1:.2e}")
        plt.plot(T[0:n_points], m2[0:n_points], label=f"Model2, Rel_Likelihood={rel_likelihood_2:.2e}")
        plt.axvspan(T[0], T[n_points], alpha=0.5, color='gray', label="Fitted data")
        plt.xlabel("Time")
        plt.ylabel("MSD")
        plt.legend()
        plt.show()



def adc_analysis(tracks: list, R: float=1/6, nFitPoints=None, useNormalization=True):
    """Revised analysis using the apparent diffusion coefficient
    Parameters
    ----------
    R: float
        Point scanning across the field of view.
    """
    # Calculate MSD for each track
    msd_list = []
    for track in tracks:
        # Normalize the track
        if useNormalization:
            track = normalize(track)
        msd_list.append([*MSD(track["x"], track["y"]), track["t"][1]-track["t"][0]])

    for this_msd, this_msd_error, dt in msd_list:
        # Number time frames for this track
        N = len(this_msd)

        # Define the number of points to use for fitting
        if nFitPoints is None:
            n_points = int(0.25 * N)
        else:
            n_points = int(nFitPoints * N)

        # Asserting that the nFitPoints is valid
        assert n_points >= 2, f"nFitPoints={n_points} is not enough"
        if n_points > int(0.25 * N):
            warnings.warn(
                "Using too many points for the fit means including points which have higher measurment errors.")
            # Selecting more points than 25% should be possible, but not advised

        # Time coordinates
        T = np.linspace(dt, dt*N, N, endpoint=True)  # This is the time array, as the fits will be MSD vs T

        # Compute  the time-dependent apparent diffusion coefficient.
        Dapp = this_msd / (4 * T * (1 - 2*R*dt / T))

        # Define the models to fit the Dapp
        model_brownian = lambda t, D, delta: D + delta**2 / (2 * t * (1 - 2*R*dt/t))
        model_confined = lambda t, D_micro, delta, tau: D_micro * (tau/t) * (1 - np.exp(-tau/t)) + delta ** 2 / (2 * t * (1 - 2 * R * dt / t))
        model_hop = lambda t, D_macro, D_micro, delta, tau: D_macro + D_micro * (tau/t) * (1 - np.exp(-tau/t)) + delta ** 2 / (2 * t * (1 - 2 * R * dt / t))

        # Perform fits.
        r_brownian = optimize.curve_fit(model_brownian, T[0:n_points], Dapp[0:n_points], sigma=this_msd_error[0:n_points])
        r_confined = optimize.curve_fit(model_confined, T[0:n_points], Dapp[0:n_points], sigma=this_msd_error[0:n_points])
        r_hop = optimize.curve_fit(model_hop, T[0:n_points], Dapp[0:n_points], sigma=this_msd_error[0:n_points])

        # Compute BIC for each model.
        pred_brownian = model_brownian(T, *r_brownian[0])
        pred_confined = model_confined(T, *r_confined[0])
        pred_hop = model_hop(T, *r_hop[0])
        bic_brownian = BIC(pred_brownian[0:n_points], Dapp[0:n_points], len(r_brownian[0]), 1)
        bic_confined = BIC(pred_confined[0:n_points], Dapp[0:n_points], len(r_confined[0]), 1)
        bic_hop = BIC(pred_hop[0:n_points], Dapp[0:n_points], len(r_hop[0]), 1)
        bic_min = min([bic_brownian, bic_confined, bic_hop])
        if bic_min == bic_brownian:
            category = "brownian"
        elif bic_min == bic_confined:
            category = "confined"
        elif bic_min == bic_hop:
            category = "hop"
        else:
            category = "unknown"

        print("Brownian diffusion parameters: D={}, delta={}, BIC={}".format(*r_brownian[0], bic_brownian))
        print("Confined diffusion parameters: D_micro={}, delta={}, tau={}, BIC={}".format(*r_confined[0], bic_confined))
        print("Hop diffusion parameters: D_macro={}, D_micro={}, delta={}, tau={}, BIC={}".format(*r_hop[0], bic_hop))
        print("Diffusion category: {}".format(category))

        # Calculate the relative likelihood for each model
        rel_likelihood_brownian = np.exp((bic_brownian - bic_min) * 0.5)
        rel_likelihood_confined = np.exp((bic_confined - bic_min) * 0.5)
        rel_likelihood_hop = np.exp((bic_hop - bic_min) * 0.5)

        # Plot the results
        plt.semilogx(T, Dapp, label="Data", marker="o")
        plt.semilogx(T[0:n_points], pred_brownian[0:n_points], label=f"Brownian, Rel_Likelihood={rel_likelihood_brownian:.2e}")
        plt.semilogx(T[0:n_points], pred_confined[0:n_points], label=f"Confined, Rel_Likelihood={rel_likelihood_confined:.2e}")
        plt.semilogx(T[0:n_points], pred_hop[0:n_points], label=f"Hop, Rel_Likelihood={rel_likelihood_hop:.2e}")
        plt.axvspan(T[0], T[n_points], alpha=0.25, color='gray', label="Fitted data")
        plt.xlabel("Time [step]")
        plt.ylabel("Normalized ADC")
        plt.title("Diffusion Category: {}".format(category))
        plt.legend()
        plt.show()


def smartAveraging():
    """Average tracks by category, and report average track fit results and summary statistics"""
    pass

def squaredDisplacementAnalysis(tracks: list, dt: float=1.0, display_fit: bool=False):
    """Squared Displacement Analysis strategy to obtain apparent diffusion coefficient.
    Parameters
    ----------
    tracks: list
        list of tracks to be analysed
    dt: float
        timestep
    display_fit: bool
        display fit for every timepoint
    """
    # We define a list of timepoints at which to calculate the distribution
    J = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,60,70,80,90,100] # can be more, I don't think less.

    for track in tracks:
        # Perform the analysis for a sigle track
        dapp_list = []
        for j in J:
            # Calculate the SD
            x = np.array(track["x"])
            y = np.array(track["y"])
            sd = SD(x, y, j)
            
            t_lag = j * dt

            x_fit = np.sqrt(sd)
            reg = rayleigh.fit(x_fit)  # Fit Rayleigh PDF to SD data

            if display_fit:
                # Use Freedman Diaconis Rule for binning
                hist_SD, bins = np.histogram(x_fit, bins='fd', density=True)
                plt.bar(bins[:-1], hist_SD, width=(bins[1] - bins[0]), align='edge', alpha=0.5, label="Data")
                # Plot the fit
                eval_x = np.linspace(bins[0], bins[-1], 100)
                plt.plot(eval_x, rayleigh.pdf(eval_x, *reg), label="Fit")
                plt.legend()
                plt.title("$n = {}$".format(j))
                plt.show()
            
            sigma = reg[1]
            dapp = sigma**2 / (2 * t_lag)
            dapp_list.append(dapp)
            
        plt.semilogx(np.array(J) * dt, dapp_list); 
        plt.xlabel("Time")
        plt.ylabel("Estimated $D_{app}$")
        plt.show()