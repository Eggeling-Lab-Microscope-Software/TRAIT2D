#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tqdm
import warnings
from scipy import optimize
from scipy import interpolate
import matplotlib.pyplot as plt

import itertools
import os
from concurrent.futures import ProcessPoolExecutor

class ListOfTracks:
    def __init__(self, tracks: list=None):
        self._tracks = tracks

    def __repr__(self):
        return ("<%s instance at %s>\n"
                "Number of tracks: %s\n") % (self.__class__.__name__,
                                            id(self),
                                            len(self._tracks))

    def load_trajectories(self, filename: str):
        # TODO
        pass
    
    def msd_analysis(self):
        for track in self._tracks:
            track.msd_analysis()

    def adc_analysis(self):
        for track in self._tracks:
            track.adc_analysis()

    def sd_analysis(self):
        for track in self._tracks:
            track.sd_analysis()

    def smart_averaging(self):
        """Average tracks by category, and report average track fit results and summary statistics"""

        track_length = self._tracks[0]._x.size
        average_D_app_brownian = np.zeros(track_length - 3)
        average_D_app_confined = np.zeros(track_length - 3)
        average_D_app_hop = np.zeros(track_length - 3)
        
        average_MSD_brownian = np.zeros(track_length - 3)
        average_MSD_confined = np.zeros(track_length - 3)
        average_MSD_hop = np.zeros(track_length - 3)

        counter_brownian = 0
        counter_confined = 0
        counter_hop = 0

        for track in self._tracks:
            if not track.is_dapp_calculated() or not track.is_msd_calculated():
                raise ValueError("All tracks have to be analysed before averaging!")
            if track._x.size != track_length:
                raise ValueError("Encountered track with incorrect track length! (Got {}, expected {} for track {}.)".format(track._x.size, track_length - 3, k + 1))
            if track._MSD.size != track_length - 3:
                raise ValueError("Encountered MSD with incorrect length! (Got {}, expected {} for track {}.)".format(track._MSD.size, track_length - 3, k + 1))
            if track._Dapp.size != track_length - 3:
                raise ValueError("Encountered D_app with incorrect length!(Got {}, expected {} for track {}.)".format(track._Dapp.size, track_length - 3, k + 1))

        for track in self._tracks:
            if track._model == "brownian":
                counter_brownian += 1
                average_D_app_brownian += track._Dapp
                average_MSD_brownian += track._MSD
            elif track._model == "confined":
                counter_confined += 1
                average_D_app_confined += track._Dapp
                average_MSD_confined += track._MSD
            elif track._model == "hop":
                counter_hop += 1
                average_D_app_hop += track._Dapp
                average_MSD_hop += track._MSD
            elif track._model == "unknown":
                continue
            else:
                raise ValueError('Invalid model name encountered: {}. Allowed are "brownian", "confined", "hop" and "unknown".'.format(track._model))

        average_D_app_brownian /= counter_brownian
        average_D_app_confined /= counter_confined
        average_D_app_hop /= counter_hop

        average_MSD_brownian /= counter_brownian
        average_MSD_confined /= counter_confined
        average_MSD_hop /= counter_hop

        counter_sum = counter_brownian + counter_confined + counter_hop
        sector_brownian_area = counter_brownian / counter_sum
        sector_confined_area = counter_confined / counter_sum
        sector_hop_area = counter_hop / counter_sum

        # TODO: The whole fitting routine from ADC.
        # TODO: Plot results.
        # TODO: Print fit results.

        print(sector_brownian_area)
        print(sector_confined_area)
        print(sector_hop_area)

        plt.semilogx(average_D_app_brownian)
        plt.semilogx(average_D_app_confined)
        plt.semilogx(average_D_app_hop)

        fig1, ax1 = plt.subplots()
        ax1.pie([sector_brownian_area, sector_confined_area, sector_hop_area], labels=["brownian", "confined", "hop"], autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax1.legend()

        plt.show()

class Track:
    def __init__(self, x=None, y=None, t=None):
        """Create a track.
        Parameters
        ----------
        x: array_like

        y: array_like

        t: array_like
        """
        self._x = np.array(x)
        self._y = np.array(y)
        self._t = np.array(t)

        self._MSD = None
        self._MSD_error = None

        self._Dapp = None

        self._model = "unknown"

    def __repr__(self):
        return ("<%s instance at %s>\n"
                "MSD calculated: %s\n"
                "Dapp calculated: %s\n"
                "Model class: %s") % (self.__class__.__name__,
                                      id(self),
                                      self.is_msd_calculated(),
                                      self.is_dapp_calculated(),
                                      self._model)

    def is_msd_calculated(self):
        """Returns True if the MSD of this track has already been calculated.
        """
        return self._MSD is not None

    def is_dapp_calculated(self):
        """Returns True if the D_app of this track has already been calculated.
        """
        return self._Dapp is not None

    def calculate_sd_at(self, j: int):
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
        length_array = self._x.size # Length of the track

        pos_x = np.array(self._x)
        pos_y = np.array(self._y)

        idx_0 = np.arange(0, length_array-j-1, 1)
        idx_t = idx_0 + j

        SD = (pos_x[idx_t] - pos_x[idx_0])**2 + (pos_y[idx_t] - pos_y[idx_0])**2

        SD.sort()

        return SD

    def normalized(self):
        """Normalize the track.
        Returns
        -------
        Instance of NormalizedTrack containing the normalized track data.
        """
        if self.__class__ == NormalizedTrack:
            warnings.warn("Track is already an instance of NormalizedTrack. This will do nothing.")

        x = self._x
        y = self._y
        t = self._t

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

        # Create normalized Track object
        return NormalizedTrack(x, y, t, xy_min, xy_max, tmin, tmax)

    def calculate_msd(self, N: int=None, numWorkers: int=None, chunksize: int=100):
        """Mean squared displacement calculation
        Parameters
        ----------
        N: int
            Maximum MSD length to consider (if none, will be computed from the track length)
        numWorkers: int
            Number or processes used for calculation. Defaults to number of system cores.
        chunksize: int
            Chunksize for process pool mapping. Small number might have negative performance impacts.
        """

        if N is None:
            N = self._x.size

        MSD = np.zeros((N-3,))
        MSD_error = np.zeros((N-3,))
        pos_x = self._x
        pos_y = self._y

        if numWorkers == None:
            workers = os.cpu_count()
        else:
            workers = numWorkers
        
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                i = range(1, N-2)
                results = list(tqdm.tqdm(executor.map(MSD_loop, i,
                                                                itertools.repeat(pos_y),
                                                                itertools.repeat(pos_x),
                                                                itertools.repeat(N),
                                                    chunksize=chunksize),
                                                total=len(i),
                                                desc="MSD calculation (workers: {})".format(workers)))
            i = 0
            for (MSD_i, MSD_error_i) in results:
                MSD[i] = MSD_i
                MSD_error[i] = MSD_error_i
                i += 1
        else:
            for i in tqdm.tqdm(range(1, N-2), desc="MSD calculation"):
                idx_0 = np.arange(1, N-i-1, 1)
                idx_t = idx_0 + i
                this_msd = (pos_x[idx_t] - pos_x[idx_0])**2 + (pos_y[idx_t] - pos_y[idx_0])**2

                MSD[i-1] = np.mean(this_msd)
                MSD_error[i-1] = np.std(this_msd) / np.sqrt(len(this_msd))

        self._MSD = MSD
        self._MSD_error = MSD_error

    def msd_analysis(self, fractionFitPoints: float=0.25, nFitPoints: int=None, dt: float=1.0, linearPlot=False, numWorkers: int=None, chunksize: int=100):
        """ Classical Mean Squared Displacement Analysis for single track
        Parameters
        ----------
        fractionFitPoints: float
            Fraction of points to use for fitting if nFitPoints is not specified.
        nFitPoints: int
            Number of points to user for fitting. Will override fractionFitPoints.
        dt: float
            Timestep.
        linearPlot: bool
            Plot results on a liner scale instead of default logarithmic.
        numWorkers: int
            Number or processes used for calculation. Defaults to number of system cores.
        chunksize: int
            Chunksize for process pool mapping. Small number might have negative performance impacts.
        """

        # Calculate MSD if this has not been done yet.
        if self._MSD is None:
            self.calculate_msd()

        # Number time frames for this track
        N = self._MSD.size

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
        reg1 = optimize.curve_fit(model1, T[0:n_points], self._MSD[0:n_points], sigma=self._MSD_error[0:n_points])
        # print(f"reg1 parameters: {reg1[0]}") # Debug
        reg2 = optimize.curve_fit(model2, T[0:n_points], self._MSD[0:n_points], [*reg1[0][0:2], 1.0], sigma=self._MSD_error[0:n_points])
        # reg2 = optimize.curve_fit(model2, T[0:n_points], this_msd[0:n_points], sigma=this_msd_error[0:n_points])
        # print(f"reg2 parameters: {reg2[0]}") #Debug

        # Compute BIC for both models
        m1 = model1(T, *reg1[0])
        m2 = model2(T, *reg2[0])
        bic1 = BIC(m1[0:n_points], self._MSD[0:n_points], 2, 1)
        bic2 = BIC(m2[0:n_points], self._MSD[0:n_points], 2, 1)
        print(bic1, bic2) # FIXME: numerical instabilities due to low position values. should normalize before analysis, and then report those adimentional values.

        # Relative Likelihood for each model
        rel_likelihood_1 = np.exp((bic1 - min([bic1, bic2])) * 0.5)
        rel_likelihood_2 = np.exp((bic2 - min([bic1, bic2])) * 0.5)

        # Plot the results
        if linearPlot:
            plt.plot(T, self._MSD, label="Data")
        else:
            plt.semilogx(T, self._MSD, label="Data")
        plt.plot(T[0:n_points], m1[0:n_points], label=f"Model1, Rel_Likelihood={rel_likelihood_1:.2e}")
        plt.plot(T[0:n_points], m2[0:n_points], label=f"Model2, Rel_Likelihood={rel_likelihood_2:.2e}")
        plt.axvspan(T[0], T[n_points], alpha=0.5, color='gray', label="Fitted data")
        plt.xlabel("Time")
        plt.ylabel("MSD")
        plt.legend()
        plt.show()

    def adc_analysis(self, R: float = 1/6, nFitPoints=None, maxfev=1000):
        """Revised analysis using the apparent diffusion coefficient
        Parameters
        ----------
        R: float
            Point scanning across the field of view.
        nFitPoints: int
            Number of points used for fitting. Defaults to 25 % of total points.
        maxfev: int
            maxfev used by Scipy fitting routine.
        """
        # Calculate MSD if this has not been done yet.
        if self._MSD is None:
            self.calculate_msd()

        dt = self._t[1] - self._t[0]

        N = self._MSD.size

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
        Dapp = self._MSD / (4 * T * (1 - 2*R*dt / T))

        # Define the models to fit the Dapp
        model_brownian = lambda t, D, delta: D + delta**2 / (2 * t * (1 - 2*R*dt/t))
        model_confined = lambda t, D_micro, delta, tau: D_micro * (tau/t) * (1 - np.exp(-tau/t)) + delta ** 2 / (2 * t * (1 - 2 * R * dt / t))
        model_hop = lambda t, D_macro, D_micro, delta, tau: D_macro + D_micro * (tau/t) * (1 - np.exp(-tau/t)) + delta ** 2 / (2 * t * (1 - 2 * R * dt / t))

        # Perform fits.
        r_brownian = optimize.curve_fit(model_brownian, T[0:n_points], Dapp[0:n_points], sigma=self._MSD_error[0:n_points], maxfev=maxfev)
        r_confined = optimize.curve_fit(model_confined, T[0:n_points], Dapp[0:n_points], sigma=self._MSD_error[0:n_points], maxfev=maxfev)
        r_hop = optimize.curve_fit(model_hop, T[0:n_points], Dapp[0:n_points], sigma=self._MSD_error[0:n_points], maxfev=maxfev)

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
        self._model = category

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

        self._Dapp = Dapp

    def sd_analysis(self, dt: float=1.0, display_fit: bool=False, binsize_nm: float = 10.0,
                    J: list=[1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,60,70,80,90,100]):
        """Squared Displacement Analysis strategy to obtain apparent diffusion coefficient.
        Parameters
        ----------
        dt: float
            timestep
        display_fit: bool
            display fit for every timepoint
        binsize_nm: float
            binsize in nm
        J: list
            list of timepoints to consider
        """
        # Convert binsize to m
        binsize = binsize_nm * 1e-9

        # We define a list of timepoints at which to calculate the distribution
        # can be more, I don't think less.

        # Perform the analysis for a single track
        dapp_list = []
        for j in tqdm.tqdm(J, desc="SD analysis for single track"):
            # Calculate the SD
            sd = self.calculate_sd_at(j)
            
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
            popt, pcov = optimize.curve_fit(rayleighPDF, bin_mids, hist_SD, p0=binsize)

            if display_fit:
                # Plot binned data
                plt.bar(bins[:-1], hist_SD, width=(bins[1] - bins[0]), align='edge', alpha=0.5, label="Data")
                plt.gca().set_xlim(0, 4.0e-7)
                # Plot the fit
                eval_x = np.linspace(bins[0], bins[-1], 100)
                plt.plot(eval_x, rayleighPDF(eval_x, popt[0]), label="Rayleigh Fit")
                plt.legend()
                plt.title("$n = {}$".format(j))
                plt.show()
            
            sigma = popt[0]
            dapp = sigma**2 / (2 * t_lag)
            dapp_list.append(dapp)
            
        plt.semilogx(np.array(J) * dt, dapp_list); 
        plt.xlabel("Time")
        plt.ylabel("Estimated $D_{app}$")
        plt.show()           

class NormalizedTrack(Track):
    def __init__(self, x=None, y=None, t=None, xy_min = None, xy_max = None, tmin=None, tmax=None):
        Track.__init__(self, x, y, t)
        self._xy_min = xy_min
        self._xy_max = xy_max
        self._tmin = tmin
        self._tmax = tmax

def MSD_loop(i, pos_x, pos_y, N):
    idx_0 = np.arange(1, N-i-1, 1)
    idx_t = idx_0 + i
    this_msd = (pos_x[idx_t] - pos_x[idx_0])**2 + (pos_y[idx_t] - pos_y[idx_0])**2

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