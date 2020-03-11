#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tqdm
import warnings
import logging
from scipy import optimize
from scipy import interpolate
import matplotlib.pyplot as plt

import itertools
import os
from concurrent.futures import ProcessPoolExecutor


class ListOfTracks:
    def __init__(self, tracks: list = None):
        self.__tracks = tracks

    def __repr__(self):
        return ("<%s instance at %s>\n"
                "Number of tracks: %s\n") % (self.__class__.__name__,
                                             id(self),
                                             len(self.__tracks))

    def get_tracks(self):
        return self.__tracks

    def get_track(self, idx):
        return self.__tracks[idx]

    def load_trajectories(self, filename: str):
        # TODO
        pass

    def msd_analysis(self, **kwargs):
        for track in self.__tracks:
            track.msd_analysis(**kwargs)

    def adc_analysis(self, **kwargs):
        for track in self.__tracks:
            track.adc_analysis(**kwargs)

    def sd_analysis(self, **kwargs):
        for track in self.__tracks:
            track.sd_analysis(**kwargs)

    def smart_averaging(self):
        """Average tracks by category, and report average track fit results and summary statistics"""

        track_length = self.__tracks[0].get_x().size
        average_D_app_brownian = np.zeros(track_length - 3)
        average_D_app_confined = np.zeros(track_length - 3)
        average_D_app_hop = np.zeros(track_length - 3)

        average_MSD_brownian = np.zeros(track_length - 3)
        average_MSD_confined = np.zeros(track_length - 3)
        average_MSD_hop = np.zeros(track_length - 3)

        counter_brownian = 0
        counter_confined = 0
        counter_hop = 0

        k = 0
        for track in self.__tracks:
            k += 1
            if track.get_adc_analysis_results()["analyzed"] == False:
                raise ValueError(
                    "All tracks have to be analyzed using adc_analysis() before averaging!")
            if track.get_x().size != track_length:
                raise ValueError("Encountered track with incorrect track length! (Got {}, expected {} for track {}.)".format(
                    track.get_x().size, track_length - 3, k + 1))
            if track.get_msd().size != track_length - 3:
                raise ValueError("Encountered MSD with incorrect length! (Got {}, expected {} for track {}.)".format(
                    track.get_msd().size, track_length - 3, k + 1))
            if track.get_adc_analysis_results()["Dapp"].size != track_length - 3:
                raise ValueError("Encountered D_app with incorrect length!(Got {}, expected {} for track {}.)".format(
                    track.get_adc_analysis_results()["Dapp"].size, track_length - 3, k + 1))

        for track in self.__tracks:
            if track.get_adc_analysis_results()["model"] == "brownian":
                counter_brownian += 1
                average_D_app_brownian += track.get_adc_analysis_results()[
                    "Dapp"]
                average_MSD_brownian += track.get_msd()
            elif track.get_adc_analysis_results()["model"] == "confined":
                counter_confined += 1
                average_D_app_confined += track.get_adc_analysis_results()[
                    "Dapp"]
                average_MSD_confined += track.get_msd()
            elif track.get_adc_analysis_results()["model"] == "hop":
                counter_hop += 1
                average_D_app_hop += track.get_adc_analysis_results()["Dapp"]
                average_MSD_hop += track.get_msd()
            elif track.get_adc_analysis_results()["model"] == "unknown":
                continue
            else:
                raise ValueError(
                    'Invalid model name encountered: {}. Allowed are "brownian", "confined", "hop" and "unknown".'.format(track._model))

        if counter_brownian:
            average_D_app_brownian /= counter_brownian
            average_MSD_brownian /= counter_brownian

        if counter_confined:
            average_D_app_confined /= counter_confined
            average_MSD_confined /= counter_confined

        if counter_hop:
            average_D_app_hop /= counter_hop
            average_MSD_hop /= counter_hop

        counter_sum = counter_brownian + counter_confined + counter_hop
        if counter_sum == 0:
            raise ValueError("No tracks are categorized!")
        sector_brownian_area = counter_brownian / counter_sum
        sector_confined_area = counter_confined / counter_sum
        sector_hop_area = counter_hop / counter_sum

        # TODO: The whole fitting routine from ADC.
        # TODO: Plot results.
        # TODO: Print fit results.

        plt.semilogx(average_D_app_brownian)
        plt.semilogx(average_D_app_confined)
        plt.semilogx(average_D_app_hop)

        fig1, ax1 = plt.subplots()
        ax1.pie([sector_brownian_area, sector_confined_area, sector_hop_area], labels=["brownian", "confined", "hop"], autopct='%1.1f%%',
                shadow=True, startangle=90)
        # Equal aspect ratio ensures that pie is drawn as a circle.
        ax1.axis('equal')
        ax1.legend()

        plt.show()

        return {"sector_brownian_area": sector_brownian_area, "sector_confined_area": sector_confined_area, "sector_hop_area": sector_hop_area}


class Track:
    def __init__(self, x=None, y=None, t=None):
        """Create a track.
        Parameters
        ----------
        x: array_like

        y: array_like

        t: array_like
        """
        self.__x = np.array(x)
        self.__y = np.array(y)
        self.__t = np.array(t)

        self.__msd = None
        self.__msd_error = None

        self.__msd_analysis_results = {"analyzed": False, "results": None}
        self.__adc_analysis_results = {
            "analyzed": False, "model": "unknown", "Dapp": None, "results": None}
        self.__sd_analysis_results = {
            "analyzed": False, "model": "unknown", "Dapp": None, "J": None, "results": None}

    @classmethod
    def from_dict(cls, dict):
        """Create a track from a dictionary.
        Parameters
        ----------
        dict: dict
            Dictionary of the track. Has to contain the fields "x", "y" and "t".
        """
        return cls(dict["x"], dict["y"], dict["t"])

    @classmethod
    def from_file(cls, filename):
        """Create a track from a file containing a single track.
        Parameters
        ----------
        filename: str
            Name of the file.
        """
        ext = os.path.splitext(filename)
        if ext == ".csv":
            # TODO: .csv-specific import
            pass
        elif ext == ".json":
            # TODO: .json-specific import
            pass
        elif ext == ".pcl":
            # TODO: .pcl-specific import
            pass

    def __repr__(self):
        return ("<%s instance at %s>\n"
                "MSD calculated: %s\n"
                "MSD analysis done: %s\n"
                "SD analysis done: %s\n"
                "ADC analysis done: %s\n") % (self.__class__.__name__,
                                              id(self),
                                              self.is_msd_calculated(),
                                              self.__msd_analysis_results["analyzed"],
                                              self.__sd_analysis_results["analyzed"],
                                              self.__adc_analysis_results["analyzed"])

    def get_msd_analysis_results(self):
        return self.__msd_analysis_results

    def get_sd_analysis_results(self):
        return self.__sd_analysis_results

    def get_adc_analysis_results(self):
        return self.__adc_analysis_results

    def delete_msd_analysis_results(self):
        self.__msd_analysis_results = {"analyzed": False, "results": None}

    def delete_sd_analysis_results(self):
        self.__sd_analysis_results = {
            "analyzed": False, "model": "unknown", "Dapp": None, "results": None}

    def delete_adc_analysis_results(self):
        self.__adc_analysis_results = {
            "analyzed": False, "model": "unknown", "Dapp": None, "J": None, "results": None}

    def plot_msd_analysis_results(self, linearPlot: bool = False):
        if self.get_msd_analysis_results()["analyzed"] == False:
            raise ValueError(
                "Track as not been analyzed using msd_analysis yet!")

        # Definining the models used for the fit
        def model1(t, D, delta2): return 4 * D * t + 2 * delta2
        def model2(t, D, delta2, alpha): return 4 * D * t**alpha + 2 * delta2

        results = self.get_msd_analysis_results()["results"]
        N = self.__x.size
        T = np.linspace(1, N, N-3,  endpoint=True)
        n_points = results["n_points"]
        reg1 = results["model1"]["params"]
        reg2 = results["model2"]["params"]
        m1 = model1(T, *reg1)
        m2 = model2(T, *reg2)
        rel_likelihood_1 = results["model1"]["rel_likelihood"]
        rel_likelihood_2 = results["model2"]["rel_likelihood"]
        # Plot the results
        if linearPlot:
            plt.plot(T, self.__msd, label="Data")
        else:
            plt.semilogx(T, self.__msd, label="Data")
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

    def plot_adc_analysis_results(self):
        if self.get_adc_analysis_results()["analyzed"] == False:
            raise ValueError(
                "Track has not been analyzed using adc_analysis yet!")

        dt = self.__t[1] - self.__t[0]
        N = self.__t.size
        T = np.linspace(1, N, N-3,  endpoint=True) * dt

        results = self.get_adc_analysis_results()["results"]

        r_brownian = results["brownian"]["params"]
        r_confined = results["confined"]["params"]
        r_hop = results["hop"]["params"]

        rel_likelihood_brownian = results["brownian"]["rel_likelihood"]
        rel_likelihood_confined = results["confined"]["rel_likelihood"]
        rel_likelihood_hop = results["hop"]["rel_likelihood"]

        n_points = results["n_points"]
        R = results["R"]

        Dapp = self.get_adc_analysis_results()["Dapp"]
        model = self.get_adc_analysis_results()["model"]

        # Define the models to fit the Dapp
        def model_brownian(t, D, delta): return D + \
            delta**2 / (2 * t * (1 - 2*R*dt/t))
        def model_confined(t, D_micro, delta, tau): return D_micro * (tau/t) * \
            (1 - np.exp(-tau/t)) + delta ** 2 / (2 * t * (1 - 2 * R * dt / t))

        def model_hop(t, D_macro, D_micro, delta, tau): return D_macro + D_micro * \
            (tau/t) * (1 - np.exp(-tau/t)) + \
            delta ** 2 / (2 * t * (1 - 2 * R * dt / t))

        pred_brownian = model_brownian(T, *r_brownian)
        pred_confined = model_confined(T, *r_confined)
        pred_hop = model_hop(T, *r_hop)

        plt.semilogx(T, Dapp, label="Data", marker="o")
        plt.semilogx(T[0:n_points], pred_brownian[0:n_points],
                     label=f"Brownian, Rel_Likelihood={rel_likelihood_brownian:.2e}")
        plt.semilogx(T[0:n_points], pred_confined[0:n_points],
                     label=f"Confined, Rel_Likelihood={rel_likelihood_confined:.2e}")
        plt.semilogx(T[0:n_points], pred_hop[0:n_points],
                     label=f"Hop, Rel_Likelihood={rel_likelihood_hop:.2e}")
        plt.axvspan(T[0], T[n_points], alpha=0.25,
                    color='gray', label="Fitted data")
        plt.xlabel("Time [step]")
        plt.ylabel("Normalized ADC")
        plt.title("Diffusion Category: {}".format(model))
        plt.legend()
        plt.show()

    def plot_sd_analysis_results(self):
        if self.get_sd_analysis_results()["analyzed"] == False:
            raise ValueError(
                "Track has not been analyzed using sd_analysis yet!")

        J = self.get_sd_analysis_results()["J"]

        dt = self.__t[1] - self.__t[0]
        T = J * dt

        results = self.get_sd_analysis_results()["results"]

        r_brownian = results["brownian"]["params"]
        r_confined = results["confined"]["params"]
        r_hop = results["hop"]["params"]

        rel_likelihood_brownian = results["brownian"]["rel_likelihood"]
        rel_likelihood_confined = results["confined"]["rel_likelihood"]
        rel_likelihood_hop = results["hop"]["rel_likelihood"]

        n_points = results["n_points"]
        R = results["R"]

        Dapp = self.get_sd_analysis_results()["Dapp"]
        model = self.get_sd_analysis_results()["model"]

        # Define the models to fit the Dapp
        def model_brownian(t, D, delta): return D + \
            delta**2 / (2 * t * (1 - 2*R*dt/t))
        def model_confined(t, D_micro, delta, tau): return D_micro * (tau/t) * \
            (1 - np.exp(-tau/t)) + delta ** 2 / (2 * t * (1 - 2 * R * dt / t))

        def model_hop(t, D_macro, D_micro, delta, tau): return D_macro + D_micro * \
            (tau/t) * (1 - np.exp(-tau/t)) + \
            delta ** 2 / (2 * t * (1 - 2 * R * dt / t))

        pred_brownian = model_brownian(T, *r_brownian)
        pred_confined = model_confined(T, *r_confined)
        pred_hop = model_hop(T, *r_hop)

        plt.semilogx(T, Dapp, label="Data", marker="o")
        plt.semilogx(T[0:n_points], pred_brownian[0:n_points],
                     label=f"Brownian, Rel_Likelihood={rel_likelihood_brownian:.2e}")
        plt.semilogx(T[0:n_points], pred_confined[0:n_points],
                     label=f"Confined, Rel_Likelihood={rel_likelihood_confined:.2e}")
        plt.semilogx(T[0:n_points], pred_hop[0:n_points],
                     label=f"Hop, Rel_Likelihood={rel_likelihood_hop:.2e}")
        plt.axvspan(T[0], T[n_points], alpha=0.25,
                    color='gray', label="Fitted data")
        plt.xlabel("Time [step]")
        plt.ylabel("Normalized ADC")
        plt.title("Diffusion Category: {}".format(model))
        plt.legend()
        plt.show()

    def get_x(self):
        return self.__x

    def get_y(self):
        return self.__y

    def get_t(self):
        return self.__t

    def is_msd_calculated(self):
        """Returns True if the MSD of this track has already been calculated.
        """
        return self.__msd is not None

    def get_msd(self):
        return self.__msd

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
        length_array = self.__x.size  # Length of the track

        pos_x = np.array(self.__x)
        pos_y = np.array(self.__y)

        idx_0 = np.arange(0, length_array-j-1, 1)
        idx_t = idx_0 + j

        SD = (pos_x[idx_t] - pos_x[idx_0])**2 + \
            (pos_y[idx_t] - pos_y[idx_0])**2

        SD.sort()

        return SD

    def normalized(self):
        """Normalize the track.
        Returns
        -------
        Instance of NormalizedTrack containing the normalized track data.
        """
        if self.__class__ == NormalizedTrack:
            warnings.warn(
                "Track is already an instance of NormalizedTrack. This will do nothing.")

        x = self.__x
        y = self.__y
        t = self.__t

        # Getting the span of the diffusion.
        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()
        tmin = t.min()
        tmax = t.max()

        # Normalizing the coordinates
        xy_min = min([xmin, ymin])
        xy_max = min([xmax, ymax])
        x = (x - xy_min) / (xy_max - xy_min)
        y = (y - xy_min) / (xy_max - xy_min)
        t = (t - tmin) / (tmax - tmin)

        # Create normalized Track object
        return NormalizedTrack(x, y, t, xy_min, xy_max, tmin, tmax)

    def calculate_msd(self, N: int = None, numWorkers: int = None, chunksize: int = 100):
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
            N = self.__x.size

        MSD = np.zeros((N-3,))
        MSD_error = np.zeros((N-3,))
        pos_x = self.__x
        pos_y = self.__y

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
                this_msd = (pos_x[idx_t] - pos_x[idx_0])**2 + \
                    (pos_y[idx_t] - pos_y[idx_0])**2

                MSD[i-1] = np.mean(this_msd)
                MSD_error[i-1] = np.std(this_msd) / np.sqrt(len(this_msd))

        self.__msd = MSD
        self.__msd_error = MSD_error

    def msd_analysis(self, fractionFitPoints: float = 0.25, nFitPoints: int = None, dt: float = 1.0, linearPlot=False, numWorkers: int = None, chunksize: int = 100):
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
        if self.__msd is None:
            self.calculate_msd()

        # Number time frames for this track
        N = self.__msd.size

        # Define the number of points to use for fitting
        if nFitPoints is None:
            n_points = int(fractionFitPoints * N)
        else:
            n_points = int(nFitPoints)

        # Asserting that the nFitPoints is valid
        assert n_points >= 2, f"nFitPoints={n_points} is not enough"
        if n_points > int(0.25 * N):
            warnings.warn(
                "Using too many points for the fit means including points which have higher measurment errors.")
            # Selecting more points than 25% should be possible, but not advised

        # This is the time array, as the fits will be MSD vs T
        T = np.linspace(1, N, N,  endpoint=True)

        # Definining the models used for the fit
        def model1(t, D, delta2): return 4 * D * t + 2 * delta2
        def model2(t, D, delta2, alpha): return 4 * D * t**alpha + 2 * delta2

        # Fit the data to these 2 models using weighted least-squares fit
        # TODO: normalize the curves to make the fit easier to perform.
        reg1 = optimize.curve_fit(
            model1, T[0:n_points], self.__msd[0:n_points], sigma=self.__msd_error[0:n_points])
        # print(f"reg1 parameters: {reg1[0]}") # Debug
        reg2 = optimize.curve_fit(model2, T[0:n_points], self.__msd[0:n_points], [
                                  *reg1[0][0:2], 1.0], sigma=self.__msd_error[0:n_points])
        # reg2 = optimize.curve_fit(model2, T[0:n_points], this_msd[0:n_points], sigma=this_msd_error[0:n_points])
        # print(f"reg2 parameters: {reg2[0]}") #Debug

        # Compute BIC for both models
        m1 = model1(T, *reg1[0])
        m2 = model2(T, *reg2[0])
        bic1 = BIC(m1[0:n_points], self.__msd[0:n_points], 2, 1)
        bic2 = BIC(m2[0:n_points], self.__msd[0:n_points], 2, 1)
        # FIXME: numerical instabilities due to low position values. should normalize before analysis, and then report those adimentional values.

        # Relative Likelihood for each model
        rel_likelihood_1 = np.exp((bic1 - min([bic1, bic2])) * 0.5)
        rel_likelihood_2 = np.exp((bic2 - min([bic1, bic2])) * 0.5)

        self.__msd_analysis_results["analyzed"] = True
        self.__msd_analysis_results["results"] = {"model1": {"params": reg1[0], "BIC": bic1, "rel_likelihood": rel_likelihood_1},
                                                  "model2": {"params": reg2[0], "BIC": bic2, "rel_likelihood": rel_likelihood_2},
                                                  "n_points": n_points}

        return self.__msd_analysis_results

    def adc_analysis(self, R: float = 1/6, fractionFit=0.25, maxfev=1000, numWorkers=None, chunksize=chunksize):
        print(fractionFit)
        """Revised analysis using the apparent diffusion coefficient
        Parameters
        ----------
        R: float
            Point scanning across the field of view.
        nFitPoints: int
            Number of points used for fitting. Defaults to 25 % of total points.
        maxfev: int
            maxfev used by Scipy fitting routine.
        categorize: bool
            categorize the track model after Dapp calculation
        """
        # Calculate MSD if this has not been done yet.
        if self.__msd is None:
            self.calculate_msd(numWorkers=numWorkers, chunksize=chunksize)

        dt = self.__t[1] - self.__t[0]

        N = self.__msd.size

        # Time coordinates
        # This is the time array, as the fits will be MSD vs T
        T = np.linspace(dt, dt*N, N, endpoint=True)

        # Compute  the time-dependent apparent diffusion coefficient.
        Dapp = self.__msd / (4 * T * (1 - 2*R*dt / T))

        model, results = self.__categorize(np.array(Dapp), np.arange(
            1, N+1), fractionFit=fractionFit, maxfev=maxfev)

        self.__adc_analysis_results["analyzed"] = True
        self.__adc_analysis_results["Dapp"] = np.array(Dapp)
        self.__adc_analysis_results["model"] = model
        self.__adc_analysis_results["results"] = results

        return self.__adc_analysis_results

    def sd_analysis(self, display_fit: bool = False, binsize_nm: float = 10.0,
                    J: list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100], fractionFit: float = 0.25, maxfev=1000):
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
        categorize: bool
            categorize the track model after Dapp calculation
        """
        # Convert binsize to m
        binsize = binsize_nm * 1e-9

        dt = self.__t[1] - self.__t[0]

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
            popt, pcov = optimize.curve_fit(
                rayleighPDF, bin_mids, hist_SD, p0=(max_x-min_x))
            if display_fit:
                # Plot binned data
                plt.bar(bins[:-1], hist_SD, width=(bins[1] - bins[0]),
                        align='edge', alpha=0.5, label="Data")
                plt.gca().set_xlim(0, 4.0e-7)
                # Plot the fit
                eval_x = np.linspace(bins[0], bins[-1], 100)
                plt.plot(eval_x, rayleighPDF(
                    eval_x, popt[0]), label="Rayleigh Fit")
                plt.legend()
                plt.title("$n = {}$".format(j))
                plt.show()

            sigma = popt[0]
            dapp = sigma**2 / (2 * t_lag)
            dapp_list.append(dapp)

        plt.semilogx(np.array(J) * dt, dapp_list)
        plt.xlabel("Time")
        plt.ylabel("Estimated $D_{app}$")
        plt.show()

        model, results = self.__categorize(np.array(dapp_list), np.array(
            J), fractionFit=fractionFit, maxfev=maxfev)

        self.__sd_analysis_results["analyzed"] = True
        self.__sd_analysis_results["Dapp"] = np.array(dapp_list)
        self.__sd_analysis_results["J"] = np.array(J)
        self.__sd_analysis_results["model"] = model
        self.__sd_analysis_results["results"] = results

    def __categorize(self, Dapp, J, R: float = 1/6, fractionFit: float = 0.25, maxfev=1000):
        if fractionFit > 0.25:
            warnings.warn(
                "Using too many points for the fit means including points which have higher measurment errors.")

        dt = self.__t[1] - self.__t[0]
        T = J * dt

        n_points = np.argmax(J > fractionFit * J[-1])
        # Define the models to fit the Dapp
        def model_brownian(t, D, delta): return D + \
            delta**2 / (2 * t * (1 - 2*R*dt/t))
        def model_confined(t, D_micro, delta, tau): return D_micro * (tau/t) * \
            (1 - np.exp(-tau/t)) + delta ** 2 / (2 * t * (1 - 2 * R * dt / t))

        def model_hop(t, D_macro, D_micro, delta, tau): return D_macro + D_micro * \
            (tau/t) * (1 - np.exp(-tau/t)) + \
            delta ** 2 / (2 * t * (1 - 2 * R * dt / t))

        # Perform fits.
        if self.is_msd_calculated():
            error = self.__msd_error[J[0:n_points]]
        else:
            error = None

        r_brownian = optimize.curve_fit(
            model_brownian, T[0:n_points], Dapp[0:n_points], sigma=error, maxfev=maxfev)
        r_confined = optimize.curve_fit(
            model_confined, T[0:n_points], Dapp[0:n_points], sigma=error, maxfev=maxfev)
        r_hop = optimize.curve_fit(
            model_hop, T[0:n_points], Dapp[0:n_points], sigma=error, maxfev=maxfev)

        # Compute BIC for each model.
        pred_brownian = model_brownian(T, *r_brownian[0])
        pred_confined = model_confined(T, *r_confined[0])
        pred_hop = model_hop(T, *r_hop[0])
        bic_brownian = BIC(
            pred_brownian[0:n_points], Dapp[0:n_points], len(r_brownian[0]), 1)
        bic_confined = BIC(
            pred_confined[0:n_points], Dapp[0:n_points], len(r_confined[0]), 1)
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

        # Calculate the relative likelihood for each model
        rel_likelihood_brownian = np.exp((bic_brownian - bic_min) * 0.5)
        rel_likelihood_confined = np.exp((bic_confined - bic_min) * 0.5)
        rel_likelihood_hop = np.exp((bic_hop - bic_min) * 0.5)

        return category, {"brownian": {"params": r_brownian[0], "bic": bic_brownian, "rel_likelihood": rel_likelihood_brownian},
                          "confined": {"params": r_confined[0], "bic": bic_confined, "rel_likelihood": rel_likelihood_confined},
                          "hop": {"params": r_hop[0], "bic": bic_hop, "rel_likelihood": rel_likelihood_hop}, "n_points": n_points, "R": R}


class NormalizedTrack(Track):
    def __init__(self, x=None, y=None, t=None, xy_min=None, xy_max=None, tmin=None, tmax=None):
        Track.__init__(self, x, y, t)
        self.__xy_min = xy_min
        self.__xy_max = xy_max
        self.__tmin = tmin
        self.__tmax = tmax


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
