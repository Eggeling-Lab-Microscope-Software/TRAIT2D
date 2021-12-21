#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tqdm
import warnings
import logging
import csv
from scipy import optimize
from scipy import interpolate
import pandas as pd

from trait2d.exceptions import *

import itertools
import os
from concurrent.futures import ProcessPoolExecutor

class Borg:
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state

class ModelDB(Borg):
    """Singleton class holding all models that should be used in analysis."""
    models = []
    def __init__(self):
        Borg.__init__(self)
    def add_model(self, model):
        """Add a new model class to the ModelDB.

        Parameters
        ----------
        model:
            Model class (*not* an instance) to add. There are predefined models available in
            trait2d.analysis.models. Example usage:

            .. code-block:: python

                from trait2d.analysis.models import ModelConfined
                ModelDB().add_model(ModelConfined)
        """
        for m in self.models:
            if m.__class__ == model:
                raise ValueError("ModelDB already contains an instance of the model {}.".format(model.__name__))
        self.models.append(model())

    def get_model(self, model):
        """
        Return the model instance from ModelDB.

        Parameters
        ----------
        model:
            Model class (*not* and instance) to remove. Example usage:

            .. code-block:: python
            
                from trait2d.analysis.models import ModelConfined
                ModelDB().get_model(ModelConfined).initial = [1.0e-12, 1.0e-9, 0.5e-3]
        """

        for i in range(len(self.models)):
            if model == self.models[i].__class__:
                return self.models[i]
        raise ValueError("ModelDB does not contain an instance of the model {}.".format(model.__name__))

    def remove_model(self, model):
        """
        Remove a model from ModelDB.

        Parameters
        ----------
        model:
            Model class (*not* an instance) to remove. Example usage:

            .. code-block:: python

                from trait2d.analysis.models import ModelConfined
                ModelDB().remove_model(ModelConfined)
        """
        for i in range(len(self.models)):
            if model == self.models[i].__class__:
                self.models.pop(i)
                return
        raise ValueError("ModelDB does not contain an instance of the model {}.".format(model.__name__))

    def cleanup(self):
        """
        Remove all models from ModelDB. It is good practice to call this
        at the end of your scripts since other scripts might share the same
        instance.
        """
        self.models = []

class ListOfTracks:
    """Create an object that can hold multiple tracks and analyze them in bulk.

    Parameters
    ----------
    tracks : list
        A Python list containing the tracks as `Track` objects.
    """

    def __init__(self, tracks: list = None):
        self._tracks = tracks

    @classmethod
    def from_file(cls, filename, format=None, col_name_x='x', col_name_y='y', col_name_t='t', col_name_id='id', unit_length='metres', unit_time='seconds'):
        """Create a ListOfTracks from a file containing multiple tracks. Currently only supports '.csv' files.
        The file must contain the fields 'x', 'y', 't' as well as 'id'. Different column names can also be
        specified using the appropriate arguments.

        Parameters
        ----------
        filename: str
            Name of the file.
        format: str
            Either 'csv' or 'json' or 'pcl'. Only csv is implemented at the moment.
        col_name_x: str
            Column title of x positions.
        col_name_y: str
            Column title of y positions.
        col_name_t: str
            Column title of time.
        col_name_id: str
            Column title of track IDs.
        unit_length: str
            Length unit of track data. Either 'metres', 'millimetres', 'micrometres' or 'nanometres'.
        unit_time: str
            Time unit of track data. Either 'seconds', 'milliseconds', 'microseconds' or 'nanoseconds'.
        id: int
            Track ID in case the file contains more than one track.

        Raises
        ------
        LoadTrackIdNotFoundError
            When no track with the given id is found
        LodTrackIdMissingError
            When the file contains multiple tracks but no id is specified.
        """
        df = pd.read_csv(filename)
        ids = df[col_name_id].unique()
        tracks = []
        for id in ids:
            tracks.append(Track.from_dataframe(df, col_name_x, col_name_y, col_name_t, col_name_id, unit_length, unit_time, id))
        return cls(tracks)

    def __repr__(self):
        return ("<%s instance at %s>\n"
                "Number of tracks: %s\n") % (self.__class__.__name__,
                                             id(self),
                                             len(self._tracks))

    def get_tracks(self):
        """Return a Python list that contains the tracks.

        Returns
        -------
        tracks : list
            List of tracks. Each element will be a `Track` object.
        """
        return self._tracks

    def get_track(self, idx):
        """Return a single track at the specified index.

        Parameters
        ----------
        idx : int
            Index of track.

        Returns
        -------
        track: Track
            Track at index idx.
        """
        return self._tracks[idx]

    def get_track_by_id(self, id):
        """Return a single track with the specified ID.
        If there are multiple tracks with the same ID,
        only the first track is returned.

        Parameters
        ----------
        id: int
            ID of the track.

        Returns
        -------
        track: Track
            Track with the specified ID.
        """
        for track in self._tracks:
            if track.get_id() == id:
                return track

    def get_sublist(self, model):
        """Return a new ListOfTracks containing only tracks categorized as
        the specified model using ADC analysis.

        Parameters
        ----------
        model
            Class (*not* an instance) of the model the tracks should be categorized as.
            Predefined models can be found at `trait2d.analysis.models`.

        Returns
        -------
        track_list: ListOfTracks
            ListOfTracks containing the tracks that meet the criteria.
            Note: The tracks in the new list will still be references to to
            the tracks in the original list.
        """
        track_list = []
        for track in self._tracks:
            if track.get_adc_analysis_results() is not None:
                if track.get_adc_analysis_results()["best_model"] == model.__name__:
                    track_list.append(track)
        return ListOfTracks(track_list)

    def plot_trajectories(self, cmap="plasma"):
        """Plot all trajectories.

        Parameters
        ----------
        cmap : str
            Name of the colormap to use (see https://matplotlib.org/tutorials/colors/colormaps.html
            for a list of possible values)
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.ticker import FuncFormatter
        from matplotlib.cm import get_cmap
        cmap = get_cmap(cmap)
        plt.figure()
        ax = plt.gca()
        xmin = 0
        xmax = 0
        ymin = 0
        ymax = 0
        for track in self._tracks:
            segs = []
            colors = []
            x = track.get_x()
            y = track.get_y()
            t = track.get_t()
            x -= x[0]
            y -= y[0]
            t -= t[0]
            tmax = t.max()
            tmin = t.min()
            tdif = tmax - tmin
            for i in range(1, t.size):
                segs.append([(float(x[i-1]), float(y[i-1])),
                             (float(x[i]), float(y[i]))])
                colors.append(cmap(float(t[i] - tmin) / tdif))
            lc = LineCollection(segs, colors=colors)
            ax.add_collection(lc)
            xmin = min(xmin, x.min())
            xmax = max(xmax, x.max())
            ymin = min(ymin, y.min())
            ymax = max(ymax, y.max())
        ax.set_title("Plot of {} trajectories".format(len(self._tracks)))
        xspan = xmax - xmin
        yspan = ymax - ymin
        ax.set_xlim(xmin - 0.1 * xspan, xmax + 0.1 * xspan)
        ax.set_ylim(ymin - 0.1 * yspan, ymax + 0.1 * yspan)
        ax.axhline(0, linewidth=0.5, color='black', zorder=-1)
        ax.axvline(0, linewidth=0.5, color='black', zorder=-1)
        ax.set_aspect('equal', 'datalim')
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: "%d" % int(x * 1e9)))
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: "%d" % int(x * 1e9)))
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        plt.show()

    def normalize(self, **kwargs):
        """Normalize all tracks. All Track instances of the ListOfTracks will be
        replaced by NormalizedTrack instances containing individual information about
        the normalization.

        Parameters
        ----------
            Keyword arguments to be used by Track.normalized() for each track.
        """
        self._tracks = [track.normalized(**kwargs) for track in self._tracks]

    def msd_analysis(self, **kwargs):
        """Analyze all tracks using MSD analysis.

        Parameters
        ----------
        **kwargs
            Keyword arguments to be used by msd_analysis for each track.

        Returns
        -------
        list_failed
            List containing the indices of the tracks for which the analysis failed.
        """
        list_failed = []
        i = 0
        for track in self._tracks:
            try:
                track.msd_analysis(**kwargs)
            except:
                list_failed.append(i)
            i += 1

        if len(list_failed) > 0:
            warnings.warn("MSD analysis failed for {}/{} tracks. \
                Consider raising the maximum function evaluations using \
                the maxfev keyword argument. \
                To get a more detailed stacktrace, run the MSD analysis \
                for a single track.".format(len(list_failed), len(self._tracks)))

        return list_failed

    def adc_analysis(self, **kwargs):
        """Analyze all tracks using ADC analysis.

        Parameters
        ----------
        **kwargs
            Keyword arguments to be used by adc_analysis for each track.

        Returns
        -------
        list_failed
            List containing the indices of the tracks for which the analysis failed.
        """
        list_failed = []
        i = 0
        for track in self._tracks:
            try:
                track.adc_analysis(**kwargs)
            except:
                list_failed.append(i)
            i+=1
        
        if len(list_failed) > 0:
            warnings.warn("ADC analysis failed for {}/{} tracks. "
                "Consider raising the maximum function evaluations using "
                "the maxfev keyword argument. "
                "To get a more detailed stacktrace, run the ADC analysis "
                "for a single track.".format(len(list_failed), len(self._tracks)))

        return list_failed

    def adc_summary(self, ensemble_average = False, avg_only_params = False, interpolation = False, plot_msd = False, plot_dapp = False, plot_pie_chart = False):
        """Average tracks by model and optionally of the whole ensemble and optionally plot the results.

        Parameters
        ----------
        ensemble_average: bool
            averages the MSD and D_app, but not the parameters, for all the tracks and adds it to the results
        avg_only_params: bool
            Only average the model parameters but not D_app and MSD
        interpolation: bool
            User linear interpolation of averaging. Has to be used when
            not all tracks have the same uniform time step size.
        plot_msd: bool
            Plot the averaged MSD for each model.
        plot_dapp: bool
            Plot the averaged D_app for each model.
        plot_pie_chart: bool
            Plot a pie chart showing the relative fractions of each model.

        Returns
        -------
        results : dict
            Relative shares and averaged values of each model.
        """

        if avg_only_params and (plot_msd or plot_dapp):
            warnings.warn("avg_only_params is True. plot_msd or plot_dapp will have no effect.")

        track_length = 0
        max_t = 0.0
        t = None
        for track in self._tracks:
            if track.get_t()[-1] > max_t:
                max_t = track.get_t()[-1]
                track_length = track.get_x().size
                t = track.get_t()

        average_D_app = {}
        average_MSD = {}
        average_params = {}
        sampled = {}
        counter = {}

        dt = t[1] - t[0]

        k = 0
        for track in self._tracks:
            k += 1
            if track.get_t()[1] - track.get_t()[0] != dt and not avg_only_params and not interpolation:
                raise ValueError("Cannot average MSD and D_app: Encountered track with incorrect time step size! "
                                 "(Got {}, expected {} for track {}.) Use the flag avg_only_params = True or "
                                 "enable interpolation with interpolation = True.".format(
                    track.get_t()[1] - track.get_t()[0], dt, k + 1))

        # Parameter averaging
        for track in self._tracks:
            if track.get_adc_analysis_results() is None:
                continue
            model = track.get_adc_analysis_results()["best_model"]
            if model is not None:
                if not model in average_params.keys():
                    average_params[model] = len(track.get_adc_analysis_results()["fit_results"][model]["params"]) * [0.0]
                    counter[model] = 0
                counter[model] += 1
                average_params[model] += track.get_adc_analysis_results()["fit_results"][model]["params"]
        
        for model in average_params.keys():
            average_params[model] /= counter[model]
        
        k = 0
        for track in self._tracks:
            k += 1
            if track.get_t()[1] - track.get_t()[0] != dt and not avg_only_params and not interpolation:
                raise ValueError("Cannot average MSD and D_app: Encountered track with incorrect time step size! "
                                 "(Got {}, expected {} for track {}.) Use the flag avg_only_params = True or "
                                 "enable interpolation with interpolation = True.".format(
                    track.get_t()[1] - track.get_t()[0], dt, k + 1))

        if not avg_only_params:
            for track in self._tracks:
                if track.get_adc_analysis_results() is None:
                    continue

                model = track.get_adc_analysis_results()["best_model"]

                D_app = np.zeros(track_length - 3)
                MSD = np.zeros(track_length - 3)
                if interpolation:
                    interp_MSD = interpolate.interp1d(track.get_t()[0:-3], track.get_msd(), bounds_error = False, fill_value = 0)
                    interp_D_app = interpolate.interp1d(track.get_t()[0:-3], track.get_adc_analysis_results()["Dapp"], bounds_error = False, fill_value = 0)
                    MSD = interp_MSD(t[0:-3])
                    D_app = interp_D_app(t[0:-3])
                else:
                    D_app[0:track.get_adc_analysis_results()["Dapp"].size] = track.get_adc_analysis_results()["Dapp"]
                    MSD[0:track.get_msd().size] = track.get_msd()
                mask = np.zeros(track_length - 3)
                np.put(mask, np.where(MSD != 0.0), 1)


                if not model in average_D_app.keys():
                    average_D_app[model] = np.zeros(track_length - 3)
                    average_MSD[model] = np.zeros(track_length - 3)
                    sampled[model] = np.zeros(track_length - 3)

                average_D_app[model] += D_app
                average_MSD[model] += MSD
                sampled[model] += mask

        counter_sum = 0
        for model in counter:
            counter_sum += counter[model]
            if counter[model]:
                average_D_app[model] /= sampled[model]
                average_MSD[model] /= sampled[model]

        if counter_sum == 0:
            warnings.warn("No tracks are categorized!")

        sector = {}
        for model in counter:
            sector[model] = counter[model] / len(self._tracks)
        sector["not catergorized"] = (len(self._tracks) - counter_sum) / len(self._tracks)
        
        ###ENSEMBLE AVERAGE - Added by FR###
        
        
        if ensemble_average:
            ens_D_app = np.zeros(track_length - 3)
            ens_MSD = np.zeros(track_length - 3)
            sampled_total = np.zeros(track_length - 3)
            
            for track in self._tracks:
                if track.get_adc_analysis_results() is None:
                    continue
                
                D_app = np.zeros(track_length - 3)
                MSD = np.zeros(track_length - 3)

                if interpolation:
                    interp_MSD = interpolate.interp1d(track.get_t()[0:-3], track.get_msd(), bounds_error = False, fill_value = 0)
                    interp_D_app = interpolate.interp1d(track.get_t()[0:-3], track.get_adc_analysis_results()["Dapp"], bounds_error = False, fill_value = 0)
                    MSD = interp_MSD(t[0:-3])
                    D_app = interp_D_app(t[0:-3])
                else:
                    D_app[0:track.get_adc_analysis_results()["Dapp"].size] = track.get_adc_analysis_results()["Dapp"]
                    MSD[0:track.get_msd().size] = track.get_msd()
                    
                mask_total = np.zeros(track_length - 3)
                np.put(mask_total, np.where(MSD != 0.0), 1)
                
                sampled_total += mask_total
                ens_D_app += D_app
                ens_MSD += MSD
            
            #Final averaging operation#
            average_MSD['Ensemble'] = ens_MSD/sampled_total
            average_D_app['Ensemble'] = ens_D_app/sampled_total
            
        ###END OF ADDITION###

        if plot_msd and not avg_only_params:
            import matplotlib.pyplot as plt
            plt.figure()
            ax = plt.gca()
            ax.set_xlabel("t")
            ax.set_ylabel("Average MSD")
            for model in average_MSD.keys():
                ax.semilogx(t[0:-3], average_MSD[model], label=model)
            ax.legend()
            plt.show()
            
        if plot_dapp and not avg_only_params:
            import matplotlib.pyplot as plt
            plt.figure()
            ax = plt.gca()
            ax.set_xlabel("t")
            ax.set_ylabel("Average ADC")
            for model in average_D_app.keys():
                ax.semilogx(t[0:-3], average_D_app[model], label=model)
            ax.legend()
            plt.show()

        if plot_dapp and avg_only_params:
            import matplotlib.pyplot as plt
            plt.figure()
            min_val = 9999999.9
            max_val = 0.0
            ax = plt.gca()
            ax.set_title("curves generated from the average parameters from the ADC analysis ")
            ax.set_xlabel("t")
            ax.set_ylabel("Average ADC")
            for model in counter:
                new_min = np.min(average_D_app[model])
                new_max = np.max(average_D_app[model])
                min_val = min(min_val, new_min)
                max_val = max(max_val, new_max)
                l, = ax.semilogx(t[0:-3], average_D_app[model], label=model)
                r = average_params[model]
                for c in ModelDB().models:
                    if c.__class__.__name__ == model:
                        m = c
                        pred = m(t, *r)
                        plt.semilogx(t[0:-3], pred[0:-3], linestyle='dashed', color=l.get_color())
                    else:
                        continue
            ax.set_ylim(0.95*min_val, 1.05*max_val)
            ax.legend()
            plt.show()

        if plot_pie_chart:
            import matplotlib.pyplot as plt
            plt.figure()
            ax = plt.gca()
            ax.pie(sector.values(),
            labels=sector.keys())    
            plt.show() 
  
        return {"sectors": sector,
                "average_params": average_params,
                "t":t,
                "average_msd": average_MSD,
                "average_dapp": average_D_app}

    def get_msd(self, interpolation = False): # Get average MSD
        """Get the mean squared displacement averaged over all tracks.

        Parameters
        ----------
        interpolation: bool
            Linearly interpolate all msd values over the time points of the first track.
            Use when working with differently spaced tracks.

        Returns
        -------
        t : ndarray
            1-dimensional array containing time points at which msd was sampled
        msd: ndarray
            1-dimensional array containing averaged msd values
        """
        track_length = 0
        max_t = 0.0
        t = None
        for track in self._tracks:
            if track.get_t()[-1] > max_t:
                max_t = track.get_t()[-1]
                track_length = track.get_x().size
                t = track.get_t()
        average_MSD = np.zeros(track_length - 3)
        average_MSD_err = np.zeros(track_length - 3)
        sampled = np.zeros(track_length - 3)
        for track in self._tracks:
            if track.get_msd() is None:
                continue

            MSD = np.zeros(track_length - 3)
            MSD_err = np.zeros(track_length - 3)
            if interpolation:
                interp_MSD = interpolate.interp1d(track.get_t()[0:-3], track.get_msd(), bounds_error = False, fill_value = 0)
                interp_MSD_err = interpolate.interp1d(track.get_t()[0:-3], track.get_msd(), bounds_error = False, fill_value = 0)
                MSD = interp_MSD(t[0:-3])
                MSD_err = interp_MSD_err(t[0:-3])
            else:
                MSD[0:track.get_msd().size] = track.get_msd()
                MSD_err[0:track.get_msd().size] = track._msd_error
            mask = np.zeros(track_length - 3)
            np.put(mask, np.where(MSD != 0.0), 1)

            average_MSD += MSD
            average_MSD_err += MSD_err
            sampled += mask
        average_MSD /= sampled
        average_MSD_err /= sampled
        return t[0:-3], average_MSD, average_MSD_err

    def plot_msd(self, interpolation = False):
        """Plot the mean squared displacement averaged over all tracks.

        Parameters
        ----------
        interpolation: bool
            Linearly interpolate all msd values over the time points of the first track.
            Use when working with differently spaced tracks.
        """

        t, msd, err= self.get_msd(interpolation)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.grid(linestyle='dashed', color='grey')
        plt.xlabel("t")
        plt.ylabel("Average MSD")
        plt.semilogx(t, msd, color='black')
        plt.fill_between(t, msd-err, msd+err, color='black', alpha=0.5)

    def get_dapp(self, interpolation = False):
        """Get the apparent diffusion coefficient averaged over all tracks.

        Parameters
        ----------
        interpolation: bool
            Linearly interpolate all D_app values over the time points of the first track.
            Use when working with differently spaced tracks.

        Returns
        -------
        t : ndarray
            1-dimensional array containing time points at which msd was sampled
        average_dapp: ndarray
            1-dimensional array containing averaged D_app values
        """
        track_length = 0
        max_t = 0.0
        t = None
        for track in self._tracks:
            if track.get_t()[-1] > max_t:
                max_t = track.get_t()[-1]
                track_length = track.get_x().size
                t = track.get_t()
        average_dapp = np.zeros(track_length - 3)
        sampled = np.zeros(track_length - 3)
        for track in self._tracks:
            if track.get_adc_analysis_results() is None:
                continue

            D_app = np.zeros(track_length - 3)
            if interpolation:
                interp_D_app = interpolate.interp1d(track.get_t()[0:-3], track.get_adc_analysis_results()["Dapp"], bounds_error = False, fill_value = 0)
                D_app = interp_D_app(t[0:-3])
            else:
                D_app[0:track.get_adc_analysis_results()["Dapp"].size] = track.get_adc_analysis_results()["Dapp"]
            mask = np.zeros(track_length - 3)
            np.put(mask, np.where(D_app != 0.0), 1)

            average_dapp += D_app
            sampled += mask
        average_dapp /= sampled
        return t[0:-3], average_dapp

    def plot_dapp(self):
        """Plot the apparent diffusion coefficient averaged over all tracks.

        Parameters
        ----------
        interpolation: bool
            Linearly interpolate all D_app values over the time points of the first track.
            Use when working with differently spaced tracks.
        """
        t, dapp = self.get_dapp()
        import matplotlib.pyplot as plt
        plt.figure()
        ax = plt.gca()
        ax.set_xlabel("t")
        ax.set_ylabel("Average Dapp")
        ax.semilogx(t, dapp)
        ax.legend()

    def average(self, interpolation=False):
        """Get the mean squared displacement averaged over all tracks.

        Parameters
        ----------
        interpolation: bool
            Linearly interpolate all msd values over the time points of the first track.
            Use when working with differently spaced tracks.
        use_averaged_msd: bool
            Initialize the track with a MSD curve which is the average of all MSD curves
            in the list. This is *not* neccessarily equal to the result of performing
            `calculate_msd` on the averaged track.

        Returns
        -------
        track : Track
            A single track containing the average 
        """
        t, msd, msd_err = self.get_msd()
        return MSDTrack(msd, msd_err, t)

class Track:
    """Create a track that can hold trajectory and analysis information.

    Parameters
    ----------
    x: array_like
        x coordinates of trajectory.
    y: array_like
        y coordinates of trajectory.
    t: array_like
        time coordinates of trajectory.
    """

    def __init__(self, x=None, y=None, t=None, id=None):
        self._x = np.array(x, dtype=float)
        self._y = np.array(y, dtype=float)
        self._t = np.array(t, dtype=float)

        self._id = id

        self._msd = None
        self._msd_error = None
        self._msd_SEM = None

        self._msd_analysis_results = None
        self._adc_analysis_results = None

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
    def from_dataframe(cls, df, col_name_x='x', col_name_y='y', col_name_t='t', col_name_id='id', unit_length='metres', unit_time='seconds', id=None):
        """Create a single track from a DataFrame. Currently only supports '.csv' tracks.
        The DataFrame must contain the fields 'x', 'y', 't' as well as 'id'. Different column names can also be
        specified using the appropriate arguments.

        Parameters
        ----------
        df: pandas.DataFrame
            DataFrame containing the data.
        col_name_x: str
            Column title of x positions.
        col_name_y: str
            Column title of y positions.
        col_name_t: str
            Column title of time.
        col_name_id: str
            Column title of track IDs.
        unit_length: str
            Length unit of track data. Either 'metres', 'millimetres', 'micrometres' or 'nanometres'.
        unit_time: str
            Time unit of track data. Either 'seconds', 'milliseconds', 'microseconds' or 'nanoseconds'.
        id: int
            Track ID in case the file contains more than one track.

        Raises
        ------
        LoadTrackIdNotFoundError
            When no track with the given id is found
        LodTrackIdMissingError
            When the file contains multiple tracks but no id is specified.
        """
        length_factor = None
        if unit_length == "metres":
            length_factor = 1
        elif unit_length == "millimetres":
            length_factor = 1e-3
        elif unit_length == "micrometres":
            length_factor = 1e-6
        elif unit_length == "nanometres":
            length_factor = 1e-9
        else:
            raise ValueError(
                "unit must be metres, millimetres, micrometres, or nanometres")

        time_factor = None
        if unit_time == "seconds":
            time_factor = 1
        elif unit_time == "milliseconds":
            time_factor = 1e-3
        elif unit_time == "microseconds":
            time_factor = 1e-6
        elif unit_time == "nanoseconds":
            time_factor = 1e-9
        else:
            raise ValueError(
                "unit must be metres, millimetres, micrometres, or nanometres")

        x = None
        y = None
        t = None
        if col_name_id in df:
            if np.min(df[col_name_id]) == np.max(df[col_name_id]):
                id = np.min(df[col_name_id])  # If there is only one id, we just select it
            if id == None:
                raise LoadTrackMissingIdError("The file seems to contain more than one track. Please specify a track id using the keyword argument id.")
            df = df.loc[df[col_name_id] == int(id)]
            if df.empty:
                raise LoadTrackIdNotFoundError("There is no track associated with the specified id!")

        x = np.array(df[col_name_x])
        y = np.array(df[col_name_y])
        t = np.array(df[col_name_t])

        return cls(x * length_factor, y * length_factor, t * time_factor, id=id)

    @classmethod
    def from_file(cls, filename, format=None, col_name_x='x', col_name_y='y', col_name_t='t', col_name_id='id', unit_length='metres', unit_time='seconds', id=None):
        """Create a single track from a file. Currently only supports '.csv' tracks.
        The DataFrame must contain the fields 'x', 'y', 't' as well as 'id'. Different column names can also be
        specified using the appropriate arguments.

        Parameters
        ----------
        filename: str
            Name of the file.
        format: str
            Either 'csv' or 'json' or 'pcl'. Only csv is implemented at the moment.
        col_name_x: str
            Column title of x positions.
        col_name_y: str
            Column title of y positions.
        col_name_t: str
            Column title of time.
        col_name_id: str
            Column title of track IDs.
        unit_length: str
            Length unit of track data. Either 'metres', 'millimetres', 'micrometres' or 'nanometres'.
        unit_time: str
            Time unit of track data. Either 'seconds', 'milliseconds', 'microseconds' or 'nanoseconds'.
        id: int
            Track ID in case the file contains more than one track.

        Raises
        ------
        LoadTrackIdNotFoundError
            When no track with the given id is found
        LodTrackIdMissingError
            When the file contains multiple tracks but no id is specified.
        """
        if format == None:
            format = os.path.splitext(filename)[1].replace(".", "")
            if format == "":
                raise ValueError(
                    'You must supply a format "csv", "json" or "pcl" if the file has no extension.')

        if format != "csv" and format != "json" and format != "pcl":
            raise ValueError("Unknown format: {}".format(format))
        if format == "csv":
            df = pd.read_csv(filename)
            return cls.from_dataframe(df, col_name_x, col_name_y, col_name_t, col_name_id, unit_length, unit_time, id)
        elif format == "json":
            # TODO: .json-specific import
            raise NotImplementedError(
                "Import from json is not yet implemented.")
        elif format == "pcl":
            # TODO: .pcl-specific import
            raise NotImplementedError(
                "Import from pcl is not yet implemented.")

    def __repr__(self):
        return ("<%s instance at %s>\n"
                "------------------------\n"
                "Track length:%s\n"
                "Track ID:%s\n"
                "------------------------\n"
                "MSD calculated:%s\n"
                "MSD analysis done:%s\n"
                "ADC analysis done:%s\n") % (
            self.__class__.__name__,
            id(self),
            str(self._t.size).rjust(11, ' '),
            str(self._id).rjust(15, ' '),
            str(self.is_msd_calculated()).rjust(9, ' '),
            str(self._msd_analysis_results is not None).rjust(6, ' '),
            str(self._adc_analysis_results is not None).rjust(6, ' ')
        )

    from ._msd import msd_analysis, get_msd_analysis_results,\
                      delete_msd_analysis_results, plot_msd_analysis_results
    from ._adc import adc_analysis, get_adc_analysis_results,\
                      delete_adc_analysis_results, plot_adc_analysis_results

    def plot_trajectory(self, cmap='plasma'):
        """Plot the trajectory.

        Parameters
        ----------
        cmap : str
            Name of the colormap to use (see https://matplotlib.org/tutorials/colors/colormaps.html
            for a list of possible values)
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.ticker import FuncFormatter
        from matplotlib.cm import get_cmap
        cmap = get_cmap(cmap)
        plt.figure()
        ax = plt.gca()
        segs = []
        colors = []

        tmax = self._t.max()
        tmin = self._t.min()
        tdif = tmax - tmin

        for i in range(1, self._t.size):
            segs.append([(float(self._x[i-1]), float(self._y[i-1])),
                         (float(self._x[i]), float(self._y[i]))])
            colors.append(
                cmap(float(self._t[i] - tmin) / tdif))
        lc = LineCollection(segs, colors=colors)
        ax.axhline(self._y[0], linewidth=0.5, color='black', zorder=-1)
        ax.axvline(self._x[0], linewidth=0.5, color='black', zorder=-1)
        ax.add_collection(lc)
        ax.set_title("Trajectory")
        xspan = self._x.max() - self._x.min()
        yspan = self._y.max() - self._y.min()
        ax.set_xlim(self._x.min() - 0.1 * xspan, self._x.max() + 0.1 * xspan)
        ax.set_ylim(self._y.min() - 0.1 * yspan, self._y.max() + 0.1 * yspan)
        ax.set_aspect('equal', 'datalim')
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: "%d" % int(x * 1e9)))
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: "%d" % int(x * 1e9)))
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        plt.show()

    def plot_msd(self):
        t = self._t[0:-3]
        msd = self._msd
        err = self._msd_error
        import matplotlib.pyplot as plt
        plt.figure()
        plt.grid(linestyle='dashed', color='grey')
        plt.xlabel("t")
        plt.ylabel("MSD")
        plt.semilogx(t, msd, color='black')
        plt.fill_between(t, msd-err, msd+err, color='black', alpha=0.5)

    def get_x(self):
        """Return x coordinates of trajectory."""
        return self._x

    def get_y(self):
        """Return y coordinates of trajectory."""
        return self._y

    def get_t(self):
        """Return time coordinates of trajectory."""
        return self._t

    def get_id(self):
        """Return ID of the track."""
        return self._id

    def get_size(self):
        """Return number of points of the trajectory."""
        return self._t.size

    def get_trajectory(self):
        """Returns the trajectory as a dictionary."""
        return {"t": self._t.tolist(), "x": self._x.tolist(), "y": self._y.tolist()}

    def is_msd_calculated(self):
        """Returns True if the MSD of this track has already been calculated."""
        return self._msd is not None

    def get_msd(self):
        """Returns the MSD values of the track."""
        return self._msd

    def get_msd_error(self):
        """Returns the MSD error values of the track."""
        return self._msd_error
    
    def get_msd_SEM(self):
        """Returns the MSD SEM of the track."""
        return self._msd_SEM

    def normalized(self, normalize_t = True, normalize_xy = True):
        """Normalize the track.

        Parameters
        ----------
        normalize_t: bool
            Normalize the time component of the track by setting the first time in the
            track to zero.
        normalize_xy: bool
            Normalize the x and y coordinates of the track by setting the initial
            position in the track to zero.

        Returns
        -------
        Instance of NormalizedTrack containing the normalized track data.
        """

        x = self._x
        y = self._y
        t = self._t
        
        xmin = 0.0
        ymin = 0.0
        tmin = 0.0

        # Normalizing the coordinates
        if normalize_xy:
            xmin = x.min()
            ymin = y.min()
            x = x - xmin
            y = y - ymin
        if normalize_t:
            tmin = t.min()
            t = t - tmin

        # Create normalized Track object
        return NormalizedTrack(x, y, t, xmin, ymin, tmin, id=self._id)

    def calculate_msd(self):
        
        """Calculates the track's mean squared displacement (msd) and stores it in 'track' object. Also deletes temporary values and colelcts them.
           Furthermore, calculates the standard deviation per point of the MSD array (msd_error) and the standard error of the mean (SEM) [legacy version].
        """
                
        N = len(self._x)
        col_Array  = np.zeros(N-3)
        Err_col_Array = np.zeros(N-3)

        data_tmp = np.column_stack((self._x, self._y))
    
        for i in tqdm.tqdm(range(1,N-2)):
            calc_tmp = np.sum(np.abs((data_tmp[1+i:N,:] - data_tmp[1:N - i,:]) ** 2), axis=1)
            col_Array[i-1] = np.mean(calc_tmp)
            Err_col_Array[i-1] = np.std(calc_tmp)
            
        d = np.arange(self._x.shape[0] - 1, 2, -1)

        self._msd = col_Array                       #store MSD
        self._msd_error = Err_col_Array             #store stdev of MSD
        self._msd_SEM = Err_col_Array/(np.sqrt(d))  #calculate and store SEM of MSD
        

    def _categorize(self, Dapp, J, Dapp_err = None, R: float = 1/6, fraction_fit_points: float = 0.25, fit_max_time: float=None, maxfev=1000, enable_log_sampling = False, log_sampling_dist = 0.2, weighting = 'error'):
        if fraction_fit_points > 0.25:
            warnings.warn(
                "Using too many points for the fit means including points which have higher measurment errors.")

        dt = self._t[1] - self._t[0]
        T = J * dt

        # Get number of points for fit from either fit_max_time or fraction_fit_points
        if fit_max_time is not None:
            n_points = int(np.argwhere(T < fit_max_time)[-1])
        else:
            n_points = np.argmax(J > fraction_fit_points * J[-1])

        cur_dist = 0
        idxs = []
        if enable_log_sampling:
            # Get indexes that are (approximately) logarithmically spaced
            idxs.append(0)
            for i in range(1, n_points):
                cur_dist += np.log10(T[i]/T[i-1])
                if cur_dist >= log_sampling_dist:
                    idxs.append(i)
                    cur_dist = 0
        else:
            # Get every index up to n_points
            idxs = np.arange(0, n_points, dtype=int)

        error = None
        if not Dapp_err is None:
            error = Dapp_err[idxs]

        # Perform fits for all included models
        fit_results = {}

        bic_min = 999.9
        category = None
        sigma = None
        if weighting == 'error':
            sigma = error
        elif weighting == 'inverse_variance':
            sigma = np.power(error, 2.0)
        elif weighting == 'variance':
            sigma = 1 / np.power(error, 2.0)
        elif weighting == 'disabled':
            sigma = None
        else:
            raise ValueError("Unknown weighting method: {}. Possible values are: 'error', 'variance', 'inverse_variance', and 'disabled'.".format(weighting))

        for model in ModelDB().models:
            model.R = R
            model.dt = dt
            model_name = model.__class__.__name__

            r = optimize.curve_fit(model, T[idxs], Dapp[idxs], p0 = model.initial,
                        sigma = sigma, maxfev = maxfev, method='trf', bounds=(model.lower, model.upper))

            perr = np.sqrt(np.diag(r[1]))
            pred = model(T, *r[0])
            bic = BIC(pred[idxs], Dapp[idxs], len(r[0]), len(idxs))
            if bic < bic_min:
                bic_min = bic
                category = model_name

            from scipy.stats import kstest
            test_results = kstest(Dapp[idxs], pred[idxs], N = len(idxs))

            fit_results[model_name] = {"params": r[0], "errors": perr, "bic" : bic, "KSTestStat": test_results[0], "KStestPValue": test_results[1]}

        # Calculate the relative likelihood for each model
        for model in ModelDB().models:
            model_name = model.__class__.__name__
            rel_likelihood = np.exp((-fit_results[model_name]["bic"] + bic_min) * 0.5)
            fit_results[model_name]["rel_likelihood"] = rel_likelihood

        fit_indices = idxs
        return category, fit_indices, fit_results


class NormalizedTrack(Track):
    """A track with normalized coordinates and additional information about the normalization."""

    def __init__(self, x=None, y=None, t=None, xmin=None, ymin=None, tmin=None, id=None):
        Track.__init__(self, x, y, t, id=id)
        self._xmin = xmin
        self._ymin = ymin
        self._tmin = tmin

class MSDTrack(Track):
    """A special track class that holds an MSD curve but no trajectory data. Can be used for analysis like an ordinary :class:`trait2d.analysis.Track` object but certain functions might not be available."""

    def __init__(self, msd, msd_err, t, id=None):
        t = np.append(t, np.zeros(3))
        Track.__init__(self, t=t, id=id)
        self._msd = msd
        self._msd_error = msd_err

    def normalized(normalize_t=True, normalize_xy=True):
        TypeError("A MSDTrack instance cannot be normalized.")

    def calculate_msd(self):
        TypeError("It is not possible to calculate the MSD of a MSDTrack instance.")

    def plot_trajectory(self):
        TypeError("It is not possible to plot the trajectory of a MSDTrack instance.")

    def __repr__(self):
        return ("<%s instance at %s>\n"
                "------------------------\n"
                "Track length:%s\n"
                "Track ID:%s\n"
                "------------------------\n"
                "MSD analysis done:%s\n"
                "ADC analysis done:%s\n") % (
            self.__class__.__name__,
            id(self),
            str(self._t.size).rjust(11, ' '),
            str(self._id).rjust(15, ' '),
            str(self._msd_analysis_results is not None).rjust(6, ' '),
            str(self._adc_analysis_results is not None).rjust(6, ' ')
        )


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