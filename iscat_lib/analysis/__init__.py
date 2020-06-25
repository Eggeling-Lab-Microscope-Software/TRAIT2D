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

from iscat_lib.exceptions import *

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
        for m in self.models:
            if m.__class__ == model.__class__:
                raise ValueError("ModelDB already contains an instance of the model {}.".format(model.__class__.__name__))
        self.models.append(model)
    def remove_model(self, model):
        for i in range(len(self.models)):
            if model == self.models[i].__class__:
                self.models.pop(i)
                return
        raise ValueError("ModelDB does not contain an instance of the model {}.".format(model.__name__))

class ListOfTracks:
    """Create an object that can hold multiple tracks and analyze them in bulk.

    Parameters
    ----------
    tracks : list
        A Python list containing the tracks.
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
        ids = df["id"].unique()
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
            List of tracks.
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

    def get_sublist(self, method, model):
        """Return a new ListOfTracks containing only tracks categorized as
        the specified model using the specified method.

        Parameters
        ----------
        method: str
            Method used for classificiation can be either 'adc' or 'sd'.
        model: str
            Model the tracks are classified as. Can be one of 'brownian',
            'confined' or 'hop'.

        Returns
        -------
        track_list: ListOfTracks
            ListOfTracks containing the tracks that meet the criteria.
            Note: The tracks in the new list will still be references to to
            the tracks in the original list.
        """
        if not method in ['adc', 'sd']:
            raise ValueError("Method must be one of 'adc' or 'sd'.")
        track_list = []
        for track in self._tracks:
            if method == 'adc':
                if track.get_adc_analysis_results()["model"] == model.__name__:
                    track_list.append(track)
            elif method == 'sd':
                if track.get_sd_analysis_results()["model"] == model.__name__:
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
            for i in range(1, t.size):
                segs.append([(x[i-1], y[i-1]),
                             (x[i], y[i])])
                colors.append(cmap(t[i] / tmax))
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

    def sd_analysis(self, **kwargs):
        """Analyze all tracks using SD analyis.

        Parameters
        ----------
        **kwargs
            Keyword arguments to be used by sd_analysis for each track.
        """
        for track in self._tracks:
            track.sd_analysis(**kwargs)

    def summary(self, avg_only_params = False, interpolation = False, plot_msd = False, plot_dapp = False, plot_pie_chart = False):
        """Average tracks by model and optionally plot the results.

        Parameters
        ----------
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

        for track in self._tracks:
            if track.get_adc_analysis_results()["analyzed"] == False:
                continue
            model = track.get_adc_analysis_results()["model"]
            if not model in average_params.keys():
                average_params[model] = len(track.get_adc_analysis_results()["results"]["models"][model]["params"]) * [0.0]
            if not model in counter.keys():
                counter[model] = 0
            counter[model] += 1
            average_params[model] += track.get_adc_analysis_results()["results"]["models"][model]["params"]
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
                if track.get_adc_analysis_results()["analyzed"] == False:
                    continue

                model = track.get_adc_analysis_results()["model"]

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
            raise ValueError("No tracks are categorized!")

        sector = {}
        for model in counter:
            sector[model] = counter[model] / counter_sum

        if plot_msd and not avg_only_params:
            import matplotlib.pyplot as plt
            plt.figure()
            ax = plt.gca()
            ax.set_xlabel("t")
            ax.set_ylabel("Average MSD")
            for model in counter:
                ax.semilogx(t[0:-3], average_MSD[model], label=model)
            ax.legend()

        if plot_dapp and not avg_only_params:
            import matplotlib.pyplot as plt
            plt.figure()
            ax = plt.gca()
            ax.set_xlabel("t")
            ax.set_ylabel("Average D_app")
            for model in counter:
                l, = ax.semilogx(t[0:-3], average_D_app[model], label=model)
                r = average_params[model]
                for c in ModelDB().models:
                    if c.__class__.__name__ == model:
                        m = c
                pred = m(t, *r)
                plt.semilogx(t[0:-3], pred[0:-3], linestyle='dashed', color=l.get_color())
            ax.legend()

        if plot_pie_chart:
            import matplotlib.pyplot as plt
            plt.figure()
            ax = plt.gca()
            ax.pie(sector.values(),
                   labels=sector.keys())

        return {"sectors": sector,
                "average_params": average_params, 
                "average_msd": average_MSD,
                "average_dapp": average_D_app}

    def get_msd(self, interpolation = False): # Get average MSD
        track_length = 0
        max_t = 0.0
        t = None
        for track in self._tracks:
            if track.get_t()[-1] > max_t:
                max_t = track.get_t()[-1]
                track_length = track.get_x().size
                t = track.get_t()
        average_MSD = np.zeros(track_length - 3)
        sampled = np.zeros(track_length - 3)
        for track in self._tracks:
            if track.get_msd() is None:
                continue

            MSD = np.zeros(track_length - 3)
            if interpolation:
                interp_MSD = interpolate.interp1d(track.get_t()[0:-3], track.get_msd(), bounds_error = False, fill_value = 0)
                MSD = interp_MSD(t[0:-3])
            else:
                MSD[0:track.get_msd().size] = track.get_msd()
            mask = np.zeros(track_length - 3)
            np.put(mask, np.where(MSD != 0.0), 1)

            average_MSD += MSD
            sampled += mask
        average_MSD /= sampled
        return t[0:-3], average_MSD

    def plot_msd(self):
        t, msd = self.get_msd()
        import matplotlib.pyplot as plt
        plt.figure()
        ax = plt.gca()
        ax.set_xlabel("t")
        ax.set_ylabel("Average MSD")
        ax.semilogx(t, msd)
        ax.legend()

    def get_dapp(self, interpolation = False): # Get average MSD
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
            if track.get_adc_analysis_results()["analyzed"] == False:
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

    def plot_msd(self):
        t, msd = self.get_msd()
        import matplotlib.pyplot as plt
        plt.figure()
        ax = plt.gca()
        ax.set_xlabel("t")
        ax.set_ylabel("Average MSD")
        ax.semilogx(t, msd)
        ax.legend()

    def plot_dapp(self):
        t, dapp = self.get_dapp()
        import matplotlib.pyplot as plt
        plt.figure()
        ax = plt.gca()
        ax.set_xlabel("t")
        ax.set_ylabel("Average Dapp")
        ax.semilogx(t, dapp)
        ax.legend()

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

        self._msd_analysis_results = {"analyzed": False, "results": None}
        self._adc_analysis_results = {
            "analyzed": False, "model": "unknown", "Dapp": None, "results": None}
        self._sd_analysis_results = {
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
                "SD analysis done:%s\n"
                "ADC analysis done:%s\n") % (
            self.__class__.__name__,
            id(self),
            str(self._t.size).rjust(11, ' '),
            str(self._id).rjust(15, ' '),
            str(self.is_msd_calculated()).rjust(9, ' '),
            str(self._msd_analysis_results["analyzed"]).rjust(6, ' '),
            str(self._sd_analysis_results["analyzed"]).rjust(7, ' '),
            str(self._adc_analysis_results["analyzed"]).rjust(6, ' ')
        )

    from ._msd import msd_analysis, get_msd_analysis_results,\
                      delete_msd_analysis_results, plot_msd_analysis_results
    from ._adc import adc_analysis, get_adc_analysis_results,\
                      delete_adc_analysis_results, plot_adc_analysis_results
    from ._sd import sd_analysis, get_sd_analysis_results,\
                     delete_sd_analysis_results, plot_sd_analysis_results,\
                     _calculate_sd_at

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
        for i in range(1, self._t.size):
            segs.append([(self._x[i-1], self._y[i-1]),
                         (self._x[i], self._y[i])])
            colors.append(
                cmap(self._t[i] / (self._t.max() - self._t.min())))
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

    def normalized(self, normalize_t = True, normalize_xy = True):
        """Normalize the track.
        Returns
        -------
        Instance of NormalizedTrack containing the normalized track data.

        Parameters
        ----------
        normalize_t: bool
            Normalize the time component of the track by setting the first time in the
            track to zero.
        normalize_xy: bool
            Normalize the x and y coordinates of the track by setting the initial
            position in the track to zero.
        """
        if self.__class__ == NormalizedTrack:
            warnings.warn(
                "Track is already an instance of NormalizedTrack. This will do nothing.")

        x = self._x
        y = self._y
        t = self._t

        # Getting the span of the diffusion.
        xmin = x.min()
        ymin = y.min()
        tmin = t.min()

        # Normalizing the coordinates
        if normalize_xy:
            x = x - xmin
            y = y - ymin
        if normalize_t:
            t = t - tmin

        # Create normalized Track object
        return NormalizedTrack(x, y, t, xmin, ymin, tmin, id=self._id)

    def calculate_msd(self, N: int = None, num_workers: int = None, chunksize: int = 100):
        """Calculates the mean squared displacement of the track.

        Parameters
        ----------
        N: int
            Maximum MSD length to consider (if none, will be computed from the track length)
        num_workers: int
            Number or processes used for calculation. Defaults to number of system cores.
        chunksize: int
            Chunksize for process pool mapping. Small numbers might have negative performance impacts.
        """

        if N is None:
            N = self._x.size

        MSD = np.zeros((N-3,))
        MSD_error = np.zeros((N-3,))
        pos_x = self._x
        pos_y = self._y

        if num_workers == None:
            workers = os.cpu_count()
        else:
            workers = num_workers

        if N < 100 * chunksize:
            warnings.warn("Track is not very long, switching to single worker as "
                "this will probably be faster than setting up the process pool. Use "
                "num_workers = 1 to suppress this warning.")
            workers = 1

        def MSD_loop(i, pos_x, pos_y, N):
            idx_0 = np.arange(1, N-i-1, 1)
            idx_t = idx_0 + i
            this_msd = (pos_x[idx_t] - pos_x[idx_0])**2 + \
                (pos_y[idx_t] - pos_y[idx_0])**2

            MSD = np.mean(this_msd)
            MSD_error = np.std(this_msd) / np.sqrt(len(this_msd))

            return MSD, MSD_error

        verbose = False
        if verbose:
            tqdm_wrapper = tqdm.tqdm
        else:
            tqdm_wrapper = lambda x, **kwargs: x

        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                i = range(1, N-2)
                results = list(tqdm_wrapper(executor.map(MSD_loop, i,
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
            for i in tqdm_wrapper(range(1, N-2), desc="MSD calculation"):
                MSD[i-1], MSD_error[i-1] = MSD_loop(i, pos_x, pos_y, N)

        self._msd = MSD
        self._msd_error = MSD_error

    def _categorize(self, Dapp, J, Dapp_err = None, R: float = 1/6, fraction_fit_points: float = 0.25, fit_max_time: float=None, initial_guesses = {}, maxfev=1000, enable_log_sampling = False, log_sampling_dist = 0.2, weighting = 'error'):
        if fraction_fit_points > 0.25:
            warnings.warn(
                "Using too many points for the fit means including points which have higher measurment errors.")

        dt = self._t[1] - self._t[0]
        T = J * dt

        print(fit_max_time)
        if fit_max_time is not None:
            n_points = int(np.argwhere(T < fit_max_time)[-1])
        else:
            n_points = np.argmax(J > fraction_fit_points * J[-1])

        cur_dist = 0
        idxs = []
        if enable_log_sampling:
            idxs.append(0)
            for i in range(1, n_points):
                cur_dist += np.log10(T[i]/T[i-1])
                if cur_dist >= log_sampling_dist:
                    idxs.append(i)
                    cur_dist = 0
        else:
            idxs = np.arange(0, n_points, dtype=int)
        print(T[idxs[-1]])
        error = None
        if not Dapp_err is None:
            error = Dapp_err[idxs]

        # Perform fits for all included models
        results = {"models": {}, "indexes": {}}
        bic_min = 999.9
        category = "unknown"
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
            bic = BIC(pred[idxs], Dapp[idxs], len(r[0]), 1)
            if bic < bic_min:
                bic_min = bic
                category = model_name

            results["models"][model_name] = {"params": r[0], "errors": perr, "bic" : bic}

        # Calculate the relative likelihood for each model
        for model in ModelDB().models:
            model_name = model.__class__.__name__
            rel_likelihood = np.exp((-results["models"][model_name]["bic"] + bic_min) * 0.5)
            results["models"][model_name]["rel_likelihood"] = rel_likelihood

        results["indexes"] = idxs
        return category, results


class NormalizedTrack(Track):
    """A track with normalized coordinates and additional information about the normalization."""

    def __init__(self, x=None, y=None, t=None, xmin=None, ymin=None, tmin=None, id=None):
        Track.__init__(self, x, y, t, id=id)
        self._xmin = xmin
        self._ymin = ymin
        self._tmin = tmin

def MSD_loop(i, pos_x, pos_y, N):
    idx_0 = np.arange(1, N-i-1, 1)
    idx_t = idx_0 + i
    this_msd = (pos_x[idx_t] - pos_x[idx_0])**2 + \
        (pos_y[idx_t] - pos_y[idx_0])**2

    MSD = np.mean(this_msd)
    MSD_error = np.std(this_msd) / np.sqrt(len(this_msd))

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