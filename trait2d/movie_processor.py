#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def background_substraction(movie, saturation = 0.3, nFrames = 1000, invert=True):
    """iScat Background substraction based on the  original thesis processing
    Parameters
    ----------
    movie : ndarray
        The movie to process, an ndarray of shape (time, row, col)
    saturation : float
        The pixel value saturation used for brightness adjustment (between 0 and 100)
    nFrames : int
        Number of frames to use at the beginning of the movie for flat-field computation
    invert : bool
        Bool to invert the intensity.

    Returns
    -------
    movie_p : ndarray
        Processed movie

    """
    dtype = movie.dtype
    movie_p = np.copy(movie).astype(np.float32)

    # Compute flat field from first 1000 frames (lateral shift)
    bg_median = np.median(movie_p[0:nFrames,...], axis=0)
    movie_p = movie_p / bg_median

    # Temporal average
    bg_avg = np.mean(movie_p, axis=0)

    # Background removal
    movie_p = movie_p - bg_avg

    # Normalization
    imin = movie_p.min()
    imax = movie_p.max()
    movie_p = (movie_p - imin) / (imax - imin)

    # Brightness adjustment
    isat = np.percentile(movie_p, 100 - saturation)
    movie_p = movie_p / isat
    movie_p[movie_p>1] = 1

    # Inverting the contrast
    if invert:
        movie_p = 1 - movie_p

    # Convert back to the original pixel format
    movie_p = (np.iinfo(dtype).max * movie_p).astype(dtype)

    return movie_p
