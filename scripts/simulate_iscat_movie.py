#!/usr/bin/env python3
# coding: utf-8

"""Simulate iScat movie using existing tracks and point-spread function"""

import argparse
import csv
import imageio as io
from skimage import util as sk_util
from scipy.ndimage import convolve
from scipy.signal import fftconvolve, convolve2d
import logging
import numpy as np
import matplotlib.pyplot as plt
import tqdm

# Notes
# TODO: Add PSF & Object appearance inputs
# TODO: Add z-phase jitter for the PSF instead of using a fixed plane
# The PSF can be generated with the FIJI plugin 'DeconvolutionLab2'. Create a python wrapper?
# TODO: Load a simulation parameters file instead of passing everything in the commande line

def parse_arguments():
    """Parse the input arguments for the simulator"""

    # Define the arguments
    parser = argparse.ArgumentParser("iScat Movie Simulator")
    parser.add_argument("tracks", help="Tracks csv file")
    parser.add_argument("output", help="Output movie file")
    parser.add_argument("-r", "--resolution", default=1, type=float,
                        help="Reconstruction resolution in px (default=%(default)s)")
    parser.add_argument("--square", action="store_true", help="Make the simulated movie square")
    parser.add_argument("--psf", help="Optional PSF file")
    parser.add_argument("--snr", default=25, type=float, help="Reconstruction SNR (default=%(default)s)")
    parser.add_argument("--background_intensity", default=0.1, type=float, help="Background intensity %(default)s")
    parser.add_argument("--gaussian_noise", action="store_true", help="Add gaussian noise before PSF convolution")
    parser.add_argument("--gaussian_noise_variance", default=1e-3, type=float, help="Gaussian noise variance (default=%(default)s)")
    parser.add_argument("--poisson_noise", action="store_true", help="Add poisson noise after PSF convolution")
    parser.add_argument("-v", "--verbose", action="store_true") # TODO: Setup the verbose flag with logging

    # Parse the arguments and return
    args = parser.parse_args()

    # Set the logging level
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def load_tracks(filename):
    """Load the tracks from a csv file"""
    # Load the csv file
    with open(filename, "r") as csvfile:
        # Detect the csv format
        dialect = csv.Sniffer().sniff(csvfile.read())

        # Create a reader
        csvfile.seek(0)
        reader = csv.reader(csvfile, dialect)

        tracks = {"x": [], "y": [], "track_id": [], "frame": [], "track_frame": []}
        for i, row in enumerate(reader):
            if i == 0:
                column_names = row
            else:
                tracks["x"].append(float(row[column_names.index("Position X")]))
                tracks["y"].append(float(row[column_names.index("Position Y")]))
                tracks["track_id"].append(int(row[column_names.index("TrackID")]))
                tracks["frame"].append(int(row[column_names.index("ID")]))
                tracks["track_frame"].append(int(row[column_names.index("OriginalID")]))

    return tracks

def main():
    # Parse input arguments
    args = parse_arguments()

    # Load the tracks
    tracks = load_tracks(args.tracks)
    n_spots = len(tracks["x"])
    logging.debug("There are {} spots forming the tracks".format(n_spots))

    # Get the number of frames
    t_min = 0
    t_max = np.max(tracks["frame"])
    n_frames = t_max - t_min + 1
    logging.debug("Movie number of frames: {}".format(n_frames))

    # Get the movie size
    xmin = 0 #np.min(tracks['x'])
    xmax = np.max(tracks['x'])
    ymin = 0 #np.min(tracks['y'])
    ymax = np.max(tracks['y'])
    if args.square:
        xmax = max((xmax,ymax))
        ymax = max((xmax,ymax))
    logging.debug("Tracks X range: {}".format((xmin, xmax)))
    logging.debug("Tracks Y range: {}".format((ymin, ymax)))

    # Create the grid # TODO: Use input size as alternative
    x = np.linspace(xmin, xmax, int((xmax - xmin)/args.resolution))
    y = np.linspace(ymin, ymax, int((ymax - ymin)/args.resolution))
    nx = len(x) + 1
    ny = len(y) + 1
    logging.debug("Movie size at resolution={:f} is {}x{}x{}".format(args.resolution, n_frames, nx, ny))

    # Prepare the movie array
    movie = np.ones((n_frames, nx, ny)) * args.background_intensity

    if args.gaussian_noise:
        logging.debug("Adding Gaussian noise")
        movie = sk_util.random_noise(movie, mode="gaussian", var=args.gaussian_noise_variance)

    # Populate the tracks
    logging.debug("Reconstructing the tracks")
    for this_spot in range(n_spots):
        mx = int(np.floor((tracks['x'][this_spot]-xmin) / args.resolution))
        my = int(np.floor((tracks['y'][this_spot]-ymin) / args.resolution))
        mf = tracks["frame"][this_spot]
        if (xmin <= mx < xmax) and (ymin <= my < ymax):
            movie[mf, mx, my] = (1+args.snr) * args.background_intensity

    # Convolve by PSF if provided
    if args.psf is not None:
        psf = io.volread(args.psf).squeeze()
        psf_2d = psf[int(psf.shape[0]/2), ...].squeeze()
        psf_2d = psf_2d / psf_2d.sum()

        # Pad the movie
        px, py = psf_2d.shape
        movie = np.pad(movie, ((0,0),(px//2, px//2),(py//2, py//2)), mode="reflect")

        # Apply convolution
        for i in tqdm.tqdm(range(n_frames), desc="Convolving with PSF"): # TODO: link tqdm with logging
            movie[i,...] = fftconvolve(movie[i,...], psf_2d, mode='same')

        # Unpad
        movie = movie[:,px//2:px//2+nx, py//2:py//2+ny]

    # Add Poisson noise
    logging.debug("Adding Poisson noise")
    if args.poisson_noise:
        movie = sk_util.random_noise(movie, mode="poisson", clip=False)
        movie[movie<0] = 0

    # Saving the movie
    logging.debug("Saving the movie")
    io.volwrite(args.output, movie.astype(np.float32))

if __name__ == "__main__":
    main()
