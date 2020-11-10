#!/usr/bin/env python3
# coding: utf-8

"""Simulate iScat movie using existing tracks and point-spread function"""

import argparse
import csv
import logging
import imageio as io
from skimage import util as sk_util
from scipy.signal import fftconvolve
import numpy as np
import tqdm
from trait2d import simulators

def parse_arguments():
    """Parse the input arguments for the simulator"""

    # Define the arguments
    parser = argparse.ArgumentParser("iScat Movie Simulator")
    parser.add_argument("tracks", help="Tracks csv file")
    parser.add_argument("output", help="Output movie file")
    parser.add_argument("-r", "--resolution", default=1, type=float,
                        help="Reconstruction resolution in px (default=%(default)s)")
    parser.add_argument("-tr", "--time_resolution", default=1, type=float,
                        help="Reconstruction resolution in frame/seconds (default=%(default)s)")
    parser.add_argument("--square", action="store_true", help="Make the simulated movie square")
    parser.add_argument("--psf", help="Optional PSF file")
    parser.add_argument("--snr", default=25, type=float, help="Reconstruction SNR (default=%(default)s)")
    parser.add_argument("--background_intensity", default=0.1, type=float, help="Background intensity %(default)s")
    parser.add_argument("--gaussian_noise", action="store_true", help="Add gaussian noise before PSF convolution")
    parser.add_argument("--gaussian_noise_variance", default=1e-3, type=float, help="Gaussian noise variance (default=%(default)s)")
    parser.add_argument("--poisson_noise", action="store_true", help="Add poisson noise after PSF convolution")
    parser.add_argument("-v", "--verbose", action="store_true")

    # Parse the arguments and return
    args = parser.parse_args()

    # Set the logging level
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    # Parse input arguments
    args = parse_arguments()

    # Create the simulator
    s = simulators.iscat_movie(args.tracks,
                               resolution=args.resolution,
                               dt=args.time_resolution,
                               snr=args.snr,
                               background=args.background_intensity,
                               noise_poisson=args.poisson_noise)
    if args.gaussian_noise:
        s.noise_gaussian = args.gaussian_noise_variance
    if args.square:
        s.ratio = "square"
    else:
        s.ratio = None

    # Run simulation
    s.run()

    # Saving the movie
    logging.debug("Saving the movie")
    s.save(args.output)

if __name__ == "__main__":
    main()
