#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import warnings
from scipy import spatial
from skimage import draw
import matplotlib.pyplot as plt
import tqdm
import json
import csv
import pprint
from pathlib import Path
import pickle
from skimage import util as sk_util
from scipy.signal import fftconvolve
import imageio as io
from pathlib import Path

DEBUG = True


# Abstract diffusion object
class Diffusion(object):
    """Abstract diffusion simulator

    The methods to modify after inherition are: self.run, and te self.params_list, and assign the input variables
    """
    def __init__(self, Tmax=1.0, dt=1e-3, L=1, dL=1e-3, d=1e-2, seed: int=None, quantize=True):

        ## Initializations
        self.params_list = ["Tmax", "dt", "L", "dL", "d", "seed", "quantize"]
        self.Tmax = Tmax
        self.dt = dt
        self.L = L
        self.dL = dL
        self.d = d
        self.seed = seed
        self.quantize = quantize

        # Set random seed
        np.random.seed(self.seed)

    def run(self):
        # Initialize
        self.trajectory = dict()
        self.trajectory["x"] = []
        self.trajectory["y"] = []
        self.trajectory["t"] = []
        self.trajectory["id"] = []

    def plot_trajectory(self, time_resolution=None, limit_fov=False, alpha=0.8, title="Diffusion"):
        """Display the simulated trajectory.

        Parameters
        ----------

        param time_resolution:
            [s]
        """
        assert hasattr(self, "trajectory"), "You must first run the simulator"
        if time_resolution is not None:
            time_interval = int(np.ceil(time_resolution / self.dt))
        else:
            time_interval = 1
        x = np.array(self.trajectory["x"][0::time_interval])
        y = np.array(self.trajectory["y"][0::time_interval])
        plt.plot(x, y, alpha=alpha)

        if limit_fov:
            plt.xlim((0, self.parameters["L"]))
            plt.ylim((0, self.parameters["L"]))

        plt.title(title)

    def _gather_parameters(self):
        self.parameters = dict()
        for this_param in self.params_list:
            self.parameters[this_param] = getattr(self, this_param)

    def _set_parameters(self, parameters):
        # Parameters must be a dict
        for this_param in self.params_list:
            if this_param in parameters:
                setattr(self, this_param, parameters[this_param])

        # Set the random seed if given
        if "seed" in parameters:
            np.random.seed(parameters["seed"])

    def print_parameters(self):
        """Print the simulation parameters, accessible from self._gather_parameters()
        """
        self._gather_parameters()
        pprint.pprint(self.parameters)

    def save_trajectory(self, filename, format=None):
        """Save the simulated trajectory as either a json, cvs or pcl file with fields t, x, and y
        """
        supported_formats = ["json", "csv", "pcl"]
        if format is None:
            format = Path(filename).suffix.replace(".","")
        assert hasattr(self, "trajectory"), "You must first run the simulator to obtain a trajectory"
        assert format in supported_formats, f"Supported formats are: {supported_formats}"

        if format == "json":
            with open(filename, "w") as f:
                json.dump(self.trajectory, f)
        elif format == "csv":
            with open(filename, "w", newline='') as f:
                fieldnames = ["t", "x", "y", "id"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for i in range(len(self.trajectory["x"])):
                    this_row = {"t": self.trajectory["t"][i],
                                "x": self.trajectory["x"][i],
                                "y": self.trajectory["y"][i],
                               "id": 0}
                    writer.writerow(this_row)

        elif format == "pcl":
            with open(filename, "wb") as f:
                pickle.dump(self.trajectory, f)

    def save_parameters(self, filename, mkdir=True):
        """Save the simulation parameters, accessible through _gather_parameter
        """
        self._gather_parameters()

        # Create the parent directory
        if mkdir:
            Path(filename).parent().mkdir(parents=True, exist_ok=True)

        # Save the parameters as a JSON file
        with open(filename, "w") as f:
            json.dumps(f, self.parameters)

    def load_parameters(self, filename):
        """Load the simulation parameters
        """
        # Load the parameters from a JSON file
        with open(filename, 'r') as f:
            parameters = json.load(f)

        # Set the loaded parameters
        self._set_parameters(parameters)

# Brownian diffusion simulation
class BrownianDiffusion(Diffusion):
    """BrownianDiffusion Initialization

    Parameters
    ----------
    Tmax: float
        Maximum simulation time (s)
    dt: float
        Simulation time resolution (s)
    L: float
        Simulation domain size (m)
    dL: float
        Simulation spatial resolution (m)
    d: float
        Diffusion coefficient (m^2/s)
    seed: int
        Seed to initialize the random generator (for reproducibility)
    quantize: bool
        Quantize the position to the simulation spatial resolution grid.
    """
    def __init__(self, **kwargs):
        super(BrownianDiffusion, self).__init__(**kwargs)

    def run(self):
        """Run a random walk simulation (Brownian diffusion)
        """
        # Starting values is in the center.
        x = self.L / 2.0
        y = self.L / 2.0
        diffuse = True
        iteration = 0

        x_list = [x]
        y_list = [y]
        t_list = [0]

        pbar = tqdm.tqdm(desc="Brownian Diffusion Simulation", total=int(self.Tmax / self.dt))
        while iteration * self.dt < self.Tmax and diffuse:
            # Update position and compartment
            x0 = x + np.sqrt(2 * self.d * self.dt) * np.random.randn()
            y0 = y + np.sqrt(2 * self.d * self.dt) * np.random.randn()

            # Outside of area, stop it here
            if not(0 <= x0 <= self.L) or not(0 <= y0 <= self.L):
                diffuse = False
                continue

            # Update the position and compartment
            x = x0
            y = y0
            x_list.append(x)
            y_list.append(y)
            t_list.append(t_list[-1]+self.dt)
            iteration += 1
            pbar.update()
        pbar.close()
        print(f"Free diffusion simulation was completed in {iteration} iterations.")

        # Make sure the simulated track is expressed as dL spatial resolution
        if self.quantize:
            x_list = (np.round(np.array(x_list) / self.dL) * self.dL).tolist()
            y_list = (np.round(np.array(y_list) / self.dL) * self.dL).tolist()

        self.trajectory = {"x": x_list, "y": y_list, "t": t_list}

# Hopping diffusion simulation
class HoppingDiffusion(Diffusion):
    """Simulates a hopping diffusion trajectory of a single molecule for a
    hopping diffusion model (i.e. free diffusion inside of compartments but
    changing from one compartment to the next only with a certain probability)

    **Syntax**:

    .. code-block:: python

        simulator = hopping_trajectories(Tmax, dt, L, Df, HL, HP, seed)
        simulator.run()

    **Authors**:

    | Jan Keller-Findeisen (jkeller1@gwdg.de)
    | Department of NanoBiophotonics
    | Max Planck Institute of Biophysical Chemistry
    | Am Fassberg 11, 3077 Goettingen, Germany

    Converted to Python by Joel Lefebvre

    Parameters
    ----------
    Tmax : float
        Maximal total length [s]
    dt : float
        Time step [s]
    L : float
        Sandbox length [m]
    dL : float
        Compartment map pixel size [m]
    Df : float
        Free diffusion coefficient [m^2/s]
    HL : float
        Average compartment diameter/length [m]
    HP : float
        Hopping probability [0-1]
    seed : float
        Random generator seed (nonnegative integer)
    quantize: bool
        Quantize the position to the simulation spatial resolution grid.
    """
    def __init__(self, Tmax=100, dt=50e-6, L=10e-6, dL=20e-9, Df=8e-13, HL=40e-9, HP=0.01, seed: int = None, quantize=True):

        # super(HoppingDiffusion, self).__init__(seed=seed)

        # Set parameters
        self.params_list = ["Tmax", "dt", "L", "dL", "Df", "HL", "HP", "seed", "quantize"]
        self.Tmax = Tmax
        self.dt = dt
        self.L = L
        self.dL = dL
        self.Df = Df
        self.HL = HL
        self.HP = HP
        self.seed = seed
        self.quantize = quantize

        # Set random seed
        np.random.seed(self.seed)

        # Mean displacement in two directions
        self.md = np.sqrt(4 * self.Df * self.dt)
        if self.md > self.HL / 2.0:
            warnings.warn("mean displacement (2D) is more than half of average compartment length, results might be compromised")
        self.d = self.md / np.sqrt(2)  # we need the mean displacement in 1D later

        # Number of compartments
        self.num = int(np.round((self.L / self.HL)**2))

    def create_hopping_map(self):
        ## Create hopping map
        centers = self.L * np.random.rand(self.num, 2)

        # Compute Voronoi diagram
        vor = spatial.Voronoi(centers)

        # Fill map with ids
        xm, ym = np.meshgrid(np.arange(0, self.L, self.dL), np.arange(0, self.L, self.dL))
        m = np.zeros(xm.shape)
        id = 0
        nc = len(vor.regions)
        for kc in range(nc):
            v = vor.regions[kc]
            v = [x for x in v if x != -1]
            if len(v) > 0:
                v.append(v[0])
                p = np.round(vor.vertices[v] / self.dL)
                id += 1
                # cv2.polylines(m, v, isClosed=True, color=id)
                rr, cc = draw.polygon(p[:,0], p[:,1], shape=xm.shape)
                m[rr,cc] = id

        # Add the hopping map to the internals
        self.hopping_map = m
        self.centers = centers
        self.voronoi_vertices = vor.vertices
        self.voronoi_regions = vor.regions

    def run(self, update_hoppingMap=False):
        ## Create hopping trajectory

        # Create or update the hopping map
        if not hasattr(self, "hopping_map") or update_hoppingMap:
            self.create_hopping_map()

        # Starting values, in the center of the map
        x = self.L / 2.0
        y = self.L / 2.0
        id = self.hopping_map[int(np.floor(x/self.dL)), int(np.floor(y/self.dL))]
        diffuse = True
        iteration = 0

        buffer = np.zeros((int(1e5), 3))
        buffer_counter = 0
        store = dict()
        store_counter = 0
        t = 0

        pbar = tqdm.tqdm(desc="Simulation", total=int(self.Tmax / self.dt))
        while iteration * self.dt < self.Tmax and diffuse: # the indefinitely long loop for each time step
            # Update position and compartment
            x0 = x + self.d * np.random.randn()
            y0 = y + self.d * np.random.randn()

            # Outside of area, stop it here
            if x0 < 0 or x0 > self.L or y0 < 0 or y0 > self.L:
                diffuse = False
                continue

            # Get the new position id
            id0 = self.hopping_map[int(np.floor(x0/self.dL)), int(np.floor(y0/self.dL))]

            # Different compartment, and we do not hop
            if id0 != id and np.random.rand() > self.HP:
                # Retry until compartment is the same again
                while id0 != id:
                    x0 = x + self.d * np.random.randn()
                    y0 = y + self.d * np.random.randn()
                    ix = int(np.floor(x0 / self.dL))
                    iy = int(np.floor(y0 / self.dL))
                    if 0 <= ix < self.hopping_map.shape[0] and 0 <= iy < self.hopping_map.shape[1]:
                        id0 = self.hopping_map[ix,iy]

            # Update the position and compartment
            x = x0
            y = y0
            id = id0
            iteration += 1

            # Store position and compartment
            buffer[buffer_counter,:] = [x, y, id]
            buffer_counter += 1

            if buffer_counter >= buffer.shape[0]:
                store[store_counter] = buffer
                store_counter += 1
                buffer = np.zeros((int(1e5), 3))
                buffer_counter = 0

            t += self.dt
            pbar.update()
            if t > self.Tmax:
                diffuse = False
        pbar.close()
        print(f"Hopping diffusion simulation was completed in {iteration} iterations.")

        # Trim buffer
        buffer = buffer[0:buffer_counter,:]
        store[store_counter] = buffer

        # Gather the simulation results
        x_list = []
        y_list = []
        id_list = []
        for key in store.keys():
            x_list.extend(list(store[key][:, 0]))
            y_list.extend(list(store[key][:, 1]))
            id_list.extend(list(store[key][:, 2]))

        # Quantize the position
        if self.quantize:
            x_list = (np.round(np.array(x_list) / self.dL) * self.dL).tolist()
            y_list = (np.round(np.array(y_list) / self.dL) * self.dL).tolist()

        t_list = list(np.linspace(0, len(x_list) * self.dt, len(x_list)))
        self.trajectory = dict()
        self.trajectory["x"] = x_list
        self.trajectory["y"] = y_list
        self.trajectory["t"] = t_list
        self.trajectory["id"] = id_list

    def display_hopping_map(self):
        assert hasattr(self, "hopping_map"), "No hopping map was set or created."
        plt.imshow(self.hopping_map)
        plt.title("Hopping Map")

# Moving Acquisition simulation
class movie_simulator(object):
    """Generate a syntetic iscat movie from a set of tracks.

    **Syntax**:

    .. code-block:: python

        movie_simulator = iscat_movie(tracks)
        movie_simulator.run()

    **Authors**:

    | Joël Lefebvre (lefebvre.joel@uqam.ca)
    | Department of Computer Sciences
    | Université du Québec a Montréal (UQAM)
    | 201, Av. du Président-Kennedy, Montréal (Qc), Canada (H3C 3P8)

    Parameters
    ----------
    tracks : dict or CSV filename
        A dictionary or a CSV filename containing the set of tracks to simulate. The dictionary must include the keys
        `x`, `y`, `t` and `id`
    resolution : float
        Spatial resolution [m/px]
    dt : float
        Temporal resolution  [frame/sec]
    contrast : float
        Contrast between the simulated particle and the background (Contrast = particle intensity - background intensity)
    background : float
        Background intensity between 0 and 1
    noise_gaussian : float
        Gaussian noise variance
    noise_poisson : bool
        If True, Poisson noise will be added.
    ratio : str
        Aspect ratio of the simulated movie. Available ("square"). If none is given,
        the aspect ratio will be inferred from the tracks position.
    """

    # Notes
    # TODO: Add PSF & Object shape inputs (instead of only psf)
    # TODO: Add z-phase jitter for the PSF instead of using a fixed plane
    # TODO: Load a simulation parameters file instead of passing everything in the command line
    # TODO: Use input size as alternative
    # TODO: link tqdm with logging
    # TODO: Create a python wrapper for the ImageJ plugin 'DeconvolutionLab2' to generate PSF in the script?
    # TODO: Background noise with different statistics (similar to transcient particles)
    def __init__(self, tracks=None, resolution=1.0, dt=1, contrast=5, background=0.3,
                 noise_gaussian=0.15, noise_poisson=True, ratio="square"):
        # Prepare the simulator
        self.resolution = resolution
        self.contrast = contrast  # Contrast between the simulated particle and the background
        self.background = background  # Background intensity
        self.noise_gaussian = noise_gaussian  # Gaussian noise variance
        self.noise_poisson = noise_poisson  # Poisson noise variance
        self.dt = dt  # Temporal resolution
        self.ratio = ratio
        self.initialized = False

        if isinstance(tracks, dict):
            self.tracks = tracks
        elif isinstance(tracks, str) or isinstance(tracks, Path):
            self.load_tracks(tracks)

    def initialize(self):
        """Initialize the simulator"""
        assert hasattr(self, 'tracks'), "You must load a tracks file or set a tracks dict first"
        self.n_spots = len(self.tracks["x"])

        # Get the number of frames
        self.tmin = 0
        self.tmax = np.max(self.tracks["t"])
        self.n_frames = int((self.tmax - self.tmin)/self.dt) + 1


        # Get the movie shape
        self.xmin = np.min(self.tracks["x"])
        self.ymin = np.min(self.tracks["y"])
        self.xmax = np.max(self.tracks["x"])
        self.ymax = np.max(self.tracks["y"])
        if self.ratio == "square":
            self.xmin = min(self.xmin, self.ymin)
            self.ymin = min(self.xmin, self.ymin)
            self.xmax = max(self.xmax, self.ymax)
            self.ymax = max(self.xmax, self.ymax)
        else:
            print(f"Unknown ratio: {self.ratio}")

        # Initialize the simulation grid
        x = np.linspace(self.xmin, self.xmax, int((self.xmax - self.xmin) / self.resolution))
        y = np.linspace(self.ymin, self.ymax, int((self.ymax - self.ymin) / self.resolution))
        self.nx = len(x) + 1
        self.ny = len(y) + 1

        self.initialized = True
        
        print(" - - - - - - - - ")
        print(self.tmax)
        print(len(x))
        print("self.xmin", self.xmin)
        print("self.ymin", self.ymin)
        print("self.xmax", self.xmax)

        print(f"Movie shape will be: ({self.nx}, {self.ny}) with ({self.n_frames}) frames")

    def get_estimated_size(self):
        """Return the estimated movie size in MB"""
        return self.nx * self.ny * self.n_frames * 8 / 1000**2 # Using double precision float (float64)

    def run(self, reinitialize=False):
        """Run the movie simulation

        Parameters
        ----------
        reinitialize: bool
            If `True`, the simulator will be reinitialized.
        """
        if reinitialize or not(self.initialized):
            self.initialize()

        # Create the movie array
        print("Creating an empty movie")
        movie = np.ones((self.n_frames, self.nx, self.ny), dtype=np.float64) * self.background

        # Add Gaussian noise to the background
        if self.noise_gaussian is not None:
            print("Adding gaussian noise to the background")
            movie = sk_util.random_noise(movie, mode="gaussian", var=self.noise_gaussian)

        # Populate the tracks
        for this_spot in tqdm.tqdm(range(self.n_spots), "Adding tracks"):
            mx = int(np.round((self.tracks['x'][this_spot] - self.xmin) / self.resolution))
            my = int(np.round((self.tracks['y'][this_spot] - self.ymin) / self.resolution))
            mt = int(self.tracks["t"][this_spot] / self.dt)
            if isinstance(mx, list):
                for x, y, t in zip(mx, my, mt):
                    if (0 <= mx < self.nx) and (0 <= my < self.ny):
                        movie[t, x, y] += self.contrast
            else:
                if (0 <= mx < self.nx) and (0 <= my < self.ny):
                    movie[mt, mx, my] += self.contrast

        # Convolve by PSF if provided
        if hasattr(self, "psf_2d"):
            px, py = self.psf_2d.shape
            movie = np.pad(movie, ((0, 0), (px // 2, px // 2), (py // 2, py // 2)), mode="reflect")

            # Apply convolution
            for i in tqdm.tqdm(range(self.n_frames), desc="Convolving with PSF"):
                movie[i, ...] = fftconvolve(movie[i, ...], self.psf_2d, mode='same')

            # Unpad
            movie = movie[:, px // 2:px // 2 + self.nx, py // 2:py // 2 + self.ny]

        # Add Poisson noise
        if self.noise_poisson:
            print("Adding Poisson noise")
            movie = sk_util.random_noise(movie, mode="poisson", clip=False)
            movie[movie < 0] = 0

        self.movie = movie
        print("Movie generation is done.")

    def save(self, filename):
        """Save the simulated movie.

        Parameters
        ----------
        filename : str
            Output volume filename. Must be a volume format supported by `imageio.volwrite`

        Note
        ----
        The volume will be converted to single precision float (`numpy.float32`)
        """
        assert hasattr(self, "movie"), "You must first run the simulation"
        io.volwrite(filename, self.movie.astype(np.float32))

    def load_tracks(self, filename, field_x="x", field_y="y", field_t="t", field_id="id", file_format=None): # TODO: Load other tracks format
        """Load the tracks from a csv file.
        
        Parameters
        ----------
        filename : str
            Path to a csv filename
        field_x : str
            Column name in the CSV corresponding to the tracks X positions.
        field_y : str
            Column name in the CSV corresponding to the tracks Y positions.
        field_t : str
            Column name in the CSV corresponding to the tracks time.
        field_id : str
            Column name in the CSV corresponding to the tracks ID.
        file_format : str
            Specify the file format (available are cvs, json, pcl). If none is given, it will be inferred from the filename
        """
        tracks = {"x": [], "y": [], "t": [], "id": []}
        if Path(filename).suffix == ".csv" or file_format == "csv":
            # Load the csv file
            with open(filename, "r") as csvfile:
                #  Detect the csv format
                dialect = csv.Sniffer().sniff(csvfile.read())

                #  Create a reader
                csvfile.seek(0)
                reader = csv.reader(csvfile, dialect)

                for i, row in enumerate(reader):
                    if i == 0:
                        column_names = row
                    else:
                        tracks["x"].append(float(row[column_names.index(field_x)]))
                        tracks["y"].append(float(row[column_names.index(field_y)]))
                        tracks["t"].append(float(row[column_names.index(field_t)]))
                        tracks["id"].append(int(row[column_names.index(field_id)]))
        elif Path(filename).suffix == ".json" or file_format == "json":
            with open(filename, "r") as f:
                content = json.load(f)
            tracks["x"] = content[field_x]
            tracks["y"] = content[field_y]
            tracks["t"] = content[field_t]
            tracks["id"] = content[field_id]

        elif Path(filename).suffix == ".pcl" or file_format == "pcl":
            with open(filename, "rb") as f:
                content = pickle.load(f)
            tracks["x"] = content[field_x]
            tracks["y"] = content[field_y]
            tracks["t"] = content[field_t]
            tracks["id"] = content[field_id]

        self.tracks = tracks

    def load_psf(self, filename):
        """Load a Point-Spread Function (PSF) from a file

        Parameters
        ----------
        filename: str
            Input volume filename. Must be a volume format supported by `imageio.volwrite`

        Note
        ----
        Only the middle slice along the first dimension will be used

        .. code-block:: python
            psf = psf[int(psf.shape[0]/2), ...]
        """
        psf = io.volread(filename).squeeze()
        psf_2d = psf[int(psf.shape[0] / 2), ...].squeeze()
        psf_2d = psf_2d / psf_2d.sum()
        self.psf_2d = psf_2d
