#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import warnings
from scipy import spatial
from skimage import draw
import matplotlib.pyplot as plt
import tqdm
import json
import pprint
from pathlib import Path

DEBUG = True

# Hopping diffusion simulation
class hopping_diffusion(object):
    def __init__(self, Tmax=100, dt=50e-6, L=10e-6, dL=20e-9, Df=8e-13, HL=40e-9, HP=0.01, seed: int = None):
        """ Simulates a hopping diffusion trajectory of a single molecule for a
        hopping diffusion model (i.e. free diffusion inside of compartments but
        changing from one compartment to the next only with a certain probability)

        Syntax
        ------
        simulator = hopping_trajectories(Tmax, dt, L, Df, HL, HP, seed)
        simulator.run()

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

        Authors
        -------
        Jan Keller-Findeisen (jkeller1@gwdg.de)
        Department of NanoBiophotonics
        Max Planck Institute of Biophysical Chemistry
        Am Fassberg 11, 3077 Goettingen, Germany

        Converted to Python by Joel Lefebvre
        """

        # Set parameters
        self.Tmax = Tmax
        self.dt = dt
        self.L = L
        self.dL = dL
        self.Df = Df
        self.HL = HL
        self.HP = HP
        self.seed = seed

        ## Initializations
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
        print("Create hopping map")
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
        print("Simulate hopping diffusion")

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

        pbar = tqdm.tqdm(desc="Simulation", total=self.Tmax)
        while iteration < self.Tmax and diffuse: # the indefinitely long loop for each time step
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
                    id0 = self.hopping_map[int(np.floor(x0 / self.dL)), int(np.floor(y0 / self.dL))]

            # Update the position and compartment
            x = x0
            y = y0
            id = id0

            # Store position and compartment
            buffer[buffer_counter,:] = [x, y, id]
            buffer_counter += 1

            if buffer_counter >= buffer.shape[0]:
                store[store_counter] = buffer
                store_counter += 1
                buffer = np.zeros((int(1e5), 3))
                buffer_counter = 0

            t += self.dt
            pbar.update(self.dt)
            if t > self.Tmax:
                diffuse = False
        pbar.close()

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

    def display_trajectory(self, time_resolution=0.5e-3, limit_fov=False, alpha=0.8):
        """ Display the simulated trajectory.
        :param time_resolution: [s]
        :param limit_fov
        :return:
        """
        assert hasattr(self, "trajectory"), "You must first run the simulator"
        time_interval = int(np.ceil(time_resolution / self.parameters["dt"]))
        x = self.trajectory["x"][0::time_interval]
        y = self.trajectory["y"][0::time_interval]
        plt.plot(x, y, alpha=alpha)

        if limit_fov:
            plt.xlim((0, self.parameters["L"]))
            plt.ylim((0, self.parameters["L"]))

    def _gather_parameters(self):
        self.parameters = dict()
        self.parameters["Tmax"] = self.Tmax
        self.parameters["dt"] = self.dt
        self.parameters["L"] = self.L
        self.parameters["dL"] = self.dL
        self.parameters["Df"] = self.Df
        self.parameters["HL"] = self.HL
        self.parameters["HP"] = self.HP
        self.parameters["seed"] = self.seed
        self.parameters["md"] = self.md
        self.parameters["d"] = self.d
        self.parameters["num"] = self.num  # Number of compartments

    def _set_parameters(self, parameters):
        # Parameters must be a dict
        params_list = ["Tmax", "dt", "L", "dL", "Df", "HL", "HP", "seed", "md", "d", "num"]
        for this_param in params_list:
            if this_param in parameters:
                setattr(self, this_param, parameters[this_param])

        # Set the random seed if given
        if "seed" in parameters:
            np.random.seed(parameters["seed"])

    def print_parameters(self):
        self._gather_parameters()
        pprint.pprint(self.parameters)

    def save_trajectory(self, filename): # TODO: Check which tracking file format to use.
        assert hasattr(self, "trajectory"), "You must first run the simulator to obtain a trajectory"
        pass

    def save_parameters(self, filename, mkdir=True):
        self._gather_parameters()

        # Create the parent directory
        if mkdir:
            Path(filename).parent().mkdir(parents=True, exist_ok=True)

        # Save the parameters as a JSON file
        with open(filename, "w") as f:
            json.dumps(f, self.parameters)

    def load_parameters(self, filename):
        # Load the parameters from a JSON file
        with open(filename, 'r') as f:
            parameters = json.load(f)

        # Set the loaded parameters
        self._set_parameters(parameters)

    # TODO: save the options (voronoi diagram, vertices, centers, regions, ...)