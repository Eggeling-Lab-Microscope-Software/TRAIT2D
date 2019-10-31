#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import warnings
from scipy import spatial
from skimage import draw
import matplotlib.pyplot as plt
import tqdm

DEBUG = True

# Hopping diffusion simulation
def hopping_trajectories(Tmax=100, dt=50e-6, L=10e-6, dL=20e-9, Df=8e-13, HL=40e-9, HP=0.01, seed=None):
    """ Simulates a hopping diffusion trajectory of a single molecule for a
    hopping diffusion model (i.e. free diffusion inside of compartments but
    changing from one compartment to the next only with a certain probability)

    Syntax
    ------
    [trajectory, grid] = hopping_trajectories(Tmax, dt, L, Df, HL, HP, seed)

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

    Returns
    -------
    trajectory : ndarray
        vector Nx4 with order (time [s], x [m], y [m], compartment [id])
    grid:
        characterization of the simulated

    Authors
    -------
    Jan Keller-Findeisen (jkeller1@gwdg.de)
    Department of NanoBiophotonics
    Max Planck Institute of Biophysical Chemistry
    Am Fassberg 11, 3077 Goettingen, Germany

    Converted to Python by Joel Lefebvre
    """

    ## Initializations
    # Set random seed
    np.random.seed(seed)

    # Mean displacement in two directions
    md = np.sqrt(4 * Df * dt)
    if md > HL / 2.0:
        warnings.warn("mean displacement (2D) is more than half of average compartment length, results might be compromised")
    d = md / np.sqrt(2)  # we need the mean displacment in 1D later

    # Number of compartments
    num = int(np.round((L / HL)**2))

    ## Create hopping map
    print(" Create hopping map")
    centers = L * np.random.rand(num, 2)

    # Compute Voronoi diagram
    vor = spatial.Voronoi(centers)

    # Fill map with ids
    xm, ym = np.meshgrid(np.arange(0, L, dL), np.arange(0, L, dL))
    m = np.zeros(xm.shape)
    id = 0
    nc = len(vor.regions)
    for kc in range(nc):
        v = vor.regions[kc]
        v = [x for x in v if x != -1]
        if len(v) > 0:
            v.append(v[0])
            p = np.round(vor.vertices[v] / dL)
            id += 1
            # cv2.polylines(m, v, isClosed=True, color=id)
            rr, cc = draw.polygon(p[:,0], p[:,1], shape=xm.shape)
            m[rr,cc] = id
    if DEBUG:
        plt.imshow(m); plt.title("Voronoi Map"); plt.show()

    ## Create hopping trajectory
    print(" Simulate hopping diffusion")

    # Starting values, in the center of the map
    x = L / 2.0
    y = L / 2.0
    id = m[int(np.floor(x/dL)), int(np.floor(y/dL))]
    diffuse = True
    iteration = 0

    buffer = np.zeros((int(1e5), 3))
    buffer_counter = 0
    store = dict()
    store_counter = 0
    t = 0

    pbar = tqdm.tqdm(desc="Diffuse", total=Tmax)
    while iteration < Tmax and diffuse: # the indefinitely long loop for each time step
        # Update position and compartment
        x0 = x + d * np.random.randn()
        y0 = y + d * np.random.randn()

        # Outside of area, stop it here
        if x0 < 0 or x0 > L or y0 < 0 or y0 > L:
            diffuse = False
            continue

        # Get the new position id
        id0 = m[int(np.floor(x0/dL)), int(np.floor(y0/dL))]

        # Different compartment, and we do not hop
        if id0 != id and np.random.rand() > HP:
            # Retry until compartment is the same again
            while id0 != id:
                x0 = x + d * np.random.randn()
                y0 = y + d * np.random.randn()
                id0 = m[int(np.floor(x0 / dL)), int(np.floor(y0 / dL))]

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

        t += dt
        pbar.update(dt)
        if t > Tmax:
            diffuse = False
    pbar.close()

    # Trim buffer
    buffer = buffer[0:buffer_counter,:]
    store[store_counter] = buffer

    ## Save
    output = dict()
    output["m"] = m
    output["vertices"] = vor.vertices
    output["regions"] = vor.regions
    output["centers"] = centers

    x_list = []
    y_list = []
    id_list = []
    for key in store.keys():
        x_list.extend(list(store[key][:, 0]))
        y_list.extend(list(store[key][:, 1]))
        id_list.extend(list(store[key][:, 2]))

    t_list = list(np.linspace(0, len(x_list) * dt, len(x_list)))
    trajectory = np.stack([t_list, x_list, y_list, id_list])

    return trajectory, output


hopping_trajectories(HL=160e-9)