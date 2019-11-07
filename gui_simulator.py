#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI iSCAT simulator

@author: mariaa
"""

import matplotlib
matplotlib.use('TkAgg') # This is a bug fix in order to use the GUI on Mac

import sys
from iscat_lib.detectors import Detectors
from iscat_lib.movie_processor import background_substraction
from iscat_lib.tracker import Tracker

import skimage
from skimage import io
import matplotlib.pyplot as plt

import json
import numpy as np
import cv2
import random
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import csv
# for plotting
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
# default Matplotlib key bindings
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import imageio



class MainVisual(tk.Frame):
    '''
    class of GUI for simulator
    '''

    def __init__(self, master):

        #define a window
        tk.Frame.__init__(self, master)
        self.master = master
        master.title("iSCAT simulator 0.0 ")
        master.configure(background='gray')
        master.protocol('WM_DELETE_WINDOW', self.close_app)

        # # # parameters # # #
        #trajectory generation
        self.Tmax = 100 # Maximal total length [s]
        self.dt = 50e-6 #Time step [s]
        self.L = 10e-6 #Sandbox length [m]
        self.dL = 20e-9 #Compartment map pixel size [m]
        self.Df = 8e-13 # Free diffusion coefficient [m^2/s]
        self.HL = 40e-9 # Average compartment diameter/length [m]
        self.HP = 0.01 #Hopping probability [0-1]
        self.seed = 23 #Random generator seed (nonnegative integer)

        self.dynamics_switch=0 # dynamics type: 0 - diffusion 1 - hopping diffusion
        
        self.trajectory={} # data of the tracks prepared for the csv file
        
     # # # # # # menu to choose files and set tracker parameters # # # # # #

        # name the section

        lbl1 = tk.Label(master=root, text=" TRAJECTORY ", bg='gray', compound=tk.LEFT)
        lbl1.grid(row=0, column=1, columnspan=1, pady=5)

        # separator - line 
        self._separator = ttk.Separator(master=root, orient=tk.VERTICAL)
        self._separator.grid(row=1, column=1, columnspan=4, pady=5, sticky="ew")

        # type of dynamics
        var = tk.IntVar() # the switch variable

        # variable update
        def update_switch():
            self.dynamics_switch=var.get()

        # dynamics type switch: # 0 - diffusion 1 - hopping diffusion
#        self.R1 = tk.Radiobutton(root, text=" diffusion ", variable=var, value=0, bg='gray', command =update_switch )
#        self.R1.grid(row=2, column=1, pady=5)

        self.R2 = tk.Radiobutton(root, text=" hopping diffusion ", variable=var, value=1, bg='gray',command = update_switch ) #  command=sel)
        self.R2.grid(row=2, column=3, pady=5)        
        
        # setting trajectory parameters
        lbl1 = tk.Label(master=root, text="parameters: ", width=30, bg='gray')
        lbl1.grid(row=3, column=1, columnspan=3, pady=5)


        lbl3 = tk.Label(master=root, text=" Maximal total length, s", width=35, bg='gray', compound=tk.LEFT)
        lbl3.grid(row=4, column=1, columnspan=2)
        v = tk.StringVar(root, value=str(self.Tmax))
        self.param_Tmax = tk.Entry(root, width=10, text=v)
        self.param_Tmax.grid(row=4, column=3, columnspan=2)

        lbl4 = tk.Label(master=root, text=" Time step, s", width=35, bg='gray', compound=tk.LEFT)
        lbl4.grid(row=5, column=1, columnspan=2)
        v = tk.StringVar(root, value=str(self.dt))
        self.param_dt = tk.Entry(root, width=10, text=v)
        self.param_dt.grid(row=5, column=3, columnspan=2)

        lbl5 = tk.Label(master=root, text=" Sandbox length, m", width=30, bg='gray', compound=tk.LEFT)
        lbl5.grid(row=6, column=1, columnspan=2)
        v = tk.StringVar(root, value=str(self.L))
        self.param_L = tk.Entry(root, width=10, text=v)
        self.param_L.grid(row=6, column=3, columnspan=2)

        lbl2 = tk.Label(master=root, text=" Compartment map pixel size, m", width=35, bg='gray', compound=tk.LEFT)
        lbl2.grid(row=7, column=1, columnspan=2)
        v = tk.StringVar(root, value=str(self.dL))
        self.param_dL = tk.Entry(root, width=10, text=v)
        self.param_dL.grid(row=7, column=3, columnspan=2)


        lbl6 = tk.Label(master=root, text=" Free diffusion coefficient, $m^2/s$", width=35, bg='gray', compound=tk.LEFT)
        lbl6.grid(row=8, column=1, columnspan=2)
        v = tk.StringVar(root, value=str(self.Df))
        self.param_Df = tk.Entry(root, width=10, text=v)
        self.param_Df.grid(row=8, column=3, columnspan=2)

        lbl6 = tk.Label(master=root, text=" Average compartment diameter/length, m", width=35, bg='gray', compound=tk.LEFT)
        lbl6.grid(row=9, column=1, columnspan=2)
        v = tk.StringVar(root, value=str(self.HL))
        self.param_HL = tk.Entry(root, width=10, text=v)
        self.param_HL.grid(row=9, column=3, columnspan=2)
        
        lbl6 = tk.Label(master=root, text=" Hopping probability [0:1]", width=35, bg='gray', compound=tk.LEFT)
        lbl6.grid(row=10, column=1, columnspan=2)
        v = tk.StringVar(root, value=str(self.HP))
        self.param_HP = tk.Entry(root, width=10, text=v)
        self.param_HP.grid(row=10, column=3, columnspan=2)

        lbl6 = tk.Label(master=root, text=" Random generator seed (int)", width=35, bg='gray', compound=tk.LEFT)
        lbl6.grid(row=11, column=1, columnspan=2)
        v = tk.StringVar(root, value=str(self.seed))
        self.param_seed = tk.Entry(root, width=10, text=v)
        self.param_seed.grid(row=11, column=3, columnspan=2)



        #preview button
        self.button2 = tk.Button(text="    GENERATE    ", command=self.generate_trajectory, width=20, bg='gray') #, height=30)
        self.button2.grid(row=12, column=1, columnspan=2,pady=5)

        # button to run the tracker and save results
        self.button2 = tk.Button(text="    SAVE   ", command=self.save_trajectory, width=20, bg='gray')
        self.button2.grid(row=12, column=3, columnspan=2, pady=5)

##################################################

        # separator - line 
        self._separator = ttk.Separator(master=root, orient=tk.VERTICAL)
        self._separator.grid(row=20, column=1, columnspan=4, pady=5, sticky="ew")        
                # name the section

        lbl1 = tk.Label(master=root, text=" IMAGE SEQUENCE ", bg='gray', compound=tk.LEFT)
        lbl1.grid(row=21, column=1, columnspan=1, pady=5)

        # separator - line 
        self._separator = ttk.Separator(master=root, orient=tk.VERTICAL)
        self._separator.grid(row=22, column=1, columnspan=4, pady=5, sticky="ew")
        
        # button to load trajectory
        self.button2 = tk.Button(text="    load trajectory    ", command=self.load_trajectory, width=20, bg='gray')
        self.button2.grid(row=23, column=3, columnspan=2, pady=5)
        
        # setting image parameters
        lbl1 = tk.Label(master=root, text="parameters: ", width=30, bg='gray')
        lbl1.grid(row=24, column=1, columnspan=3, pady=5)


        lbl3 = tk.Label(master=root, text=" Parameter #1", width=35, bg='gray', compound=tk.LEFT)
        lbl3.grid(row=25, column=1, columnspan=2)
        v = tk.StringVar(root, value=str(self.HP))
        self.param2_sigma = tk.Entry(root, width=10, text=v)
        self.param2_sigma.grid(row=25, column=3, columnspan=2)

        lbl4 = tk.Label(master=root, text=" Parameter #2", width=35, bg='gray', compound=tk.LEFT)
        lbl4.grid(row=26, column=1, columnspan=2)
        v = tk.StringVar(root, value=str(self.HP))
        self.param3_threshold = tk.Entry(root, width=10, text=v)
        self.param3_threshold.grid(row=26, column=3, columnspan=2)

        lbl5 = tk.Label(master=root, text=" Parameter #3", width=30, bg='gray', compound=tk.LEFT)
        lbl5.grid(row=27, column=1, columnspan=2)
        v = tk.StringVar(root, value=str(self.HP))
        self.param4_peak = tk.Entry(root, width=10, text=v)
        self.param4_peak.grid(row=27, column=3, columnspan=2)

        lbl2 = tk.Label(master=root, text=" Parameter #4", width=35, bg='gray', compound=tk.LEFT)
        lbl2.grid(row=28, column=1, columnspan=2)
        v = tk.StringVar(root, value=str(self.HP))
        self.param1_diameter = tk.Entry(root, width=10, text=v)
        self.param1_diameter.grid(row=28, column=3, columnspan=2)


        lbl6 = tk.Label(master=root, text=" Parameter #5", width=30, bg='gray', compound=tk.LEFT)
        lbl6.grid(row=29, column=1, columnspan=2)
        v = tk.StringVar(root, value=str(self.HP))
        self.param5_distance = tk.Entry(root, width=10, text=v)
        self.param5_distance.grid(row=29, column=3, columnspan=2)

        #preview button
        self.button2 = tk.Button(text="    GENERATE    ", command=self.generate_images, width=20, bg='gray') #, height=30)
        self.button2.grid(row=30, column=1, columnspan=2,pady=5)

        # button to run the tracker and save results
        self.button2 = tk.Button(text="    SAVE   ", command=self.save_images, width=20, bg='gray')
        self.button2.grid(row=30, column=3, columnspan=2, pady=10, padx=20)
    


    def generate_trajectory(self):
        '''
        function to generate trajectory
        '''
        
        # update the parameters        
        self.read_parameters()
        
        print("generate_trajectory(self)")
        
        self.trajectory={}
        
        
        
    def save_trajectory(self):
        '''
        save trajectory
        '''
        filename = tk.filedialog.askopenfilename()
        root.update()

         
        print("save trajectory into a file: ", filename)

    def read_parameters(self):
        '''
        read parameters from the GUI
        '''
        if self.param_Tmax.get()!='':
            self.Tmax=int(self.param_Tmax.get())

#        if self.param2_sigma.get()!='':
#            self.sigma=float(self.param2_sigma.get())
#
#        if self.param3_threshold.get()!='':
#            self.threshold=float(self.param3_threshold.get())
#
#        if self.param4_peak.get()!='':
#            self.min_peak=float(self.param4_peak.get())
#
#        if self.param5_distance.get()!='':
#            self.max_dist=float(self.param5_distance.get())
#
#        if self.param6_framegap.get()!='':
#            self.frame_gap=float(self.param6_framegap.get())
#            
#        if self.param6_framegap.get()!='':
#            self.min_track_length=float(self.param7_framegap.get())

    def load_trajectory(self):
        '''
        function to load trajectory
        '''
        
        print("load_trajectory(self)")
        
        self.trajectory={}
        

    def generate_images(self):
        '''
        function to generate image sequence
        '''
        # update the parameters        
        self.read_parameters()
        
        print("generate images")

    def save_images(self):
        '''
        function to save image sequence
        '''
        
        print("save images")


    def close_app(self):
        '''
        quit all the proccesses while closing the GUI
        '''
        self.quit()


if __name__ == "__main__":
    root = tk.Tk()
    MainVisual(root)
    root.mainloop()
