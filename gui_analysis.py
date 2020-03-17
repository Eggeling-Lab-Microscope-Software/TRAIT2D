#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI iSCAT tracking

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
import csv
# for plotting
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
# default Matplotlib key bindings
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import imageio



class MainVisual(tk.Frame):
    '''
    class of GUI for iSCAT trajectory analysis
    '''

    def __init__(self, master):

        #define a window
        tk.Frame.__init__(self, master)
        self.master = master
        master.title("iSCAT trajectory analysis 1.0 ")
        master.configure(background='white')
        
        # get the monitor size

        self.monitor_width=master.winfo_screenwidth()
        self.monitor_height = master.winfo_screenheight()

        master.geometry(str(int(self.monitor_width/3))+"x"+str(int(self.monitor_height/3))) #"1200x1000")

        self.button_size=int(self.monitor_width/25)
        master.protocol('WM_DELETE_WINDOW', self.close_app)

        # # # parameters # # #
        #general
        self.trajectory_file=" " # path 
       
 
        
     # # # # # # menu to choose files and set parameters # # # # # #

        # button to select trajectory
        self.button1 = tk.Button(text="       Select file with trajectorie       ", command=self.load_trajectory, width=int(self.button_size/3), bg='gray')
        self.button1.grid(row=0, column=1, columnspan=3, pady=5)

        # show selected trajectory path
        lbl1 = tk.Label(master=root, text="selected file: "+self.trajectory_file, bg='white')
        lbl1.grid(row=1, column=1, columnspan=3, pady=5)

        # setting parameters

        lbl1 = tk.Label(master=root, text="PARAMETERS: ", width=int(self.button_size/2), bg='white')
        lbl1.grid(row=3, column=1, columnspan=3, pady=5)


        lbl3 = tk.Label(master=root, text=" Parameter 1", width=int(self.button_size/2), bg='white')
        lbl3.grid(row=4, column=1)
        v = tk.StringVar(root, value=str(0))
        self.param_1 = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.param_1.grid(row=4, column=2)

        lbl4 = tk.Label(master=root, text=" Parameter 2", width=int(self.button_size/2), bg='white')
        lbl4.grid(row=5, column=1)
        v = tk.StringVar(root, value=str(100))
        self.param_2 = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.param_2.grid(row=5, column=2)



        lbl7 = tk.Label(master=root, text="Parameter 3", width=int(self.button_size/2), bg='white')
        lbl7.grid(row=10, column=1)
        v = tk.StringVar(root, value=str(0.001))
        self.param3 = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.param3.grid(row=10, column=2)

	
        var = tk.IntVar() # the switch variable

        # variable update
        def update_switch():
            self.spot_switch=var.get()

        # spot type switch: # 0 - option 0, 1 - option 1
        self.R1 = tk.Radiobutton(root, text=" option 0 ", variable=var, value=0, bg='white', command =update_switch)
        self.R1.grid(row=11, column=1, pady=5)

        self.R2 = tk.Radiobutton(root, text=" option 1 ", variable=var, value=1, bg='white',command = update_switch ) #  command=sel)
        self.R2.grid(row=11, column=2, pady=5)


        # process button
        self.button2 = tk.Button(text="    process   ", command=self.process_trajectory, width=int(self.button_size/3), bg='gray') #, height=30)
        self.button2.grid(row=12, column=1, columnspan=1,pady=5)

    def load_trajectory(self):
        '''
	 load file with trajectories in it
        '''
        print("load trajectory")

    def process_trajectory(self):
        '''
	 process trajectory
        '''
        print("process trajectory")


    def close_app(self):
        '''
        quit all the proccesses while closing the GUI
        '''
        self.quit()


if __name__ == "__main__":
    root = tk.Tk()
    MainVisual(root)
    root.mainloop()
