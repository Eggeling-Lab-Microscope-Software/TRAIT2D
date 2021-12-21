#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI iSCAT simulator

@author: mariaa, joel
"""

import matplotlib
matplotlib.use('TkAgg') # This is a bug fix in order to use the GUI on Mac

from trait2d.simulators import HoppingDiffusion, movie_simulator, BrownianDiffusion
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import font
from tkinter import messagebox
import webbrowser

# for plotting
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# default Matplotlib key bindings

# Global Variables
HOPPING_DIFFUSION = 0  # Label for the hopping diffusion
FREE_DIFFUSION = 1  # Label for the free diffusion

class MainVisual(tk.Frame):
    '''
    class of GUI for simulator
    '''

    def __init__(self, master):

        #define a window
        tk.Frame.__init__(self, master)
        self.master = master
        master.title("TRAIT Simulator")
        master.configure(background='gray')
        # get the monitor size
        self.dpi = 100
        self.monitor_width = master.winfo_screenwidth()
        self.monitor_height = master.winfo_screenheight()
        self.button_size = int(self.monitor_width/25)
        self.fig_size = (int(self.monitor_height/2/self.dpi), int(self.monitor_height/2/self.dpi))
        
        master.protocol('WM_DELETE_WINDOW', self.close_app)

        # # # parameters # # #
        # trajectory generation
        self.Tmax = 10  # Maximal total length [s]
        self.dt = 5e-3  #Time step [s]
        self.L = 10e-6  #Sandbox length [m]
        self.dL = 20e-9  #Compartment map pixel size [m]
        self.Df = 8e-13  # Free diffusion coefficient [m^2/s]
        self.HL = 40e-9  # Average compartment diameter/length [m]
        self.HP = 0.01  #Hopping probability [0-1]
        self.seed = None  #Random generator seed (nonnegative integer)

        self.dynamics_switch = 0  # dynamics type: 0 - diffusion 1 - hopping diffusion
        
        self.trajectory_file_type = "csv"  # file type to save the trajectories
        self.trajectory = {}  # data of the tracks prepared for the csv file
        
        # image generation
        self.resolution = 1e-7
        self.dt_image = 5e-3
        self.contrast = 5
        self.background = 0.3
        self.noise_gaussian = 0.15
        self.noise_poisson = True
        self.ratio = "square"
        
        # trajectory generator
        self.TG = HoppingDiffusion(Tmax=self.Tmax, dt=self.dt, L=self.L, dL=self.dL,
                                   Df=self.Df, HL=self.HL, HP=self.HP, seed=self.seed)
        
        # image generator
        self.IG = movie_simulator(tracks=None, resolution=self.resolution, dt=self.dt_image,
                                  contrast=self.contrast, background=self.background,
                                  noise_gaussian=self.noise_gaussian,
                                  noise_poisson=self.noise_poisson, ratio=self.ratio)
        
     # # # # # # menu to choose files and set tracker parameters # # # # # #

        # name the section

        lbl1 = tk.Label(master=root, text=" TRAJECTORY ", bg='gray', compound=tk.LEFT)
        lbl1.grid(row=0, column=1, columnspan=1, pady=5)

        # separator - line 
        self._separator = ttk.Separator(master=root, orient=tk.VERTICAL)
        self._separator.grid(row=1, column=1, columnspan=8, pady=5, sticky="ew")

        # type of dynamics
        var = tk.IntVar()  # the switch variable

        # variable update
        def update_switch():
            self.dynamics_switch = var.get()

        # dynamics type switch: # 1 - diffusion 0 - hopping diffusion
        self.R1 = tk.Radiobutton(root, text=" Free Diffusion ", variable=var,
                                 value=1, bg='gray', command=update_switch)
        self.R1.grid(row=2, column=1, columnspan=2)

        self.R2 = tk.Radiobutton(root, text=" Hopping Diffusion ", variable=var,
                                 value=0, bg='gray', command=update_switch) #  command=sel)
        self.R2.grid(row=2, column=3, columnspan=3, pady=5)
        
        # setting trajectory parameters
        lbl1 = tk.Label(master=root, text=" Parameters: ", width=30, bg='gray')
        lbl1.grid(row=3, column=1, columnspan=7, pady=5)


        lbl3 = tk.Label(master=root, text="Maximal total length [Tmax, s]",
                        width=self.button_size, bg='gray', compound=tk.LEFT)
        lbl3.grid(row=4, column=1, columnspan=2)
        v = tk.StringVar(root, value=str(self.Tmax))
        self.param_Tmax = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.param_Tmax.grid(row=4, column=3, columnspan=6)

        lbl4 = tk.Label(master=root, text="Time step [dt, s]",
                        width=self.button_size, bg='gray', compound=tk.LEFT)
        lbl4.grid(row=5, column=1, columnspan=2)
        v = tk.StringVar(root, value=str(self.dt))
        self.param_dt = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.param_dt.grid(row=5, column=3, columnspan=6)

        lbl5 = tk.Label(master=root, text=" Sandbox length [L, m]",
                        width=self.button_size, bg='gray', compound=tk.LEFT)
        lbl5.grid(row=6, column=1, columnspan=2)
        v = tk.StringVar(root, value=str(self.L))
        self.param_L = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.param_L.grid(row=6, column=3, columnspan=6)

        lbl2 = tk.Label(master=root, text=" Compartment map pixel size [dL, m]",
                        width=self.button_size, bg='gray', compound=tk.LEFT)
        lbl2.grid(row=7, column=1, columnspan=2)
        v = tk.StringVar(root, value=str(self.dL))
        self.param_dL = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.param_dL.grid(row=7, column=3, columnspan=6)


        lbl6 = tk.Label(master=root, text=" Free diffusion coefficient [Df, "+r'm^2/s]',
                        width=self.button_size, bg='gray', compound=tk.LEFT)
        lbl6.grid(row=8, column=1, columnspan=2)
        v = tk.StringVar(root, value=str(self.Df))
        self.param_Df = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.param_Df.grid(row=8, column=3, columnspan=6)

        lbl6 = tk.Label(master=root, text=" Average compartment diameter/length [HL, m]",
                        width=self.button_size, bg='gray', compound=tk.LEFT)
        lbl6.grid(row=9, column=1, columnspan=2)
        v = tk.StringVar(root, value=str(self.HL))
        self.param_HL = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.param_HL.grid(row=9, column=3, columnspan=6)
        
        lbl6 = tk.Label(master=root, text=" Hopping probability [HP, [0,1]]",
                        width=self.button_size, bg='gray', compound=tk.LEFT)
        lbl6.grid(row=10, column=1, columnspan=2)
        v = tk.StringVar(root, value=str(self.HP))
        self.param_HP = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.param_HP.grid(row=10, column=3, columnspan=6)

        lbl6 = tk.Label(master=root, text=" Random generator seed [seed, integer]",
                        width=self.button_size, bg='gray', compound=tk.LEFT)
        lbl6.grid(row=11, column=1, columnspan=2)
        if (self.seed == None):
            v = tk.StringVar(root, value='')
        else:
            v = tk.StringVar(root, value=str(self.seed))
        self.param_seed = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.param_seed.grid(row=11, column=3, columnspan=6)

        #generate button
        self.button2 = tk.Button(text="    GENERATE    ", command=self.generate_trajectory,
                                 width=int(self.button_size/4), bg='gray') #, height=30)
        self.button2.grid(row=12, column=1, columnspan=2,pady=5)

        # button to save results
        self.button2 = tk.Button(text="    SAVE   ", command=self.save_trajectory,
                                 width=int(self.button_size/4), bg='gray')
        self.button2.grid(row=12, column=3, columnspan=6, pady=5)
        
        #preview button
        self.button2 = tk.Button(text="  SHOW TRACK  ", command=self.show_trajectory,
                                 width=int(self.button_size/4), bg='gray') #, height=30)
        self.button2.grid(row=13, column=1, columnspan=2,pady=5)
        
        # choose the file type
        # type of dynamics
        var_2 = tk.StringVar()  # the switch variable
        var_2.set("csv")
        # variable update
        def update_switch_2():
            self.trajectory_file_type=var_2.get()

        
        # trajectory file type
        self.F1 = tk.Radiobutton(root, text=" csv", variable=var_2,
                                 value="csv", bg='gray',command = update_switch_2 ) #  command=sel)
        self.F1.grid(row=13, column=4, columnspan=1, pady=5)     

        self.F2 = tk.Radiobutton(root, text=" json", variable=var_2,
                                 value="json", bg='gray',command = update_switch_2 ) #  command=sel)
        self.F2.grid(row=13, column=5, columnspan=1, pady=5)  

        self.F3 = tk.Radiobutton(root, text=" pcl", variable=var_2,
                                 value="pcl", bg='gray',command = update_switch_2 ) #  command=sel)
        self.F3.grid(row=13, column=6, columnspan=1, pady=5)     
##################################################

        # separator - line 
        self._separator = ttk.Separator(master=root, orient=tk.VERTICAL)
        self._separator.grid(row=20, column=1, columnspan=8, pady=5, sticky="ew")        
                # name the section

        lbl1 = tk.Label(master=root, text=" IMAGE SEQUENCE ", bg='gray', compound=tk.LEFT)
        lbl1.grid(row=21, column=1, columnspan=1, pady=5)

        # separator - line 
        self._separator = ttk.Separator(master=root, orient=tk.VERTICAL)
        self._separator.grid(row=22, column=1, columnspan=8, pady=5, sticky="ew")
        
        # button to load trajectory
        self.button2 = tk.Button(text="    load trajectory    ", command=self.load_trajectory,
                                 width=int(self.button_size/4), bg='gray')
        self.button2.grid(row=23, column=3, columnspan=6, pady=5)
        
        # setting image parameters
        lbl1 = tk.Label(master=root, text="Parameters: ", width=int(self.button_size/1.5), bg='gray')
        lbl1.grid(row=24, column=1, columnspan=7, pady=5)


        lbl3 = tk.Label(master=root, text="Resolution [resolution, m/pix]",
                        width=self.button_size, bg='gray', compound=tk.LEFT)
        lbl3.grid(row=25, column=1, columnspan=2)
        v = tk.StringVar(root, value=str(self.resolution))
        self.param_resolution = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.param_resolution.grid(row=25, column=3, columnspan=6)

        lbl4 = tk.Label(master=root, text="Contrast [contrast, float>0]",
                        width=self.button_size, bg='gray', compound=tk.LEFT)
        lbl4.grid(row=26, column=1, columnspan=2)
        v = tk.StringVar(root, value=str(self.contrast))
        self.param_contrast = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.param_contrast.grid(row=26, column=3, columnspan=6)

        lbl5 = tk.Label(master=root, text=" Background intensity [background, [0,1]]",
                        width=self.button_size, bg='gray', compound=tk.LEFT)
        lbl5.grid(row=27, column=1, columnspan=2)
        v = tk.StringVar(root, value=str(self.background))
        self.param_background = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.param_background.grid(row=27, column=3, columnspan=6)

        lbl2 = tk.Label(master=root, text=" Gaussian noise variance [noise_gaussian]",
                        width=self.button_size, bg='gray', compound=tk.LEFT)
        lbl2.grid(row=28, column=1, columnspan=2)
        v = tk.StringVar(root, value=str(self.noise_gaussian))
        self.param_noise_gaussian = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.param_noise_gaussian.grid(row=28, column=3, columnspan=6)


        lbl6 = tk.Label(master=root, text=" Poisson noise variance [noise_poisson, bool]",
                        width=self.button_size, bg='gray', compound=tk.LEFT)
        lbl6.grid(row=29, column=1, columnspan=2)
        v = tk.StringVar(root, value=str(self.noise_poisson))
        self.param_noise_poisson = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.param_noise_poisson.grid(row=29, column=3, columnspan=6)

        lbl2 = tk.Label(master=root, text=" Temporal resolution [dt_image, frame/sec]",
                        width=self.button_size, bg='gray', compound=tk.LEFT)
        lbl2.grid(row=30, column=1, columnspan=2)
        v = tk.StringVar(root, value=str(self.dt_image))
        self.param_dt_image = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.param_dt_image.grid(row=30, column=3, columnspan=6)


        lbl6 = tk.Label(master=root, text=" Ratio [ratio]",
                        width=self.button_size, bg='gray', compound=tk.LEFT)
        lbl6.grid(row=31, column=1, columnspan=2)
        v = tk.StringVar(root, value=str(self.ratio))
        self.param_ratio = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.param_ratio.grid(row=31, column=3, columnspan=6)
        
        #preview button
        self.button2 = tk.Button(text=" load psf", command=self.load_psf,
                                 width=int(self.button_size/4), bg='gray') #, height=30)
        self.button2.grid(row=32, column=1, columnspan=2,pady=5)
        
 
        #preview button
        self.button2 = tk.Button(text=" GENERATE and SHOW ", command=self.generate_images,
                                 width=int(self.button_size/4), bg='gray') #, height=30)
        self.button2.grid(row=33, column=1, columnspan=2,pady=5)

        # button to run the tracker and save results
        self.button2 = tk.Button(text="    SAVE   ", command=self.save_images,
                                 width=int(self.button_size/4), bg='gray')
        self.button2.grid(row=33, column=3, columnspan=6, pady=10, padx=30)

        # Link to GUI documentation.
        label_font = font.nametofont(ttk.Style().lookup("TLabel", "font"))
        link_font = font.Font(**label_font.configure())
        link_font.configure(underline=1, weight='bold')
        lblDocs = tk.Label(master=root, text="Documentation", font=link_font, fg='blue', bg='gray')
        lblDocs.grid(row=34, column=1, columnspan=1, pady=5)
        lblDocs.bind("<Button-1>", lambda e: webbrowser.open_new("https://eggeling-lab-microscope-software.github.io/TRAIT2D/simulator_gui.html#description-of-the-gui"))

    def generate_trajectory(self):
        '''
        function to generate trajectory
        '''
        # update the parameters        
        self.read_parameters()
        
        if self.dynamics_switch == HOPPING_DIFFUSION:
            self.TG = HoppingDiffusion(Tmax=self.Tmax, dt=self.dt, L=self.L, dL=self.dL,
                                       Df=self.Df, HL=self.HL, HP=self.HP, seed=self.seed)
            self.TG.run()
        elif self.dynamics_switch == FREE_DIFFUSION:
            self.TG = BrownianDiffusion(Tmax=self.Tmax, dt=self.dt, L=self.L, dL=self.dL,
                                        d=self.Df,  seed=self.seed)
            self.TG.run()            
                 
        # print("generate_trajectory(self)") ## Debug print
        self.trajectory = self.TG.trajectory

    def save_trajectory(self):
        '''
        save trajectory
        '''
        # update the parameters        
        self.read_parameters()
        # select file location and name
        save_file = tk.filedialog.asksaveasfilename()
        if not(save_file.endswith(self.trajectory_file_type)):
                save_file += "."+self.trajectory_file_type
                
        self.TG.save_trajectory(save_file, self.trajectory_file_type)

         
        print("save trajectory into a file: ", save_file)
        
    def show_trajectory(self):
        '''
        plot trajectory in separate window
        '''
        # Update the parameters
        self.read_parameters()

        # DrawingArea
        novi = tk.Toplevel()
        novi.title("trajectory plot")

        if self.dynamics_switch == HOPPING_DIFFUSION:
            title = "Hopping diffusion"
        elif self.dynamics_switch == FREE_DIFFUSION:
            title = "Free diffusion"
        else:
            title = "Diffusion"
        
        fig = plt.figure(figsize=self.fig_size)
        self.TG.plot_trajectory(time_resolution=0.5e-3, limit_fov=False, alpha=0.8,
                                   title=title)
        plt.xlabel("x")
        plt.ylabel("y")            
                # DrawingArea
        canvas = FigureCanvasTkAgg(fig, master=novi)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0)

        # print("trajectory is plotted") # DEBUG print

    def read_parameters(self):
        '''
        read parameters from the GUI
        '''
        # trajectory
        if self.param_Tmax.get()!='':
            self.Tmax=float(self.param_Tmax.get())

        if self.param_dt.get()!='':
            self.dt=float(self.param_dt.get())

        if self.param_L.get()!='':
            self.L=float(self.param_L.get())

        if self.param_dL.get()!='':
            self.dL=float(self.param_dL.get())

        if self.param_Df.get()!='':
            self.Df=float(self.param_Df.get())

        if self.param_HL.get()!='':
            self.HL=float(self.param_HL.get())

        if self.param_HP.get()!='':
            self.HP=float(self.param_HP.get())

        if self.param_seed.get()!='':
            self.seed=int(self.param_seed.get())
        else:
            self.seed = None # To use a random seed
            
        # image sequence

        if self.param_resolution.get()!='':
            self.resolution=float(self.param_resolution.get())

        if self.param_dt_image.get()!='':
            self.dt_image=float(self.param_dt_image.get())

        if self.param_contrast.get()!='':
            self.contrast=float(self.param_contrast.get())

        if self.param_background.get()!='':
            self.background=float(self.param_background.get())

        if self.param_noise_gaussian.get()!='':
            self.noise_gaussian=float(self.param_noise_gaussian.get())

        if self.param_noise_poisson.get()!='':
            self.noise_poisson=bool(self.param_noise_poisson.get())

        if self.param_ratio.get()!='':
            self.ratio=str(self.param_ratio.get())
            

    def load_trajectory(self):
        '''
        function to load trajectory
        '''
        
        print("load_trajectory(self)")

        filename = tk.filedialog.askopenfilename()
        root.update()   
        
        # load from csv file
        
        self.IG.load_tracks(filename, field_x="x", field_y="y", field_t="t", field_id="id")
        
        
        self.trajectory=self.IG.tracks
        

    def generate_images(self):
        '''
        function to generate image sequence
        '''
        # update the parameters        
        self.read_parameters()
        
        # define image generator class with new parameters and the trajectory
        self.IG.tracks=self.trajectory
        self.IG.resolution=self.resolution
        self.IG.dt=self.dt_image
        self.IG.contrast=self.contrast
        self.IG.background=self.background
        self.IG.noise_gaussian=self.noise_gaussian
        self.IG.noise_poisson=self.noise_poisson
        self.IG.ratio=self.ratio

        # Compute an estimated file size
        self.IG.initialize()
        size_mb = self.IG.get_estimated_size()
        continue_simulation = messagebox.askyesno("Continue simulation?",f"Estimated memory size is {size_mb:.2f}MB. Do you wish to continue?")
        if not(continue_simulation):
            msg = "Aborting the movie simulation"
            print(msg)
            messagebox.showwarning("Warning", msg)
            return

        # run the generator
        self.IG.run()
        
        # plot the average image
        img = np.average(self.IG.movie, axis=0)
        
        # DrawingArea
        novi = tk.Toplevel()
        novi.title("Projection of the image sequence")
        
        fig = plt.figure(figsize=self.fig_size)

#        time_interval = int(np.ceil(0.5e-3 / self.dt))
#        x = self.trajectory["x"][0::time_interval]
#        y = self.trajectory["y"][0::time_interval]
#        plt.plot(x, y, alpha=0.8)
        plt.imshow(img.T, origin='lower', cmap="gray")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Projection of the image sequence")
            
                # DrawingArea
        canvas = FigureCanvasTkAgg(fig, master=novi)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0)
        print("image is generated")

    def save_images(self):
        '''
        function to save image sequence
        '''
        # select file location and name
        save_file_name = tk.filedialog.asksaveasfilename()
        if not(save_file_name.endswith("tiff")) and not(save_file_name.endswith("tif")):
            save_file_name += ".tiff"

        self.IG.save(save_file_name)
        print(f"save images: {save_file_name}")
        
    def load_psf(self):
        '''
        load psf
        '''
        filename = tk.filedialog.askopenfilename()
        root.update() 
        
        self.IG.load_psf(filename)

    def close_app(self):
        '''
        quit all the proccesses while closing the GUI
        '''
        self.quit()

def main():
    global root
    root = tk.Tk()
    MainVisual(root)
    root.mainloop()

if __name__ == "__main__":
    main()
