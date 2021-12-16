#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRAIT tracker GUI 

"""

import matplotlib
matplotlib.use('TkAgg') # This is a fix to use the GUI on Mac

import sys
from trait2d.detectors import Detectors
from trait2d.movie_processor import background_substraction
from trait2d.tracker import Tracker

import skimage
from skimage import io
import matplotlib.pyplot as plt

import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import font, ttk
import csv
import webbrowser

# for plotting
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# default Matplotlib key bindings
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import imageio

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

class MainVisual(tk.Frame):
    '''
    GUI for TRAIT tracker
    '''

    def __init__(self, master):

        #define the window
        tk.Frame.__init__(self, master)
        self.master = master
        master.title("TRAIT tracker")
        master.configure(background='white')
        
        # get the monitor size
        self.dpi=100
        self.monitor_width=master.winfo_screenwidth()
        self.monitor_height = master.winfo_screenheight()
        self.button_size=int(self.monitor_width/25)
        self.figsize_value=(int(self.monitor_height/3/self.dpi), int(self.monitor_height/3/self.dpi)) # parameters for the figure
        self.scale_length=int(self.monitor_height/3)
        
        master.protocol('WM_DELETE_WINDOW', self.close_app)

        # # # parameters # # #
        #general
        self.movie_file=" " # path to the movie file
        self.movie=np.ones((1,400,400))*0.8 # matrix with data
        self.movie_processed=np.ones((1, 400,400))*0.8 # matrix with processed data
        self.movie_length=0 # length of the original movie
        self.frame_pos=0 # frame location
        
        
        self.img_resolution=1 # nm/pix
        self.frame_rate=1 # sec/frame
        
        self.start_frame=0
        self.end_frame=0

        
        
        # detection and tracking parameters
        self.maximum_diameter=10 # size of the field for gaussian fitting
        self.sigma=6. # defines gaussian in spot-enhancing filter
        self.threshold=4. # defines threshold in spot-enhancing filter
        self.min_peak=0.2 # defines parameter for local maxima [0,1]

        self.max_dist=15 # number of pixels between two possible connections
        self.frame_gap=15 # maximum number of possible consequently skipped frames in a track
        self.spot_switch=0 # spot type: 0 - dark spot, 1 - light spot
        
        self.min_track_length=100 # track with length less than the value will be removed

        # list of colors for trajectory plots
        self.color_list=[(200, 0, 0), (200, 0, 127), (0, 0, 255), (200, 155, 0),
                    (0, 255, 59), (255, 10, 120), (255, 127, 255),
                    (127, 0, 255), (0, 255, 0), (177, 0, 20),
                    (12, 200, 0), (0, 114, 255), (255, 20, 0),
                    (0, 255, 255), (255, 100, 100), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]

        self.tracks_data=[] # data of the tracks prepared for the csv file
        
     # # # # # # menu to choose files and set tracker parameters # # # # # #

        # button to select movie
        self.button1 = tk.Button(text="       Select movie file       ", command=self.select_movie, width=int(self.button_size/3), bg='gray')
        self.button1.grid(row=0, column=1, columnspan=3, pady=5)

        # show selected movie name
        self.lbl1 = tk.Label(master=root, text="Selected file: "+self.movie_file, bg='white')
        self.lbl1.grid(row=1, column=1, columnspan=3, pady=5)

        # button for preprocessing step
        self.button2 = tk.Button(text="    Run pre-processing step    ", command=self.processing, width=int(self.button_size/3), bg='gray')
        self.button2.grid(row=2, column=1, columnspan=3, pady=5)

        # setting tracker parameters

        lbl1 = tk.Label(master=root, text="PARAMETERS: ", width=int(self.button_size/2), bg='white')
        lbl1.grid(row=3, column=1, columnspan=3, pady=5)


        lbl3 = tk.Label(master=root, text="SEF: sigma", width=int(self.button_size/2), bg='white')
        lbl3.grid(row=4, column=1)
        v = tk.StringVar(root, value=str(self.sigma))
        self.param2_sigma = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.param2_sigma.grid(row=4, column=2)

        lbl4 = tk.Label(master=root, text="SEF: threshold [0.01,10]", width=int(self.button_size/2), bg='white')
        lbl4.grid(row=5, column=1)
        v = tk.StringVar(root, value=str(self.threshold))
        self.param3_threshold = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.param3_threshold.grid(row=5, column=2)

        lbl5 = tk.Label(master=root, text="SEF: min peak value [0,1]", width=int(self.button_size/2), bg='white')
        lbl5.grid(row=6, column=1)
        v = tk.StringVar(root, value=str(self.min_peak))
        self.param4_peak = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.param4_peak.grid(row=6, column=2)

        lbl2 = tk.Label(master=root, text="  Patch size (even number), px", width=int(self.button_size/2), bg='white')
        lbl2.grid(row=7, column=1)
        v = tk.StringVar(root, value=str(self.maximum_diameter))
        self.param1_diameter = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.param1_diameter.grid(row=7, column=2)


        lbl6 = tk.Label(master=root, text="Linking: max distance, px", width=int(self.button_size/2), bg='white')
        lbl6.grid(row=8, column=1)
        v = tk.StringVar(root, value=str(self.max_dist))
        self.param5_distance = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.param5_distance.grid(row=8, column=2)

        lbl6 = tk.Label(master=root, text="Linking: frame gap, frame", width=int(self.button_size/2), bg='white')
        lbl6.grid(row=9, column=1)
        v = tk.StringVar(root, value=str(self.frame_gap))
        self.param6_framegap = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.param6_framegap.grid(row=9, column=2)

        lbl7 = tk.Label(master=root, text="Minimum track length, frames", width=int(self.button_size/2), bg='white')
        lbl7.grid(row=10, column=1)
        v = tk.StringVar(root, value=str(self.min_track_length))
        self.param7_framegap = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.param7_framegap.grid(row=10, column=2)
        
        
        # resolution in time and space                
        res_lb = tk.Label(master=root, text=" Resolution (\u03BCm per pix) : ", width=int(self.button_size/2), bg='white')
        res_lb.grid(row=11, column=1)
        v = tk.StringVar(root, value=str(self.img_resolution))
        self.res_parameter = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.res_parameter.grid(row=11, column=2)
            
        lbl3 = tk.Label(master=root, text="Frame rate (sec per frame) : ", width=int(self.button_size/2), bg='white')
        lbl3.grid(row=12, column=1)
        v = tk.StringVar(root, value=str(self.frame_rate))
        self.frame_parameter = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.frame_parameter.grid(row=12, column=2)  



        # type of spots (dark or light)
        var = tk.IntVar() # the switch variable

        # variable update
        def update_monitor_switch():
            self.spot_switch=var.get()

        # spot type switch: # 0 - dark spot, 1 - light spot
        self.R1 = tk.Radiobutton(root, text=" Dark spot ", variable=var, value=0, bg='white', command =update_monitor_switch)
        self.R1.grid(row=13, column=1, pady=5)

        self.R2 = tk.Radiobutton(root, text=" Light spot ", variable=var, value=1, bg='white',command = update_monitor_switch ) #  command=sel)
        self.R2.grid(row=13, column=2, pady=5)
        

        #preview button
        self.button2 = tk.Button(text="    Preview    ", command=self.preview, width=int(self.button_size/3), bg='gray') #, height=30)
        self.button2.grid(row=14, column=1, columnspan=1,pady=5, padx=5)
        
        
         # save parameters
        self.button2 = tk.Button(text=" Save parameters  ", command=self.save_parameters, width=int(self.button_size/3), bg='gray') #, height=30)
        self.button2.grid(row=15, column=1, columnspan=1,pady=5, padx=5)

         # load parameters
        self.button2 = tk.Button(text=" Load parameters ", command=self.load_parameters, width=int(self.button_size/3), bg='gray') #, height=30)
        self.button2.grid(row=16, column=1, columnspan=1,pady=5, padx=5)


        
        # test run
        self.button2 = tk.Button(text="  Test run  ", command=self.run_test, width=int(self.button_size/3), bg='gray')
        self.button2.grid(row=14, column=2, columnspan=1, pady=5, padx=5)

        # button to run the tracker and save image sequence with plotted trakectories (for visualisation)
        self.button2 = tk.Button(text="    Run tracking    ", command=self.complete_tracking, width=int(self.button_size/3), bg='gray')
        self.button2.grid(row=15, column=2, columnspan=1, pady=5, padx=5)

        # button to save csv file
        self.button2 = tk.Button(text="    Save data    ", command=self.save_data, width=int(self.button_size/3), bg='gray')
        self.button2.grid(row=16, column=2, columnspan=1, pady=5, padx=5)


        # show dark screen until movie is selected
        self.fig, self.ax = plt.subplots(1,1,figsize=self.figsize_value)
        plt.axis('off')
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        
        self.show_frame()

        def show_values(v):
            self.frame_pos=int(v)
            self.show_frame() 
          
        self.scale_movie = tk.Scale(root, from_=0, to=self.movie_processed.shape[0]-1, tickinterval=int(self.movie_processed.shape[0]/5), length=self.scale_length, width=10, orient="horizontal", command=show_values)
        self.scale_movie.set(self.frame_pos)        
        self.scale_movie.grid(row=21, column=1, columnspan=3, pady=5, padx=5) 
        
        # Link to GUI documentation.
        label_font = font.nametofont(ttk.Style().lookup("TLabel", "font"))
        link_font = font.Font(**label_font.configure())
        link_font.configure(underline=1, weight='bold')
        lblDocs = tk.Label(master=root, text="Documentation", font=link_font, fg='blue', bg='white')
        lblDocs.grid(row=22, column=1, columnspan=1, pady=5)
        lblDocs.bind("<Button-1>", lambda e: webbrowser.open_new("https://eggeling-lab-microscope-software.github.io/TRAIT2D/tracker_gui.html#description-of-the-gui"))


    def show_frame(self , centers=[]):
        '''
        show current frame 
        '''
        
        self.ax.clear() # clean the plot 
        self.ax.imshow(self.movie_processed[self.frame_pos,:,:], cmap="gray")
        self.ax.axis('off')
        for point in centers:
            self.ax.plot(point[1], point[0],  "*r")
        
        # DrawingArea
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=20, column=1, columnspan=3,pady=5)
        self.canvas.draw()
        
        
    def save_data(self):
        '''
        save csv file
        '''
        
        # select file location and name
        save_file = tk.filedialog.asksaveasfilename()
        if not(save_file.endswith(".csv")):
                save_file += ".csv"

        with open(save_file, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(self.tracks_data)

        csvFile.close()
         
        print("csv file has been saved to ", save_file)

    def read_parameters(self):
        '''
        read parameters from the GUI
        '''
        if self.param1_diameter.get()!='':
            self.maximum_diameter=int(self.param1_diameter.get())

        if self.param2_sigma.get()!='':
            self.sigma=float(self.param2_sigma.get())

        if self.param3_threshold.get()!='':
            self.threshold=float(self.param3_threshold.get())

        if self.param4_peak.get()!='':
            self.min_peak=float(self.param4_peak.get())

        if self.param5_distance.get()!='':
            self.max_dist=float(self.param5_distance.get())

        if self.param6_framegap.get()!='':
            self.frame_gap=float(self.param6_framegap.get())
            
        if self.param6_framegap.get()!='':
            self.min_track_length=float(self.param7_framegap.get())
                    
            
        if self.res_parameter.get()!='':
            self.img_resolution=float(self.res_parameter.get())
        
        if self.frame_parameter.get()!='':
            self.frame_rate=float(self.frame_parameter.get())

    def processing(self):
        '''
        preprocessing step - connected to a button
        '''
        self.movie_processed = background_substraction(self.movie.copy())

        # show the frame in the monitor
        self.show_frame()
        
    def save_parameters(self):
        '''
        save parameters to a file
        '''
        
        self.read_parameters()
        
        parameters_data=[["SEF:sigma", self.sigma],["SEF:threshold", self.threshold],["SEF:min peak value", self.min_peak],
                         ["Patch size", self.maximum_diameter],["Linking:max distance", self.max_dist],["Linking:frame gap", self.frame_gap],
                         ["Min track length", self.min_track_length],["Resolution", self.img_resolution],["frame rate", self.frame_rate]]
        
        # select file location and name
        save_file = tk.filedialog.asksaveasfilename( title="Provide filename to save parameters")
        if not(save_file.endswith(".csv")):
                save_file += ".csv"

        with open(save_file, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(parameters_data)

        csvFile.close()
        
        
        
    def load_parameters(self):
        '''
        load parameters from a file
        '''
        open_file = tk.filedialog.askopenfilename( title="Open file with parameters ")
   
        with open(open_file, newline='') as f:    
            
            reader = csv.reader(f)
            
            param_val_list=[]
            try:
                for row in reader:
                    param_val_list.append(float(row[1]))
            except csv.Error as e:
                sys.exit('file {}, line {}: {}'.format(open_file, reader.line_num, e))
         
            
            
        # update and diplayed values
        
        self.sigma=param_val_list[0]
        v = tk.StringVar(root, value=str(self.sigma))
        self.param2_sigma = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.param2_sigma.grid(row=4, column=2)

        self.threshold=param_val_list[1]
        v = tk.StringVar(root, value=str(self.threshold))
        self.param3_threshold = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.param3_threshold.grid(row=5, column=2)

        self.min_peak=param_val_list[2]
        v = tk.StringVar(root, value=str(self.min_peak))
        self.param4_peak = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.param4_peak.grid(row=6, column=2)

        self.maximum_diameter=int(param_val_list[3])
        v = tk.StringVar(root, value=str(self.maximum_diameter))
        self.param1_diameter = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.param1_diameter.grid(row=7, column=2)

        self.max_dist=param_val_list[4]
        v = tk.StringVar(root, value=str(self.max_dist))
        self.param5_distance = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.param5_distance.grid(row=8, column=2)

        self.frame_gap=param_val_list[5]
        v = tk.StringVar(root, value=str(self.frame_gap))
        self.param6_framegap = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.param6_framegap.grid(row=9, column=2)

        self.min_track_length=param_val_list[6]
        v = tk.StringVar(root, value=str(self.min_track_length))
        self.param7_framegap = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.param7_framegap.grid(row=10, column=2)
        
        self.img_resolution=param_val_list[7]        
        v = tk.StringVar(root, value=str(self.img_resolution))
        self.res_parameter = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.res_parameter.grid(row=11, column=2)
        
        self.frame_rate=param_val_list[8]            
        v = tk.StringVar(root, value=str(self.frame_rate))
        self.frame_parameter = tk.Entry(root, width=int(self.button_size/4), text=v)
        self.frame_parameter.grid(row=12, column=2)  


    def run_test(self):
        '''
        run tracking for a selected frame range
        '''  
            
        def action_cancel():
            
            try:
                self.new_window.destroy()
            except:
                pass
            
        
        def action_apply():
            
            
            
            try:
                
                # refine range
                self.start_frame=int(self.txt_position_1.get())
                    
                self.end_frame=int(self.txt_position_2.get())  

            except:
                print("Could not identidy the frame number!")
                
            try: 

                # run tracking
                self.tracking()
                action_cancel()
            except:
                pass
            
        # frame to ask for start and end frame of test tracking
        self.new_window = tk.Toplevel(root, bg='white')
        self.new_window.title(" Test run ")
        self.new_window.geometry("+50+50")

        text = tk.Label(master=self.new_window, text="Provide start and end frames for tracking test", bg='white')
        text.grid(row=0, column=0, columnspan=2, pady=5, padx=5, sticky=tk.W)

        lbpose = tk.Label(master=self.new_window, text=" start frame: ", bg='white')
        lbpose.grid(row=1, column=0, pady=5, padx=5, sticky=tk.W)  
        
        self.txt_position_1 = tk.Entry(self.new_window, width=int(self.button_size/4))
        self.txt_position_1.grid(row=1, column=1, pady=5, padx=5)                
        

        lbpose = tk.Label(master=self.new_window, text=" end frame: ", bg='white')
        lbpose.grid(row=2, column=0, pady=5, padx=5, sticky=tk.W)  
        
        self.txt_position_2 = tk.Entry(self.new_window, width=int(self.button_size/4))
        self.txt_position_2.grid(row=2, column=1, pady=5, padx=5)                
        
        self.buttonOK= tk.Button(master=self.new_window,text=" run ", command=action_apply)
        self.buttonOK.grid(row=3, column=0, pady=5, padx=5)   
        
        self.button_cancel= tk.Button(master=self.new_window,text=" cancel ", command=action_cancel)
        self.button_cancel.grid(row=3, column=1, pady=5, padx=5)     
                

               
      
    def complete_tracking(self):
        '''
        run tracking for the entire image sequence
        '''
        
        # define range
        self.start_frame=0
        self.end_frame=self.movie_processed.shape[0] 
        
        # run tracking
        self.tracking()

        

    def preview(self):
        '''
        show random frame with detection on the monitor - connected to a button
        '''
        
        #read parameters
        self.read_parameters()

        # select movie frame
        image = self.movie_processed[self.frame_pos,:,:]

        # invert image in case you are looking at the dark spot
        if self.spot_switch==0:
            image_for_process=skimage.util.invert(image)
        else:
            image_for_process=image

        # set detector
        detect_particle=Detectors()

        #MSSEF settings
        detect_particle.c=self.threshold  # coef for the thresholding
        detect_particle.sigma=self.sigma # max sigma for LOG
        detect_particle.expected_size=self.maximum_diameter # field for gaussian fitting

        #thresholding settings
        detect_particle.min_distance=5 # minimum distance between two max after MSSEF
        detect_particle.threshold_rel=self.min_peak # min peak value in relation to the image

        #run detector
        centers=detect_particle.detect(image_for_process)

        #plot the result
        self.show_frame(centers)


    def select_movie(self):
        '''
        select movie for processing - connected to a button
        '''

        filename = tk.filedialog.askopenfilename()
        root.update()
        self.movie_file=filename

        # read the file
        self.movie=skimage.io.imread(self.movie_file)
        
        #convert to 16bit to 8 bit
        if self.movie.dtype=='uint16':      
            self.movie=(self.movie - np.min(self.movie))/(np.max(self.movie)- np.min(self.movie))
            self.movie=skimage.util.img_as_ubyte(self.movie)
            
        self.movie_length=self.movie.shape[0]
        try:
            self.lbl1.destroy()
        except:
            pass
        self.lbl1 = tk.Label(master=root, text="movie file: "+self.movie_file.split("/")[-1], bg='white')
        self.lbl1.grid(row=1, column=1, columnspan=3, pady=5, padx=5)

        
        # copy proccessed
        self.movie_processed=self.movie.copy()
        
        # show first frame in the monitor
        self.show_frame()
        
        
        def show_values(v):
            self.frame_pos=int(v)
            self.show_frame() 
          
        self.scale_movie = tk.Scale(root, from_=0, to=self.movie_processed.shape[0]-1, tickinterval=int(self.movie_processed.shape[0]/5), length=self.scale_length, width=10, orient="horizontal", command=show_values)
        self.scale_movie.set(self.frame_pos)        
        self.scale_movie.grid(row=21, column=1, columnspan=3, pady=5, padx=5) 

    def tracking(self):
        '''
        detection and linking from the selected movie - connected to a button
        '''
        print("tracker is running ...")
        
        #read parameters
        self.read_parameters()
        

        def track_to_frame(data):

            # change data arrangment from tracks to frames

            track_data_framed={}
            track_data_framed.update({'frames':[]})

            for n_frame in range(self.start_frame, np.min((self.end_frame+1, self.movie_processed.shape[0]))):


                frame_dict={}
                frame_dict.update({'frame': n_frame})
                frame_dict.update({'tracks': []})

                #rearrange the data
                for track in data:
                    p=data.get(track)
                    if n_frame in p['frames']: # if the frame is in the track
                        frame_index=p['frames'].index(n_frame) # find position in the track

                        new_trace=p['trace'][0:frame_index+1] # copy all the traces before the frame
                        frame_dict['tracks'].append({'trackID': p['trackID'], 'trace': new_trace}) # add to the list


                track_data_framed['frames'].append(frame_dict) # add the dictionary
                
            return track_data_framed


        def save_movie(tracks, save_file):
            '''
            save movie with tracks
            '''

            track_data_framed=track_to_frame(tracks)

            movie_length=np.min((self.end_frame-self.start_frame+1, self.movie_processed.shape[0]))
            final_img_set = np.zeros((movie_length, self.movie_processed.shape[1], self.movie_processed.shape[2], 3))
            
            frame_pos=0
            
            for frameN in range(self.start_frame, np.min((self.end_frame+1, self.movie_processed.shape[0]))):

                plot_info=track_data_framed['frames'][frame_pos]['tracks']

                frame_img=self.movie_processed[frameN,:,:]

                # Make a colour image frame
                orig_frame = np.zeros((self.movie_processed.shape[1], self.movie_processed.shape[2], 3))
                
                orig_frame [:,:,0] = frame_img/np.max(frame_img)*256
                orig_frame [:,:,1] = frame_img/np.max(frame_img)*256
                orig_frame [:,:,2] = frame_img/np.max(frame_img)*256

                for p in plot_info:
                    trace=p['trace']
                    trackID=p['trackID']

                    clr = trackID % len(self.color_list)
                    if (len(trace) > 1):
                        for j in range(len(trace)-1):
                            # Draw trace line
                            point1=trace[j]
                            point2=trace[j+1]
                            x1 = int(point1[1])
                            y1 = int(point1[0])
                            x2 = int(point2[1])
                            y2 = int(point2[0])
                            cv2.line(orig_frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                     self.color_list[clr], 2)

                cv2.imshow('Tracking', orig_frame)

                ################### to save #################

                final_img_set[frame_pos,:,:,:]=orig_frame

                frame_pos+=1
                


            # save results

            final_img_set=final_img_set/np.max(final_img_set)*255
            final_img_set=final_img_set.astype('uint8')
            
            if not(save_file.endswith(".tif") or save_file.endswith(".tiff")):
                save_file += ".tif"
            print("saving to ", save_file)  
            
            imageio.volwrite(save_file, final_img_set)
            
            cv2.destroyAllWindows()



        # detector settings
        detect_particle=Detectors()
        
        #MSSEF settings
        detect_particle.c=self.threshold # coef for thresholding
        detect_particle.sigma=self.sigma # max sigma for LOG

        #other settings
        detect_particle.min_distance=5 # minimum distance between two max after MSSEF
        detect_particle.threshold_rel=self.min_peak # min peak value in relation to the image
        detect_particle.expected_size=self.maximum_diameter # expected particle size


        tracker = Tracker(self.max_dist, self.frame_gap,  0)

        print(self.movie_processed.shape[0])
        # frame to frame detection and linking loop 
        for frameN in range(self.start_frame, np.min((self.end_frame+1, self.movie_processed.shape[0]))):
            print('frame', frameN)
            
            #detection
            frame_img=self.movie_processed[frameN,:,:]
            if self.spot_switch==0:
                frame_img=skimage.util.invert(frame_img)

            centers=detect_particle.detect(frame_img)

            #tracking
            tracker.update(centers,  frameN)

        #add remaining tracks
        for trackN in range(0, len(tracker.tracks)):
            tracker.completeTracks.append(tracker.tracks[trackN])


        # rearrange the data for saving
         
        self.tracks_data=[]
        self.tracks_data.append(['X', 'Y', 'TrackID',
                                 't'])

        data_tracks={}
        trackID=0
        for trackN in range(0, len(tracker.completeTracks)):
            #save trajectories 
            
            #if track is long enough:
            if len(tracker.completeTracks[trackN].trace)>=self.min_track_length:
                trackID+=1
              
                # check the track for missing detections
                frames=tracker.completeTracks[trackN].trace_frame
                trace=tracker.completeTracks[trackN].trace
                pos=0
                new_frames=[]
                new_trace=[]
                for frame_pos in range(frames[0], frames[-1]+1):
                    frame=frames[pos]
                    
                    if frame_pos==frame:
                        new_frames.append(frame_pos)
                        new_trace.append(trace[pos])
                        pos=pos+1
                        
                    else:
                        new_frames.append(frame_pos)
                        frame_img=self.movie_processed[frame_pos,:,:]
                        
                        # find  particle location
                        
                        point=trace[pos] # previous frame
                        
                        # define ROI 
                        data=np.zeros((detect_particle.expected_size,detect_particle.expected_size))
            
                        #start point
                        start_x=int(point[0]-detect_particle.expected_size/2)
                        start_y=int(point[1]-detect_particle.expected_size/2)
                        
                        #end point
                        end_x=int(point[0]+detect_particle.expected_size/2)
                        end_y=int(point[1]+detect_particle.expected_size/2)
                        
                        x_0=0
                        x_1=detect_particle.expected_size
                        y_0=0
                        y_1=detect_particle.expected_size
                        
                        # define ROI coordinates
                        
                        if start_x<0:
                            start_x=0
                            end_x=detect_particle.expected_size
                            
                        if start_y<0:
                            start_y=0
                            end_y=detect_particle.expected_size
                            
                        if end_x>frame_img.shape[0]:
                            end_x=frame_img.shape[0]
                            start_x=frame_img.shape[0]-detect_particle.expected_size
            
                        if end_y>frame_img.shape[1]:
                            end_y=frame_img.shape[1]
                            start_y=frame_img.shape[1]-detect_particle.expected_size
                        

                        data[x_0:x_1,y_0:y_1]=frame_img[start_x:end_x, start_y:end_y]
                        
                        # subpixel localisatopm
                        x,y=detect_particle.radialsym_centre(data)
                        
                        # check that the centre is inside of the spot            
                        if y<detect_particle.expected_size and x<detect_particle.expected_size and y>=0 and x>=0:               
                            new_trace.append([x+int(point[0]-detect_particle.expected_size/2),y+int(point[1]-detect_particle.expected_size/2)])
   
                        else: # if not use the previous point
                            new_trace.append(trace[pos])                
                
                
                for pos in range(0, len(new_trace)):
                    point=new_trace[pos]
                    frame=new_frames[pos]
                    self.tracks_data.append([ point[1]*self.img_resolution, point[0]*self.img_resolution, trackID, frame*self.frame_rate])

                #save for plotting tracks
                data_tracks.update({tracker.completeTracks[trackN].track_id:{
                        'trackID':trackID,
                        'trace': new_trace,
                        'frames': new_frames,
                        'skipped_frames': 0
                        }})

        save_file = tk.filedialog.asksaveasfilename(title="save the movie with trajectories")
        if save_file:
            save_movie(data_tracks, save_file)


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
