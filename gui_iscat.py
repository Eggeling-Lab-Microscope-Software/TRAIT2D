#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI iSCAT tracking

@author: mariaa
"""
import sys
sys.path.append("/home/mariaa/Eggeling_lab/iSCAT_project/code/iscat_lib")

from detectors import Detectors
from movie_processor import background_substraction
import skimage
from skimage import io
import matplotlib.pyplot as plt
from tracker import Tracker
import json
import numpy as np
import cv2
import random
import tkinter as tk
from tkinter import filedialog

# for plotting
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
# default Matplotlib key bindings
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure



class MainVisual(tk.Frame):
    '''
    class of GUI widgets and functions
    '''
    # choose the files and visualise the tracks on the data
    def __init__(self, master):

        #define a window
        tk.Frame.__init__(self, master)
        self.master = master
        master.title("iSCAT tracker 1.0 ")
        master.configure(background='white')
#        master.geometry("1100x1000")

        # # # parameters # # #
        #general
        self.movie_file=" " # path to the move file
        self.movie=[] # matrix with data
        self.movie_length=0 # length of the original movie
        self.figsize_value=(6,6) # parameters for the figure size

        # detection and tracking parameters
        self.maximum_diameter=10 # size of the field for gaussian fitting
        self.sigma=2. # defines gaussian in spot-enhancing filter
        self.threshold=4. # defines threshold in spot-enhancing filter
        self.min_peak=0.2 # defines parameters for local maxima [0,1]

        self.max_dist=10 # number of pixels between two possible connections
        self.frame_gap=5 # maximum number of possible consequently skipped frames in a track
        self.spot_switch=0 # spot type: 0 - dark spot, 1 - light spot

        # list of colors for trajectories plots
        self.color_list=[(200, 0, 0), (200, 0, 127), (0, 0, 255), (200, 155, 0),
                    (100, 255, 5), (255, 10, 120), (255, 127, 255),
                    (127, 0, 255), (0, 255, 0), (177, 0, 20),
                    (12, 200, 0), (0, 114, 255), (255, 20, 0),
                    (0, 255, 255), (255, 100, 100), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]

     # # # # # # menu to choose files and set tracker parameters # # # # # #

        # button to select movie
        self.button1 = tk.Button(text="       Select movie file       ", command=self.select_movie, width=40)
        self.button1.grid(row=0, column=1, columnspan=3, pady=5)

        # show selected movie name
        lbl1 = tk.Label(master=root, text="selected file: "+self.movie_file, bg='white')
        lbl1.grid(row=1, column=1, columnspan=3, pady=5)

        # button for preprocessing step
        self.button2 = tk.Button(text="    Process data before tracking    ", command=self.processing, width=40)
        self.button2.grid(row=2, column=1, columnspan=3, pady=5)

        # setting tracker parameters

        lbl1 = tk.Label(master=root, text="PARAMETERS: ", width=30, bg='white')
        lbl1.grid(row=3, column=1, columnspan=3, pady=5)

        lbl2 = tk.Label(master=root, text="maximum diameter(even number), px", width=35, bg='white')
        lbl2.grid(row=4, column=1)
        self.param1_diameter = tk.Entry(root, width=10)
        self.param1_diameter.grid(row=4, column=2)

        lbl3 = tk.Label(master=root, text="sigma", width=35, bg='white')
        lbl3.grid(row=5, column=1)
        self.param2_sigma = tk.Entry(root, width=10)
        self.param2_sigma.grid(row=5, column=2)

        lbl4 = tk.Label(master=root, text="threshold [0.01,10]", width=35, bg='white')
        lbl4.grid(row=6, column=1)
        self.param3_threshold = tk.Entry(root, width=10)
        self.param3_threshold.grid(row=6, column=2)

        lbl5 = tk.Label(master=root, text="minimum peak value [0,1]", width=30, bg='white')
        lbl5.grid(row=7, column=1)
        self.param4_peak = tk.Entry(root, width=10)
        self.param4_peak.grid(row=7, column=2)

        lbl6 = tk.Label(master=root, text="maximum distance, px", width=30, bg='white')
        lbl6.grid(row=8, column=1)
        self.param5_distance = tk.Entry(root, width=10)
        self.param5_distance.grid(row=8, column=2)

        lbl6 = tk.Label(master=root, text="frame gap, frame", width=30, bg='white')
        lbl6.grid(row=9, column=1)
        self.param6_framegap = tk.Entry(root, width=10)
        self.param6_framegap.grid(row=9, column=2)

        #preview button
        self.button2 = tk.Button(text="    preview    ", command=self.preview, width=20) #, height=30)
        self.button2.grid(row=10, column=1, columnspan=1,pady=5)

        # button to run the tracker and save results
        self.button2 = tk.Button(text="    Run tracking algorithm    ", command=self.tracking, width=20)
        self.button2.grid(row=10, column=2, columnspan=1, pady=5)


        # type of spots (dark or light)
        var = tk.IntVar() # the switch variable

        # variable update
        def update_monitor_switch():
            self.spot_switch=var.get()

        # spot type switch: # 0 - dark spot, 1 - light spot
        self.R1 = tk.Radiobutton(root, text=" dark spot ", variable=var, value=0, bg='white', command =update_monitor_switch )
        self.R1.grid(row=11, column=1)

        self.R2 = tk.Radiobutton(root, text=" light spot ", variable=var, value=1, bg='white',command = update_monitor_switch ) #  command=sel)
        self.R2.grid(row=11, column=2)


        # show the movie frame: use matplotlib imshow and plotting

        # show dark screen until movie is selected
        bg_img=np.ones((400,400))*0.8
        fig = plt.figure(figsize=self.figsize_value)
        plt.axis('off')
        self.im = plt.imshow(bg_img) # for later use self.im.set_data(new_data)

        # DrawingArea
        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=15, column=1, columnspan=3,pady=5)

    def read_parameters(self):

        # read parameters from the GUI

        if self.param1_diameter.get()=='':
            self.maximum_diameter=10
        else:
            self.maximum_diameter=int(self.param1_diameter.get())

        if self.param2_sigma.get()=='':
            self.sigma=8
        else:
            self.sigma=float(self.param2_sigma.get())

        if self.param3_threshold.get()=='':
            self.threshold=5
        else:
            self.threshold=float(self.param3_threshold.get())

        if self.param4_peak.get()=='':
            self.min_peak=0.1
        else:
            self.min_peak=float(self.param4_peak.get())

        if self.param5_distance.get()=='':
            self.max_dist=10
        else:
            self.max_dist=float(self.param5_distance.get())

        if self.param6_framegap.get()=='':
            self.frame_gap=5
        else:
            self.frame_gap=float(self.param6_framegap.get())

    def processing(self):
        # preprocessing step
        self.movie_processed = background_substraction(self.movie)

    def preview(self):
        # show random frame with detection on the monitor

        #read parameters
        self.read_parameters()

        # select movie frame
        pos=random.randrange(0, self.movie.shape[0]-1)
        self.image = self.movie[pos,:,:]

        # invert image in case you are looking at the dark spot
        if self.spot_switch==0:
            image_for_process=skimage.util.invert(self.image)
        else:
            image_for_process=self.image

        # set detector
        detect_particle=Detectors()

        #MSSEF settings
        detect_particle.c=self.threshold #0.01 # coef for the thresholding
        detect_particle.sigma=self.sigma # max sigma for LOG
        detect_particle.expected_size=self.maximum_diameter # field for gaussian fitting

        #thresholding settings
        detect_particle.min_distance=5 # minimum distance between two max after MSSEF
        detect_particle.threshold_rel=self.min_peak # min peak value in relation to the image

        #run detector
        centers=detect_particle.detect(image_for_process)

        #plot the result
        plt.close()
        fig = plt.figure(figsize=self.figsize_value)
        plt.axis('off')
        self.im = plt.imshow(self.image) # for later use self.im.set_data(new_data)

        for point in centers:
            plt.plot(point[1], point[0],  "*r")

        # DrawingArea
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().grid(row=15, column=1, columnspan=3, pady=5)


    def select_movie(self):

        # Select movie for processing - connect to the button

        filename = tk.filedialog.askopenfilename(filetypes = [("All files", "*.*")])
        self.movie_file=filename

        # read the file
        self.movie=skimage.io.imread(self.movie_file)
        self.movie_length=self.movie.shape[0]
        lbl1 = tk.Label(master=root, text="movie file: "+self.movie_file.split("/")[-1], bg='white')
        lbl1.grid(row=1, column=1, columnspan=3, pady=5)

        # show first frame in the monitor

        self.image = self.movie[1,:,:]
        plt.close()
        fig = plt.figure(figsize=self.figsize_value)
        plt.axis('off')
        self.im = plt.imshow(self.image) # for later use self.im.set_data(new_data)

        # DrawingArea
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().grid(row=15, column=1, columnspan=3, pady=5)

    def tracking(self):
        # detection and linking from the selected movie - connect to the button
        print("tracker running")
        #read parameters
        self.read_parameters()

        def track_to_frame(data):

            # change data arrangment from tracks to frames

            track_data_framed={}
            track_data_framed.update({'frames':[]})

            for n_frame in range(0, self.movie.shape[0]):


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

            track_data_framed=track_to_frame(tracks)

            final_img_set = np.zeros((self.movie.shape[0], self.movie.shape[1], self.movie.shape[2], 3))

            for frameN in range(0, self.movie.shape[0]):

                plot_info=track_data_framed['frames'][frameN]['tracks']
                frame_img=self.movie[frameN,:,:]
                # Make a colour image frame
                orig_frame = np.zeros((self.movie.shape[1], self.movie.shape[2], 3))

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

                # Display the resulting tracking frame
                cv2.imshow('Tracking', orig_frame)

                ################### to save #################
                final_img_set[frameN,:,:,:]=orig_frame


                    # save results

            final_img_set=final_img_set/np.max(final_img_set)*255
            final_img_set=final_img_set.astype('uint8')
            skimage.io.imsave(save_file, final_img_set)
            cv2.destroyAllWindows()



        # detector settings

        detect_particle=Detectors()
        #MSSEF settings

        detect_particle.c=self.threshold #0.01 # coef for the thresholding
        detect_particle.sigma=self.sigma # max sigma for LOG

        #thresholding
        detect_particle.min_distance=5 # minimum distance between two max after MSSEF
        detect_particle.threshold_rel=self.min_peak # min picl value in relation to the image
        detect_particle.expected_size=self.maximum_diameter

        # tracker settings
        duration_shreshold=50 # minimum  track length

        tracker = Tracker(self.max_dist, self.frame_gap, self.movie.shape[0], 0)


        # tracking itself
        for frameN in range(0, self.movie.shape[0]):
            print('frame', frameN)
            #detection

            frame_img=self.movie[frameN,:,:]
            if self.spot_switch==0:
                frame_img=skimage.util.invert(frame_img)

            centers=detect_particle.detect(frame_img)

            #tracking
            tracker.update(centers, frameN)

        for trackN in range(0, len(tracker.tracks)):
            tracker.completeTracks.append(tracker.tracks[trackN])


         # # rearrange the data into disctionary  and save it

        data_tracks={}

        for trackN in range(0, len(tracker.completeTracks)):
            if len(tracker.completeTracks[trackN].trace)>=duration_shreshold:
                data_tracks.update({tracker.completeTracks[trackN].track_id:{
                        'trackID':tracker.completeTracks[trackN].track_id,
                        'trace': tracker.completeTracks[trackN].trace,
                        'frames':tracker.completeTracks[trackN].trace_frame,
                        'skipped_frames': tracker.completeTracks[trackN].skipped_frames
                        }})

        save_file = tk.filedialog.asksaveasfilename(filetypes = [("All files", "*.*")])
        save_movie(data_tracks, save_file)

        with open('test.txt', 'w') as f:
            json.dump(data_tracks, f, ensure_ascii=False)





if __name__ == "__main__":
    root = tk.Tk()
    MainVisual(root)
    root.mainloop()
