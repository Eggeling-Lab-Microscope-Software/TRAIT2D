#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI iSCAT tracking

@author: mariaa
"""
import sys
sys.path.append("/home/mariaa/Eggeling_lab/iSCAT_project/code/iscat_lib")

from detectors import Detectors
import skimage
from skimage import io
import matplotlib.pyplot as plt
from tracker import Tracker
import json
import numpy as np
import cv2

import tkinter as tk
from tkinter import filedialog

# for plotting
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

class MainVisual(tk.Frame):
    # choose the files and visualise the tracks on the data
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.master = master
        master.title("iSCAT tracker 1.0 ")
        master.configure(background='white')
#        master.geometry("1100x1000")
        
        self.movie_file=" " # path to the move file
        self.track_file=" "# path to the file with tracking data (json format)
        self.movie=[] # matrix with data
        self.track_data_original={}
        self.track_data={} # original tracking data
        self.track_data_filtered={}  # filtered tracking data  
        self.track_data_framed={}  # tracking data arranged by frames  
        self.filter_duration=[0, 1000]
        self.filter_length=[0, 10000]   
        self.frame_pos=0
        self.movie_length=0
        self.monitor_switch=0 # 0- show tracks and track numbers, 1- only tracks, 2 - nothing
        
        # 
        self.figsize_value=(6,6)
      
     # # # # # # menu to choose files and print data # # # # # #
        
        self.button1 = tk.Button(text="       Select movie file       ", command=self.select_movie, width=40)
        self.button1.grid(row=0, column=2, pady=5)

        
        self.button2 = tk.Button(text="    Run tracking algorithm    ", command=self.tracker, width=40)
        self.button2.grid(row=1, column=2, pady=5) 
    
        
      # # # # # # movie  # # # # # # 

              # movie name 
        lbl1 = tk.Label(master=root, text="movie file: "+self.movie_file, bg='white')
        lbl1.grid(row=3, column=1, columnspan=3, pady=5)
        
        # plot bg
        bg_img=np.ones((400,400))*0.8
        fig = plt.figure(figsize=self.figsize_value)
        plt.axis('off')
        self.im = plt.imshow(bg_img) # for later use self.im.set_data(new_data)


        # DrawingArea
        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=8, column=1, columnspan=3,pady=5)

            
    def select_movie(self):
        # Allow user to select movie
        filename = tk.filedialog.askopenfilename(filetypes = [("All files", "*.*")])
        self.movie_file=filename
        # read files 
        self.movie=skimage.io.imread(self.movie_file)
        self.movie_length=self.movie.shape[0]  
        lbl1 = tk.Label(master=root, text="movie file: "+self.movie_file.split("/")[-1], bg='white')
        lbl1.grid(row=3, column=1, columnspan=3, pady=5)
        
                # plot image
        self.show_tracks()

    def show_tracks(self):
        # read data from the selected filesa and show tracks      

        # plot image

        self.image = self.movie[1,:,:]
        plt.close()
        fig = plt.figure(figsize=self.figsize_value)
        plt.axis('off')
        self.im = plt.imshow(self.image) # for later use self.im.set_data(new_data)

        # DrawingArea
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().grid(row=8, column=1, columnspan=3, pady=5)
        
    def tracker(self):
        print("tracker running")
        
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
                    
                    clr = trackID % len(color_list)
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
                                     color_list[clr], 2)
        
        #                point=trace[0]
        #                cv2.putText(orig_frame,str(trackID) ,(int(point[1]),int(point[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,self.color_list[clr],1,cv2.LINE_AA)
                # Display the resulting tracking frame
                cv2.imshow('Tracking', orig_frame)
        
                ################### to save #################
                final_img_set[frameN,:,:,:]=orig_frame
                
            
                    # save results
            
            final_img_set=final_img_set/np.max(final_img_set)*65535
            final_img_set=final_img_set.astype('uint16')
            skimage.io.imsave(save_file, final_img_set)
            cv2.destroyAllWindows()
        
        color_list=[(200, 0, 0), (200, 0, 127), (0, 0, 255), (200, 155, 0),
                    (100, 255, 5), (255, 10, 120), (255, 127, 255),
                    (127, 0, 255), (0, 255, 0), (177, 0, 20), 
                    (12, 200, 0), (0, 114, 255), (255, 20, 0),
                    (0, 255, 255), (255, 100, 100), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]
        
        # detector settings
        
        detect_particle=Detectors()
        #MSSEF settings
        detect_particle.c=4 #0.01 # coef for the thresholding
        detect_particle.k_max=3 # end of  the iteration
        detect_particle.k_min=1 # start of the iteration
        detect_particle.sigma_min=0.1 # min sigma for LOG
        detect_particle.sigma_max=8 # max sigma for LOG     
        
        #thresholding
        detect_particle.min_distance=5 # minimum distance between two max after MSSEF
        detect_particle.threshold_rel=0.1 # min pixl value in relation to the image
        
        detect_particle.int_size=3 # define number of pixels near centre for the thresholding calculation
        detect_particle.box_size=32 # bounding box size for detection
        
        # tracker settings
        
        distance_threshold=10
        max_skip_frame=3
        max_track_length=self.movie.shape[0]
        
        tracker = Tracker(distance_threshold, max_skip_frame, max_track_length, 0)
        
        # setting of the script:
        duration_shreshold=50
        
        for frameN in range(0, self.movie.shape[0]):
            print('frame', frameN)
            #detection
            
            frame_img=self.movie[frameN,:,:]    
            centers=detect_particle.detect(frame_img)
        
            #tracking
            tracker.update(centers, frameN)
            
        for trackN in range(0, len(tracker.tracks)):
            tracker.completeTracks.append(tracker.tracks[trackN])
                
            ######################## run tracklinking ##############################
         # # rearrange the data into disctionary   
        
        data_tracks={}
        
        for trackN in range(0, len(tracker.completeTracks)):
            if len(tracker.completeTracks[trackN].trace)>duration_shreshold:
                data_tracks.update({tracker.completeTracks[trackN].track_id:{
                        'trackID':tracker.completeTracks[trackN].track_id,
                        'trace': tracker.completeTracks[trackN].trace,
                        'frames':tracker.completeTracks[trackN].trace_frame,
                        'skipped_frames': tracker.completeTracks[trackN].skipped_frames
                        }})
        print(data_tracks)   
        save_file = tk.filedialog.asksaveasfilename(filetypes = [("All files", "*.*")])
        save_movie(data_tracks, save_file) 
        
        with open('test.txt', 'w') as f:
            json.dump(data_tracks, f, ensure_ascii=False)
        
        
        

        
if __name__ == "__main__":
    root = tk.Tk()
    MainVisual(root)
    root.mainloop()