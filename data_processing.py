#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iSCAT data - detection and tracking

@author: mariaa
"""

import sys
from trait2d.detectors import Detectors
import skimage
from skimage import io
import matplotlib.pyplot as plt
from trait2d.tracker import Tracker
import json
import numpy as np
import scipy as sp
import cv2

def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = sp.optimize.leastsq(errorfunction, params)
    return p, success



def track_to_frame(data, movie):
    # change data arrangment from tracks to frames
    track_data_framed={}
    track_data_framed.update({'frames':[]})

    for n_frame in range(0, movie.shape[0]):


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


def save_movie(movie, tracks, save_file):

    track_data_framed=track_to_frame(tracks, movie)

    final_img_set = np.zeros((movie.shape[0], movie.shape[1], movie.shape[2], 3))

    for frameN in range(0, movie.shape[0]):

        plot_info=track_data_framed['frames'][frameN]['tracks']
        frame_img=movie[frameN,:,:]
        # Make a colour image frame
        orig_frame = np.zeros((movie.shape[1], movie.shape[2], 3))

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

# read the file
img_filename='/home/mariaa/Eggeling_lab/iSCAT_project/data/20_nm_gold_nanoparticles_on_cell_surface/event52_MF_avF.tif'
#img_filename='/home/mariaa/Eggeling_lab/iSCAT_project/data/flavidin_on_lipid_bilayer/event4_MF_avF.tif'

filename_tracks='/home/mariaa/Eggeling_lab/iSCAT_project/code/results/tracks_flavidin_on_lipid_bilayer.txt'
movie=skimage.io.imread(img_filename)

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
max_track_length=movie.shape[0]

tracker = Tracker(distance_threshold, max_skip_frame, max_track_length, 0)

# setting of the script:
duration_shreshold=50

for frameN in range(0, movie.shape[0]):
    print('frame', frameN)
    #detection

    frame_img=movie[frameN,:,:]
    centers=detect_particle.detect(frame_img)
    new_centers=[]
    expected_size=24
    # fitting gaussian
    for point in centers:
        # extract subarray
        data=frame_img[int(point[0]-expected_size/2):int(point[0]+expected_size/2), int(point[1]-expected_size/2):int(point[1]+expected_size/2)]

    # plot results to check
    plt.figure()

    plt.subplot(121)
    plt.imshow(frame_img, cmap='gray')
    plt.title('original', fontsize='large')

    plt.subplot(122)
    plt.imshow(frame_img, cmap='gray')
    if len(centers)>0:
        for point in centers:
            plt.plot(point[1], point[0], '*r')
        for point in new_centers:
            plt.plot(point[1], point[0], '*g')

    plt.title('coordinates', fontsize='large')

#    plt.show()

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
save_movie(movie, data_tracks, 'test.tif')

with open(filename_tracks, 'w') as f:
    json.dump(data_tracks, f, ensure_ascii=False)
