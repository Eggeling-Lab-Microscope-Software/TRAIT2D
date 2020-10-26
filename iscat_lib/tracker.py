#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Class for detection linking
'''


import numpy as np
from scipy.optimize import linear_sum_assignment

class Track(object):

    def __init__(self, first_point, first_frame, trackIdCount):

        self.track_id = trackIdCount  # track ID
        self.trace_frame = [first_frame]  # frame
        self.skipped_frames = 0  # number of frames skipped undetected
        self.trace = [first_point]  # trace path


class Tracker(object):
    """
    links detections frame to frame
    
    """

    def __init__(self, dist_thresh=30, max_frames_to_skip=20, trackIdCount=0):

        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.tracks = []
        self.trackIdCount = trackIdCount
        self.completeTracks=[]
        
        print(" - - - - - tracker: - - - - - - - ")
        print("dist_thresh ", dist_thresh)
        print("max_frames_to_skip ", max_frames_to_skip)
        
    def cost_calculation(self, detections):
        '''
        calculates cost matrix based on the distance
        '''
        N = len(self.tracks)
        M = len(detections)
        cost = np.zeros((N, M))   # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(detections)):
                try:
                    diff = np.array(self.tracks[i].trace[len(self.tracks[i].trace)-1]) - np.array(detections[j])
                    distance = np.sqrt((diff[0])**2 +(diff[1])**2 )

                    cost[i][j] = distance
                except:
                    pass
        cost_array=np.asarray(cost)
        cost_array[cost_array>self.dist_thresh]=10000
        cost=cost_array.tolist()
        return cost
    
    def assignDetectionToTracks(self, cost):
        '''
        assignment based on Hungerian Algorithm
        '''

        N = len(self.tracks)
        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]
        
        return assignment
    
    
    def update(self, detections, frameN):
        '''
        main linking function
        '''

        # create tracks if no tracks  found
        if (len(self.tracks) == 0):
            for i in range(len(detections)):
                track = Track(detections[i], frameN, self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)

        # tracking the targets if there were tracks before
        else:
            
            # Calculate cost using sum of square distance between predicted vs detected centroids
            cost=self.cost_calculation(detections)

            # assigning detection to tracks
            assignment=self.assignDetectionToTracks(cost)

 
            # add the position to the assigned tracks and detect annasigned tracks
            un_assigned_tracks = []
            
            for i in range(len(assignment)):
                if (assignment[i] != -1):
                    # check with the cost distance threshold and unassign if cost is high
                    if (cost[i][assignment[i]] > self.dist_thresh):
                        assignment[i] = -1
                        un_assigned_tracks.append(i)
                        self.tracks[i].skipped_frames += 1

                    else: # add the detection to the track
                        self.tracks[i].trace.append(detections[assignment[i]])
                        self.tracks[i].trace_frame.append(frameN)
                        self.tracks[i].skipped_frames =0
                        
                else:
                    un_assigned_tracks.append(i)
                    self.tracks[i].skipped_frames += 1                

                        
            # Unnasigned detections
            un_assigned_detects = []
            for i_det in range(len(detections)):
                    if i_det not in assignment:
                        un_assigned_detects.append(i_det)
    
            # Start new tracks
            if(len(un_assigned_detects) != 0):
                for i in range(len(un_assigned_detects)):
                    track = Track(detections[un_assigned_detects[i]], frameN,
                                  self.trackIdCount)              


                    self.trackIdCount += 1
                    self.tracks.append(track)
                        
                    
            del_tracks = [] # list of tracks to delete
            
        #remove tracks which have too many skipped frames
            for i in range(len(self.tracks)):
                if (self.tracks[i].skipped_frames > self.max_frames_to_skip):
                    del_tracks.append(i)        
        
        
        
        # delete track
     
            if len(del_tracks) > 0:   

                val_compensate_for_del=0
                for id in del_tracks:
                    new_id=id-val_compensate_for_del

                    self.completeTracks.append(self.tracks[new_id])
                    del self.tracks[new_id]
                    val_compensate_for_del+=1


                      
        print("number of detections: ", len(detections))
        print("number of tracks: ", len(self.tracks))
        print() 

