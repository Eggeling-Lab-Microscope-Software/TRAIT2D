'''
Tracker Using  Hungarian Algorithm
'''

# Import python libraries
import numpy as np
from scipy.optimize import linear_sum_assignment
from math import atan2,degrees

class Track(object):

    def __init__(self, first_point, first_frame, trackIdCount):

        self.track_id = trackIdCount  # identification of each track object
        self.trace_frame = [first_frame]  # predicted centroids (x,y)
        self.skipped_frames = 0  # number of frames skipped undetected
        self.trace = [first_point]  # trace path
        self.direction=720
    
    def trackAngle(self):
        if len(self.trace)>=2:
            if len(self.trace)>=5:
                p1=self.trace[-5]
            else:
                p1=self.trace[0]                
            
            p2=self.trace[-1]
            xDiff = p2[0] - p1[0]
            yDiff = p2[1] - p1[1]
            self.direction= int(degrees(atan2(yDiff, xDiff)))
        else:
            self.direction=720


class Tracker(object):

    def __init__(self, dist_thresh=30, max_frames_to_skip=5, max_trace_length=100,
                 trackIdCount=0):

        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.tracks = []
        self.trackIdCount = trackIdCount
        self.completeTracks=[]
        
        print(" - - - - - tracker: - - - - - - - ")
        print("dist_thresh ", dist_thresh)
        print("max_frames_to_skip ", max_frames_to_skip)
        print("max_trace_length ", max_trace_length)
        
    def cost_calculation(self, detections):
        N = len(self.tracks)
        M = len(detections)
        cost = np.zeros((N, M))   # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(detections)):
                try:
                    diff = np.array(self.tracks[i].trace[len(self.tracks[i].trace)-1]) - np.array(detections[j])
                    distance = np.sqrt((diff[0]*diff[0])**2 +(diff[1]*diff[1])**2 )
    #                   print("distance: ", distance)
                    cost[i][j] = distance
                except:
                    pass
        
        return cost
    
    def assignDetectionToTracks(self, cost):
        # based on Hungerian Algorithm
        N = len(self.tracks)
        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]
        
        return assignment
    
    
    def update(self, detections, frameN):

# Create tracks if no tracks  found
        if (len(self.tracks) == 0):
            for i in range(len(detections)):
                track = Track(detections[i], frameN, self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)

        else:
# tracking the targets if there were tracks before
            # Calculate cost using sum of square distance between predicted vs detected centroids
            cost=self.cost_calculation(detections)
    
            # Hungarian Algorithm assigning detection to tracks:
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
                        self.tracks[i].trackAngle()
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
#                    track.trace_frame.append(frameN)

                    self.trackIdCount += 1
                    self.tracks.append(track)
                        
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
     
            #  remove tracks, which has high skipped_frame value
            del_tracks = []
            for i in range(len(self.tracks)):
#                print("track ", i, " length: ", len(self.tracks[i].trace_frame))
                if ((self.tracks[i].trace_frame[-1]-self.tracks[i].trace_frame[0]) > self.max_trace_length):
                    del_tracks.append(i)
#                    print("track ", i,  " (", self.tracks[i].track_id, ") ", " will be deleted because of ", len(self.tracks[i].trace_frame), " frames length")

        #remove track which are longer than the max_length
            for i in range(len(self.tracks)):
                if (self.tracks[i].skipped_frames > self.max_frames_to_skip):
                    del_tracks.append(i)        
#                    print("track ", i,  " (", self.tracks[i].track_id, ") ", " will be deleted because of ", self.tracks[i].skipped_frames, " skipped frames")
        
        
        
     # when there are some tracks to delete:    
            if len(del_tracks) > 0:   
#                print(del_tracks)
                val_compensate_for_del=0
                for id in del_tracks:
                    new_id=id-val_compensate_for_del
#                    print("track ", new_id,  " (", self.tracks[new_id].track_id, ") ", " is deleted")
                    self.completeTracks.append(self.tracks[new_id])
                    del self.tracks[new_id]
                    val_compensate_for_del+=1



    
        #remove track which are lonfer than the max_length
                      
        print()         
        print("number of tracks: ", len(self.tracks))
        print("next track ID: ", self.trackIdCount)

