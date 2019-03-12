'''
    Cargo detector
    Python Version    : 3.6
'''

# Import python libraries
import numpy as np

import scipy as sp
import matplotlib.pyplot as plt
import skimage
from skimage import exposure, filters # to import file

from skimage.feature import peak_local_max # find local max on the image





class Detectors(object):
    """
    Detectors class to detect objects in video frame
    """
    def __init__(self):
        """Initialize variables
        """

        self.img_mssef=[]
        self.binary_mssef=[]
        
        # parameters for approach
        #MSSEF
        self.c=0.8 #0.01 # coef for the thresholding
        self.k_max=20 # end of  the iteration
        self.k_min=1 # start of the iteration
        self.sigma_min=0.1 # min sigma for LOG
        self.sigma_max=3. # max sigma for LOG     
        
        #thresholding
        self.min_distance=4 # minimum distance between two max after MSSEF
        self.threshold_rel=0.1 # min picl value in relation to the image
        
        self.int_size=3 # define number of pixels near centre for the thresholding calculation
        self.box_size=32 # bounding box size for detection
        
        
    def sef(self, img, img_sef_bin_prev, sigma, c, print_val=0):   
        '''
        spot enhancing filter
        '''
    #function to calculate spot-enhancing filter for a single scale
        
        img_filtered=img*img_sef_bin_prev # multiply image with the binary from the pervious iteration
        img_sef1=sp.ndimage.gaussian_laplace(img_filtered, sigma) # calculate laplacian of gaussian
        img_sef1=abs(img_sef1-np.abs(np.max(img_sef1))) # remove negative values keeping the proportion b/w pixels

    # thresholding
        th=np.mean(img_sef1)+c*np.std(img_sef1) # calculate threshold value
        img_sef=np.copy(img_sef1) # copy the image
        img_sef[img_sef<th]=0 # thresholding 
    
        # create a binary mask for the next step
        img_sef_bin = np.copy(img_sef)
        img_sef_bin[img_sef_bin<th]=0
        img_sef_bin[img_sef_bin>=th]=1
    
        # plot the image if print_val==1
        if print_val==1:
            fig = plt.figure()
            plt.gray()
            ax1=fig.add_subplot(411)
            ax2=fig.add_subplot(412)
            ax3=fig.add_subplot(413)
            ax4=fig.add_subplot(414)
            ax1.imshow(img)
            ax2.imshow(img_sef1)
            ax3.imshow(img_sef_bin)
            ax4.imshow(img_sef)
            plt.show()
    
        return img_sef, img_sef_bin
    
    def mssef(self, img, c, k_max, k_min, sigma_min, sigma_max):
        '''
        Multi-scale spot enhancing filter
        '''
#### the code is based on the paper "Tracking virus particles in fluorescence microscopy images using two-step multi-frame association"  Jaiswal,Godinez, Eils, Lehmann, Rohr 2015 ###
        
        img_bin=np.ones(img.shape) # original array
        N=k_max-k_min # number of steps
    
    # Multi-scale spot-enhancing filter loop
        for k in range(k_min, k_max):
    #        print ('sigma: ', sigma)
            sigma=sigma_max-(k-1)*(sigma_max-sigma_min)/(N-1) #assign sigma
            result, img_sef_bin=self.sef(img, img_bin, sigma, c, print_val=0) # SEF for a single scale
            img_bin=img_sef_bin
            
        return result, img_sef_bin
    
    
    def img_segmentation(self, img_segment, int_size, box_size, lm, printVal=False):
        '''
        the function segments the image based on the thresholding and watershed segmentation
        the only center part of the segmented part is taked into account. The region is defined based on the segmentation
        '''

        dict_lm={}
        dict_lm.update({'local_max': lm})

    # calculate threshold based on the centre
        threshold=np.mean(img_segment[int(box_size/2-int_size):int(box_size/2+int_size), int(box_size/2-int_size):int(box_size/2+int_size)])*0.75
    #    thresholding to get the mask
        mask=np.zeros(np.shape(img_segment))
        mask[img_segment>threshold]=1
    
        # separate the objects in image
    ## Generate the markers as local maxima of the distance to the background
        distance = sp.ndimage.distance_transform_edt(mask)
        local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=mask)
        markers = sp.ndimage.label(local_maxi)[0]
    
        # segment the mask
        segment = skimage.morphology.watershed(-distance, markers, mask=mask)
       
    # save the segment which is only in the centre
        val=segment[int(box_size/2), int(box_size/2)]
        segment[segment!=val]=0
        segment[segment==val]=1
        
        # finding bounding box:
        region_val = np.argwhere(segment==1)
    
       #place the segment on the final image 
        dict_lm.update({'xmax': region_val[:,0].max()+lm[0]-box_size/2+1})
        dict_lm.update({'xmin': region_val[:,0].min()+lm[0]-box_size/2-1})
        dict_lm.update({'ymax': region_val[:,1].max()+lm[1]-box_size/2+1})
        dict_lm.update({'ymin': region_val[:,1].min()+lm[1]-box_size/2-1})
        x_centre=(dict_lm['xmax']-dict_lm['xmin'])/2+dict_lm['xmin']
        y_centre=(dict_lm['ymax']-dict_lm['ymin'])/2+dict_lm['ymin']
       
        new_centre=[x_centre, y_centre]
        dict_lm.update({'local_max': new_centre})
 #   mean_reg=np.mean(img_segment[region_val[:,0].min():region_val[:,0].max()+1, region_val[:,1].min():region_val[:,1].max()+1])
        if printVal==True:
            plt.figure()
            plt.subplot(141)
            plt.imshow(img_segment, cmap='gray')
            plt.title('original', fontsize='large')
            plt.subplot(142)
            plt.imshow(mask, cmap='gray')
            plt.title('mask', fontsize='large')
            plt.subplot(143)
            plt.imshow(segment, cmap='gray')
            plt.title('segmented', fontsize='large')
           # segment[region_val[:,0].min():region_val[:,0].max()+1, region_val[:,1].min():region_val[:,1].max()+1]=5
            plt.subplot(144)
            plt.imshow(segment, cmap='gray')
            plt.title('new region', fontsize='large')
            plt.show()  
#    input("Press Enter to continue...")
#        print("old: ", lm)          
#        print("bounding box new: ", dict_lm)
#        print() 
        return dict_lm, segment # return the image and the dictionary with the local_max and refined region
    


    def detect(self, frame):
        """
        Detect vesicles
        """

        # Convert BGR to GRAY
        gray = frame

            # MSSEF

#        self.img_mssef, self.binary_mssef=self.mssef(gray, self.c, self.k_max, self.k_min, self.sigma_min, self.sigma_max)
    
        self.img_mssef, self.binary_mssef=self.sef(gray, np.ones(gray.shape), self.sigma_max, self.c)       #img, img_sef_bin_prev, sigma, c
        # 3. find local maximum in the 
        peaks_coor=peak_local_max(self.img_mssef, min_distance=self.min_distance, threshold_rel=self.threshold_rel) # min distance between peaks and threshold_rel - min value of the peak - in relation to the max value

        # remove the area where the membrane is
        coordinates=[]  
        for point in peaks_coor:
            new_point=[int(point[0]),int(point[1])]
            coordinates.append(new_point)


        return coordinates
