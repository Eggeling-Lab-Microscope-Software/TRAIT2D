'''
    Cargo detector
    Python Version    : 3.6
'''

# Import python libraries
import numpy as np

import scipy as sp
import matplotlib.pyplot as plt
#from skimage import exposure, filters # to import file

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
        self.sigma=3. # max sigma for LOG     
        
        #thresholding
        self.min_distance=4 # minimum distance between two max after MSSEF
        self.threshold_rel=0.1 # min picl value in relation to the image
        
        self.expected_size=20 # expected size of the particle
        
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

    def gaussian(self, height, center_x, center_y, width_x, width_y):
        """Returns a gaussian function with the given parameters"""
        width_x = float(width_x)
        width_y = float(width_y)
        return lambda x,y: height*np.exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)
    
    def moments(self, data):
        '''
         Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution by calculating its
        moments 
        '''
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
    
    def fitgaussian(self, data):
        '''
        nonlinear leastsquare fit of gaussian parameters of a 2D distribution: 
        Returns (height, x, y, width_x, width_y)
        '''
         
        params = self.moments(data)
        errorfunction = lambda p: np.ravel(self.gaussian(*p)(*np.indices(data.shape)) -
                                     data)
        p, success = sp.optimize.leastsq(errorfunction, params)
        return p

    def detect(self, frame):
        '''
        Detect vesicles
        '''

            # MSSEF

#        self.img_mssef, self.binary_mssef=self.mssef(gray, self.c, self.k_max, self.k_min, self.sigma_min, self.sigma_max)
    
        self.img_mssef, self.binary_mssef=self.sef(frame, np.ones(frame.shape), self.sigma, self.c)       #img, img_sef_bin_prev, sigma, c
        # 3. find local maximum in the 
        peaks_coor=peak_local_max(self.img_mssef, min_distance=self.min_distance, threshold_rel=self.threshold_rel) # min distance between peaks and threshold_rel - min value of the peak - in relation to the max value

        # remove the area where the membrane is
        coordinates=[]  
        sigmas=[]
        for point in peaks_coor:
            
#            print("point ", point)
#            print("size ", frame.shape)
            data=np.zeros((self.expected_size,self.expected_size))
            
            #start point
            start_x=int(point[0]-self.expected_size/2)
            start_y=int(point[1]-self.expected_size/2)
            
            #end point
            end_x=int(point[0]+self.expected_size/2)
            end_y=int(point[1]+self.expected_size/2)
            
            x_0=0
            x_1=self.expected_size
            y_0=0
            y_1=self.expected_size
            
            #possible cases to avoid our of boudary case
            
            if start_x<0:
                start_x=0
                x_0=int(self.expected_size/2-point[0])
                
            if start_y<0:
                start_y=0
                y_0=int(self.expected_size/2-point[1]) 
                
            if end_x>frame.shape[0]:
                end_x=frame.shape[0]
                x_1=int(frame.shape[0]-point[0]+self.expected_size/2)

            if end_y>frame.shape[1]:
                end_y=frame.shape[1]
                y_1=int(frame.shape[1]-point[1]+self.expected_size/2)
            
            
            data[x_0:x_1,y_0:y_1]=frame[start_x:end_x, start_y:end_y]
            
            # nonlinear least square fitting
            params  = self.fitgaussian(data)
            (height, y, x, width_x, width_y) = params
#            print(params)
            
            # check that the centre is inside of the spot
            
            if y<self.expected_size and x<self.expected_size and y>=0 and x>=0:               
            # insurt another approach
                coordinates.append([x+int(point[0]-self.expected_size/2),y+int(point[1]-self.expected_size/2)])
                sigmas.append([width_x, width_y])

#                print("point: ", (x,y), "  ", [x+int(point[0]-self.expected_size/2),y+int(point[1]-self.expected_size/2)])
            
#            #plotting
#            fit = self.gaussian(*params)
#            plt.figure()
#            plt.imshow(data)
#            plt.contour(fit(*np.indices(data.shape)), cmap=plt.cm.copper)
#            plt.plot(x, y, '*r')
#            plt.show() 


        return coordinates, sigmas
