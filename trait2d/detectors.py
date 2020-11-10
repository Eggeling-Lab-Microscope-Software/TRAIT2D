'''
class for particle detection
'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from skimage.feature import peak_local_max # find local max on the image


class Detectors(object):
    """
    detectors class for particle detection and subpixel localisation
    """
    def __init__(self):
        """Initialise variables
        """

        self.img_sef=[]
        self.binary_sef=[]
        
        # parameters for approach
        #SEF
        self.c=0.8 # coef for the thresholding
        self.sigma=3. # max sigma for LOG     
        
        #thresholding
        self.min_distance=4 # minimum distance between two max after SEF
        self.threshold_rel=0.1 # min picl value in relation to the image
        
        self.expected_size=20 # expected size of the particle
        
    def sef(self, img, sigma, c):   
        '''
        spot enhancing filter
        '''
        img_filtered=img*np.ones(img.shape)
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
    
        return img_sef, img_sef_bin
    
    def radialsym_centre(self, img):
        '''
         Calculates the center of a 2D intensity distribution (calculation of radial symmetry centers)  
    
        '''
        
        def lsradialcenterfit(m, b, w):
            '''
            least squares solution to determine the radial symmetry center
            '''
            wm2p1=np.divide(w,(np.multiply(m,m)+1))
            sw=np.sum(wm2p1)
            smmw = np.sum(np.multiply(np.multiply(m,m),wm2p1))
            smw  = np.sum(np.multiply(m,wm2p1))
            smbw = np.sum(np.multiply(np.multiply(m,b),wm2p1))
            sbw  = np.sum(np.multiply(b,wm2p1))
            det = smw*smw - smmw*sw
            xc = (smbw*sw - smw*sbw)/det # relative to image center
            yc = (smbw*smw - smmw*sbw)/det # relative to image center
                
            return xc, yc
    
        # GRID
        #  number of grid points
        Ny, Nx = img.shape
        
        # for x
        val=int((Nx-1)/2.0-0.5)
        xm_onerow = np.asarray(range(-val,val+1))
        xm = np.ones((Nx-1,Nx-1))*xm_onerow
        
        # for y
        val=int((Ny-1)/2.0-0.5)
        ym_onerow = np.asarray(range(-val,val+1))
        ym = (np.ones((Ny-1,Ny-1))*ym_onerow).transpose()
    
        # derivate along 45-degree shidted coordinates
    
        dIdu = np.subtract(img[0:Nx-1, 1:Ny].astype(float),img[1:Nx, 0:Ny-1].astype(float))
        dIdv = np.subtract(img[0:Nx-1, 0:Ny-1].astype(float),img[1:Nx, 1:Ny].astype(float))
        
        
        #smoothing
        filter_core=np.ones((3,3))/9
        fdu=sp.signal.convolve2d(dIdu,filter_core,  mode='same', boundary='fill', fillvalue=0)
        fdv=sp.signal.convolve2d(dIdv,filter_core,  mode='same', boundary='fill', fillvalue=0)
    
        dImag2=np.multiply(fdu,fdu)+np.multiply(fdv,fdv)
    
        #slope of the gradient
        m = np.divide(-(fdv + fdu), (fdu-fdv))
        
        # if some of values in m is NaN 
        m[np.isnan(m)]=np.divide(dIdv+dIdu, dIdu-dIdv)[np.isnan(m)]
        
        # if some of values in m is still NaN
        m[np.isnan(m)]=0 
        
        
        # if some of values in m  are inifinite
        
        m[np.isinf(m)]=10*np.max(m)
        
        #shortband b
        b = ym - m*xm
        
        #weighting
        sdI2=np.sum(dImag2)
        
        xcentroid = np.sum(np.multiply(dImag2, xm))/sdI2
        ycentroid = np.sum(np.multiply(dImag2, ym))/sdI2
        w=np.divide(dImag2, np.sqrt(np.multiply((xm-xcentroid),(xm-xcentroid))+np.multiply((ym-ycentroid),(ym-ycentroid))))
        
        # least square minimisation
        xc,yc=lsradialcenterfit(m, b, w)
        
        # output replated to upper left coordinate
        x=xc + (Nx+1)/2 # xc + (Nx+1)/2
        y=yc + (Ny+1)/2 # yc + (Ny+1)/2
        
        return x, y

    def detect(self, frame):
        '''
        detect vesicles
        '''

        # Spot enhancing filter   
        self.img_sef, self.binary_sef=self.sef(frame, self.sigma, self.c)
        
        # find local maximum
        peaks_coor=peak_local_max(self.img_sef, min_distance=self.min_distance, threshold_rel=self.threshold_rel) # min distance between peaks and threshold_rel - min value of the peak - in relation to the max value


        coordinates=[]  
        for point in peaks_coor:
            
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
            
            # define ROI coordinates
            
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
            
            # radial symmetry centers
            x,y=self.radialsym_centre(data)
            
            # check that the centre is inside of the spot            
            if y<self.expected_size and x<self.expected_size and y>=0 and x>=0:               
                coordinates.append([x+int(point[0]-self.expected_size/2),y+int(point[1]-self.expected_size/2)])

        return coordinates
