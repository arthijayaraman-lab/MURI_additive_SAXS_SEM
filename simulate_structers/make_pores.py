
import numpy as np 
import torch
from utils.Gaussian_3d import Gaussian_3d
from utils.utils import * 
import scipy.ndimage as spim
from scipy.special import erfc
import cv2 

class make_pores():
    """
    Logic adapted from porespy.simulate.blob. and made effecient to handle large tensors. 
    """
    def __init__(self, shape, blobiness = 5, porosity = 0.05, device=torch.device("cpu"), compare = False):
        self.shape = shape
        self.blobiness = blobiness
        self.porosity = porosity
        self.compare = compare
        self.device = device

    def simulate_pores(self):
        im = np.random.random(self.shape).astype(np.float32)
        self.sigma = np.array([np.mean(self.shape)/(40*self.blobiness)]*len(self.shape))
        self.r = np.round(4*self.sigma)
        self.K = (2*self.r+1).astype(np.uint16)
        Gaussian_fun = Gaussian_3d( channels=1, kernel_size=tuple(self.K), sigma=self.sigma, device = self.device, dim=3).to(self.device)
        inp = torch.from_numpy(np.expand_dims(np.expand_dims(im, axis=0), axis=0).astype(np.float32)).to(self.device)
        blur = Gaussian_fun(inp).cpu().detach().numpy()
        
        if self.compare:
            im_spim = spim.gaussian_filter(im, sigma=self.sigma)

        norm_blur = norm_to_uniform(blur, scale=[0, 1])
        norm_blur = norm_blur < self.porosity

        if self.compare:
            norm_im_spim = norm_to_uniform(im_spim, scale=[0, 1])
            norm_im_spim = norm_im_spim < self.porosity

        norm_blur = (norm_blur[0,0,:,:,:])
        #norm_blur_coord = np.argwhere(norm_blur).astype(np.uint16)
        return norm_blur