
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
        im = np.expand_dims(np.expand_dims(np.random.random(self.shape).astype(np.float32), axis=0), axis=0).astype(np.float32)
        self.sigma = np.array([np.mean(self.shape)/(40*self.blobiness)]*len(self.shape))
        self.r = np.round(4*self.sigma)
        self.K = (2*self.r+1).astype(np.uint16)
        blur = np.zeros_like(im)

        for w_i in range(0,self.shape[0],500):
            for h_j in range(0,self.shape[1],500):
                for d_k in range(0,self.shape[2],500):
                    print(w_i, "-", h_j, "-", d_k)
                    Gaussian_fun = Gaussian_3d( channels=1, kernel_size=tuple(self.K), sigma=self.sigma, device = self.device, dim=3).to(self.device)
                    inp = torch.from_numpy(im[:, :, w_i:w_i+500, h_j:h_j+500, d_k:d_k+500]).to(self.device)
                    blur[0, 0, w_i:w_i+500, h_j:h_j+500, d_k:d_k+500] = Gaussian_fun(inp).cpu().detach().numpy()
                    torch.cuda.empty_cache()

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