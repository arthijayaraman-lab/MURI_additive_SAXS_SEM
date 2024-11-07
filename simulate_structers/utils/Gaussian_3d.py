import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import cv2


class Gaussian_3d(nn.Module):
    """
    Apply gaussian smoothing on a 3d tensor. 
    """
    def __init__(self, channels, kernel_size, sigma, device, dim=3):
        super(Gaussian_3d, self).__init__()

        kernel = cv2.getGaussianKernel(kernel_size[0], sigma[0])
        self.dim = dim 
        self.device = device
        #kernel = np.einsum('ij,k->ijk', (kernel @ kernel.T).copy(), kernel[:,0].copy()).copy() 
        kernel_1d = torch.from_numpy(kernel[:,0].astype(np.float32))
        
        self.kernel_3d=[]
        for i in range(dim):
            axis_rep = [1]*5
            axis_rep[2+i]=kernel.shape[0]
            self.kernel_3d.append(kernel_1d.view(tuple(axis_rep)).to(device))
        
        #kernel = torch.stack(kernel_3d, dim=0)
        
        #self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        out_shape = input.shape[-3:]

        for i in range(self.dim):
            input = self.conv(input, weight=self.kernel_3d[i], groups=self.groups, padding=0)

        return F.interpolate(input, size=out_shape, mode="trilinear")
        

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using - ", device)
    k_size = [15]*3
    sigma = [12]*3
    shape = [1,1,300,300,300]
    im = np.random.random(shape).astype(np.float32)
    smoothing = Gaussian_3d( channels=1, kernel_size=k_size, sigma=sigma, device = device, dim=3).to(device)
    inp = torch.from_numpy(im.astype(np.float32)).to(device)
    output = smoothing(inp).cpu().detach().numpy()
    print("Completed")