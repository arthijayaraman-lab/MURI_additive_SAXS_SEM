import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as spim
from scipy.special import erfc
import math
import warnings

def save_structure_to_txt(coords, filename):
    np.savetxt(filename, coords, fmt='%d', comments='')
    
def make_fcc_lattice(n):
    lattice = np.zeros((n, n, n), dtype=bool)
    i, j, k = np.indices((n, n, n), dtype=np.uint16)
    lattice[(i + j + k) % 2 == 0] = True
    del i,j,k
    return lattice 

def plot_structure(coords, title, color, gaps=None, legend_items=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[:,0], coords[:,1], coords[:,2], color=color, s=800)
    if gaps is not None:
        for gap in gaps:
            ax.scatter(gap[0], gap[1], gap[2], color=gap[3], s=800)  # Increasing the size of removed spheres for better visibility
    ax.set_title(title)
    plt.show()
    
def norm_to_uniform(im, scale=None):
    """
    Adapted from porespy
    Take an image with normally distributed greyscale values and convert it to
    a uniform (i.e. flat) distribution.

    Parameters
    ----------
    im : ndarray
        The image containing the normally distributed scalar field
    scale : [low, high]
        A list or array indicating the lower and upper bounds for the new
        randomly distributed data.  The default is ``None``, which uses the
        ``max`` and ``min`` of the original image as the the lower and upper
        bounds, but another common option might be [0, 1].

    Returns
    -------
    image : ndarray
        A copy of ``im`` with uniformly distributed greyscale values spanning
        the specified range, if given.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/norm_to_uniform.html>`_
    to view online example.

    """
    if scale is None:
        scale = [im.min(), im.max()]
    im = (im - np.mean(im)) / np.std(im)
    im = 1 / 2 * erfc(-im / np.sqrt(2))
    im = (im - im.min()) / (im.max() - im.min())
    im = im * (scale[1] - scale[0]) + scale[0]
    return im
