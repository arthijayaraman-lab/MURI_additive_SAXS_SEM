import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as spim
from scipy.special import erfc
import math
import warnings

from scipy.spatial import Voronoi
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.nn import functional as F
from scipy.spatial.transform import Rotation as R
import math 
import time 
import torch


def make_grain_voxels(n, out_size):
    """
    Input-
    out_size(int)   - Length of a side in cube 
    n(int)          - Number of grains in cube

    Return-
    cube_index(3d- ndarray(np.uint8)) scaled from 1-n

    """
    points = np.random.rand(n, 3)

    #vor = Voronoi(points)
    
    init_cube_dim = out_size//2
    x = np.linspace(0, 1, init_cube_dim)
    y = np.linspace(0, 1, init_cube_dim)
    z = np.linspace(0, 1, init_cube_dim)
    X, Y, Z = np.meshgrid(x, y, z)

    query_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    del X, Y, Z, x, y, z    

    tree = cKDTree(points)
    closest_point_index = tree.query(query_points)[1]
        
    target_size = out_size

    cube_recon = np.expand_dims(np.expand_dims(closest_point_index.reshape(init_cube_dim,init_cube_dim,init_cube_dim).astype(np.float32), axis=0), axis=0)

    cube_recon_torch = torch.from_numpy(cube_recon)
    cube_interpolated = F.interpolate(cube_recon_torch, size=[target_size]*3, mode="nearest")  # trilinear
    closest_point_index = cube_interpolated.numpy().flatten()

    del cube_interpolated, cube_recon_torch, query_points, tree

    closest_point_index = np.round(closest_point_index).astype(np.uint8)
    cube_index = closest_point_index.reshape([target_size]*3)
    cube_index+=1
    return cube_index

def transform_coordinates(coordinates, rotation_matrix):
    transformed_coords = coordinates@rotation_matrix.T
    return transformed_coords
    
def make_fcc_affine(n, rotation_angles=[0,0,0]):
    """
    Affine transformation of Fcc coordinates w/o translation 
    Input-
    rotation_angles([int,int,int])   - roation angles in afine transform 
    n(int)          - Dim of cube

    Return-
    transformed_coordinates(3d- ndarray(np.uint16)) np.uint16 might change to float 
    """
    lattice = make_fcc_lattice(n)
    fcc_coord = get_coords_from_3d(lattice)  

    # center coordinate at origin
    center = (n - 1) / 2 
    fcc_coord_centered = fcc_coord - center


    rotation_matrix = R.from_euler('xyz', rotation_angles, degrees=True).as_matrix().astype(np.float16)
    transformed_coordinates = transform_coordinates(fcc_coord_centered, rotation_matrix)
    
    # Translate back to the center and round
    transformed_coordinates += center
    transformed_coordinates = np.round(transformed_coordinates).astype(int)
    
    # Clip the coordinates to ensure they stay within bounds
    transformed_coordinates = np.clip(transformed_coordinates, 0, n - 1)
    
    # Create a new lattice with the rotated coordinates
    rotated_lattice = np.zeros_like(lattice, dtype=bool)
    rotated_lattice[transformed_coordinates[:, 0], transformed_coordinates[:, 1], transformed_coordinates[:, 2]] = True

    return transformed_coordinates, rotated_lattice

def delete_rows_exceeding_value(arr, value):
    mask = np.all(np.abs(arr) <= value, axis=1)  # Create a mask where all elements in a row are <= value
    filtered_arr = arr[mask]  # Use the mask to filter rows
    return filtered_arr


def delete_rows_suseding_value(arr, value):
    """
    Deletes rows from the array where any element in the row is greater than the specified value.

    Parameters:
        arr (np.ndarray): Input 2D NumPy array.
        value (float or int): Threshold value.
    
    Returns:
        np.ndarray: Array with rows removed.
    """
    mask = np.all(np.abs(arr) >= value, axis=1)  # Create a mask where all elements in a row are <= value
    filtered_arr = arr[mask]  # Use the mask to filter rows
    return filtered_arr
def delete_rows_having_value(arr, value):
    """
    Deletes rows from the array where any element in the row is greater than the specified value.

    Parameters:
        arr (np.ndarray): Input 2D NumPy array.
        value (float or int): Threshold value.
    
    Returns:
        np.ndarray: Array with rows removed.
    """
    mask = np.all(np.abs(arr) != value, axis=1)  # Create a mask where all elements in a row are <= value
    filtered_arr = arr[mask]  # Use the mask to filter rows
    return filtered_arr

def delete_rows_bounds(arr, lb, ub):
    """
    Deletes rows from the array where any element in the row is greater than the specified value.

    Parameters:
        arr (np.ndarray): Input 2D NumPy array.
        value (float or int): Threshold value.
    
    Returns:
        np.ndarray: Array with rows removed.
    """
    mask = np.all((arr >= lb) & (arr <= ub), axis=1)
    filtered_arr = arr[mask]  # Use the mask to filter rows
    return filtered_arr


def make_fcc_affine_float_coords(n, fcc_coord, rotation_angles=[0,0,0]):
    """
    Affine transformation of Fcc coordinates w/o translation 
    Input-
    rotation_angles([int,int,int])   - roation angles in afine transform 
    n(int)          - Dim of cube

    Return-
    transformed_coordinates(3d- ndarray(np.uint16)) np.uint16 might change to float 
    """
    n_orig = n 
    n=int(n*2)#(2/(2**0.5)))    

    # center coordinate at origin
    center = (n - 1) / 2 
    fcc_coord_centered = fcc_coord - center

    rotation_matrix = R.from_euler('xyz', rotation_angles, degrees=True).as_matrix().astype(np.float32)

    st = time.time()
    transformed_coordinates = transform_coordinates(fcc_coord_centered, rotation_matrix)
    #print("transfor c00rds", time.time()-st)
    st = time.time()

    transformed_coordinates = delete_rows_exceeding_value(transformed_coordinates, n//4)
    # Translate back to the center and round
    #new_center = ((n//(2/(2**0.5))) - 1) / 2 
    new_center = ((n//2) - 1) / 2 
    transformed_coordinates += new_center
    #transformed_coordinates = np.round(transformed_coordinates).astype(int)
    
    # Clip the coordinates to ensure they stay within bounds
    #transformed_coordinates = np.clip(transformed_coordinates, 0, (n_orig) - 1)
    
    # Create a new lattice with the rotated coordinates
    #rotated_lattice = np.zeros_like(lattice, dtype=bool)
    #rotated_lattice[transformed_coordinates[:, 0], transformed_coordinates[:, 1], transformed_coordinates[:, 2]] = True
    #transformed_coordinates = delete_rows_exceeding_value(transformed_coordinates, (n_orig) - 1)
    #transformed_coordinates = delete_rows_suseding_value(transformed_coordinates, 0)
    transformed_coordinates = delete_rows_bounds(transformed_coordinates, 0, (n_orig) - 1)

    #print("aux affine", time.time()-st)
    return transformed_coordinates

def make_grains(n, out_size):
    """
    Input-
    out_size(int)   - Length of a side in cube 
    n(int)          - Number of grains in cube
    """
    grains_loc_cube = make_grain_voxels(n, out_size)
    rot_angl = [0,0,0]
    for i in range(n):
        coords_fcc_affine = make_fcc_affine(out_size, rotation_angles= [np.random.choice(rot_angl), np.random.choice(rot_angl), np.random.choice(rot_angl)]) 
        fcc_affine = np.zeros_like(grains_loc_cube).astype(bool)
        for j in range(coords_fcc_affine.shape[0]):
            fcc_affine[coords_fcc_affine[j]] =  True 
        np.logical_and(grains_loc_cube[grains_loc_cube==i], fcc_affine)

def get_coords_from_3d(lattice):
    coords = np.argwhere(lattice).astype(np.uint16)
    return coords
    
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
