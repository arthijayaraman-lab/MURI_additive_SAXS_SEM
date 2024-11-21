import numpy as np 
from utils.utils import *
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import math 

def plot_scatter_points(coords, n):
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.set_xlim([0, n])
    #ax.set_ylim([0, n])
    #ax.set_zlim([0, n])
    
    if int(coords.shape[1])>3:
        print(np.unique(coords.T[3]))
        ax.scatter(coords.T[0], coords.T[1], coords.T[2], c=plt.cm.viridis(coords.T[3]/np.max(coords.T[3])), s= 55)
    else:
        ax.scatter(coords.T[0], coords.T[1], coords.T[2], c=[0]*int(coords.shape[0]), s= 55)
        
    plt.show()
    return None

def compute_pair_wise_dist(fcc_array, grain_array, grain_no=0, threshold=1):
    """
    Checks if any point in array1 is within a threshold distance of any point in array2.

    Parameters:
        array1 (np.ndarray): A 2D NumPy array of shape (N, D), where N is the number of points and D is the dimension.
        array2 (np.ndarray): A 2D NumPy array of shape (M, D), where M is the number of points and D is the dimension.
        threshold (float): The distance threshold.
    
    Returns:
        bool: True if any point in array1 is within the threshold of any point in array2, False otherwise.
    """
    # Compute the squared distances between all points in array1 and array2
    distances = np.linalg.norm(fcc_array[:, np.newaxis, :] - grain_array[np.newaxis, :, :], axis=-1)  

    mask = np.any(distances <= threshold, axis=1) 
    filtered_arr = fcc_array[mask]
    
    filtered_arr = np.append(filtered_arr, np.array([grain_no]*int(filtered_arr.shape[0])).reshape(int(filtered_arr.shape[0]),1), axis = 1)
    
    return  filtered_arr #np.any(distances <= threshold)


if __name__ == "__main__":
    """
    make grains tested and OK!
    """
    target_size=10
    n_grains = 2
    grain_loc_cube = make_grain_voxels(n_grains,target_size).astype(np.uint16)
    closest_point_index = grain_loc_cube.ravel()

    colors = plt.cm.viridis(closest_point_index/np.max(closest_point_index))

    # Plotting
    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d')
    # Set plot limits to the unit cube

    x = np.linspace(0, 1, target_size)
    y = np.linspace(0, 1, target_size)
    z = np.linspace(0, 1, target_size)
    X, Y, Z = np.meshgrid(x, y, z)
    query_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    del X, Y, Z, x, y, z
 
    ax.scatter(query_points.T[0], query_points.T[1], query_points.T[2], c = colors, s= 55)
    plt.show()
    #"""

    output = np.zeros_like(grain_loc_cube).astype(bool) 
    
    out_size = target_size
    rot_angl = [100,30,60]
    
    out = np.empty((0, 4))

    for i in range(1, n_grains+1): 
        fcc_coords_affine = make_fcc_affine_float_coords(out_size, rotation_angles= [np.random.choice(rot_angl), np.random.choice(rot_angl), np.random.choice(rot_angl)])
        fcc_coords_affine = fcc_coords_affine.astype(np.float32)
        coords_grain_loc = np.argwhere(grain_loc_cube==i)
        grain_lattice_coords = compute_pair_wise_dist(fcc_coords_affine, coords_grain_loc, grain_no=i*10, threshold=1)   
        out = np.append(out, grain_lattice_coords, axis=0)
    
    plot_scatter_points(out, out_size)
        
    #closest_point_index = np.argwhere(output) 
    
    #plot_scatter_points(closest_point_index, out_size)
    #"""