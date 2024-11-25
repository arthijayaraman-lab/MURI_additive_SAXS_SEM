import numpy as np 
from utils.utils import *
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import math 
from scipy.spatial import KDTree
import numpy as np
import time 
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed

import numpy as np
from scipy.spatial import KDTree
from joblib import Parallel, delayed
from utils.utils import *

def compute_pair_wise_dist(fcc_array, grain_array, grain_no=0, threshold=1, num_jobs=17):

    # Filter fcc_array based on min and max bounds with threshold
    for i in range(3):
        mask = fcc_array[:, i] > (np.min(grain_array, axis=0)[i] - threshold)
        fcc_array = fcc_array[mask]

    for i in range(3):
        mask = fcc_array[:, i] < (np.max(grain_array, axis=0)[i] + threshold)
        fcc_array = fcc_array[mask]
    
    # Build KDTree for the grain_array
    tree = KDTree(grain_array)

    # Function to process a batch
    def query_batch(batch):
        return tree.query_ball_point(batch, threshold)

    # Calculate the size of each batch based on the number of jobs
    total_points = fcc_array.shape[0]
    batch_size = total_points // num_jobs + (total_points % num_jobs > 0)  # Ensure all points are included

    # Create batches based on the number of jobs
    batches = [fcc_array[i:i + batch_size] for i in range(0, total_points, batch_size)]
    
    # Run the queries in parallel using joblib
    results = Parallel(n_jobs=num_jobs)(delayed(query_batch)(batch) for batch in batches)

    # Collect all indices into a single list
    all_indices = [idx for sublist in results for idx in sublist]

    # Create mask based on the presence of neighbors
    mask = np.array([len(idx) > 0 for idx in all_indices])

    # Filtered points from fcc_array
    filtered_arr = fcc_array[mask]
    
    # Add the grain_no column if there are any filtered points
    if filtered_arr.size > 0:  # Ensure non-empty array before stacking
        filtered_arr = np.hstack((filtered_arr, np.full((filtered_arr.shape[0], 1), grain_no)))

    return filtered_arr
"""

def compute_pair_wise_dist(fcc_array, grain_array, grain_no=0, threshold=1, batch_size=100000):
    # Filter fcc_array based on min and max bounds with threshold
    for i in range(3):
        mask = fcc_array[:, i] > (np.min(grain_array, axis=0)[i] - threshold)
        fcc_array = fcc_array[mask]

    for i in range(3):
        mask = fcc_array[:, i] < (np.max(grain_array, axis=0)[i] + threshold)
        fcc_array = fcc_array[mask]
    
    # Build KDTree for the grain_array
    tree = KDTree(grain_array)

    # Function to process batches
    def query_batch(batch):
        return tree.query_ball_point(batch, threshold)

    print(fcc_array.shape[0])
    # Prepare to run queries in parallel
    indices_list = []
    with ThreadPoolExecutor() as executor:
        # Create batches and submit them to the thread pool
        for start in range(0, fcc_array.shape[0], batch_size):
            end = min(start + batch_size, fcc_array.shape[0])
            batch = fcc_array[start:end]
            indices_list.append(executor.submit(query_batch, batch))

    # Collect results
    all_indices = []
    for future in indices_list:
        all_indices.extend(future.result())

    # Create mask based on the presence of neighbors
    mask = np.array([len(idx) > 0 for idx in all_indices])

    # Filtered points from fcc_array
    filtered_arr = fcc_array[mask]
    
    # Add the grain_no column if there are any filtered points
    if filtered_arr.size > 0:  # Ensure non-empty array before stacking
        filtered_arr = np.hstack((filtered_arr, np.full((filtered_arr.shape[0], 1), grain_no)))

    return filtered_arr
"""


"""
def compute_pair_wise_dist(fcc_array, grain_array, grain_no=0, threshold=1):
    for i in range(3):
        mask = fcc_array[:, i]>(np.min(grain_array, axis = 0)[i]-threshold)
        fcc_array = fcc_array[mask]

    for i in range(3):
        mask = fcc_array[:, i]<(np.max(grain_array, axis = 0)[i]+threshold)
        fcc_array = fcc_array[mask]
    

    tree = KDTree(grain_array)

    # Can be made more effecient by multi threading currently uses only one thread  
    indices = tree.query_ball_point(fcc_array, threshold)
    
    mask = np.array([len(idx) > 0 for idx in indices])
    
    filtered_arr = fcc_array[mask]
    
    if filtered_arr.size > 0:  # Ensure non-empty array before reshaping
        filtered_arr = np.hstack((filtered_arr, np.full((filtered_arr.shape[0], 1), grain_no)))
    
    return filtered_arr
"""

def plot_scatter_points(coords, n):
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.set_xlim([0, n])
    #ax.set_ylim([0, n])
    #ax.set_zlim([0, n])
    
    if int(coords.shape[1])>3:
        print(np.unique(coords.T[3]))
        ax.scatter(coords.T[0], coords.T[1], coords.T[2], c=plt.cm.viridis(coords.T[3]/np.max(coords.T[3])), s= 5)
    else:
        ax.scatter(coords.T[0], coords.T[1], coords.T[2], c=[0]*int(coords.shape[0]), s= 55)
        
    plt.show()
    return None

"""
def compute_pair_wise_dist(fcc_array, grain_array, grain_no=0, threshold=1):
    Checks if any point in array1 is within a threshold distance of any point in array2.

    Parameters:
        array1 (np.ndarray): A 2D NumPy array of shape (N, D), where N is the number of points and D is the dimension.
        array2 (np.ndarray): A 2D NumPy array of shape (M, D), where M is the number of points and D is the dimension.
        threshold (float): The distance threshold.
    
    Returns:
        bool: True if any point in array1 is within the threshold of any point in array2, False otherwise.
    # Compute the squared distances between all points in array1 and array2
    distances = np.linalg.norm(fcc_array[:, np.newaxis, :] - grain_array[np.newaxis, :, :], axis=-1)  

    mask = np.any(distances <= threshold, axis=1) 
    filtered_arr = fcc_array[mask]
    
    filtered_arr = np.append(filtered_arr, np.array([grain_no]*int(filtered_arr.shape[0])).reshape(int(filtered_arr.shape[0]),1), axis = 1)
    
    return  filtered_arr #np.any(distances <= threshold)
"""

if __name__ == "__main__":
    """
    make grains tested and OK!
    """
    target_size=100
    n_grains = 20

    #st = time.time()
    grain_loc_cube = make_grain_voxels(n_grains,target_size).astype(np.uint16)
    #print("time for graim_voxels", time.time()-st)
    closest_point_index = grain_loc_cube.ravel()

    """
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
        #st = time.time()
        fcc_coords_affine = make_fcc_affine_float_coords(out_size, rotation_angles= [np.random.choice(rot_angl), np.random.choice(rot_angl), np.random.choice(rot_angl)])
        #print("time for make_fcc_affine_float_coords", time.time()-st)
        fcc_coords_affine = fcc_coords_affine.astype(np.float32)
        coords_grain_loc = np.argwhere(grain_loc_cube==i)
        
        st = time.time()
        grain_lattice_coords = compute_pair_wise_dist(fcc_coords_affine, coords_grain_loc, grain_no=i*10, threshold=1)   
        print("time for compute_pair_wise_dist", time.time()-st)
        
        print(grain_lattice_coords.shape)
        print(grain_lattice_coords[0])
        out = np.append(out, grain_lattice_coords, axis=0)
    
    del grain_lattice_coords, fcc_coords_affine

    #plot_scatter_points(out, out_size)
    out = out[:,:-1].copy()
    save_structure_to_txt(out, "grains_100_20_fcc_affine.txt")
    print("Save Complete")
    #closest_point_index = np.argwhere(output) 
    
    #plot_scatter_points(closest_point_index, out_size)
    #"""