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
import random
import os
import numpy as np
from scipy.spatial import KDTree
from joblib import Parallel, delayed
from utils.utils import *
from make_pores import make_pores
import psutil
from tqdm import tqdm


"""
def testing_dist(fcc_array, grain_array, chunk_size=1000):
    min_bounds = np.min(grain_array, axis=0) - threshold
    max_bounds = np.max(grain_array, axis=0) + threshold

    # Create mask for fcc_array based on bounds
    mask = np.all((fcc_array >= min_bounds) & (fcc_array <= max_bounds), axis=1)
    fcc_array = fcc_array[mask]

    tree = KDTree(grain_array)

    # Initialize mask
    n_points = fcc_array.shape[0]
    mask = np.zeros(n_points, dtype=bool)

    # Process in chunks
    for i in range(0, n_points, chunk_size):
        batch = fcc_array[i:i + chunk_size]
        indices = tree.query_ball_point(batch, threshold)
        mask[i:i + chunk_size] = [len(idx) > 0 for idx in indices]
    return fcc_array[mask]
"""
def create_bool_array(shape, true_coords):
    # Initialize a 3D array of False values
    bool_array = np.zeros(shape, dtype=bool)
    
    # Unpack the true_coords into separate arrays for each dimension
    #z, y, x = zip(*true_coords)  # This will unpack true_coords into three lists: z, y, x
    z = true_coords[:,2]
    y = true_coords[:,1]
    x = true_coords[:,0]
    # Set specified coordinates to True
    bool_array[np.array(z), np.array(y), np.array(x)] = True

    return bool_array


def testing_dist(size, fcc_array, grain_array, threshold=1):
    min_bounds = np.min(grain_array, axis=0) - threshold
    max_bounds = np.max(grain_array, axis=0) + threshold

    # Create mask for fcc_array based on bounds
    mask = np.all((fcc_array >= min_bounds) & (fcc_array <= max_bounds), axis=1)
    fcc_array = fcc_array[mask]
    
    fcc_array_int = np.round(fcc_array)
    grain_voxel = create_bool_array([size]*3, grain_array)
    extracted_values = grain_voxel[tuple(fcc_array_int.T.astype(np.uint16))]
        
    return fcc_array[extracted_values]
    

def compute_pair_wise_dist(fcc_array, grain_array, grain_no=0, threshold=1, num_jobs=15):

    # Filter fcc_array based on min and max bounds with threshold
    for i in range(3):
        mask = fcc_array[:, i] > (np.min(grain_array, axis=0)[i] - threshold)
        fcc_array = fcc_array[mask]

    for i in range(3):
        mask = fcc_array[:, i] < (np.max(grain_array, axis=0)[i] + threshold)
        fcc_array = fcc_array[mask]
    
    print(fcc_array.shape)
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
def get_ram_usage_gb():
    """Gets the RAM usage in gigabytes."""
    mem = psutil.virtual_memory()
    return mem.used / (1024 ** 3)  # Convert bytes to GB


def select_random_file():
    directory = "/lustre/jayaraman_lab/users/3352/MURI_additive_SAXS_SEM/simulate_structers/affine_fcc_coords/"
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    if files:
        random_file = random.choice(files)
        full_path = os.path.join(directory, random_file)
        return full_path, random_file
    else:
        return "No files found in the directory"

def random_read_fcc():
    path, file_name = select_random_file()
    with open(path, 'rb') as f:
        inp = np.load(f)["arr_0"].copy()
    inp=np.array(inp).astype(np.float16)
    rot_angle = [int(x) for x in file_name.split("_")[1:4]]
    return inp, rot_angle


def main():
    """
    make grains tested and OK!
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("PyTorch is using GPU")
    else:
        print("PyTorch is not using GPU")

    target_size=700 # fixed 600
    n_grains = int(np.random.choice(range(2,80), size=1, replace=True)) # vary 2-80
    blobiness = int(np.random.choice(range(1,15), size=1, replace=True)) # 
    porosity=random.uniform(0.01, 0.5) # 0.01 - 0.5 left squewed    
    out_size = target_size
    rot_angl = list(np.random.choice(range(90), size=3, replace=True))
    coored_dump_folder = "/lustre/jayaraman_lab/users/3352/MURI_additive_SAXS_SEM/simulate_structers/coords_dump/"

    print("-----Parameters------")
    print("n_grains - ",n_grains)
    print("blobiness - ",blobiness)
    print("porosity - ",porosity)
    print("rot_angl - ",rot_angl)

    st = time.time()
    grain_loc_cube = make_grain_voxels(n_grains,target_size).astype(np.uint16)
    #print(grain_loc_cube.min(),"-", grain_loc_cube.max())
    print("time to make grains - ", time.time()-st)
    print(f"RAM usage: {get_ram_usage_gb():.2f} GB")    
    
    st = time.time()
    pores_inst = make_pores(shape=[target_size]*3, blobiness = blobiness, porosity = porosity, device=device)
    pores = pores_inst.simulate_pores().copy()
    if np.all(pores==False):
        grain_loc_cube[~pores]=1
    else:        
        grain_loc_cube[~pores]=0
    del pores, pores_inst
    print("time to make pores - ", time.time()-st)
    print(f"RAM usage: {get_ram_usage_gb():.2f} GB")    
    
    #print("grain_loc_cube", grain_loc_cube.shape)
    #print("grain_loc_cube unique", len(np.unique(grain_loc_cube)))
    #print("closest_point_index", closest_point_index.shape)
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
    out = np.empty((0, 3))
    

    st = time.time()
    for i in tqdm(np.unique(grain_loc_cube)):

        #fcc_coords_affine = make_fcc_affine_float_coords(out_size, fcc_coord, rotation_angles=rot_angl)
        fcc_coords_affine, rot_angl = random_read_fcc()

        #print("time for make_fcc_affine_float_coords", time.time()-st)
        fcc_coords_affine = fcc_coords_affine.astype(np.float16)
        coords_grain_loc = np.argwhere(grain_loc_cube==i)
        
        
        #grain_lattice_coords = compute_pair_wise_dist(fcc_coords_affine, coords_grain_loc, grain_no=i*10, threshold=1) 
        grain_lattice_coords = testing_dist(out_size, fcc_coords_affine, coords_grain_loc)
        
        out = np.append(out, grain_lattice_coords, axis=0)
        
    del grain_lattice_coords, fcc_coords_affine
    #print("time to make affine grains - ", time.time()-st)
    print(f"RAM usage: {get_ram_usage_gb():.2f} GB")    
    
    #plot_scatter_points(out, out_size)
    #out = out[:,:-1].copy()
    st = time.time()
    save_structure_to_txt(out, os.path.join(coored_dump_folder, "porous_{}_{}_grains_{}_{}_fcc_affine_{}_{}_{}.txt".format( blobiness, porosity, target_size, n_grains, rot_angl[0], rot_angl[1], rot_angl[2])))
    print("Save Complete")
    print("time to save - ", time.time()-st)
    print(f"RAM usage: {get_ram_usage_gb():.2f} GB")    
    
    #closest_point_index = np.argwhere(output) 
    
    #plot_scatter_points(closest_point_index, out_size)
    #"""
    return None

if __name__ == "__main__":
    for i in range(400):
        main()