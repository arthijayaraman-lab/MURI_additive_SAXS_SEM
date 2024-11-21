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
    x = np.linspace(0, n, 1)
    y = np.linspace(0, n, 1)
    z = np.linspace(0, n, 1)
    X, Y, Z = np.meshgrid(x, y, z)

    query_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    del X, Y, Z, x, y, z
 
    ax.scatter(coords.T[0], coords.T[1], coords.T[2], s= 55)
    plt.show()

if __name__ == "__main__":
    """
    make grains tested and OK!
    """
    target_size=10
    n_grains = 2
    grain_loc_cube = make_grain_voxels(n_grains,target_size)

    """
    print(grain_loc_cube.shape)
    closest_point_index = grain_loc_cube.ravel()
    print(closest_point_index.shape)

    colors = plt.cm.viridis(closest_point_index/np.max(closest_point_index))

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Set plot limits to the unit cube
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])

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
    rot_angl = [0,90,270]

    for i in range(n_grains): 
        fcc_coords_affine, lattice_fcc = make_fcc_affine(out_size, rotation_angles= [np.random.choice(rot_angl), np.random.choice(rot_angl), np.random.choice(rot_angl)])
        lattice_fcc = lattice_fcc.astype(bool)
        fcc_coords_affine = fcc_coords_affine.astype(np.uint16)

        #plot_scatter_points(fcc_coords_affine, out_size)
        sin_grain_cube = np.zeros_like(grain_loc_cube).astype(bool) 
        sin_grain_cube[grain_loc_cube==i] = True 
        output = np.logical_xor(output, np.logical_and(sin_grain_cube, lattice_fcc).copy()) 
        
    closest_point_index = np.argwhere(output) 
    
    plot_scatter_points(closest_point_index, out_size)
    #"""