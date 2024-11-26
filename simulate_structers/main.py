import numpy as np
import matplotlib.pyplot as plt
import time
import os
import torch
from utils.utils import *
from make_pores import make_pores
from tempfile import mkdtemp
import os.path as path
import sys




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("PyTorch is using GPU")
    else:
        print("PyTorch is not using GPU")
    
    n = 500
    blobiness = 5
    porosity=0.01
    
    st = time.time()

    # make FCC
    filename = path.join(mkdtemp(), 'newfile.dat')
    lattice = np.memmap(filename, dtype=bool, mode='w+', shape=tuple([n]*3))
    lattice[:,:,:] = make_fcc_lattice(n).astype(bool)
    print("Completed lattice making")

    # add code to make grains

    # add code to out fcc orientation 

    
    # make pores 
    pores_inst = make_pores(shape=[n]*3, blobiness = blobiness, porosity = porosity, device=device)
    pores = pores_inst.simulate_pores().copy()
    del pores_inst
    print("Generated Pores")

    lattice = lattice[:pores.shape[0],:pores.shape[1],:pores.shape[2]].copy()

    structures = np.logical_and(lattice, pores).copy()
    
    structures = lattice.copy()
    del pores
    print("Logical ops done")

    plt.imshow(structures[5,:,:])
    plt.savefig("cs_imgs/structure_cs_{}_{}_{}_fcc.png".format(n, blobiness, porosity))
    
    structures_coords = get_coords_from_3d(structures)
    #structures_coords = delete_rows_having_value(structures_coords, 0)
    #structures_coords = delete_rows_having_value(structures_coords, 499)

    print(structures_coords.min(),"-",structures_coords.max())

    print("size of structures_coords - ", sys.getsizeof(structures_coords))
    save_structure_to_txt(structures_coords, "lattice_coords/{}_{}_{}_fcc_perfect.txt".format(n, blobiness, porosity))
    
    print(structures.shape)
    
    print("Time consumed - ", st-time.time())
