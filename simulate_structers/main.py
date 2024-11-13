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
    
    n = 2500
    blobiness = 5
    porosity=0.01
    
    st = time.time()

    # make FCC
    filename = path.join(mkdtemp(), 'newfile.dat')
    lattice = np.memmap(filename, dtype=bool, mode='w+', shape=tuple([n]*3))
    lattice[:,:,:] = make_fcc_lattice(n).astype(bool)
    print("Completed lattice making")

    
    # make pores 
    pores_inst = make_pores(shape=[n]*3, blobiness = blobiness, porosity = porosity, device=device)
    pores = pores_inst.simulate_pores().copy()
    del pores_inst
    print("Generated Pores")

    lattice = lattice[:pores.shape[0],:pores.shape[1],:pores.shape[2]].copy()
    
    structures = np.logical_and(lattice, pores).copy()
    del pores
    print("Logical ops done")

    plt.imshow(structures[5,:,:])
    plt.savefig("structure_cs_{}_{}_{}_fcc.png".format(n, blobiness, porosity))
    
    
    structures_coords = get_coords_from_3d(structures, strd=500)
    
    print("size of structures_coords - ", sys.getsizeof(structures_coords))

    save_structure_to_txt(structures_coords, "{}_{}_{}_fcc.txt".format(n, blobiness, porosity))
    
    print(structures.shape)
    
    print("Time consumed - ", st-time.time())

