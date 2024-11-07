import numpy as np
import matplotlib.pyplot as plt
import time
import os
import torch
from utils.utils import *
from make_pores import make_pores
from tempfile import mkdtemp
import os.path as path


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("PyTorch is using GPU")
    else:
        print("PyTorch is not using GPU")
    
    n = 700
    blobiness = 5
    porosity=0.05
    
    st = time.time()

    # make FCC
    filename = path.join(mkdtemp(), 'newfile.dat')
    lattice = np.memmap(filename, dtype=bool, mode='w+', shape=tuple([n]*3))
    lattice[:,:,:] = make_fcc_lattice(n).astype(bool)

    # make pores 
    pores_inst = make_pores(shape=[n]*3, blobiness = blobiness, porosity = porosity, device=device)
    pores = pores_inst.simulate_pores()

    structures = np.logical_and(lattice, pores).copy()

    del pores, pores_inst
    
    save_structure_to_txt(structures, "{}_{}_{}_fcc.txt".format(n, blobiness, porosity))

    print(structures.shape)
    
    print("Time consumed - ", st-time.time())
    

