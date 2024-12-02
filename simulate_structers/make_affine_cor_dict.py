import numpy as np 
from utils.utils import *
import os
import random 
import time 


if __name__ == "__main__":
    dst = "/lustre/jayaraman_lab/users/3352/MURI_additive_SAXS_SEM/simulate_structers/affine_fcc_coords/"
    out_size=700
    st = time.time()
    fcc_coord = get_coords_from_3d(make_fcc_lattice(out_size*2))  
    print("time to prefect FCC - ", time.time()-st)
    rot_all = []
    for i in range(5000):
        rot_angl = list(np.random.choice(range(90), size=3, replace=True))
        if rot_angl not in rot_all: 
            st = time.time()
            fcc_coords_affine = make_fcc_affine_float_coords(out_size, fcc_coord, rotation_angles=rot_angl).astype(np.float16)
            np.savez_compressed(os.path.join(dst, "{}_{}_{}_{}_fcc_affine".format(out_size, rot_angl[0], rot_angl[1], rot_angl[2])), fcc_coords_affine)
            rot_all.append(rot_angl)
            print("iter time - ", time.time()-st)
