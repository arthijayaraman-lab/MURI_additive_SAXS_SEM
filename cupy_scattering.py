# python .\gpu_scattering_1D.py .\sample1.dump results.txt lammpsdump
# Code to compute 1D scattering profile using GPU resources 
import sys
import cupy as cp
import time
import numpy as np

def fibonacci_sphere(samples=1000, direction=[0, 1, 0]):
    phi = cp.pi * (3. - cp.sqrt(5.))  # golden angle in radians

    if samples == 1:
        point = cp.array([direction])
        return point / cp.linalg.norm(point, axis=1)

    # Vectorized approach
    indices = cp.arange(samples)
    y = 1 - (indices / (samples - 1)) * 2
    radius = cp.sqrt(1 - y * y)

    theta = phi * indices

    x = cp.cos(theta) * radius
    z = cp.sin(theta) * radius

    points = cp.column_stack((x, y, z))
    return points

def single_loop_minus_box(qrange, len_box, box_shift, rs, fis, apply_center_correction=True, direction=[0, 1, 0], total_points=300):
    v_array = fibonacci_sphere(total_points, direction).transpose()
    ret = cp.zeros((len(qrange), v_array.shape[1]), dtype=cp.float32)
    batch_size = 10000  # Adjust based on profiling
    num_batches = len(rs[0]) // batch_size + (len(rs[0]) % batch_size > 0)
    tic = time.perf_counter()

    for qi, q in enumerate(qrange):
        #if qi % 5 == 0: print(qi)
        q_vecs = q * v_array
        sum_exp = cp.zeros(v_array.shape[1], dtype=cp.complex64)

        for batch in range(num_batches):
            start = batch * batch_size
            end = min((batch + 1) * batch_size, len(rs[0]))
            r_batch = cp.array(rs[0][start:end])
            if apply_center_correction:
                for i in range(3):
                    r_batch[:,i] -= box_shift[i]

            rvqs = cp.matmul(r_batch, q_vecs)
            sum_exp += cp.exp(-1j * rvqs).sum(axis=0)

        box_ff = cp.sinc(q_vecs[0] * len_box[0] / 2 / cp.pi) * cp.sinc(q_vecs[1] * len_box[1] / 2 / cp.pi) * cp.sinc(q_vecs[2] * len_box[2] / 2 / cp.pi)
        sum_exp = sum_exp / len(rs[0]) - box_ff
        ret[qi, :] = cp.real(sum_exp * cp.conj(sum_exp))

    omega = ret.mean(axis=1).get()
    toc = time.perf_counter()
    print("Elapsed time: "+str(toc - tic)+" seconds")
    return omega


def Iq1():
    file = '/lustre/jayaraman_lab/users/3352/MURI_additive_SAXS_SEM/simulate_structers/1000_5_0.01_fcc.txt'
    lattice_spacing = 1 #d_i
    pos = np.loadtxt(file)
    print("Loaded file")

    len_box = np.max(pos) - np.min(pos) + lattice_spacing/2 
    qmax = 2
    qmin = (2 * np.pi)/len_box
    
    q_range = np.logspace(np.log10(qmin), qmax, 250)

    print("distance range - ", (2 * np.pi)/(10**qmax), " to ", len_box)
    
    ff_rad = 0.353 
    
    tic = time.perf_counter()
    structure_factor = single_loop_minus_box(q_range, len_box, [pos], [1], ff_rad = ff_rad, origin_centered = False, I1q=True)
    toc = time.perf_counter()
    print(toc - tic)
    plt.loglog(q_range, structure_factor)
    plt.savefig("Iq1.png")
    plt.show()
    return q_range, structure_factor


if __name__ == '__main__':

    #read the lamps file 
    file = = '/lustre/jayaraman_lab/users/3352/MURI_additive_SAXS_SEM/simulate_structers/1000_5_0.01_fcc.txt'
    lattice_spacing = 1

    pos = cp.loadtxt(file)
    
    print("Loaded file")

    len_box = cp.max(pos) - cp.min(pos) + lattice_spacing/2 
    qmax = 2
    qmin = (2 * np.pi)/len_box
    
    q_range = cp.logspace(np.log10(qmin), qmax, 250)

    print("distance range - ", (2 * np.pi)/(10**qmax), " to ", len_box)

    ff_rad = 0.353 
    
    tic = time.perf_counter()
    structure_factor = single_loop_minus_box(q_range, len_box, [pos], [1], ff_rad = ff_rad, origin_centered = False, I1q=True)
    toc = time.perf_counter()
    print(toc - tic)
    plt.loglog(q_range, structure_factor)
    plt.savefig("Iq1.png")
    plt.show()
    return q_range, structure_factor




    


    len_box[0] = xhi-xlo; len_box[1] = yhi-ylo; len_box[2] = zhi- zlo
    box_shift[0] = (xhi+xlo)/2; box_shift[1] = (yhi+ylo)/2; box_shift[2] = (zhi+zlo)/2
    q_range = cp.logspace(-2, 2, 100)
    omega = single_loop_minus_box(q_range, len_box, box_shift, [pts], [1], apply_center_correction=True)

    #plt.loglog(cp.asnumpy(q_range), cp.asnumpy(omega))
    #plt.show()
    
    q_range = cp.asnumpy(q_range)
    omega = cp.asnumpy(omega)
    with open(str(sys.argv[2]), 'w') as output_file:
        for i in range(len(q_range)):
            output_file.write(str(q_range[i])+" "+str(omega[i])+'\n')