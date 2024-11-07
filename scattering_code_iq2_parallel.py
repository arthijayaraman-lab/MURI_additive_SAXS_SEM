#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 15:12:26 2023

@author: skronen

"""
import math
import numpy as np
import numexpr as ne
import time
import matplotlib.pyplot as plt
import joblib
import math
from joblib import Parallel, delayed


def crystal_dirs(sample_num = 6):
    dirs = []
    for i in range(-sample_num,sample_num+1):
        for j in range(-sample_num, sample_num+1):
            for k in range(-sample_num, sample_num + 1):
                if i==0 and j ==0 and k == 0:
                    continue
                else:
                    vec = [i, j, k]
                    dirs.append(np.array(vec)/np.linalg.norm(vec))
    return np.array(dirs)
                

def fibonacci_sphere(samples=1000, direction = [0,1,0]):
    """
    Gets direction vectors reletively evenly spaced over the surface of
    the unit sphere using the fibonnaci sphere. These are used
    as directions for the q-vectors, and are averaged over in the scattering calculation
    to improve sampling.
    
    Parameters
    ----------
    samples : int, optional
        The number of sampled vectors. With previous testing, I've found the scattering
        profile changes if you use less than 300, but above 300, it becomes fairly constant.
        The default is 1000.
    direction : 1x3 list, optional
        if samples == 1, this direction is used. The default is [0,1,0].

    Returns
    -------
    points : n_samples x 3 array 
        Array of sampled unit vectors

    """
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians
    
    if samples == 1:
        print(direction)
        point = np.array([direction])
        point = point/np.linalg.norm(point, axis = 1)
        return point
    
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))
    points = np.array(points)
    return points

def single_loop_minus_box(qrange,len_box, rs,fis, origin_centered = False, direction = [0,1,0], num_qdirs = 300, ff_rad = None, crystal = False, Iq1_ff=None, I1q=True):
    """
    Parameters
    ----------
    qrange : list or np.array-like
        a list or array of all the q values.
    rs : list/array of list/array
        an array of coordinates of all atoms, grouped in types.
        e.g., if there are two atom types, A and B, 
        rs should be [an array of all A atoms, an array of all B atoms]
        each array should be of shape Nx3.
    fis : list or np.array-like.
        array containing SLDs of all the atom types.
    total_points : int
        Total number of angular q-vectors used to average for isotropic I(q) at each q point. 
        The default is 100.
    origin_centered: bool
        Whether the supplied points are centered around the origin. If false: it is 
        assumed the box ranges from [0,L]. If true [-L/2, L/2].
     direction : 1x3 list, optional
         if total_points == 1, this direction is used. The default is [0,1,0].
    num_qdirs : int, optional
        See "samples" in the fibonacci sphere function
        The default is 300.
   

    Returns
    -------
    omega: array.
    An array of the same length of qrange, containing all I(q) values.

    """

    if crystal: 
        v_array = crystal_dirs(num_qdirs).T
    else:
        v_array = fibonacci_sphere(num_qdirs, direction).transpose()
    ret = np.zeros((len(qrange), v_array.shape[1]))  # look into 

    num_scatterers = len(rs[0]) 
    rho_avg = num_scatterers/len_box**3 # density of scatters 
    
    print(f"Starting scattering with {num_scatterers} scatterers")
        

    if I1q:
        def q_vect_loop(qi,q):
            
            if qi%5 == 0: print(qi)
            q_vecs = q*v_array # scale vectors of sphere with q value 
            rvqs = []
            for ri,r in enumerate(rs): #iterate over sets of points   
                r_array = np.array(r)
                if not origin_centered: r_array -= len_box/2 # center co-ordinates
                rvqs.append(np.matmul(r_array,q_vecs))
        
            sum_exp = np.zeros(v_array.shape[1]).astype('complex128') # check this shape and why is it here 
            
            # FFT of cube 
            box_ff = np.sinc(q_vecs[0]*len_box/2/np.pi)*np.sinc(q_vecs[1]*len_box/2/np.pi)*np.sinc(q_vecs[2]*len_box/2/np.pi) #compute box_ff for each vector at current q aka form factor of a cube formula     
            
            # form factor of sphere i.e its the Fourier transform of sphere 
            qr = q * ff_rad # scale radius with q range 
            Vparticle = 4/3 * np.pi * ff_rad**3 # volume of sphere 
            total_particle_volume = num_scatterers * Vparticle  # total scaterers volume  
            ff_particle = 3 *Vparticle* (np.sin(qr) - qr*np.cos(qr)) / qr**3 # form factor of shere formula 
                
            for rvqi,rvq in enumerate(rvqs):
                sum_exp += ne.evaluate("sum(exp(-1j*rvq),axis=0)") #sum across all scatterers
                
            sum_exp = sum_exp * ff_particle
            sum_exp = sum_exp - total_particle_volume * box_ff
            #sum_exp = sum_exp - box_ff
            
            sum_conj = sum_exp* np.conj(sum_exp) 
            #print(ret)
            #ret[qi,:] += np.real(sum_conj).copy() #average across all directions
            return np.real(sum_conj).copy() 

        results = Parallel(n_jobs=10)(delayed(q_vect_loop)(qi, q) for qi, q in enumerate(qrange))

        for qi, result in enumerate(results):
            ret[qi, :] = result  # Safely update ret in the main thread
            
        structure_factor = ret.mean(1)
    else:      
        def q_vect_loop_iq2(qi,q):
            
            if qi%5 == 0: print(qi)
            q_vecs = q*v_array
            rvqs = []
            for ri,r in enumerate(rs): #iterate over sets of points
                r_array = np.array(r)
                if not origin_centered: r_array -= len_box/2
                rvqs.append(np.matmul(r_array,q_vecs))
        
            sum_exp = np.zeros(v_array.shape[1]).astype('complex128')
            
            
            for rvqi,rvq in enumerate(rvqs):
                sum_exp += ne.evaluate("sum(exp(-1j*rvq),axis=0)") #sum across all scatterers
                
            """REMOVE BOX EFFECTS"""
            box_ff = np.sinc(q_vecs[0]*len_box/2/np.pi)*np.sinc(q_vecs[1]*len_box/2/np.pi)*np.sinc(q_vecs[2]*len_box/2/np.pi) #compute box_ff for each vector at current q    
            Vparticle = (4/3) * np.pi * (ff_rad**3) # volume of sphere 
            total_particle_volume = num_scatterers * Vparticle  # total scaterers volume  

            if not(Iq1_ff == None):
                sum_exp = sum_exp*Iq1_ff[q] 

            sum_exp = sum_exp/num_scatterers #normalize amplitude 
            #sum_exp = sum_exp - total_particle_volume * box_ff
            sum_exp = sum_exp - box_ff
            sum_conj = sum_exp* np.conj(sum_exp) 
            #ret[qi,:] += np.real(sum_conj) #average across all directions 
            return np.real(sum_conj).copy() 
        
        results = Parallel(n_jobs=10)(delayed(q_vect_loop_iq2)(qi, q) for qi, q in enumerate(qrange))
        
        for qi, result in enumerate(results):
            ret[qi, :] = result  # Safely update ret in the main thread
            
        structure_factor = ret.mean(1) 
    return structure_factor 

def Iq1(q_range=None):
    file = '700_5_0.05_fcc.txt'
    lattice_spacing = 1 #d_i
    pos = np.loadtxt(file)
    print("Loaded file")

    len_box = np.max(pos) - np.min(pos) + lattice_spacing/2 
    qmax = 2
    qmin = (2 * np.pi)/len_box
    
    if q_range == None:
        q_range = np.logspace(np.log10(qmin), qmax, 250)
    else:
        q_range = q_range[np.abs(q_range - qmin).argmin():]  

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


def Iq2():
    file = 'fcc_prefect_100d.txt'
    lattice_spacing = 100 #d_i
    pos = np.loadtxt(file)

    len_box = np.max(pos) - np.min(pos) + lattice_spacing/2 
    print("len_box", len_box)
    qmax = 2
    print("distance range - ", (2 * np.pi)/(10**qmax)," to ", len_box)

    qmin = (2 * np.pi)/len_box
    q_range = np.logspace(np.log10(qmin), qmax, 250)
    #ff_rad = 0.433 # 43.301
    ff_rad = 0.353*lattice_spacing

    Iq1_q_range, Iq1_structure_factor = Iq1(q_range)
    print(Iq1_structure_factor.shape)
    Iq1_structure_factor = np.append(np.ones(250-Iq1_q_range.shape[0]),Iq1_structure_factor, axis=0)
    print(Iq1_structure_factor.shape)

    Iq1_ff = dict(zip(q_range, Iq1_structure_factor))
    tic = time.perf_counter()
    structure_factor = single_loop_minus_box(q_range, len_box, [pos], [1], ff_rad = ff_rad, origin_centered = False, Iq1_ff=Iq1_ff, I1q=False)
    #structure_factor = single_loop_minus_box(q_range, len_box, [pos], [1], ff_rad = ff_rad, origin_centered = False, Iq1_ff=None, I1q=False)
    
    toc = time.perf_counter()
    print(toc - tic)
    plt.loglog(q_range, structure_factor)
    plt.savefig("Iq2.png")
    plt.show()

    return q_range, structure_factor

def Sq2():
    file = 'fcc_prefect_100d.txt'
    lattice_spacing = 100 #d_i
    pos = np.loadtxt(file)

    len_box = np.max(pos) - np.min(pos) + lattice_spacing/2 
    qmax = 2
    print("distance range - ", (2 * np.pi)/(10**qmax)," to ", len_box)

    qmin = (2 * np.pi)/len_box
    q_range = np.logspace(np.log10(qmin), qmax, 250)
    ff_rad = 0.353*lattice_spacing

    tic = time.perf_counter()
    structure_factor = single_loop_minus_box(q_range, len_box, [pos], [1], ff_rad = ff_rad, origin_centered = False, Iq1_ff=None, I1q=False)
    
    toc = time.perf_counter()
    print(toc - tic)
    plt.loglog(q_range, structure_factor)
    plt.savefig("Sq2.png")
    plt.show()

    return q_range, structure_factor

if __name__ == '__main__':
    q_range, structure_factor = Iq1()
    #q_range, structure_factor = Iq2()
    #q_range, structure_factor = Sq2() 
    #plt.plot(q_range, structure_factor)
