"""
Make a numpy array
>>> np_array = np.random.random((30, 30))
Write it in h5 format
>>> h5_name = 'np_array.h5'
>>> write_dh5_np(h5_name, np_array)
Read it in h5 format and check contents
>>> np_array_from_h5 = read_hd5_np(h5_name)
>>> np.allclose(np_array_from_h5, np_array)
True
Write h5 file in txt format
>>> txt_name = 'np_array.txt'
>>> txt_name = convert_to_txt(h5_name, txt_name)
Read it in txt format and check contents
>>> np_array_from_txt = np.loadtxt(txt_name)
>>> np.allclose(np_array_from_txt, np_array)
True
"""

import h5py
import numpy as np
import os
import sys
import doctest


def write_dh5_np(h5_name, np_array):
    """
    :param h5_name: Name of a h5 file
    :type h5_name: str
    :param np_array: Array of floats
    :type np_array: np.array
    """
    assert not os.path.isfile(h5_name), "Won't overwrite {}".format(h5_name)
    
    with h5py.File(h5_name, 'w') as h5_file:
        h5_file.create_dataset('np_array', data=np_array)

def read_hd5_np(h5_name):
    """
    :param h5_name: Name of a h5 file
    :type h5_name: str
    
    :returns: Data in h5 file
    :rtype: np.array
    """    
    with h5py.File(h5_name, 'r') as h5_file:
        np_array = h5_file['np_array'][()]
        
    return np_array    

def convert_to_txt(h5_name, txt_name=None):
    """
    :param h5_name: Name of a h5 file
    :type h5_name: str
    :param txt_name: Name of a txt file to be created
    :type txt_name: str
        
    :returns: Name of a txt file that was created
    :rtype: str
    """
    if txt_name is None:
        strip_name = os.path.splitext(h5_name)[0]
        txt_name = '{}.txt'.format(strip_name)
    
    assert not os.path.isfile(txt_name), "Won't overwrite {}".format(txt_name)
    
    np_array = read_hd5_np(h5_name)       
    np.savetxt(txt_name, np_array)
    
    return txt_name
    
if __name__ == "__main__":

    args = sys.argv[1:]

    if not args:
    
        doctest.testmod()   

    else:
    
        hdf5_name = sys.argv[1]
        
        try:
            txt_name = sys.argv[2]
        except IndexError:
            txt_name = None
            
        convert_to_txt(hdf5_name, txt_name)
