import h5py
filename = "/home/p51pro/UD/jayraman_lab/muri_additive/Sintered NCT Data/Apr 2023 Plate 1/sample_6/S1CJT015_RT_25.0C_00368_00001_dv_ave.h5"

"""
with h5py.File(filename, "r") as f:
    group = f["entry_0000"]   
    print(group.keys())
    print(f.get["input"])
    
    
    /entry_0000/input
/entry_0000/program_name
/entry_0000/saxsutilities/data/array
/entry_0000/saxsutilities/data/array_errors
/entry_0000/saxsutilities/data/header_array/ddummy
/entry_0000/saxsutilities/data/header_array/delta_time
/entry_0000/saxsutilities/data/header_array/dummy
/entry_0000/saxsutilities/data/header_array/errordummy
/entry_0000/saxsutilities/data/header_array/errortype
/entry_0000/saxsutilities/data/header_array/exposuretime
/entry_0000/saxsutilities/data/header_array/file_block
/entry_0000/saxsutilities/data/header_array/file_name
/entry_0000/saxsutilities/data/header_array/file_type
/entry_0000/saxsutilities/data/header_array/history_saxsutilities2_2022-10-12T20:22:36.919032
/entry_0000/saxsutilities/data/header_array/hmstarttime
/entry_0000/saxsutilities/data/header_array/intensity0shutcor
/entry_0000/saxsutilities/data/header_array/intensity1shutcor
/entry_0000/saxsutilities/data/header_array/offset_1
/entry_0000/saxsutilities/data/header_array/offset_2
/entry_0000/saxsutilities/data/header_array/q
/entry_0000/saxsutilities/data/header_array/sampledistance
/entry_0000/saxsutilities/data/header_array/short_name
/entry_0000/saxsutilities/data/header_array/time
/entry_0000/saxsutilities/data/header_array/title
/entry_0000/saxsutilities/data/header_array/titlebody
/entry_0000/saxsutilities/data/header_array/titleextension
/entry_0000/saxsutilities/data/header_array_errors/ddummy
/entry_0000/saxsutilities/data/header_array_errors/dummy
/entry_0000/saxsutilities/data/header_array_errors/errortype
/entry_0000/saxsutilities/data/q
/entry_0000/saxsutilities/data/t
/entry_0000/start_time
/entry_0000/title


"""

def get_ds_dictionaries(name, node):
  
    fullname = node.name
    if isinstance(node, h5py.Dataset):
    # node is a dataset/entry/data/data
        print(f'Dataset: {fullname}; adding to dictionary')
        ds_dict[fullname] = node
        print('ds_dict size', len(ds_dict)) 
    else:
     # node is a group
        print(f'Group: {fullname}; skipping')  
    
with h5py.File(filename,'r') as h5f:
    ds_dict = {}  
    h5f.visititems(get_ds_dictionaries) 
    
    #print((ds_dict['/entry/sample/thickness'][()]))
    
    for name in (ds_dict.keys()):
        print("name:   ",name)
        try:
            continue
            #print((ds_dict[name][()]))
            #print((ds_dict[name][()]).shape)
        except:
            pass
