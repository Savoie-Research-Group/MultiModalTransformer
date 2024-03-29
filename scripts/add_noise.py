#This function is used to assemble noised spectra as the input for noised spectra case study

import numpy as np
from include_noise import modify_array

def add_noise(input_data,src_maxlen,noise_level,return_spec):
    spec_array = input_data[:,src_maxlen:]
    if return_spec[0] == 1:
        ms_array = spec_array[:,0:999] - 288
        for i in range(ms_array.shape[0]):
            ms_array[i,0:999] = modify_array(ms_array[i,0:999],noise_level) + 288
    else:
        ms_array = spec_array[:,0:999]
    if return_spec[1] == 1:
        ir_array = spec_array[:,999:1899] - 389
        for i in range(ir_array.shape[0]):
            ir_array[i,0:900] = modify_array(ir_array[i,0:900],noise_level) + 389
    else:
        ir_array = spec_array[:,999:1899]
    if return_spec[2] == 1:
        nmr_array = spec_array[:,1899:] - 489
        for i in range(nmr_array.shape[0]):
            nmr_array[i,0:993] = modify_array(nmr_array[i,0:993],noise_level) + 489
    else:
        nmr_array = spec_array[:,1899:]
    output_data = np.concatenate((input_data[:,0:src_maxlen],ms_array,ir_array,nmr_array),axis=1)
    return output_data
            
