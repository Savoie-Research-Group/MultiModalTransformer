#randomly drop out 10% of spectra for training and validation examples

import numpy as np
def random_choice(array_input, src_maxlen, drop_rate):
    ms_drop_list = []
    ir_drop_list = []
    nmr_drop_list = []
    for i in range(array_input.shape[0]):
        rand1 = float(np.random.rand(1))
        rand2 = float(np.random.rand(1))
        rand3 = float(np.random.rand(1))
        if i % 3 == 0:
            if rand1 < drop_rate:
                if rand2 < drop_rate:
                    array_input[i,src_maxlen:src_maxlen + 999] = np.full((999),288)
                    array_input[i,src_maxlen+999:src_maxlen+1899] = np.full((900),389)
                    ms_drop_list.append(i)
                    ir_drop_list.append(i)
                else:
                    if rand3 < drop_rate:
                        array_input[i,src_maxlen+1899:] = np.full((993),489)
                        nmr_drop_list.append(i)
            else:
                if rand2 < drop_rate:
                    array_input[i,src_maxlen+999:src_maxlen+1899] = np.full((900),389)
                    ir_drop_list.append(i)
                else:
                    continue
                if rand3 < drop_rate:
                    array_input[i,src_maxlen+1899:] = np.full((993),489)
                    nmr_drop_list.append(i)
                else:
                    continue
        if i % 3 == 1:
            if rand1 < drop_rate:
                if rand2 < drop_rate:
                    array_input[i,src_maxlen+1899:] = np.full((993),489)
                    array_input[i,src_maxlen+999:src_maxlen+1899] = np.full((900),389)
                    nmr_drop_list.append(i)
                    ir_drop_list.append(i)
                else:
                    if rand3 < drop_rate:
                        array_input[i,src_maxlen:src_maxlen+999] = np.full((999),288)
                        ms_drop_list.append(i)
            else:
                if rand2 < drop_rate:
                    array_input[i,src_maxlen+1899:] = np.full((993),489)
                    nmr_drop_list.append(i)
                else:
                    continue
                if rand3 < drop_rate:
                    array_input[i,src_maxlen:src_maxlen+999] = np.full((999),288)
                    ms_drop_list.append(i)
                else:
                    continue
        if i % 3 == 2:
            if rand1 < drop_rate:
                if rand2 < drop_rate:
                    array_input[i,src_maxlen+1899:] = np.full((993),489)
                    array_input[i,src_maxlen:src_maxlen+999] = np.full((999),288)
                    nmr_drop_list.append(i)
                    ms_drop_list.append(i)
                else:
                    if rand3 < drop_rate:
                        array_input[i,src_maxlen+999:src_maxlen+1899] = np.full((900),389)
                        ir_drop_list.append(i)
            else:
                if rand2 < drop_rate:
                    array_input[i,src_maxlen:src_maxlen+999] = np.full((999),288)
                    ms_drop_list.append(i)
                else:
                    continue
                if rand3 < drop_rate:
                    array_input[i,src_maxlen+999:src_maxlen+1899] = np.full((900),389)
                    ir_drop_list.append(i)
                else:
                    continue
    return ms_drop_list, ir_drop_list, nmr_drop_list, array_input
