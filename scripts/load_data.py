import numpy as np
import torch

def load_csv(src_test_fp, tgt_test_fp, start_line, end_line):
    src_test = np.loadtxt(src_test_fp,delimiter=",",skiprows=start_line,max_rows=end_line)
    tgt_test = np.loadtxt(tgt_test_fp,delimiter=",",skiprows=start_line,max_rows=end_line)
    src_test = np.concatenate((src_test,tgt_test[:,67:]),axis=1)
    tgt_test = tgt_test[:,0:67]
    X_test = torch.Tensor(src_test)
    Y_test = torch.Tensor(tgt_test)
    return X_test, Y_test
