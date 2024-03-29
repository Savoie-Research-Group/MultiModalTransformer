import argparse
import torch
import pickle
import numpy as np
from model import deduction_transformer
from calc_acc import bs_accuracy
from parse_config_test import parse_config

parser = argparse.ArgumentParser(description = 'Evaluate deduction model performance')
parser.add_argument('-c',dest = 'config',default = 'config_eval.txt',
                    help = 'the config file for training data and working directory.')
print('parsing the config file.')
c = parse_config(parser.parse_args())

#Assign parameters
num_tokens = 288
vocab_size = 590
dim_model = 256
num_heads = 8
num_feed_forward = 2048
num_src_enc_layers = 4
num_spec_enc_layers = 2
num_src_dec_layers = 4
num_spec_dec_layers = 2
src_maxlen = 276
tgt_maxlen = 67
ms_maxlen = 999
ir_maxlen = 900
nmr_maxlen = 993
mode_lst = [int(element) for element in c['mode_lst']]
best_chk_path = c['best_chk_path']
device = torch.device('cpu')
idx_to_ch_fh = open(c['idx_to_ch_dict_path'], "rb")
idx_to_ch_dict = pickle.load(idx_to_ch_fh)
ch_to_idx_fh = open(c['ch_to_idx_dict_path'], "rb")
ch_to_idx_dict = pickle.load(ch_to_idx_fh)
target_start_token_idx = ch_to_idx_dict["<"]
target_end_token_idx = ch_to_idx_dict["$"]

#Load model checkpoint
model = deduction_transformer(vocab_size,
                              num_tokens,
                              dim_model,
                              num_heads,
                              num_feed_forward,
                              num_src_enc_layers,
                              num_spec_enc_layers,
                              num_src_dec_layers,
                              num_spec_dec_layers,
                              mode_lst,
                              device
                              )
model = model.to(device)
checkpoint = torch.load(best_chk_path,map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])

#Load test set data
src_test_fp = c['src_test']
tgt_test_fp = c['tgt_test']
src_test = np.loadtxt(src_test_fp,delimiter=",")
tgt_test = np.loadtxt(tgt_test_fp,delimiter=",")
src_test = np.concatenate((src_test,tgt_test[:,67:]),axis=1)
tgt_test = tgt_test[:,0:67]
X_test = torch.Tensor(src_test)
Y_test = torch.Tensor(tgt_test)

#Calculate overall accuracy
model.eval()
test_all = 11

#uncomment smiles output section in calc_acc.py before running test_multi.py
correct_num_top1=0
correct_num_top5=0
for i in range(0,100,100):
    batch = (X_test[i:i+100,:],Y_test[i:i+100,:])
    count_top1,count_top5 = bs_accuracy(batch,model,src_maxlen,tgt_maxlen,idx_to_ch_dict,target_start_token_idx,target_end_token_idx)
    correct_num_top1 = correct_num_top1 + count_top1
    correct_num_top5 = correct_num_top5 + count_top5

