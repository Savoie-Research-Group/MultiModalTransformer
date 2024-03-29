import argparse
import torch
import pickle
import numpy as np
from model import deduction_transformer
from calc_acc import bs_accuracy
from parse_config_noise import parse_config
from add_noise import add_noise

parser = argparse.ArgumentParser(description = 'Evaluate deduction model performance')
parser.add_argument('-c',dest = 'config',default = 'config_eval_noise.txt',
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
real_test_output_path = c['real_test_output_path']
null_test_output_path = c['null_test_output_path']
noise_level = float(c['noise_level'])
return_spec = [int(element) for element in c['return_spec']]

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
null_src_test_fp = c['null_src_test']
null_tgt_test_fp = c['null_tgt_test']

src_test = np.loadtxt(src_test_fp,delimiter=",")
tgt_test = np.loadtxt(tgt_test_fp,delimiter=",")
src_test = np.concatenate((src_test,tgt_test[:,67:]),axis=1)
src_test = add_noise(src_test,src_maxlen,noise_level,return_spec)
tgt_test = tgt_test[:,0:67]

null_src_test = np.loadtxt(null_src_test_fp,delimiter=",")
null_tgt_test = np.loadtxt(null_tgt_test_fp,delimiter=",")
null_src_test = np.concatenate((null_src_test,null_tgt_test[:,67:]),axis=1)
null_tgt_test = null_tgt_test[:,0:67]
null_src_test = add_noise(null_src_test,src_maxlen,noise_level,return_spec)

X_test = torch.Tensor(src_test)
Y_test = torch.Tensor(tgt_test)
null_X_test = torch.Tensor(null_src_test)
null_Y_test = torch.Tensor(null_tgt_test)

#Calculate overall accuracy
model.eval()
test_all = 24941
null_test_all = 14810

correct_num_top1=0
correct_num_top5=0
for i in range(0,25000,100):
    batch = (X_test[i:i+100,:],Y_test[i:i+100,:])
    count_top1,count_top5 = bs_accuracy(batch,model,src_maxlen,tgt_maxlen,idx_to_ch_dict,target_start_token_idx,target_end_token_idx)
    correct_num_top1 = correct_num_top1 + count_top1
    correct_num_top5 = correct_num_top5 + count_top5

accuracy_lst = [str(correct_num_top1)]+[str(correct_num_top5)]
text = open(real_test_output_path,"a")
for ele in accuracy_lst:
    text.write(ele)
    text.write('/24941')
    text.write("\n")
text.close()

correct_num_top1=0
correct_num_top5=0
for i in range(0,14900,100):
    batch = (null_X_test[i:i+100,:],null_Y_test[i:i+100,:])
    count_top1,count_top5 = bs_accuracy(batch,model,src_maxlen,tgt_maxlen,idx_to_ch_dict,target_start_token_idx,target_end_token_idx)
    correct_num_top1 = correct_num_top1 + count_top1
    correct_num_top5 = correct_num_top5 + count_top5

accuracy_lst = [str(correct_num_top1)]+[str(correct_num_top5)]
text = open(null_test_output_path,"a")
for ele in accuracy_lst:
    text.write(ele)
    text.write('/14810')
    text.write("\n")
text.close()
