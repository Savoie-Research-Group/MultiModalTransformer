import numpy as np
import os
import logging
import argparse
from model import deduction_transformer
from optimizer import ScheduledOptim
from model_fit import fit
from parse_config import parse_config
from random_drop import random_choice
from write_drop import write_drop
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

parser = argparse.ArgumentParser(description = 'Training model for deduction')
parser.add_argument('-c',dest = 'config',default = 'config_train.txt',
                    help = 'the config file for trianing data and working directory.')
print('parsing the config file.')
c = parse_config(parser.parse_args())

#load dataset
src_train_fp = c['src_train'] 
tgt_train_fp = c['tgt_train']
src_val_fp = c['src_val']
tgt_val_fp = c['tgt_val']
null_src_train_fp = c['null_src_train']
null_tgt_train_fp = c['null_tgt_train']
null_src_val_fp = c['null_src_val']
null_tgt_val_fp = c['null_tgt_val']

src_train = np.loadtxt(src_train_fp,delimiter=",")
tgt_train = np.loadtxt(tgt_train_fp,delimiter=",")
src_val = np.loadtxt(src_val_fp,delimiter=",")
tgt_val = np.loadtxt(tgt_val_fp,delimiter=",")
null_src_train = np.loadtxt(null_src_train_fp,delimiter=",")
null_tgt_train = np.loadtxt(null_tgt_train_fp,delimiter=",")
null_src_val = np.loadtxt(null_src_val_fp,delimiter=",")
null_tgt_val = np.loadtxt(null_tgt_val_fp,delimiter=",")

#concat reactant information (src) with spectra information (attached with tgt)
X_train = np.concatenate((src_train,tgt_train[:,67:]),axis=1)
Y_train = tgt_train[:,0:67]
X_val = np.concatenate((src_val,tgt_val[:,67:]),axis=1)
Y_val = tgt_val[:,0:67]
null_X_train = np.concatenate((null_src_train,null_tgt_train[:,67:]),axis=1)
null_Y_train = null_tgt_train[:,0:67]
null_X_val = np.concatenate((null_src_val,null_tgt_val[:,67:]),axis=1)
null_Y_val = null_tgt_val[:,0:67]
X_train = np.concatenate((X_train,null_X_train))
Y_train = np.concatenate((Y_train,null_Y_train))
X_val = np.concatenate((X_val,null_X_val))
Y_val = np.concatenate((Y_val,null_Y_val))

#Set up logger
log_name = c['loss_path']
logging.basicConfig(filename = log_name, level = logging.INFO,
                    format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(log_name)

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
epochs = 200
warmup_steps = 37500
batch_size = 20
mode_lst = [int(element) for element in c['mode_lst']]
best_chk_path = c['best_chk_path']
end_chk_path = c['end_chk_path']
device = torch.device('cuda')
val_min = 1
patience = 30

#apply dropout logic 
ms_drop_list, ir_drop_list, nmr_drop_list, X_train = random_choice(X_train,src_maxlen,0.1)
ms_val_drop, ir_val_drop, nmr_val_drop, X_val = random_choice(X_val,src_maxlen,0.1)
write_drop(ms_drop_list,'ms_train_drop.txt')
write_drop(ir_drop_list,'ir_train_drop.txt')
write_drop(nmr_drop_list,'nmr_train_drop.txt')
write_drop(ms_val_drop,'ms_val_drop.txt')
write_drop(ir_val_drop,'ir_val_drop.txt')
write_drop(nmr_val_drop,'nmr_val_drop.txt')

#Prepare input
tensor_x_train = torch.Tensor(X_train)
tensor_y_train = torch.Tensor(Y_train)
train_dataset = TensorDataset(tensor_x_train,tensor_y_train)
train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

tensor_x_val = torch.Tensor(X_val)
tensor_y_val = torch.Tensor(Y_val)
val_dataset = TensorDataset(tensor_x_val,tensor_y_val)
val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)

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

optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001, betas = (0.9, 0.98), eps = 1e-9)
sched = ScheduledOptim(optimizer, lr_mul = 1, d_model = dim_model, n_warmup_steps = warmup_steps, n_current_steps = 0)
fit(model, sched, mode_lst, train_dataloader, val_dataloader, src_maxlen, num_tokens, epochs, val_min, patience, best_chk_path, device, logger)
torch.save({
            'model_state_dict': model.state_dict(),
            'final_steps': sched.get_final_steps(),
            }, end_chk_path)
