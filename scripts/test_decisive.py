import argparse
import torch
import pickle
import numpy as np
import logging
from model_decisive import deduction_transformer
from calc_decisive import calc_decisive
from load_data import load_csv
from parse_config_decisive import parse_config

parser = argparse.ArgumentParser(description = 'Evaluate deduction model performance')
parser.add_argument('-c',dest = 'config',default = 'config_eval_decisive.txt',
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
mol_path = c['mol_path']
r_path = c['r_path']
ms_path = c['ms_path']
ir_path = c['ir_path']
nmr_path = c['nmr_path']

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

#Load test set data path
src_test_fp = c['src_test']
tgt_test_fp = c['tgt_test']
null_src_test_fp = c['null_src_test']
null_tgt_test_fp = c['null_tgt_test']

#Calculate overall accuracy
model.eval()
test_all = 24941
null_test_all = 14810

log_name = 'decisive.log'
logging.basicConfig(filename = log_name, level = logging.INFO,
                    format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(log_name)

logger.info(f'model load complete')

for i in range(0,25000,100):
    X_test, Y_test = load_csv(src_test_fp, tgt_test_fp, i, 100)
    batch = (X_test,Y_test)
    logger.info(f'data load complete')
    calc_decisive(logger,batch,model,src_maxlen,tgt_maxlen,idx_to_ch_dict,target_start_token_idx,target_end_token_idx,mol_path,r_path,ms_path,ir_path,nmr_path)
    logger.info(f'{i} done')

for i in range(0,14900,100):
    batch = (null_X_test[i:i+100,:],null_Y_test[i:i+100,:])
    calc_decisive(batch,model,src_maxlen,tgt_maxlen,idx_to_ch_dict,target_start_token_idx,target_end_token_idx,mol_path,r_path,ms_path,ir_path,nmr_path)

