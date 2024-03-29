#Regular beam search implementation on calculating top-n accuracy.

import torch
import numpy as np
import math

def beam_search(model, source, beam_width, src_maxlen, tgt_maxlen, target_start_token_idx, target_end_token_idx):
    mode_lst = model.mode_lst
    device = model.device
    bs = source.size(dim=0)
    num_hid = model.dim_model
    num_classes = model.num_tokens
    src_padding_mask = model.create_pad_mask(source[:,0:src_maxlen],0)
    src_enc = model.src_input(source[:,0:src_maxlen].long())
    src_enc = model.src_encoder(src = src_enc,src_key_padding_mask = src_padding_mask)
    if mode_lst[0] == 1:
        ms_enc = model.ms_input(source[:,src_maxlen:src_maxlen+999].long())
        ms_enc = model.ms_encoder(ms_enc)
    if mode_lst[1] == 1:
        ir_enc = model.ir_input(source[:,src_maxlen+999:src_maxlen+1899].long())
        ir_enc = model.ir_encoder(ir_enc)
    if mode_lst[2] == 1:
        nmr_enc = model.nmr_input(source[:,src_maxlen+1899:].long())
        nmr_enc = model.nmr_encoder(nmr_enc)
    seq_len = src_enc.size(dim=1)
    emb_dim = src_enc.size(dim=2)
    output_list = []

    for _ in range(bs):
        most_prob_flag = False
        out_seq = []
        dec_input = torch.ones((1,1)) * target_start_token_idx
        tgt_padding_mask = model.create_pad_mask(dec_input,0)
        sequence_length = dec_input.size(1)
        tgt_mask = model.get_tgt_mask(sequence_length).to(device)
        enc_input_src = torch.reshape(src_enc[_,:,:],(1,seq_len,emb_dim))
        src_padding_mask = model.create_pad_mask(source[_,0:src_maxlen],0)
        src_padding_mask = torch.reshape(src_padding_mask,(1,src_maxlen))
        dec_out_src = model.src_decoder(tgt = model.tgt_input(dec_input.long()),memory = enc_input_src,tgt_mask = tgt_mask,tgt_key_padding_mask = tgt_padding_mask,memory_key_padding_mask = src_padding_mask)
        dec_out_final = model.src_classifier(dec_out_src)
        if mode_lst[0] == 1:
            enc_input_ms = torch.reshape(ms_enc[_,:,:],(1,999,emb_dim))
            dec_out_ms = model.ms_decoder(tgt = model.tgt_input(dec_input.long()),memory = enc_input_ms,tgt_mask = tgt_mask,tgt_key_padding_mask = tgt_padding_mask)
            dec_out_ms = model.ms_classifier(dec_out_ms)
            dec_out_final = torch.cat((dec_out_final,dec_out_ms),axis=2)
        if mode_lst[1] == 1:
            enc_input_ir = torch.reshape(ir_enc[_,:,:],(1,900,emb_dim))
            dec_out_ir = model.ir_decoder(tgt = model.tgt_input(dec_input.long()),memory = enc_input_ir,tgt_mask = tgt_mask,tgt_key_padding_mask = tgt_padding_mask)
            dec_out_ir = model.ir_classifier(dec_out_ir)
            dec_out_final = torch.cat((dec_out_final,dec_out_ir),axis=2)
        if mode_lst[2] == 1:
            enc_input_nmr = torch.reshape(nmr_enc[_,:,:],(1,993,emb_dim))
            dec_out_nmr = model.nmr_decoder(tgt = model.tgt_input(dec_input.long()),memory = enc_input_nmr,tgt_mask = tgt_mask,tgt_key_padding_mask = tgt_padding_mask)
            dec_out_nmr = model.nmr_classifier(dec_out_nmr)
            dec_out_final = torch.cat((dec_out_final,dec_out_nmr),axis=2)
        
        prob_tensor = model.ded_classifier(dec_out_final)
        prob_tensor = torch.reshape(prob_tensor,(num_classes,))
        p_numpy = prob_tensor.detach().numpy()
        p_numpy = (p_numpy - np.amin(p_numpy))/(np.amax(p_numpy)-np.amin(p_numpy))
        prob_tensor = torch.tensor(p_numpy)
        score_tensor = torch.log(prob_tensor)
        score_tensor_final, next_ys = torch.sort(score_tensor, descending=True)
        score_tensor_final = score_tensor_final[0:beam_width]

        next_ys = next_ys[0:beam_width]
        next_ys = torch.reshape(next_ys,(beam_width,1))
        dec_input = torch.ones((beam_width,1)) * target_start_token_idx
        next_ys = torch.cat((dec_input,next_ys), -1)
        src_padding_mask = src_padding_mask.repeat((beam_width,1))
        tgt_padding_mask = model.create_pad_mask(next_ys,0)
        sequence_length = next_ys.size(1)
        tgt_mask = model.get_tgt_mask(sequence_length).to(device)
        enc_input_src = enc_input_src.repeat((beam_width,1,1))
        if mode_lst[0] == 1:
            enc_input_ms = enc_input_ms.repeat((beam_width,1,1))
        if mode_lst[1] == 1:
            enc_input_ir = enc_input_ir.repeat((beam_width,1,1))
        if mode_lst[2] == 1:
            enc_input_nmr = enc_input_nmr.repeat((beam_width,1,1))
        for i in range(tgt_maxlen-2):
            dec_out_src = model.src_decoder(tgt = model.tgt_input(next_ys.long()),memory = enc_input_src,tgt_mask = tgt_mask,tgt_key_padding_mask = tgt_padding_mask,memory_key_padding_mask = src_padding_mask)
            dec_out_final = model.src_classifier(dec_out_src)
            if mode_lst[0] == 1:
                dec_out_ms = model.ms_decoder(tgt = model.tgt_input(next_ys.long()),memory = enc_input_ms,tgt_mask = tgt_mask,tgt_key_padding_mask = tgt_padding_mask)
                dec_out_ms = model.ms_classifier(dec_out_ms)
                dec_out_final = torch.cat((dec_out_final,dec_out_ms),axis=2)
            if mode_lst[1] == 1:
                dec_out_ir = model.ir_decoder(tgt = model.tgt_input(next_ys.long()),memory = enc_input_ir,tgt_mask = tgt_mask,tgt_key_padding_mask = tgt_padding_mask)
                dec_out_ir = model.ir_classifier(dec_out_ir)
                dec_out_final = torch.cat((dec_out_final,dec_out_ir),axis=2)
            if mode_lst[2] == 1:
                dec_out_nmr = model.nmr_decoder(tgt = model.tgt_input(next_ys.long()),memory = enc_input_nmr,tgt_mask = tgt_mask,tgt_key_padding_mask = tgt_padding_mask)
                dec_out_nmr = model.nmr_classifier(dec_out_nmr)
                dec_out_final = torch.cat((dec_out_final,dec_out_nmr),axis=2)
            
            prob_tensor = model.ded_classifier(dec_out_final)
            prob_tensor = prob_tensor[:,-1,:]
            p_numpy = prob_tensor.detach().numpy()
            for j in range(beam_width):
                p_numpy[j,:] = (p_numpy[j,:] - np.amin(p_numpy[j,:]))/(np.amax(p_numpy[j,:])-np.amin(p_numpy[j,:]))
            score_tensor = torch.reshape(score_tensor_final,(beam_width,1))
            score_tensor = score_tensor.repeat(1,num_classes)
            prob_tensor = torch.tensor(p_numpy)
            score_tensor = score_tensor + torch.log(prob_tensor)
            s_numpy = score_tensor.detach().numpy()
            score_tensor = torch.reshape(score_tensor, (beam_width*num_classes,))
            score_tensor_final,score_tensor_arg = torch.sort(score_tensor, descending = True)
            score_tensor_final = score_tensor_final[0:beam_width]
            score_array_final = score_tensor_final.detach().numpy()
            idx_tuple = np.unravel_index(np.argsort(s_numpy.ravel()), s_numpy.shape)
            next_ys_zero = np.zeros((beam_width,i+3),dtype = int)
            for j in range(beam_width):
                p_numpy[j,:] = (p_numpy[j,:] - np.amin(p_numpy[j,:]))/(np.amax(p_numpy[j,:])-np.amin(p_numpy[j,:]))
            score_tensor = torch.reshape(score_tensor_final,(beam_width,1))
            score_tensor = score_tensor.repeat(1,num_classes)
            prob_tensor = torch.tensor(p_numpy)
            score_tensor = score_tensor + torch.log(prob_tensor)
            s_numpy = score_tensor.detach().numpy()
            score_tensor = torch.reshape(score_tensor, (beam_width*num_classes,))
            score_tensor_final,score_tensor_arg = torch.sort(score_tensor, descending = True)
            score_tensor_final = score_tensor_final[0:beam_width]
            score_array_final = score_tensor_final.detach().numpy()
            idx_tuple = np.unravel_index(np.argsort(s_numpy.ravel()), s_numpy.shape)
            next_ys_zero = np.zeros((beam_width,i+3),dtype = int)
            for j in range(beam_width):
                next_ys_zero[j][0:i+2]=next_ys.detach().numpy()[idx_tuple[0][-1-j]]
                next_ys_zero[j][i+2]=idx_tuple[1][-1-j]
            next_ys = torch.tensor(next_ys_zero)
            #Check whether beam search finishes
            if next_ys[0][-1] == target_end_token_idx:
                seq, seq_score = (next_ys[0,:].detach().numpy(),score_tensor_final.detach().numpy()[0])
                out_seq = out_seq + [(seq, seq_score/len(seq))]
                most_prob_flag = True
                score_array_final[0] = score_array_final[0] - 20
                if len(out_seq) >= 5:
                    break
            for i in range(beam_width):
                if i != 0:
                    if next_ys[i][-1] == target_end_token_idx:
                        seq, seq_score = (next_ys[i,:].detach().numpy(),score_tensor_final.detach().numpy()[i])
                        out_seq = out_seq + [(seq, seq_score/len(seq))]
                        score_array_final[i] = score_array_final[i] - 20
            score_tensor_final = torch.tensor(score_array_final)
            if most_prob_flag == True:
                if len(out_seq) >= 5:
                    break
            tgt_padding_mask = model.create_pad_mask(next_ys,0)
            sequence_length = next_ys.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)
        if len(out_seq) < 5:
            out_seq = out_seq + [(next_ys[0,:].detach().numpy(),-math.inf)]*beam_width
        output_list = output_list + [out_seq]
    return output_list
