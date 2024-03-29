import torch
import torch.nn as nn
from embedding import PositionEmbedding
import numpy as np
import math

class deduction_transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
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
    ):
        super().__init__()

        self.model_type = 'Transformer'
        self.dim_model = dim_model
        self.mode_lst = mode_lst
        self.num_tokens = num_tokens
        self.device = device

        self.src_input = PositionEmbedding(vocab_size, dim_model, device)
        self.src_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim_model,num_heads,num_feed_forward,dropout = 0.1, layer_norm_eps = 1e-6, batch_first=True),num_src_enc_layers)
        self.src_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(dim_model,num_heads,num_feed_forward,dropout = 0.1, layer_norm_eps = 1e-6, batch_first=True),num_src_dec_layers)
        self.src_classifier = nn.Linear(dim_model, num_tokens)
        self.tgt_input = PositionEmbedding(vocab_size, dim_model, device)
        if mode_lst[0] == 1:
            self.ms_input = PositionEmbedding(vocab_size, dim_model, device)
            self.ms_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim_model,num_heads,num_feed_forward,dropout = 0.1, layer_norm_eps = 1e-6, batch_first=True),num_spec_enc_layers)
            self.ms_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(dim_model,num_heads,num_feed_forward,dropout = 0.1, layer_norm_eps = 1e-6, batch_first=True),num_spec_dec_layers)
            self.ms_classifier = nn.Linear(dim_model, num_tokens)
        if mode_lst[1] == 1:
            self.ir_input = PositionEmbedding(vocab_size, dim_model, device)
            self.ir_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim_model,num_heads,num_feed_forward,dropout = 0.1, layer_norm_eps = 1e-6, batch_first=True),num_spec_enc_layers)
            self.ir_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(dim_model,num_heads,num_feed_forward,dropout = 0.1, layer_norm_eps = 1e-6, batch_first=True),num_spec_dec_layers)
            self.ir_classifier = nn.Linear(dim_model, num_tokens)
        if mode_lst[2] == 1:
            self.nmr_input = PositionEmbedding(vocab_size, dim_model, device)
            self.nmr_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim_model,num_heads,num_feed_forward,dropout = 0.1, layer_norm_eps = 1e-6, batch_first=True),num_spec_enc_layers)
            self.nmr_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(dim_model,num_heads,num_feed_forward,dropout = 0.1, layer_norm_eps = 1e-6, batch_first=True),num_spec_dec_layers)
            self.nmr_classifier = nn.Linear(dim_model, num_tokens)
        self.ded_classifier = nn.Linear((1+mode_lst[0]+mode_lst[1]+mode_lst[2])*num_tokens, num_tokens)

    def forward(
        self,
        x_src,
        x_ms,
        x_ir,
        x_nmr,
        y,
    ):
        src_padding_mask = self.create_pad_mask(x_src,0)
        x_src = x_src.long()
        x_src = self.src_input(x_src)
        x_src = self.src_encoder(src = x_src,src_key_padding_mask = src_padding_mask)
        y = y.long()
        sequence_length = y.size(1)
        tgt_padding_mask = self.create_pad_mask(y,0)
        tgt_mask = self.get_tgt_mask(sequence_length).to(self.device)
        y = self.tgt_input(y)
        y_src = self.src_decoder(tgt = y,memory = x_src,tgt_mask = tgt_mask,tgt_key_padding_mask = tgt_padding_mask,memory_key_padding_mask = src_padding_mask)
        y_final = self.src_classifier(y_src)
        if self.mode_lst[0] == 1:
            x_ms = x_ms.long()
            x_ms = self.ms_input(x_ms)
            x_ms = self.ms_encoder(x_ms)
            y_ms = self.ms_decoder(tgt = y,memory = x_ms,tgt_mask = tgt_mask,tgt_key_padding_mask = tgt_padding_mask)
            y_ms = self.ms_classifier(y_ms)
            y_final = torch.cat((y_final,y_ms),axis=2)
        if self.mode_lst[1] == 1:
            x_ir = x_ir.long()
            x_ir = self.ir_input(x_ir)
            x_ir = self.ir_encoder(x_ir)
            y_ir = self.ir_decoder(tgt = y,memory = x_ir,tgt_mask = tgt_mask,tgt_key_padding_mask = tgt_padding_mask)
            y_ir = self.ir_classifier(y_ir)
            y_final = torch.cat((y_final,y_ir),axis=2)
        if self.mode_lst[2] == 1:
            x_nmr = x_nmr.long()
            x_nmr = self.nmr_input(x_nmr)
            x_nmr = self.nmr_encoder(x_nmr)
            y_nmr = self.nmr_decoder(tgt = y,memory = x_nmr,tgt_mask = tgt_mask,tgt_key_padding_mask = tgt_padding_mask)
            y_nmr = self.nmr_classifier(y_nmr)
            y_final = torch.cat((y_final,y_nmr),axis=2)
        return self.ded_classifier(y_final)

    def get_tgt_mask(self, size):
        mask = torch.tril(torch.ones(size, size) == 1) 
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask

    def create_pad_mask(self, matrix, pad_token):
        return (matrix == pad_token)

    def output_tensor_to_prob(self, dec_out_tensor, num_classes, beam_width, first_flag):
        if first_flag:
            prob_tensor = self.ded_classifier(dec_out_tensor)
            prob_tensor = torch.reshape(prob_tensor,(num_classes,))
            p_numpy = prob_tensor.detach().numpy()
            p_numpy = (p_numpy - np.amin(p_numpy))/(np.amax(p_numpy)-np.amin(p_numpy))
            return p_numpy
        else:
            prob_tensor = self.ded_classifier(dec_out_tensor)
            prob_tensor = prob_tensor[:,-1,:]
            p_numpy = prob_tensor.detach().numpy()
            for j in range(beam_width):
                p_numpy[j,:] = (p_numpy[j,:] - np.amin(p_numpy[j,:]))/(np.amax(p_numpy[j,:])-np.amin(p_numpy[j,:]))
            return p_numpy

    def beam_search(self, source, beam_width, src_maxlen, tgt_maxlen, target_start_token_idx, target_end_token_idx):
        mode_lst = self.mode_lst
        device = self.device
        bs = source.size(dim=0)
        num_hid = self.dim_model
        num_classes = self.num_tokens
        src_padding_mask = self.create_pad_mask(source[:,0:src_maxlen],0)
        r_list = []
        ms_list = []
        ir_list = []
        nmr_list = []
        src_enc = self.src_input(source[:,0:src_maxlen].long())
        src_enc = self.src_encoder(src = src_enc,src_key_padding_mask = src_padding_mask)
        if mode_lst[0] == 1:
            ms_enc = self.ms_input(source[:,src_maxlen:src_maxlen+999].long())
            ms_enc = self.ms_encoder(ms_enc)
        if mode_lst[1] == 1:
            ir_enc = self.ir_input(source[:,src_maxlen+999:src_maxlen+1899].long())
            ir_enc = self.ir_encoder(ir_enc)
        if mode_lst[2] == 1:
            nmr_enc = self.nmr_input(source[:,src_maxlen+1899:].long())
            nmr_enc = self.nmr_encoder(nmr_enc)
        seq_len = src_enc.size(dim=1)
        emb_dim = src_enc.size(dim=2)
        output_list = []
        for _ in range(bs):
            r_string = ''
            ms_string = ''
            ir_string = ''
            nmr_string = ''
            most_prob_flag = False
            out_seq = []
            dec_input = torch.ones((1,1)) * target_start_token_idx
            tgt_padding_mask = self.create_pad_mask(dec_input,0)
            sequence_length = dec_input.size(1)
            tgt_mask = self.get_tgt_mask(sequence_length).to(device)
            enc_input_src = torch.reshape(src_enc[_,:,:],(1,seq_len,emb_dim))
            src_padding_mask = self.create_pad_mask(source[_,0:src_maxlen],0)
            src_padding_mask = torch.reshape(src_padding_mask,(1,src_maxlen))
            dec_out_src = self.src_decoder(tgt = self.tgt_input(dec_input.long()),memory = enc_input_src,tgt_mask = tgt_mask,tgt_key_padding_mask = tgt_padding_mask,memory_key_padding_mask = src_padding_mask)
            dec_out_all = self.src_classifier(dec_out_src)
            dec_out_0src = torch.zeros_like(dec_out_all)
            if mode_lst[0] == 1:
                enc_input_ms = torch.reshape(ms_enc[_,:,:],(1,999,emb_dim))
                dec_out_ms = self.ms_decoder(tgt = self.tgt_input(dec_input.long()),memory = enc_input_ms,tgt_mask = tgt_mask,tgt_key_padding_mask = tgt_padding_mask)
                dec_out_ms = self.ms_classifier(dec_out_ms)
                dec_out_0ms = torch.zeros_like(dec_out_ms)
                dec_out_final = torch.cat((dec_out_all,dec_out_ms),axis=2)
                dec_out_final_0src = torch.cat((dec_out_0src,dec_out_ms),axis=2)
                dec_out_final_0ms = torch.cat((dec_out_all,dec_out_0ms),axis=2)
            if mode_lst[1] == 1:
                enc_input_ir = torch.reshape(ir_enc[_,:,:],(1,900,emb_dim))
                dec_out_ir = self.ir_decoder(tgt = self.tgt_input(dec_input.long()),memory = enc_input_ir,tgt_mask = tgt_mask,tgt_key_padding_mask = tgt_padding_mask)
                dec_out_ir = self.ir_classifier(dec_out_ir)
                dec_out_0ir = torch.zeros_like(dec_out_ir)
                dec_out_final = torch.cat((dec_out_final,dec_out_ir),axis=2)
                dec_out_final_0src = torch.cat((dec_out_final_0src,dec_out_ir),axis=2)
                dec_out_final_0ms = torch.cat((dec_out_final_0ms,dec_out_ir),axis=2)
                dec_out_final_0ir = torch.cat((dec_out_all,dec_out_ms,dec_out_0ir),axis=2)
            if mode_lst[2] == 1:
                enc_input_nmr = torch.reshape(nmr_enc[_,:,:],(1,993,emb_dim))
                dec_out_nmr = self.nmr_decoder(tgt = self.tgt_input(dec_input.long()),memory = enc_input_nmr,tgt_mask = tgt_mask,tgt_key_padding_mask = tgt_padding_mask)
                dec_out_nmr = self.nmr_classifier(dec_out_nmr)
                dec_out_0nmr = torch.zeros_like(dec_out_nmr)
                dec_out_final = torch.cat((dec_out_final,dec_out_nmr),axis=2)
                dec_out_final_0src = torch.cat((dec_out_final_0src,dec_out_nmr),axis=2)
                dec_out_final_0ms = torch.cat((dec_out_final_0ms,dec_out_nmr),axis=2)
                dec_out_final_0ir = torch.cat((dec_out_final_0ir,dec_out_nmr),axis=2)
                dec_out_final_0nmr = torch.cat((dec_out_all,dec_out_ms,dec_out_ir,dec_out_0nmr),axis=2)

            p_numpy = self.output_tensor_to_prob(dec_out_final,num_classes,beam_width,first_flag=True)
            p_numpy_0src = self.output_tensor_to_prob(dec_out_final_0src,num_classes,beam_width,first_flag=True)
            p_numpy_0ms = self.output_tensor_to_prob(dec_out_final_0ms,num_classes,beam_width,first_flag=True)
            p_numpy_0ir = self.output_tensor_to_prob(dec_out_final_0ir,num_classes,beam_width,first_flag=True)
            p_numpy_0nmr = self.output_tensor_to_prob(dec_out_final_0nmr,num_classes,beam_width,first_flag=True)
            if np.argmax(p_numpy) != np.argmax(p_numpy_0src):
                r_string += 'R'
            else:
                r_string += 'F'
            if np.argmax(p_numpy) != np.argmax(p_numpy_0ms):
                ms_string += 'M'
            else:
                ms_string += 'F'
            if np.argmax(p_numpy) != np.argmax(p_numpy_0ir):
                ir_string += 'I'
            else:
                ir_string += 'F'
            if np.argmax(p_numpy) != np.argmax(p_numpy_0nmr):
                nmr_string += 'N'
            else:
                nmr_string += 'F'

            prob_tensor = torch.tensor(p_numpy)
            score_tensor = torch.log(prob_tensor)
            score_tensor_final, next_ys = torch.sort(score_tensor, descending=True)
            score_tensor_final = score_tensor_final[0:beam_width]

            next_ys = next_ys[0:beam_width]
            next_ys = torch.reshape(next_ys,(beam_width,1))
            dec_input = torch.ones((beam_width,1)) * target_start_token_idx
            next_ys = torch.cat((dec_input,next_ys), -1)
            src_padding_mask = src_padding_mask.repeat((beam_width,1))
            tgt_padding_mask = self.create_pad_mask(next_ys,0)
            sequence_length = next_ys.size(1)
            tgt_mask = self.get_tgt_mask(sequence_length).to(device)
            enc_input_src = enc_input_src.repeat((beam_width,1,1))
            if mode_lst[0] == 1:
                enc_input_ms = enc_input_ms.repeat((beam_width,1,1))
            if mode_lst[1] == 1:
                enc_input_ir = enc_input_ir.repeat((beam_width,1,1))
            if mode_lst[2] == 1:
                enc_input_nmr = enc_input_nmr.repeat((beam_width,1,1))

            for i in range(tgt_maxlen-2):
                dec_out_src = self.src_decoder(tgt = self.tgt_input(next_ys.long()),memory = enc_input_src,tgt_mask = tgt_mask,tgt_key_padding_mask = tgt_padding_mask,memory_key_padding_mask = src_padding_mask)
                dec_out_all = self.src_classifier(dec_out_src)
                dec_out_0src = torch.zeros_like(dec_out_all)
                if mode_lst[0] == 1:
                    dec_out_ms = self.ms_decoder(tgt = self.tgt_input(next_ys.long()),memory = enc_input_ms,tgt_mask = tgt_mask,tgt_key_padding_mask = tgt_padding_mask)
                    dec_out_ms = self.ms_classifier(dec_out_ms)
                    dec_out_0ms = torch.zeros_like(dec_out_ms)
                    dec_out_final = torch.cat((dec_out_all,dec_out_ms),axis=2)
                    dec_out_final_0src = torch.cat((dec_out_0src,dec_out_ms),axis=2)
                    dec_out_final_0ms = torch.cat((dec_out_all,dec_out_0ms),axis=2)
                if mode_lst[1] == 1:
                    dec_out_ir = self.ir_decoder(tgt = self.tgt_input(next_ys.long()),memory = enc_input_ir,tgt_mask = tgt_mask,tgt_key_padding_mask = tgt_padding_mask)
                    dec_out_ir = self.ir_classifier(dec_out_ir)
                    dec_out_0ir = torch.zeros_like(dec_out_ir)
                    dec_out_final = torch.cat((dec_out_final,dec_out_ir),axis=2)
                    dec_out_final_0src = torch.cat((dec_out_final_0src,dec_out_ir),axis=2)
                    dec_out_final_0ms = torch.cat((dec_out_final_0ms,dec_out_ir),axis=2)
                    dec_out_final_0ir = torch.cat((dec_out_all,dec_out_ms,dec_out_0ir),axis=2)
                if mode_lst[2] == 1:
                    dec_out_nmr = self.nmr_decoder(tgt = self.tgt_input(next_ys.long()),memory = enc_input_nmr,tgt_mask = tgt_mask,tgt_key_padding_mask = tgt_padding_mask)
                    dec_out_nmr = self.nmr_classifier(dec_out_nmr)
                    dec_out_0nmr = torch.zeros_like(dec_out_nmr)
                    dec_out_final = torch.cat((dec_out_final,dec_out_nmr),axis=2)
                    dec_out_final_0src = torch.cat((dec_out_final_0src,dec_out_nmr),axis=2)
                    dec_out_final_0ms = torch.cat((dec_out_final_0ms,dec_out_nmr),axis=2)
                    dec_out_final_0ir = torch.cat((dec_out_final_0ir,dec_out_nmr),axis=2)
                    dec_out_final_0nmr = torch.cat((dec_out_all,dec_out_ms,dec_out_ir,dec_out_0nmr),axis=2)

                p_numpy = self.output_tensor_to_prob(dec_out_final,num_classes,beam_width,first_flag=False)
                p_numpy_0src = self.output_tensor_to_prob(dec_out_final_0src,num_classes,beam_width,first_flag=False)
                p_numpy_0ms = self.output_tensor_to_prob(dec_out_final_0ms,num_classes,beam_width,first_flag=False)
                p_numpy_0ir = self.output_tensor_to_prob(dec_out_final_0ir,num_classes,beam_width,first_flag=False)
                p_numpy_0nmr = self.output_tensor_to_prob(dec_out_final_0nmr,num_classes,beam_width,first_flag=False)
                if np.argmax(p_numpy[0,:]) != np.argmax(p_numpy_0src[0,:]):
                    r_string += 'R'
                else:
                    r_string += 'F'
                if np.argmax(p_numpy[0,:]) != np.argmax(p_numpy_0ms[0,:]):
                    ms_string += 'M'
                else:
                    ms_string += 'F'
                if np.argmax(p_numpy[0,:]) != np.argmax(p_numpy_0ir[0,:]):
                    ir_string += 'I'
                else:
                    ir_string += 'F'
                if np.argmax(p_numpy[0,:]) != np.argmax(p_numpy_0nmr[0,:]):
                    nmr_string += 'N'
                else:
                    nmr_string += 'F'

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


                if next_ys[0][-1] == target_end_token_idx:
                    seq, seq_score = (next_ys[0,:].detach().numpy(),score_tensor_final.detach().numpy()[0])
                    out_seq = out_seq + [(seq, seq_score/len(seq))]
                    most_prob_flag = True
                    score_array_final[0] = score_array_final[0] - 20
                    if len(out_seq) >= beam_width:
                        break
                for i in range(beam_width):
                    if i != 0:
                        if next_ys[i][-1] == target_end_token_idx:
                            seq, seq_score = (next_ys[i,:].detach().numpy(),score_tensor_final.detach().numpy()[i])
                            out_seq = out_seq + [(seq, seq_score/len(seq))]
                            score_array_final[i] = score_array_final[i] - 20
                score_tensor_final = torch.tensor(score_array_final)
                if most_prob_flag == True:
                    if len(out_seq) >= beam_width:
                        break
                tgt_padding_mask = self.create_pad_mask(next_ys,0)
                sequence_length = next_ys.size(1)
                tgt_mask = self.get_tgt_mask(sequence_length).to(device)
            if len(out_seq) < beam_width:
                out_seq = out_seq + [(next_ys[0,:].detach().numpy(),-math.inf)]*beam_width
            output_list = output_list + [out_seq]
            r_list.append(r_string)
            ms_list.append(ms_string)
            ir_list.append(ir_string)
            nmr_list.append(nmr_string)
        return output_list, r_list, ms_list, ir_list, nmr_list
