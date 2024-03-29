import torch
import torch.nn as nn
from embedding import PositionEmbedding

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
