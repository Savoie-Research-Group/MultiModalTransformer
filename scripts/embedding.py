import torch
import torch.nn as nn

class PositionEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size,
        dim_model,
        device
    ):
        super().__init__()

        self.dim_model = dim_model
        self.token_emb = nn.Embedding(vocab_size, dim_model)
        self.device = device

    def positional_encoding(self, maxlen, dim_model):
        pos_encoding = torch.zeros(maxlen, dim_model)
        positions_list = torch.arange(0, maxlen, dtype=torch.float).view(-1, 1)
        division_term = torch.pow(10000,torch.arange(0, dim_model, 2).float() / dim_model)
        pos_encoding[:, 0::2] = torch.sin(positions_list / division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list / division_term)
        return pos_encoding

    def forward(self, x_input):
        maxlen = x_input.size(1)
        dim_model = self.dim_model
        positions = self.positional_encoding(maxlen, dim_model)
        positions = positions.to(self.device)
        emb = self.token_emb(x_input)
        return emb+positions
