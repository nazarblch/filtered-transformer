import math
from torch import nn, Tensor
import torch


def create_position_ids_from_inputs_embeds(inputs_embeds: Tensor, padding_idx=1):

    input_shape = inputs_embeds.size()[:-1]
    sequence_length = input_shape[1]

    position_ids = torch.arange(
        padding_idx + 1, sequence_length + padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
    )
    return position_ids.unsqueeze(0).expand(input_shape)


class PositionalEncoding(nn.Module):

    def __init__(self, embedding_dim: int, max_len: int = 2000):
        super().__init__()
        self.embeddings = nn.Embedding(max_len, embedding_dim)

    def forward(self, x: Tensor) -> Tensor:

        pos = create_position_ids_from_inputs_embeds(x, 1)
        x = x + self.embeddings(pos)
        return x


class PositionalEncoding2(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, :x.size(1)]
        return x


class LinearEmbedWithPos(nn.Module):

    def __init__(self, dim: int, d_model: int, multiplier: float, max_len: int = 1000):
        super().__init__()
        self.embed = nn.Linear(dim, d_model)
        self.pos_encoder = PositionalEncoding2(d_model, max_len=max_len)
        self.multiplier = multiplier

    def forward(self, x: Tensor):
        out = self.embed(x)
        return self.pos_encoder(out) * math.sqrt(self.multiplier)


class EmbedWithPos(nn.Module):

    def __init__(self, n: int, d_model: int, multiplier: float, max_len: int = 1000):
        super().__init__()
        self.embed = nn.Embedding(n, d_model)
        self.pos_encoder = PositionalEncoding2(d_model, max_len=max_len)
        self.multiplier = multiplier

    def forward(self, x: Tensor):
        out = self.embed(x)
        return self.pos_encoder(out) * math.sqrt(self.multiplier)