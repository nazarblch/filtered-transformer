import math
from torch import nn, Tensor
import torch
from torch.nn import TransformerEncoderLayer, TransformerEncoder


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


class FloatTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pre_proc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim)
        )

        d_model = hidden_dim
        dropout = 0.0
        nhead = 4
        self.pos_encoder = PositionalEncoding2(hidden_dim)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, hidden_dim, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 2)

    def forward(self, x: Tensor) -> Tensor:
        B, L, D = x.shape
        out = self.pre_proc(x.reshape(B * L, D)).reshape(B, L, -1)
        out = self.pos_encoder(out) * math.sqrt(L)
        out = self.transformer_encoder(out)

        return out