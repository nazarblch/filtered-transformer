import math
from copy import deepcopy

import torch
from gena_lm.modeling_bert import BertEncoder
from torch import nn, Tensor
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from transformers import BertModel, BertConfig
from typing import Dict

from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from models.pos_encoding import PositionalEncoding2


class RecurrentTransformer(nn.Module):

    def __init__(self,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True),
            num_layers
        )

    def forward(self, x: Tensor, state: Tensor) -> Tensor:
        xs = torch.cat([x, state], dim=1)
        return self.encoder(xs)[:, x.shape[1]:]


class BertRecurrentTransformer(RecurrentTransformer):

    def __init__(self,
                 bert: BertModel,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 2048):
        super().__init__()

        self.bert: BertModel = bert

        config = deepcopy(bert.config)

        config.num_attention_heads = nhead
        config.num_hidden_layers = num_layers
        config.intermediate_size = dim_feedforward

        self.encoder = BertEncoder(config)

    def extract_hidden(self, h: BaseModelOutputWithPoolingAndCrossAttentions) -> Tensor:
        return h['last_hidden_state']

    def forward(self, x: Dict[str, Tensor], state: Tensor) -> Tensor:
        h = self.extract_hidden(self.bert.encoder(x["input_ids"], output_hidden_states=True))
        assert state.shape[-1] == h.shape[-1]
        assert state.shape[0] == h.shape[0]
        hs = torch.cat([h, state], dim=1)
        return self.encoder(hs)['last_hidden_state'][:, h.shape[1]:]

class HierarchicalTransformer(nn.Module):
    def __init__(self, *transformers: nn.Transformer, dim: int, chunk_size: int):
        super().__init__()
        self.transformers = nn.ModuleList(transformers)
        self.hidden_dim = transformers[0].d_model
        self.pos_encoder = PositionalEncoding2(self.hidden_dim)
        self.embed = nn.Linear(dim, self.hidden_dim)
        self.chunk_size = chunk_size
        self.levels_count = len(transformers)

    def add_pos(self, x: Tensor):
        return self.pos_encoder(x) * math.sqrt(self.chunk_size)

    def chunk(self, x):
        B, L, D = x.shape
        assert L % self.chunk_size == 0
        return x.view(B * L // self.chunk_size, self.chunk_size, D)

    def compress_data(self, chunks: Tensor, state: Tensor, level: int):
        B1, count, T, D = chunks.shape
        if count % 2 == 1:
            chunks = nn.functional.pad(chunks, (0, 0, 0, 0, 0, 1))
            count += 1
        chunks = chunks.reshape(B1 * count // 2, T * 2, D)
        state = state[torch.arange(0, B1, device=state.device).repeat_interleave(count // 2)]
        compressed = self.transformers[level](state, chunks)[:, 0:T, :]
        return compressed.reshape(B1, count // 2, T, D)

    def forward(self, state: Tensor, data: Tensor):
        B, T, D = data.shape
        data = self.add_pos(self.embed(data))
        x = self.chunk(data)
        state1 = state[torch.arange(0, B, device=state.device).repeat_interleave(T // self.chunk_size)]
        x = self.transformers[0](state1, x)
        count = T // self.chunk_size
        x = x.reshape(B, count, self.chunk_size, x.shape[-1])
        res = [x[:, :, 0]]

        for l in range(1, self.levels_count):
            x = self.compress_data(x, state, l)
            res.append(x[:, :, 0])

        return res


class TransformerClassifier(nn.Module):
    def __init__(self,
                 num_classes: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True),
            num_layers
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        return self.head(self.encoder(x)[:, -1])


class BertClassifier(nn.Module):
    def __init__(self,
                 num_classes: int,
                 config: BertConfig,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048):
        super().__init__()

        config = deepcopy(config)

        config.num_attention_heads = nhead
        config.num_hidden_layers = num_layers
        config.intermediate_size = dim_feedforward

        self.encoder = BertEncoder(config)
        d_model = config.hidden_size

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        return self.head(self.encoder(x)['last_hidden_state'][:, -1])

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