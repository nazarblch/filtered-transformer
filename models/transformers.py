import math
from copy import deepcopy
from functools import reduce
import torch
from gena_lm.modeling_bert import BertEncoder
from tokenizers import Tokenizer
from torch import nn, Tensor
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel, BertConfig, PreTrainedTokenizer
from typing import Dict, List
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from models.pos_encoding import PositionalEncoding2


class RecurrentOutput:
    def __init__(self, out: Tensor, state: Tensor):
        self.state = state
        self.out = out


class RecurrentOutputWithContext(RecurrentOutput):
    def __init__(self, out: Tensor, state: Tensor, context: Tensor):
        super().__init__(out, state)
        self.context = context


class RecurrentOutputSeq:
    def __init__(self):
        self.states = []
        self.outs = []
        self.masks = []

    def append(self, os: RecurrentOutput, mask=None):
        self.states.append(os.state)
        self.outs.append(os.out)
        self.masks.append(mask)

    def get_cat_out(self):
        return torch.cat(self.outs, dim=1)

    def get_sum_mask(self):
        return reduce(lambda m1, m2: m1 + m2, self.masks)


class RecurrentTransformer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, state: Tensor) -> RecurrentOutput:
        pass



class TorchRecurrentTransformer(RecurrentTransformer):

    def __init__(self,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding2(d_model)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_layers
        )

    def forward(self, x: Tensor, state: Tensor) -> RecurrentOutput:
        x = self.pos_encoder(x) * math.sqrt(x.shape[1])
        xs = torch.cat([x, state], dim=1)
        res = self.encoder(xs.transpose(0, 1)).transpose(0, 1)

        return RecurrentOutput(res[:, :x.shape[1]], res[:, x.shape[1]:])


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

    def forward(self, x: Dict[str, Tensor], state: Tensor) -> RecurrentOutputWithContext:
        h = self.extract_hidden(self.bert(input_ids=x["input_ids"], attention_mask=x['attention_mask'], output_hidden_states=False))
        assert state.shape[-1] == h.shape[-1]
        assert state.shape[0] == h.shape[0]
        shs = torch.cat([h, state], dim=1)
        shs = self.encoder(shs)['last_hidden_state']
        new_state = shs[:, h.shape[1]:]
        out = shs[:, : h.shape[1]]

        return RecurrentOutputWithContext(out, new_state, h)


class BertRecurrentTransformerWithTokenizer(BertRecurrentTransformer):

    def __init__(self, bert: BertModel, tokenizer: PreTrainedTokenizer, max_len: int, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 2048):
        super().__init__(bert, nhead, num_layers, dim_feedforward)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def forward(self, text_seq: List[str], state: Tensor) -> RecurrentOutputWithContext:
        tokens = self.tokenizer(text_seq, max_length=self.max_len, truncation=True)
        res = {"input_ids": pad_sequence([torch.tensor(t) for t in tokens["input_ids"]], batch_first=True).cuda(),
               "attention_mask": pad_sequence([torch.tensor(t) for t in tokens["attention_mask"]], batch_first=True,
                                              padding_value=0).cuda()}

        return super().forward(res, state)


class BertRecurrentLSTM(RecurrentTransformer):

    def __init__(self,
                 bert: BertModel,
                 num_layers: int = 3,
                 dim_feedforward: int = 2048):
        super().__init__()

        self.bert: BertModel = bert
        self.num_layers = num_layers
        self.encoder = nn.LSTM(input_size=768, hidden_size=dim_feedforward, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.encoder = chrono_init(self.encoder, Tmax=4000)

    def extract_hidden(self, h: BaseModelOutputWithPoolingAndCrossAttentions) -> Tensor:
        return h['last_hidden_state']

    def forward(self, x, state: Tensor) -> Tensor:
        h = self.extract_hidden(self.bert(input_ids=x["input_ids"], attention_mask=x["attention_mask"], output_hidden_states=False))
        assert state.shape[-1] == h.shape[-1]
        assert state.shape[0] == h.shape[0]
        h1, c1 = state.transpose(0, 1)[0:self.num_layers].contiguous(), state.transpose(0, 1)[self.num_layers:self.num_layers*2].contiguous()
        out, (h2, c2) = self.encoder.forward(h, (h1, c1))
        return torch.cat([h2, c2]).transpose(0, 1)


class RecurrentTransformerFromBert(RecurrentTransformer):

    def __init__(self,
                 bert: BertModel,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 2048):
        super().__init__()

        config = deepcopy(bert.config)
        config.num_attention_heads = nhead
        config.num_hidden_layers = num_layers
        config.intermediate_size = dim_feedforward

        self.encoder = BertEncoder(config)

    def forward(self, x: Tensor, state: Tensor) -> RecurrentOutput:
        xs = torch.cat([x, state], dim=1)
        xs = self.encoder(xs)['last_hidden_state']
        new_state = xs[:, x.shape[1]:]
        out = xs[:, : x.shape[1]]

        return RecurrentOutput(out, new_state)


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