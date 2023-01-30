import math
from abc import abstractmethod
from copy import deepcopy
from typing import Dict, List, Tuple, Any
from gena_lm.modeling_bert import BertModel, BertEncoder
from torch import nn, Tensor
import torch
from torch.nn.utils.rnn import pad_sequence
from common_modules.stoch_tensor import StochasticMultinomialTensor
from memup.base import SeqDataFilter, SD, State, Info, Done


class LearnableMask:
    def __init__(self, mask: Tensor, lens: List[int], multiplier: Tensor):
        self.mask = mask
        self.lens = lens
        self.multiplier = multiplier


class LearnableBlockFilter(SeqDataFilter[SD]):

    def __init__(self, transformer: nn.TransformerEncoder, chunk_size: int, hidden_dim: int, n_chunks: int):
        super().__init__()
        self.transformer = transformer
        self.chunk_size = chunk_size
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(0.1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_chunks)
        )

    def chunk(self, x):
        B, L, D = x.shape
        assert L % self.chunk_size == 0
        return x.view(B * L // self.chunk_size, self.chunk_size, D)

    def preprocess_data(self, data: Tensor):
        B = data.shape[0]

        with torch.no_grad():
            ttx = []
            for b in range(B // 16 + int(B % 16 != 0)):
                xb = self.chunk(data[16 * b:16 * b + 16])
                xb = self.transformer(xb)
                ttx.append(xb[:, -1])
            ttx = torch.cat(ttx)

        return ttx

    def make_filter(self, data: Tensor, state: Tensor, info: Info):
        B, T = data.shape[0], data.shape[1]

        state = state[:, -1][
            torch.arange(0, B, device=state.device).repeat_interleave(T // self.chunk_size)
        ]

        if "hidden_update_n" not in info:
            info["hidden_update_n"] = 0
        else:
            info["hidden_update_n"] += 1

        if info["hidden_update_n"] % 8 == 0:
            info["hidden"] = self.preprocess_data(data)

        F = self.head(torch.cat([state, info["hidden"].cuda()], dim=-1))
        F = F.view(B, T // self.chunk_size, F.shape[1]).transpose(1, 2)
        dist = StochasticMultinomialTensor(F)
        sample = dist.sample().max(1)[0]

        Fx = self.chunk(data)[(sample > 0.5).view(-1)]
        FS = state[(sample > 0.5).view(-1)]

        Fx = self.transformer(Fx)
        F1 = self.head(torch.cat([FS, Fx[:, -1]], dim=-1))
        F1 = dist.make_diff_sample(F1).max(1)[0].repeat_interleave(self.chunk_size, dim=0)

        FD = sample.repeat_interleave(self.chunk_size, dim=1)
        lens = (FD > 0.5).type(torch.int64).sum(-1).detach().cpu().numpy().tolist()
        # x3 = data[FD > 0.5] * F1[:, None]

        return LearnableMask(FD, lens, F1[:, None])

    def apply_mask(self, x: Tensor, mask: LearnableMask, train=True):
        x = x[mask.mask > 0.5]
        if train:
            x = x * mask.multiplier

        sp = torch.split(x, mask.lens, 0)
        return pad_sequence(sp, batch_first=True)

    @abstractmethod
    def filter_data(self, data: SD, mask: LearnableMask) -> SD:
        pass

    def forward(self, data: SD, state: State, info: Info, *args) -> Tuple[SD, Done]:

        mask = self.make_filter(data, state, info)
        return self.filter_data(data, mask), False


class BertLearnableBlockFilter(SeqDataFilter[Dict[str, Any]]):

    def __init__(self, bert: BertModel, chunk_size: int, n_chunks: int):
        super().__init__()

        self.bert: BertModel = bert

        config = deepcopy(bert.config)
        config.num_attention_heads = 4
        config.num_hidden_layers = 2
        config.intermediate_size = config.hidden_size

        self.encoder = BertEncoder(config)
        hidden_dim = config.hidden_size

        self.chunk_size = chunk_size
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(0.1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_chunks)
        )

    def chunk(self, x: Dict[str, Tensor]):
        B, L = x["input_ids"].shape[0], x["input_ids"].shape[1]
        assert L % self.chunk_size == 0
        return {k: v.view(B * L // self.chunk_size, self.chunk_size, *v.shape[2:]) for k, v in x.items()}

    def transform(self, data: Dict[str, Tensor]):
        m = self.bert.get_extended_attention_mask(data["attention_mask"], data["input_ids"].shape, data["input_ids"].device)
        h = self.bert.encoder(data["input_ids"], attention_mask=m, output_hidden_states=False)['last_hidden_state']
        return self.encoder(h)['last_hidden_state']

    def encode(self, data: Dict[str, Tensor]):
        m = self.bert.get_extended_attention_mask(data["attention_mask"], data["input_ids"].shape,
                                                  data["input_ids"].device)
        h = self.bert.encoder(data["input_ids"], attention_mask=m, output_hidden_states=False)['last_hidden_state']
        return h

    def preprocess_data(self, data: Dict[str, Tensor]):

        with torch.no_grad():
            xb = self.chunk(data)
            xb = self.transform(xb)
            ttx = xb[:, -1]
        return ttx

    def filter_data(self, data: Dict[str, Tensor], state: Tensor, info: Info):
        chunk_data = self.chunk(data)
        B, T = state.shape[0], (chunk_data["input_ids"].shape[0] * self.chunk_size) // state.shape[0]

        state = state[:, -1][
            torch.arange(0, B, device=state.device).repeat_interleave(T // self.chunk_size)
        ]

        if "hidden_update_n" not in info:
            info["hidden_update_n"] = 0
        else:
            info["hidden_update_n"] += 1

        if info["hidden_update_n"] % 8 == 0:
            info["hidden"] = self.preprocess_data(data)

        F = self.head(torch.cat([state, info["hidden"].cuda()], dim=-1))
        F = F.view(B, T // self.chunk_size, F.shape[1]).transpose(1, 2)
        dist = StochasticMultinomialTensor(F)
        sample = dist.sample().max(1)[0]

        Fx = {k: v[(sample > 0.5).view(-1)] for k, v in chunk_data.items()}
        FS = state[(sample > 0.5).view(-1)]

        Fe = self.encode(Fx).detach()
        Fx = self.encoder(Fe)['last_hidden_state']
        F1 = self.head(torch.cat([FS, Fx[:, -1]], dim=-1))
        F1 = dist.make_diff_sample(F1).max(1)[0].repeat_interleave(self.chunk_size, dim=0)

        FD = sample.repeat_interleave(self.chunk_size, dim=1)
        lens = (FD > 0.5).type(torch.int64).sum(-1).detach().cpu().numpy().tolist()

        x3 = {k: v[(sample > 0.5).view(-1)].view(-1, *v.shape[2:]) for k, v in chunk_data.items()}
        x3["input_ids"] = x3["input_ids"] * F1[:, None]

        return x3, lens

    def forward(self, data: Dict[str, Tensor], state: State, info: Info, *args):

        filtered_data, lens = self.filter_data(data, state, info)
        sp = {k: torch.split(v, lens, 0) for k, v in filtered_data.items()}
        return {k: pad_sequence(v, batch_first=True) for k, v in sp.items()}, False


