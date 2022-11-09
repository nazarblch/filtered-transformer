import math
from torch import nn, Tensor
import torch
from torch.nn.utils.rnn import pad_sequence
from filter_model.base import FilterModel
from models.stoch_tensor import StochasticMultinomialTensor


class ChunkFilter(FilterModel):

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
            for b in range(B // 8):
                xb = self.chunk(data[8 * b:8 * b + 8])
                xb = self.transformer(xb)
                ttx.append(xb[:, -1])
            ttx = torch.cat(ttx)

        return ttx

    def filter_data(self, data: Tensor, hidden: Tensor, state: Tensor):
        B, T = data.shape[0], data.shape[1]

        state = state[:, -1][
            torch.arange(0, B, device=state.device).repeat_interleave(T // self.chunk_size)
        ]

        F = self.head(torch.cat([state, hidden], dim=-1))
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
        x3 = data[FD > 0.5] * F1[:, None]

        return x3, lens

    def forward(self, data: Tensor):

        hidden = [self.preprocess_data(data)]
        n = [0]

        def proc_state(state: Tensor):

            if n[0] % 8 == 0:
                hidden[0] = self.preprocess_data(data)

            filtered_data, lens = self.filter_data(data, hidden[0], state)
            sp = torch.split(filtered_data, lens, 0)
            n[0] += 1

            return pad_sequence(sp, batch_first=True)

        return proc_state


class RandomChunkFilter(FilterModel):

    def __init__(self, chunk_size: int, n_chunks: int):
        super().__init__()
        self.chunk_size = chunk_size
        self.n_chunks = n_chunks

    def chunk(self, x):
        B, L, D = x.shape
        assert L % self.chunk_size == 0
        return x.view(B * L // self.chunk_size, self.chunk_size, D)

    def filter_data(self, data: Tensor):
        B, T = data.shape[0], data.shape[1]

        F = torch.ones(B, T // self.chunk_size, self.n_chunks, device=data.device).transpose(1, 2)
        dist = StochasticMultinomialTensor(F)
        sample = dist.sample().max(1)[0]
        FD = sample.repeat_interleave(self.chunk_size, dim=1)

        lens = (FD > 0.5).type(torch.int64).sum(-1).detach().cpu().numpy().tolist()
        x3 = data[FD > 0.5]

        return x3, lens

    def forward(self, data: Tensor):

        def proc_state(state: Tensor):

            filtered_data, lens = self.filter_data(data)
            sp = torch.split(filtered_data, lens, 0)

            return pad_sequence(sp, batch_first=True)

        return proc_state