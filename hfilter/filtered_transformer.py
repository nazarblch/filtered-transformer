import math
from abc import ABC, abstractmethod
from typing import List

from torch import nn, Tensor
import torch
from torch.nn.utils.rnn import pad_sequence

from hfilter.networks import StochasticMultinomialTensor, StochasticMultinomialTensorFiltered
from models import PositionalEncoding2


class FilterModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, state: Tensor, data: Tensor):
        pass


class ChunkFilter(FilterModel):

    def __init__(self, transformer: nn.Transformer, chunk_size: int, dim: int, hidden_dim: int, sample_size: int):
        super().__init__()
        self.transformer = transformer
        self.chunk_size = chunk_size
        self.pos_encoder = PositionalEncoding2(hidden_dim)
        self.embed = nn.Linear(dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, sample_size)

    def add_pos(self, x: Tensor):
        out = self.embed(x)
        return self.pos_encoder(out) * math.sqrt(self.chunk_size)

    def chunk(self, x):
        B, L, D = x.shape
        assert L % self.chunk_size == 0
        return x.view(B * L // self.chunk_size, self.chunk_size, D)

    def de_chunk(self, x, L):
        B1 = x.shape[0]
        B = (B1 * self.chunk_size) // L
        return x.view(B, L, *x.shape[2:])

    def forward(self, state: Tensor, data: Tensor):
        B, T, D = data.shape
        data = self.add_pos(data)
        x = self.chunk(data)
        state = state[torch.arange(0, B, device=state.device).repeat_interleave(T // self.chunk_size)]
        F = self.head(self.transformer(state, x)[:, -1])
        F = F.view(B, T // self.chunk_size, F.shape[1]).transpose(1, 2)
        F = StochasticMultinomialTensor(F).sample().max(1)[0].repeat_interleave(self.chunk_size, dim=1)

        lens = (F > 0.5).type(torch.int64).sum(-1).detach().cpu().numpy().tolist()
        x3 = data[F > 0.5] * F[F > 0.5][:,  None]
        sp = torch.split(x3, lens, 0)

        return pad_sequence(sp, batch_first=True)


class RandomChunkFilter(FilterModel):

    def __init__(self, chunk_size: int, dim: int, hidden_dim: int, sample_size: int):
        super().__init__()
        self.pos_encoder = PositionalEncoding2(hidden_dim)
        self.embed = nn.Linear(dim, hidden_dim)
        self.chunk_size = chunk_size
        self.sample_size = sample_size

    def add_pos(self, x: Tensor):
        out = self.embed(x)
        return self.pos_encoder(out) * math.sqrt(self.chunk_size)

    def chunk(self, x):
        B, L, D = x.shape
        assert L % self.chunk_size == 0
        return x.view(B * L // self.chunk_size, self.chunk_size, D)

    def de_chunk(self, x, L):
        B1 = x.shape[0]
        B = (B1 * self.chunk_size) // L
        return x.view(B, L, *x.shape[2:])

    def forward(self, state: Tensor, data: Tensor):
        B, T, D = data.shape
        data = self.add_pos(data)
        F = torch.ones(B, T // self.chunk_size, self.sample_size, device=data.device).transpose(1, 2)
        F = StochasticMultinomialTensor(F).sample().max(1)[0].repeat_interleave(self.chunk_size, dim=1)

        lens = (F > 0.5).type(torch.int64).sum(-1).detach().cpu().numpy().tolist()
        x3 = data[F > 0.5] * F[F > 0.5][:,  None]
        sp = torch.split(x3, lens, 0)

        return pad_sequence(sp, batch_first=True)


class SlideFilter(FilterModel):

    def __init__(self, size: int, dim, hidden_dim):
        super().__init__()
        self.size = size
        self.pos = 0

        self.pos_encoder = PositionalEncoding2(hidden_dim)
        self.embed = nn.Linear(dim, hidden_dim)

    def add_pos(self, x: Tensor):
        out = self.embed(x)
        return self.pos_encoder(out) * math.sqrt(self.size)

    def forward(self, state: Tensor, data: Tensor):
        if state[0].abs().sum() < 1e-4 or self.pos >= data.shape[1]:
            self.pos = 0

        data = self.add_pos(data)

        fd = data[:, self.pos: self.pos + self.size]
        self.pos = self.pos + self.size

        return fd


class ChunkFilter2(FilterModel):

    def __init__(self, transformer: nn.Transformer, chunk_size: int, dim: int, hidden_dim: int, sample_size: int):
        super().__init__()
        self.transformer = transformer
        self.chunk_size = chunk_size
        self.pos_encoder = PositionalEncoding2(hidden_dim)
        self.embed = nn.Linear(dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, sample_size)

    def add_pos(self, x: Tensor):
        out = self.embed(x)
        return self.pos_encoder(out) * math.sqrt(self.chunk_size)

    def chunk(self, x):
        B, L, D = x.shape
        assert L % self.chunk_size == 0
        return x.view(B * L // self.chunk_size, self.chunk_size, D)

    def de_chunk(self, x, L):
        B1 = x.shape[0]
        B = (B1 * self.chunk_size) // L
        return x.view(B, L, *x.shape[2:])

    def forward(self, state: Tensor, data: Tensor):
        B, T, D = data.shape
        data = self.add_pos(data)
        x = self.chunk(data)
        state = state[torch.arange(0, B, device=state.device).repeat_interleave(T // self.chunk_size)]
        F = self.head(self.transformer(state, x)[:, 0])
        F = F.view(B, T // self.chunk_size, F.shape[1])
        F = StochasticMultinomialTensor(F).sample()[:, :, 0].repeat_interleave(self.chunk_size, dim=1)

        lens = (F > 0.5).type(torch.int64).sum(-1).detach().cpu().numpy().tolist()
        x3 = data[F > 0.5] * F[F > 0.5][:,  None]
        sp = torch.split(x3, lens, 0)

        return pad_sequence(sp, batch_first=True)


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


class HierarchicalChunkFilter(FilterModel):
    def __init__(self, h_transformer: HierarchicalTransformer,
                 sample_size: int):
        super().__init__()
        self.h_transformer = h_transformer
        self.sample_size = sample_size
        self.levels_count = h_transformer.levels_count
        self.chunk_size = h_transformer.chunk_size
        self.head = nn.Linear(h_transformer.hidden_dim, sample_size)
        self.rnn = nn.Sequential(
            nn.Linear(2 * h_transformer.hidden_dim, h_transformer.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_transformer.hidden_dim, h_transformer.hidden_dim)
        )

        self.head_bin = nn.ModuleList([nn.Sequential(
            nn.Linear(h_transformer.hidden_dim, h_transformer.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_transformer.hidden_dim, sample_size),
            # nn.ReLU(inplace=True),
            # nn.Linear(h_transformer.hidden_dim, 1)
        ) for _ in range(self.levels_count)])
        self.mixin_data = nn.Transformer(h_transformer.hidden_dim, 4, 2, 2, 2 * h_transformer.hidden_dim, batch_first=True)

    def expand_data(self, x: Tensor, to_size: int):
        x = x.repeat_interleave(2, dim=-1)
        if x.shape[-1] > to_size:
            assert x.shape[-1] == to_size + 1
            x = x[..., 0:to_size]
        else:
            assert x.shape[-1] == to_size

        return x

    def forward(self, state: Tensor, data: Tensor):
        res = self.h_transformer(state, data)

        F = self.head(res[-1]).transpose(1, 2)
        F = StochasticMultinomialTensor(F).sample()
        Fdata = res[-1][:, None].repeat(1, self.sample_size, 1, 1)
        Fdata = (Fdata * F[:, :, :, None]).sum(2)

        for l in range(self.levels_count - 2, -1, -1):
            B, T, D = res[l].shape
            res_1 = self.mixin_data(Fdata.repeat_interleave(T, dim=0), res[l].view(B * T, 1, D)).view(B, T, D)
            F1 = self.head_bin[l](res_1).transpose(1, 2)
            F = self.expand_data(F, F1.shape[-1])
            F = StochasticMultinomialTensorFiltered(F1, F).sample()

            Fdata_1 = res[l][:, None].repeat(1, self.sample_size, 1, 1)
            Fdata_1 = (Fdata_1 * F[:, :, :, None]).sum(2)
            Fdata = self.rnn(torch.cat([Fdata_1, Fdata], dim=-1))

        # F = F + torch.randint_like(F, 0, 1) / 3
        F = F.sum(1).repeat_interleave(self.chunk_size, dim=1)

        lens = (F > 0.5).type(torch.int64).sum(-1).detach().cpu().numpy().tolist()
        data = self.h_transformer.add_pos(self.h_transformer.embed(data))
        x3 = data[F > 0.5] * F[F > 0.5][:, None]
        sp = torch.split(x3, lens, 0)

        return pad_sequence(sp, batch_first=True)


class FilteredTransformer(nn.Module):

    def __init__(self,
                 transformer: nn.Transformer,
                 filter_model: FilterModel,
                 rollout: int):
        super().__init__()
        self.transformer = transformer
        self.filter_model = filter_model
        self.steps = rollout
        # self.embed = nn.Linear(state_dim, state_dim)

    def forward(self, s: Tensor, data: Tensor):
        # s = self.embed(s0)
        for _ in range(self.steps):
            fd = self.filter_model(s, data)
            s = self.transformer(fd, s)

        return s


if __name__ == "__main__":

    h_filter = HierarchicalChunkFilter(
        HierarchicalTransformer(
            nn.Transformer(128, 4, 3, 3, 256, batch_first=True),
            nn.Transformer(128, 4, 3, 3, 256, batch_first=True),
            nn.Transformer(128, 4, 3, 3, 256, batch_first=True),
            nn.Transformer(128, 4, 3, 3, 256, batch_first=True),
            dim=2, chunk_size=7),
        10
    ).cuda()

    res = h_filter.forward(torch.zeros(3, 10, 128).cuda(), torch.randn(3, 7 * 53, 2).cuda())

    print(res.shape)