import math
from abc import abstractmethod, ABC
from typing import Callable, List, Tuple

import numpy as np
import torch
from torch import nn, Tensor, LongTensor
import torch.multiprocessing as mp
from torch.nn.utils.rnn import pad_sequence

from hfilter.networks import StochasticMultinomialTensor
from models import PositionalEncoding2


class FilterLayer(nn.Module):
    def __init__(self, n: int, dim: int):
        super().__init__()
        self.data = nn.Parameter(torch.randn(n, dim) / math.sqrt(dim))
        self.values_data = nn.Parameter(torch.randn(n, dim) / math.sqrt(dim))
        self.n = n


class FilterNode(FilterLayer):

    def __init__(self, n: int, dim: int):
        super().__init__(n, dim)
        self.child_nodes = []


class KeysModel(nn.Module, ABC):
    @abstractmethod
    def _model_forward(self, x: Tensor, filter_data: Tensor) -> Tensor:
        pass

    def forward(self, x: Tensor, filter_data: Tensor) -> Tuple[Tensor, Tensor, np.ndarray]:
        keys = self._model_forward(x, filter_data)
        keys_sample = StochasticMultinomialTensor(keys).sample()
        int_keys = keys_sample.argmax(-1).detach().cpu().numpy().reshape(-1)
        return keys, keys_sample, int_keys


class ValuesModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: Tensor, values_data: Tensor, keys_sample: Tensor) -> Tensor:
        pass


def nodes_forward(x: Tensor,
                  prev_keys: Tensor,
                  nodes: nn.ModuleList,
                  keys_model: KeysModel,
                  values_model: ValuesModel):

    assert (prev_keys.shape[1], prev_keys.shape[2]) == (1, 1)

    filter_data = torch.stack([ch.data for ch in nodes])
    keys, keys_sample, int_keys = keys_model(x, filter_data)
    values_data = torch.stack([ch.values_data for ch in nodes])
    values = values_model(x, values_data, keys_sample) * prev_keys
    new_keys = prev_keys * keys_sample.max(-1).values[:, None, None]

    new_nodes = None
    if len(nodes[0].child_nodes) > 0:
        new_nodes = [ch.child_nodes[k] for ch, k in zip(nodes, int_keys)]

    return new_nodes, new_keys, values, keys_sample


class StateTransform(nn.Module, ABC):

    @abstractmethod
    def forward(self, x: Tensor, keys_sample: Tensor):
        pass


class FilterConstructorTree(nn.Module):

    def init_children(self, root: FilterNode):
        for i in range(self.n):
            root.child_nodes.append(FilterNode(self.n, self.dim))

    def __init__(self, n, depth, dim,
                 state_transform: StateTransform,
                 keys_model: KeysModel,
                 values_model: ValuesModel):

        super().__init__()
        self.n = n
        self.dim = dim
        self.state_transform = state_transform
        self.keys_model = keys_model
        self.values_model = values_model
        self.root = FilterNode(n, dim)
        self.depth = depth
        nodes = [self.root]
        new_nodes = []
        for d in range(depth):
            for node in nodes:
                self.init_children(node)
                new_nodes.extend(node.child_nodes)
                node.child_nodes = nn.ModuleList(node.child_nodes)
            nodes = new_nodes
            new_nodes = []

    def forward(self, x: Tensor):
        active_nodes = nn.ModuleList([self.root] * x.shape[0])
        k_sample = torch.zeros(x.shape[0], self.n, device=x.device)
        val_list = []
        prev_keys = torch.ones(x.shape[0], 1, 1, device=x.device)

        for d in range(self.depth):
            x = self.state_transform(x, k_sample)
            active_nodes, prev_keys, values, k_sample = \
                nodes_forward(x, prev_keys, active_nodes, self.keys_model, self.values_model)
            val_list.append(values)

        return val_list


class ValuesToFilterModel(nn.Module, ABC):
    @abstractmethod
    def _model_forward(self, x: Tensor, v: Tensor) -> Tensor:
        pass

    def forward(self, x: Tensor, v: Tensor) -> Tensor:
        x1 = self._model_forward(x, v)
        return StochasticMultinomialTensor(x1).sample()[:, :, 0]


class HierarchicalFilter(nn.Module):
    def __init__(self, dim: int, hidden_dim, chunk_size: int, filter_model: ValuesToFilterModel):
        super().__init__()
        self.chunk_size = chunk_size
        self.embed = nn.Linear(dim, hidden_dim)
        self.pos_encoder = PositionalEncoding2(hidden_dim)
        self.filter_model = filter_model

    def add_pos(self, x: Tensor):
        B, L, D = x.shape
        out = self.embed(x.reshape(B * L, D)).reshape(B, L, -1)
        return self.pos_encoder(out) * math.sqrt(self.chunk_size)

    def chunk(self, x):
        B, L, D = x.shape
        return x.reshape(B * L // self.chunk_size, self.chunk_size, D)

    def de_chunk(self, x, L):
        B1 = x.shape[0]
        B = B1 * self.chunk_size // L
        return x.reshape(B, L, *x.shape[2:])

    def forward(self, values: List[Tensor], data: Tensor):
        B, T, D = data.shape
        data = self.add_pos(data)
        x = self.chunk(data)
        F = None

        for v in values:
            v = v[torch.arange(0, B, device=v.device).repeat_interleave(T // self.chunk_size)]
            f = self.filter_model(x, v)
            f = self.de_chunk(f, T)
            F = f if F is None else F * f

        lens = (F > 0.5).type(torch.int64).sum(-1).detach().cpu().numpy().tolist()
        x3 = data[F > 0.5] * F[F > 0.5][:,  None]
        sp = torch.split(x3, lens, 0)

        return pad_sequence(sp, batch_first=True)


