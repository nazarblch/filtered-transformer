import math
from typing import Dict

from torch import nn, Tensor
import torch
from filter_model.base import FilterModel


class SeqFilter(FilterModel):

    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def forward(self, data: Tensor):

        pos = [0]
        length = data.shape[1]

        def proc_state(state: Tensor):

            if pos[0] >= length:
                return None

            fd = data[:, pos[0]: pos[0] + self.size]
            pos[0] += self.size

            return fd

        return proc_state


class DictSeqFilter(FilterModel):

    def __init__(self, size: int, key: str):
        super().__init__()
        self.size = size
        self.key = key

    def forward(self, data: Dict[str, Tensor]):

        pos = [0]
        length = data[self.key].shape[1]

        def proc_state(state: Tensor):

            if pos[0] >= length:
                return None, None

            fd = {k: v[:, pos[0]: pos[0] + self.size] for k, v in data.items()}
            mask: Tensor = torch.zeros(state.shape[0], length, dtype=torch.bool, device=state.device)
            mask[:, pos[0]: pos[0] + self.size] = True
            pos[0] += self.size

            return fd, mask

        return proc_state


class DictSeqFilterBidirectional(FilterModel):

    def __init__(self, size: int, key: str):
        super().__init__()
        self.size = size
        self.key = key

    def forward(self, data: Dict[str, Tensor]):

        pos = [0]
        length = data[self.key].shape[1]
        dir = [1]

        def proc_state(state: Tensor):

            if pos[0] <= 0 and dir[0] == -1:
                return None

            if pos[0] >= length:
                dir[0] = -1
                pos[0] -= self.size

            if dir[0] == 1:
                fd = {k: v[:, pos[0]: pos[0] + self.size] for k, v in data.items()}
            else:
                fd = {k: v[:, pos[0] - self.size: pos[0]] for k, v in data.items()}

            pos[0] += self.size * dir[0]

            return fd

        return proc_state







