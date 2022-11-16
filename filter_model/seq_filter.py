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
                return None

            fd = {k: v[:, pos[0]: pos[0] + self.size] for k, v in data.items()}
            pos[0] += self.size

            return fd

        return proc_state







