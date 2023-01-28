from abc import ABC, abstractmethod
from typing import Tuple

import torch

from memup.base import SeqDataFilter, SD, State, Info, Done


class SlidingWindowFilter(SeqDataFilter[SD], ABC):

    def __init__(self, window_size: int, padding: int):
        super().__init__()
        self.window_size = window_size
        self.padding = padding

    @abstractmethod
    def filter_data(self, data: SD, i1: int, i2: int, i1_pad: int, i2_pad: int) -> SD:
        pass

    def forward(self, data: SD, state: State, info: Info, *args) -> Tuple[SD, Done]:
        BS = self.window_size
        T = data.length
        assert "step" in info
        step = info["step"]
        assert step * BS < T
        done = step * BS + BS >= T

        i1 = step * BS
        i2 = i1 + BS
        i1_pad = max(0, i1 - self.padding)
        i2_pad = min(T, i2 + self.padding)

        ids = torch.arange(i1, i2)[None, ].expand(state.shape[0], -1)
        if "filter_window" not in info:
            info["filter_window"] = ids
        else:
            info["filter_window"] = torch.cat([info["filter_window"], ids], 1)

        return self.filter_data(data, i1, i2, i1_pad, i2_pad), done