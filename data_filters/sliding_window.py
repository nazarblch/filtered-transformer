from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Dict, Tuple, Generic, Any, List, Type, Set
from torch import Tensor
import torch
from memup.base import SeqDataFilter, SD, State, Info, Done


SlidingWindowWithPadding = namedtuple("SlidingWindowWithPadding", ["i1", "i2", "i1_pad", "i2_pad"])


class SlidingWindowFilter(SeqDataFilter[SD], ABC):

    def __init__(self, window_size: int, padding: int):
        super().__init__()
        self.window_size = window_size
        self.padding = padding

    @abstractmethod
    def filter_data(self, data: SD, window: SlidingWindowWithPadding) -> SD:
        pass

    def forward(self, data: SD, state: State, info: Info, *args) -> Tuple[SD, Done]:
        BS = self.window_size
        T = data["length"] if isinstance(data, dict) else data.length
        assert "step" in info
        step = info["step"]
        assert step * BS < T
        done = step * BS + BS >= T

        i1 = step * BS
        i2 = i1 + BS
        i1_pad = max(0, i1 - self.padding)
        i2_pad = min(T, i2 + self.padding)

        ids = torch.arange(i1, i2)
        if "filter_indices" not in info:
            info["filter_indices"] = ids
        else:
            info["filter_indices"] = torch.cat([info["filter_indices"], ids], 0)

        return self.filter_data(data, SlidingWindowWithPadding(i1, i2, i1_pad, i2_pad)), done


class SlidingWindowFilterTuple(Generic[SD], SlidingWindowFilter[SD]):

    def __init__(self, size: int, padding: int, pad_fields: Set[str] = set(), skip_fields: Set[str] = set()):
        super().__init__(size, padding)
        self.pad_fields = pad_fields
        self.skip_fields = skip_fields

    def filter_field(self, k, v, window: SlidingWindowWithPadding, has_batch: bool):
        i1, i2, i1_pad, i2_pad = window

        if k in self.pad_fields:
            v = v[:, i1_pad: i2_pad] if has_batch else v[i1_pad: i2_pad]
        elif k in self.skip_fields:
            v = v
        else:
            v = v[:, i1: i2] if has_batch else v[i1: i2]

        return v

    def filter_data(self, data: SD, window: SlidingWindowWithPadding) -> SD:

        data_class = data.__class__
        fields = data_class._fields
        kw = {}

        for k, v in zip(fields, data):
            if isinstance(v, (list, tuple)):
                kw[k] = [self.filter_field(k, vi, window, False) for vi in v]
            else:
                kw[k] = self.filter_field(k, v, window, True)

        return data_class(**kw)
    


class SlidingWindowFilterDict(SlidingWindowFilter[Dict[str, Any]]):

    def __init__(self, size: int, padding: int, pad_fields: Set[str] = set(), skip_fields: Set[str] = set()):
        super().__init__(size, padding)
        self.pad_fields = pad_fields
        self.skip_fields = skip_fields

    def filter_field(self, k, v, window: SlidingWindowWithPadding, has_batch: bool):
        i1, i2, i1_pad, i2_pad = window

        if k in self.pad_fields:
            v = v[:, i1_pad: i2_pad] if has_batch else v[i1_pad: i2_pad]
        elif k in self.skip_fields:
            v = v
        else:
            v = v[:, i1: i2] if has_batch else v[i1: i2]

        return v

    def filter_data(self, data: Dict[str, Any], window: SlidingWindowWithPadding) -> Dict[str, Any]:

        kw = {}

        for k, v in data.items():
            if isinstance(v, (list, tuple)):
                kw[k] = [self.filter_field(k, vi, window, False) for vi in v]
            else:
                kw[k] = self.filter_field(k, v, window, True)
                if isinstance(kw[k], Tensor):
                    kw[k] = kw[k].cuda()

        return kw