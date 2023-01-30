from abc import ABC, abstractmethod
from typing import Callable, Optional, Iterator, Tuple
import torch
from torch import nn, Tensor
from memup.base import SeqDataFilter, SD, State, Info, Done


class NStepFilter(SeqDataFilter[SD]):

    def __init__(self, steps: int, model: SeqDataFilter[SD]):
        super().__init__()
        self.steps = steps
        self.model = model

    def forward(self, data: SD, state: State, info: Info, *args) -> Tuple[SD, Done]:
        if "n_step_filter" not in info:
            info["n_step_filter"] = 0
        else:
            info["n_step_filter"] += 1

        res, done = self.model(data, state, info)

        return res, done or info["n_step_filter"] >= self.steps

