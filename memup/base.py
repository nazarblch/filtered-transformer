from abc import abstractmethod, ABC
from collections import namedtuple
from typing import TypeVar, Tuple, Dict, Generic, Callable, Iterator, List, Optional

import torch
from torch import nn, Tensor

SD = TypeVar("SD", Dict, Tuple, Tensor)
State = namedtuple("State", ["state", "extra"])
MemoryOut = Tensor
Loss = Tensor
TDWithMemory = Tuple[SD, MemoryOut, Tensor]


class SeqDataFilter(nn.Module, ABC, Generic[SD]):

    @abstractmethod
    def filter(self, data: SD, state: State, *args) -> Optional[SD]:
        pass

    @abstractmethod
    def update(self, filtered_data: SD, state: State, *args) -> State:
        pass

    @abstractmethod
    def forward(self, data: SD, state: State) -> Tuple[Optional[SD], State]:
        filtered_data = self.filter(data, state)
        new_state = self.update(filtered_data, state)
        return filtered_data, new_state


class MemUpMemory(nn.Module, ABC, Generic[SD]):
    @abstractmethod
    def forward(self, data: SD, state: State) -> Tuple[MemoryOut, State]:
        pass


class MemUpLoss(nn.Module, ABC, Generic[SD]):
    @abstractmethod
    def forward(self, data: List[TDWithMemory], state: State) -> Loss:
        pass


class MemUpLossIterator(Generic[SD]):
    def __init__(self,
                 rollout: int,
                 memory: MemUpMemory[SD],
                 loss: MemUpLoss[SD],
                 data_filter: SeqDataFilter[SD]):

        self.memory = memory
        self.loss = loss
        self.data_filter = data_filter
        self.rollout = rollout

    def forward(self, data: SD, state: State) -> Tuple[Optional[Loss], State, bool]:

        data_collection = []
        done = False

        for step in range(self.rollout):
            filtered_data, state = self.data_filter(data, state)
            if filtered_data is None:
                done = True
                break

            out, s0 = self.memory.forward(filtered_data, state[0])
            data_collection.append((filtered_data, out, s0))
            state = State(s0, state[1])

        loss = self.loss(data_collection, state)
        if torch.isnan(loss):
            loss = None

        state = State(state[0].detach(), state[1])
        return loss, state, done
