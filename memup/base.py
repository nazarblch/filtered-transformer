from abc import abstractmethod, ABC
from collections import namedtuple
from typing import TypeVar, Tuple, Dict, Generic, Callable, Iterator, List, Optional, Any
import torch
from torch import nn, Tensor

SD = TypeVar("SD", Dict, Tuple, Tensor)
State = Tensor
Info = Dict[str, Any]
MemoryOut = Tensor
Loss = Tensor
SDWithMemory = Tuple[SD, MemoryOut, State]
Done = bool


class SeqDataFilter(nn.Module, ABC, Generic[SD]):

    @abstractmethod
    def forward(self, data: SD, state: State, info: Info, *args) -> Tuple[SD, Done]:
        pass


class InfoUpdate(ABC, Generic[SD]):
    @abstractmethod
    def forward(self, data: SD, state: State, info: Info, *args) -> Info:
        pass


class MemUpMemory(nn.Module, ABC, Generic[SD]):
    @abstractmethod
    def forward(self, data: SD, state: State) -> Tuple[MemoryOut, State]:
        pass


class MemUpLoss(nn.Module, ABC, Generic[SD]):
    @abstractmethod
    def forward(self, data: List[SDWithMemory], info: Info) -> Loss:
        pass


class MemUpLossIterator(Generic[SD]):
    def __init__(self,
                 rollout: int,
                 memory: MemUpMemory[SD],
                 loss: MemUpLoss[SD],
                 data_filter: SeqDataFilter[SD],
                 info_update: List[InfoUpdate]):

        self.memory = memory
        self.loss = loss
        self.data_filter = data_filter
        self.rollout = rollout
        self.info_update = info_update

    def forward(self, data: SD, state: State, info: Info) -> Tuple[Optional[Loss], State, Info, Done]:

        data_collection = []
        done = False

        for step in range(self.rollout):

            with torch.no_grad():
                for update in self.info_update:
                    info = update.forward(data, state, info)

            filtered_data, done = self.data_filter(data, state, info)
            out, state = self.memory.forward(filtered_data, state)
            data_collection.append((filtered_data, out, state))

            if done:
                break

        loss = self.loss(data_collection, info)
        if torch.isnan(loss):
            loss = None

        state = state.detach()
        return loss, state, info, done
