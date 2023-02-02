from abc import abstractmethod, ABC
from collections import namedtuple
from typing import TypeVar, Tuple, Dict, Generic, Callable, Iterator, List, Optional, Any, Type, OrderedDict, NamedTuple
import torch
from torch import nn, Tensor

SD = TypeVar("SD", Dict, Tuple, Tensor)
CT = TypeVar("CT", Dict, Tuple, Tensor)
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


class DataCollector(Generic[SD, CT], ABC):

    def __init__(self):
        self.collection: List[CT] = []

    @abstractmethod
    def append(self, data: SD, out: MemoryOut, state: State) -> None:
        pass

    def result(self, cat_dims: Tuple[int] = (), cat_keys: Tuple[str] = ()):
        assert len(self.collection) > 0
        if isinstance(self.collection[0], (tuple, list)):
            res = list(zip(*self.collection))
            for dim in cat_dims:
                res[dim] = torch.cat(res, dim)
            res = tuple(res) if isinstance(self.collection[0], tuple) else res
            return res

        if isinstance(self.collection[0], dict):
            res = {k: [] for k in self.collection[0].keys()}
            for d in self.collection:
                for k, v in d.items():
                    v = torch.cat(v) if k in cat_keys else v
                    res[k].append(v)
            return res
        if isinstance(self.collection[0], NamedTuple):
            res = self.collection[0].__class__(*[[]] * len(self.collection[0]._fields))
            for d in self.collection:
                for k, v in d.items():
                    v = torch.cat(v) if k in cat_keys else v
                    res[k].append(v)
            return res

        return self.collection

    @abstractmethod
    def apply(self, data: SD, out: MemoryOut, state: State) -> CT:
        pass


class DataCollectorAppend(DataCollector[SD, CT], ABC):
    def append(self, data: SD, out: MemoryOut, state: State) -> None:
        self.collection.append(self.apply(data, out, state))


class DataCollectorReplace(DataCollector[SD, CT], ABC):
    def append(self, data: SD, out: MemoryOut, state: State) -> None:
        if len(self.collection) == 0:
            self.collection.append(self.apply(data, out, state))
        else:
            self.collection[0] = self.apply(data, out, state)


class DataCollectorEmpty(DataCollector[SD, CT]):
    def append(self, data: SD, out: MemoryOut, state: State) -> None:
        pass

    def apply(self, data: SD, out: MemoryOut, state: State) -> CT:
        pass


class MemUpLoss(nn.Module, ABC, Generic[SD, CT]):
    @abstractmethod
    def forward(self, collector: DataCollector[SD, CT], info: Info) -> Loss:
        pass


class MemoryRollout(Generic[SD]):
    def __init__(self,
                 steps: int,
                 memory: MemUpMemory[SD],
                 data_filter: SeqDataFilter[SD],
                 info_update: List[InfoUpdate]):

        self.memory = memory
        self.data_filter = data_filter
        self.rollout = steps
        self.info_update = info_update

    def forward(self, data: SD, state: State, info: Info, collector: DataCollector[SD, CT], steps=-1) -> Tuple[DataCollector[SD, CT], State, Info, Done]:

        done = False
        if steps < 0:
            steps = self.rollout

        for step in range(steps):

            with torch.no_grad():
                for update in self.info_update:
                    info = update.forward(data, state, info)

            filtered_data, done = self.data_filter(data, state, info)
            out, state = self.memory.forward(filtered_data, state)
            collector.append(filtered_data, out, state)

            if done:
                break

        state = state.detach()
        return collector, state, info, done


class MemoryRolloutWithLoss(Generic[SD, CT]):
    def __init__(self,
                 steps: int,
                 memory: MemUpMemory[SD],
                 loss: MemUpLoss[SD, CT],
                 data_filter: SeqDataFilter[SD],
                 info_update: List[InfoUpdate]):
        self.mem_roll = MemoryRollout[SD](steps, memory, data_filter, info_update)
        self.loss = loss

    def forward(self, data: SD, state: State, info: Info, collector: DataCollector[SD, CT]) -> Tuple[Optional[Loss], State, Info, Done]:
        collector, state, info, done = self.mem_roll.forward(data, state, info, collector)
        loss = self.loss(collector, info)
        if loss is None or torch.isnan(loss):
            loss = None
        return loss, state, info, done

