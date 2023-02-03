from collections import namedtuple
from typing import Tuple, Optional
from torch import Tensor
from common_modules.transformers import BertRecurrentTransformerWithTokenizer
from memup.base import MemUpMemory, State, DataCollectorReplace, MemoryOut, DataCollectorAppend
from memup.loss import TS

DataType = namedtuple("DataType", ["text", "target", "length"])


class MemUpMemoryImpl(MemUpMemory[DataType]):

    def __init__(self, mem_tr: BertRecurrentTransformerWithTokenizer):
        super().__init__()
        self.mem_tr = mem_tr

    def forward(self, data: DataType, state: State) -> Tuple[Optional[Tensor], State]:
        os = self.mem_tr.forward(data.text, state)
        return None, os.state


class DataCollectorTrain(DataCollectorAppend[DataType, TS]):
    def apply(self, data: DataType, out: MemoryOut, state: State) -> TS:
        return TS(data.target, state)


class DataCollectorLastState(DataCollectorReplace[DataType, TS]):
    def apply(self, data: DataType, out: MemoryOut, state: State) -> TS:
        return TS(data.target, state)