from collections import namedtuple
from typing import Iterator, Tuple, List, Callable, Optional
import torch
from torch import Tensor, nn
from memup.base import State, MemUpMemory, DataCollectorAppend, MemoryOut, DataCollectorReplace
from memup.data_filters import SlidingWindowFilter
from memup.loss import TOS, PT, TS
from metrics.base import Metric
from models.pos_encoding import EmbedWithPos, LinearEmbedWithPos
from models.transformers import TorchRecurrentTransformer, RecurrentTransformer


DataType = namedtuple("DataType", ["x", "y", "length"])


class Predictor(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(128, 4, 512, 0.1, batch_first=True),
            2
        )
        self.head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state: State):
        out = self.encoder.forward(state)[:, -1]
        return self.head(out)


class MemUpMemoryImpl(MemUpMemory):

    def __init__(self, embed: LinearEmbedWithPos, mem_tr: RecurrentTransformer):
        super().__init__()
        self.mem_tr = mem_tr
        self.embed = embed

    def forward(self, data: DataType, state: State) -> Tuple[Tensor, State]:
        x_embed = self.embed.forward(data.x.cuda())
        os = self.mem_tr.forward(x_embed, state)
        return None, os.state


class SeqDataFilterImpl(SlidingWindowFilter[DataType]):

    def __init__(self, size: int):
        super().__init__(size, padding=size // 5)

    def filter_data(self, data: DataType, i1: int, i2: int, i1_pad: int, i2_pad: int) -> DataType:
        pad_x = data.x[:, i1_pad: i2_pad]

        return DataType(pad_x, data.y, i2_pad - i1_pad)


class DataCollectorTrain(DataCollectorAppend[DataType, TS]):
    def apply(self, data: DataType, out: MemoryOut, state: State) -> TS:
        return TS(data.y, state)


class DataCollectorLastState(DataCollectorReplace[DataType, TS]):
    def apply(self, data: DataType, out: MemoryOut, state: State) -> TS:
        return TS(data.y, state)
