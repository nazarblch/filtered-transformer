from collections import namedtuple
from typing import Iterator, Tuple, List, Callable, Optional
import torch
from torch import Tensor, nn

from data_filters.sliding_window import SlidingWindowFilter, SlidingWindowWithPadding
from memup.base import State, MemUpMemory, DataCollectorAppend, MemoryOut, DataCollectorReplace
from memup.loss import TOS, PT, TS
from metrics.base import Metric
from common_modules.pos_encoding import EmbedWithPos, LinearEmbedWithPos
from common_modules.transformers import TorchRecurrentTransformer, RecurrentTransformer


DataType = namedtuple("DataType", ["x", "y", "mask", "length"])


class Predictor(nn.Module):

    def __init__(self):
        super().__init__()
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(256, 4, 512, 0.1, batch_first=True),
            2
        )

        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x: Tensor, state: State):
        out = self.decoder.forward(x, torch.cat([state, state], -1))
        return self.head(out)


class MemUpMemoryImpl(MemUpMemory):

    def __init__(self, embed: LinearEmbedWithPos, mem_tr: RecurrentTransformer):
        super().__init__()
        self.mem_tr = mem_tr
        self.embed = embed

    def forward(self, data: DataType, state: State) -> Tuple[Tensor, State]:
        x_embed = self.embed.forward(data.x.cuda())
        os = self.mem_tr.forward(x_embed, state)
        assert os.out.shape[1] >= data.y.shape[1]
        out = torch.cat([os.out, x_embed], -1)
        padding = (out.shape[1] - data.y.shape[1]) // 2
        out = out[:, padding: padding + data.y.shape[1]]
        return out, os.state


class SeqDataFilterImpl(SlidingWindowFilter[DataType]):

    def __init__(self, size: int):
        super().__init__(size, padding=0)

    def filter_data(self, data: DataType, window: SlidingWindowWithPadding) -> DataType:
        i1, i2, i1_pad, i2_pad = window

        pad_x = data.x[:, i1_pad: i2_pad]
        y = data.y[:, i1: i2]
        mask = data.mask[:, i1: i2]

        return DataType(pad_x, y, mask, i2_pad - i1_pad)



class DataCollectorTrain(DataCollectorAppend[DataType, TOS]):
    def apply(self, data: DataType, out: MemoryOut, state: State) -> TOS:
        return TOS(data.y, out, state)


class DataCollectorTrainFromState(DataCollectorAppend[DataType, TS]):
    def apply(self, data: DataType, out: MemoryOut, state: State) -> TS:
        return TS(data.y, state)


class DataCollectorLastState(DataCollectorReplace[DataType, TS]):
    def apply(self, data: DataType, out: MemoryOut, state: State) -> TS:
        return TS(data.y, state)


class DataCollectorEvalWithState(DataCollectorAppend[DataType, PT]):

    def __init__(self, predictor: nn.Module, state: Tensor):
        super().__init__()
        self.predictor = predictor
        self.state = state

    def apply(self, data: DataType, out: MemoryOut, state: State) -> PT:
        pred = self.predictor(out, self.state)
        return PT(pred, data.y)