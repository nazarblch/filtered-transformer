from collections import namedtuple
from typing import Iterator, Tuple, List, Callable, Optional
import torch
from sklearn.metrics import accuracy_score
from torch import Tensor, nn
from memup.base import SeqDataFilter, MemUpMemory, MemUpLoss, MemUpLossIterator, State, Info, Done, InfoUpdate
from memup.data_filters import SlidingWindowFilter
from metrics.base import Metric
from models.pos_encoding import EmbedWithPos, LinearEmbedWithPos
from models.transformers import TorchRecurrentTransformer, RecurrentTransformer

DataType = namedtuple("DataType", ["x", "y", "length"])


class Predictor(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = TorchRecurrentTransformer(256, 4, 2, 512, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x: Tensor, state: State):
        out = self.encoder.forward(x, torch.cat([state, state], -1)).out
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
        out = out[:, out.shape[1] - data.y.shape[1]:]
        return out, os.state


class TailAccuracyMetric(Metric):

    def __init__(self):
        super().__init__("TailAccuracy")

    @torch.no_grad()
    def __call__(self, logits: Tensor, labels: Tensor) -> float:
        T = logits.shape[1]
        logits, labels = logits[:, T - 10:], labels[:, T - 10:]
        return accuracy_score(logits.argmax(-1).reshape(-1).cpu().numpy(), labels.reshape(-1).cpu().numpy())


class SeqDataFilterImpl(SlidingWindowFilter[DataType]):

    def __init__(self, size: int):
        super().__init__(size, padding=size // 5)

    def filter_data(self, data: DataType, i1: int, i2: int, i1_pad: int, i2_pad: int) -> DataType:
        pad_x = data.x[:, i1_pad: i2_pad]
        y = data.y[:, i1: i2]

        return DataType(pad_x, y, i2_pad - i1_pad)