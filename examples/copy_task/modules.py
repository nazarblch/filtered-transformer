from collections import namedtuple
from typing import Iterator, Tuple, List, Callable, Optional
import torch
from sklearn.metrics import accuracy_score
from torch import Tensor, nn
from memup.base import SeqDataFilter, MemUpMemory, MemUpLoss, State, Info, Done, InfoUpdate, \
    DataCollector, SD, MemoryOut, CT, DataCollectorAppend
from memup.loss import TOS, PT
from metrics.base import Metric
from common_modules.pos_encoding import EmbedWithPos
from common_modules.transformers import TorchRecurrentTransformer, RecurrentTransformer


DataType = namedtuple("DataType", ["x", "y", "length"])


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
            nn.Linear(256, 10)
        )

    def forward(self, x: Tensor, state: State):
        out = self.decoder.forward(x, torch.cat([state, state], -1))
        return self.head(out)


class MemUpMemoryImpl(MemUpMemory):

    def __init__(self, embed: EmbedWithPos, mem_tr: RecurrentTransformer):
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


class TailAccuracyMetric(Metric):

    def __init__(self):
        super().__init__("TailAccuracy")

    @torch.no_grad()
    def __call__(self, logits: Tensor, labels: Tensor) -> float:
        T = logits.shape[1]
        logits, labels = logits[:, T - 10:], labels[:, T - 10:]
        return accuracy_score(logits.argmax(-1).reshape(-1).cpu().numpy(), labels.reshape(-1).cpu().numpy())


class DataCollectorTrain(DataCollectorAppend[DataType, TOS]):
    def apply(self, data: DataType, out: MemoryOut, state: State) -> TOS:
        return TOS(data.y, out, state)


class DataCollectorEval(DataCollectorAppend[DataType, PT]):

    def __init__(self, predictor: nn.Module):
        super().__init__()
        self.predictor = predictor

    def apply(self, data: DataType, out: MemoryOut, state: State) -> PT:
        pred = self.predictor(out, state)
        return PT(pred, data.y)


class DataCollectorEvalWithState(DataCollectorAppend[DataType, PT]):

    def __init__(self, predictor: nn.Module, state: Tensor):
        super().__init__()
        self.predictor = predictor
        self.state = state

    def apply(self, data: DataType, out: MemoryOut, state: State) -> PT:
        pred = self.predictor(out, self.state)
        return PT(pred, data.y)