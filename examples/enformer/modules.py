from data import EnformerDataset
from torch.utils.data import DataLoader
from collections import namedtuple
from typing import Tuple, Optional
from torch import Tensor
from common_modules.transformers import BertRecurrentTransformerWithTokenizer
from memup.base import MemUpMemory, State, DataCollectorReplace, MemoryOut, DataCollectorAppend
from memup.loss import TS, TOS
from torch import nn
from memup.preproc import IncrementStep
from metrics.base import Metric
from common_modules.transformers import BertRecurrentTransformerWithTokenizer, BertRecurrentTransformer, RecurrentTransformerFromBert, \
    BertRecurrentTransformerWithTokenizer
import torch
from data_filters.sliding_window import SlidingWindowFilter, SlidingWindowWithPadding



DataType = namedtuple("DataType", ["text", "target", "coords", "length", "tg_text"])


class SeqDataFilterImpl(SlidingWindowFilter[DataType]):

    def __init__(self, rollout, padding):
        super().__init__(rollout, padding)

    def filter_data(self, data: DataType, window: SlidingWindowWithPadding) -> DataType:

        i1, i2, i1_pad, i2_pad = window

        pad_text = [t[i1_pad:i2_pad] for t in data.text]
        filtered_target = data.target[(data.coords >= i1) * (i2 > data.coords)]\
            .view(data.target.shape[0], -1, data.target.shape[2])
        filtered_coords = data.coords[(data.coords >= i1) * (i2 > data.coords)]\
            .view(data.coords.shape[0], -1)

        tg_text = []
        # print(i1, i2)
        # if filtered_coords.shape[1] > 0:
        #     for t in data.text:
        #         for j in filtered_coords[0].numpy().reshape(-1):
        #             tg_text.append(t[j - 64: j + 64])
        
        return DataType(pad_text, filtered_target, filtered_coords, 0, tg_text)


class MemUpMemoryImpl(MemUpMemory[DataType]):

    def __init__(self, mem_tr: BertRecurrentTransformerWithTokenizer):
        super().__init__()
        self.mem_tr = mem_tr

    def forward(self, data: DataType, state: State) -> Tuple[Tensor | None, State]:
        os = self.mem_tr.forward(data.text, state)
        assert os.out.shape[1] >= data.target.shape[1]
        out = None
    
        if data.target.shape[1] > 0:
            T = data.target.shape[1]
            out = os.out[:, os.out.shape[1] - 4 * T:].view(os.out.shape[0], T, os.out.shape[2] * 4)
            # context = self.mem_tr.forward_bert(data.tg_text, 32).view(os.out.shape[0], T, -1)
            # out = torch.cat([out, context], -1)

        return out, os.state


class DataCollectorTrain(DataCollectorAppend[DataType, TS]):
    def apply(self, data: DataType, out: MemoryOut, state: State) -> TOS:
        return TOS(data.target, out, state)


class DataCollectorLastState(DataCollectorReplace[DataType, TS]):
    def apply(self, data: DataType, out: MemoryOut, state: State) -> TS:
        return TS(data.target, state)


class Predictor(nn.Module):

    def __init__(self, bert):
        super().__init__()
        # self.encoder = RecurrentTransformerFromBert(bert, 4, 3, bert.config.hidden_size * 2)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(bert.config.hidden_size, 4, bert.config.hidden_size * 2, 0.1, batch_first=True),
            3
        )

        self.head = nn.Sequential(
            # nn.Linear(bert.config.hidden_size * 7, bert.config.hidden_size * 7),
            # nn.ReLU(),
            nn.Linear(bert.config.hidden_size * 4, bert.config.hidden_size * 4),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(bert.config.hidden_size * 4, 5313)
        )

    def forward(self, x, state):
        B, D = state.shape[0], state.shape[2]
        T = x.shape[1]
        mem = x[:, :, 0:D * 4].reshape(B, -1, D)
        # context = x[:, :, D * 4:].reshape(B, -1, D)

        # cm = torch.cat([context, mem], 1)
        # out = self.encoder.forward(cm, state).out.reshape(B, T, D * 7)
        out = self.decoder.forward(mem, state)[:, -4* T:].reshape(B, T, D * 4)
        return self.head(out).abs()


class PearsonCorrLoss(nn.Module):
    def forward(self, x, y, dim=1):
        x_centered = x - x.mean(dim=dim, keepdim=True)
        y_centered = y - y.mean(dim=dim, keepdim=True)
        return -nn.functional.cosine_similarity(x_centered, y_centered, dim=dim).mean()


class PearsonCorrMetric(Metric, PearsonCorrLoss):

    def __init__(self):
        super().__init__("PearsonCorr")

    @torch.no_grad()
    def __call__(self, x: Tensor, y: Tensor) -> float:
        return -super().forward(x, y).item()