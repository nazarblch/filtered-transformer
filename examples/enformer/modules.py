from copy import deepcopy
from data import EnformerDataset
from torch.utils.data import DataLoader
from collections import namedtuple
from typing import Tuple, Optional
from torch import Tensor
from common_modules.transformers import BertRecurrentTransformerWithTokenizer, EncoderFromBert
from memup.base import MemUpMemory, State, DataCollectorReplace, MemoryOut, DataCollectorAppend
from memup.loss import TS, TOS
from torch import nn
from memup.preproc import IncrementStep
from metrics.base import Metric
from common_modules.transformers import BertRecurrentTransformerWithTokenizer, BertRecurrentTransformer, RecurrentTransformerFromBert, \
    BertRecurrentTransformerWithTokenizer
import torch
from data_filters.sliding_window import SlidingWindowFilter, SlidingWindowWithPadding
from gena_lm.modeling_bert import BertPreTrainedModel, BertModel, BertEncoder


DataType = namedtuple("DataType", ["text", "target", "coords", "length"])


class MemUpMemoryImpl(MemUpMemory[DataType]):

    def __init__(self, mem_tr: BertRecurrentTransformerWithTokenizer):
        super().__init__()
        self.mem_tr = mem_tr
        self.mem_tr.train()

    def forward(self, data: DataType, state: State) -> Tuple[None, State]:
        os = self.mem_tr.forward(data.text, state)

        return None, os.state


class DataCollectorTrain(DataCollectorAppend[DataType, TS]):
    def apply(self, data: DataType, out: MemoryOut, state: State) -> TS:
        return TS(data.target, state)


class DataCollectorLastState(DataCollectorReplace[DataType, TS]):
    def apply(self, data: DataType, out: MemoryOut, state: State) -> TS:
        return TS(data.target, state)


class Predictor(nn.Module):

    def __init__(self, bert):
        super().__init__()
        config2 = deepcopy(bert.config)
        config2.num_attention_heads = 4
        config2.num_hidden_layers = 3
        config2.intermediate_size = bert.config.hidden_size * 2

        self.encoder = BertEncoder(config2)

        self.head = nn.Sequential(
            nn.Linear(bert.config.hidden_size, bert.config.hidden_size),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(bert.config.hidden_size, 896),
            nn.Softplus()
        )

    def forward(self, x, state):
        B, D = state.shape[0], state.shape[2]
        T = x.shape[1]
        xs = torch.cat([x, state], dim=1)
        out = self.encoder.forward(xs)['last_hidden_state'][:, :T]
        return self.head(out)


class PearsonCorrLoss(nn.Module):
    def forward(self, x, y, dim=1):
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1, y.shape[-1])
        x_centered = x - x.mean(dim=dim, keepdim=True)
        y_centered = y - y.mean(dim=dim, keepdim=True)
        return -nn.functional.cosine_similarity(x_centered, y_centered, dim=dim).mean()


class PearsonCorrMetric(Metric, PearsonCorrLoss):

    def __init__(self):
        super().__init__("PearsonCorr")

    @torch.no_grad()
    def __call__(self, x: Tensor, y: Tensor) -> float:
        return -super().forward(x, y).item()