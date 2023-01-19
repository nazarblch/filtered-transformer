from collections import namedtuple
from typing import Iterator, Tuple, List, Callable, Optional
import torch
from torch.utils.tensorboard import SummaryWriter
import time
from gena_lm.modeling_bert import BertModel, BertForSequenceClassification
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from datasets.copy import CopyTask
from datasets.enformer_h5 import EnformerDataset
from memup.accumulator import Accumulator
from memup.base import SeqDataFilter, MemUpMemory, MemUpLoss, MemUpLossIterator, State, Info, Done, InfoUpdate
from memup.data_filters import SlidingWindowFilter
from memup.loss import PredictorLossWithContext, LossModule, EvalLoss
from memup.preproc import ContextPreprocessor, NStepUpdate, IncrementStep, ErrorPreprocessor, TargetsSampler, \
    TailTargets
from metrics.pearson import PearsonCorrLoss, PearsonCorrMetric
from models.pos_encoding import EmbedWithPos
from models.transformers import BertRecurrentTransformer, RecurrentTransformerFromBert, \
    BertRecurrentTransformerWithTokenizer, TorchRecurrentTransformer, TorchRecurrentNN

mem_transformer = TorchRecurrentTransformer(256, 4, 3, 512, dropout=0.1).cuda()
embed = EmbedWithPos(10, 256, 10).cuda()


class Predictor(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = TorchRecurrentTransformer(256, 4, 2, 512, dropout=0)
        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x, state):
        out = self.encoder.forward(x, state).out
        return self.head(out)


predictor = Predictor().cuda()

train_data = CopyTask(10000, 10, 500)
train_loader = DataLoader(train_data, shuffle=True, batch_size=128)
opt = torch.optim.Adam([
    {"params": mem_transformer.parameters(), "lr": 4e-5},
    {"params": embed.parameters(), "lr": 4e-5},
    {"params": predictor.parameters(), "lr": 4e-5}
])

mem_acc = Accumulator(mem_transformer, decay=0.95)
pred_acc = Accumulator(predictor, decay=0.95)
embed_acc = Accumulator(embed, decay=0.95)

DataType = namedtuple("DataType", ["x", "y", "length"])
BS = 20


class SeqDataFilterImpl(SlidingWindowFilter[DataType]):

    def __init__(self):
        super().__init__(BS, padding=BS // 5)

    def filter_data(self, data: DataType, i1: int, i2: int, i1_pad: int, i2_pad: int) -> DataType:
        pad_x = data.x[:, i1_pad: i2_pad]
        y = data.y[:, i1: i2]

        return DataType(pad_x, y, i2_pad - i1_pad)


class MemUpMemoryImpl(MemUpMemory):

    def __init__(self, embed: EmbedWithPos, mem_tr: TorchRecurrentTransformer):
        super().__init__()
        self.mem_tr = mem_tr
        self.embed = embed

    def forward(self, data: DataType, state: State) -> Tuple[Tensor, State]:
        x_embed = self.embed.forward(data.x.cuda())
        os = self.mem_tr.forward(x_embed, state)
        assert os.out.shape[1] >= data.y.shape[1]
        return os.out[:, os.out.shape[1] - data.y.shape[1]:], os.state


# data_filter = SeqDataFilterImpl()
# inc_update = IncrementStep()
#
#
# def roll(data):
#     context = []
#     s = torch.zeros(x.shape[0], 10, 128).cuda()
#     done = False
#     info = {}
#
#     while not done:
#         info = inc_update.forward(data, s, info)
#         fitered_data, done = data_filter.forward(data, s, info)
#         os = mem_acc.get_module()(embed_acc.get_module()(fitered_data.x.cuda()), s)
#         context.append(os.out[:, os.out.shape[1] - fitered_data.y.shape[1]:])
#         s = os.state
#
#     return torch.cat(context, 1)



memup_iter = MemUpLossIterator[DataType](
    rollout=2,
    memory=MemUpMemoryImpl(embed, mem_transformer),
    loss=PredictorLossWithContext(predictor, [
        LossModule(nn.CrossEntropyLoss(), "CE", 1.0),
    ], lambda data: data.y),
    data_filter=SeqDataFilterImpl(),
    info_update=[
        IncrementStep(),
        NStepUpdate(ContextPreprocessor(MemUpMemoryImpl(embed_acc.get_module(), mem_acc.get_module()), SeqDataFilterImpl()), 200),
        NStepUpdate(TailTargets(10, lambda data: data.y), 200)
    ]
)



for i in range(1000):
    print("epoch", i)
    for x, y in train_loader:
        print()

        state = torch.zeros(x.shape[0], 10, 256).cuda()
        T = x.shape[1]
        data = DataType(x, y, T)

        # with torch.no_grad():
        #     context = roll(data)
        #     c_tail = context[:, T - 10:].cuda()
        #     y_tail = y[:, T - 10:].cuda()

        done = False
        info = {}

        while not done:

            opt.zero_grad()

            loss, state, info, done = memup_iter.forward(DataType(x, y, T), state, info)

            loss.backward()
            opt.step()

        print(info['losses'])

        mem_acc.accumulate()
        pred_acc.accumulate()
        embed_acc.accumulate()

