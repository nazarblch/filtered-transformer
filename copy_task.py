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
from memup.preproc import ContextPreprocessor, NStepUpdate, IncrementStep, ErrorPreprocessor, TargetsSampler
from metrics.pearson import PearsonCorrLoss, PearsonCorrMetric
from models.transformers import BertRecurrentTransformer, RecurrentTransformerFromBert, \
    BertRecurrentTransformerWithTokenizer, TorchRecurrentTransformer, TorchRecurrentNN

mem_transformer = TorchRecurrentNN(256).cuda()
embed = nn.Embedding(20, 256).cuda()


class Predictor(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = TorchRecurrentTransformer(256, 4, 2, 512)
        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x, state):
        out = self.encoder.forward(x, state).out
        return self.head(out)


predictor = Predictor().cuda()

opt = torch.optim.Adam([
    {"params": mem_transformer.parameters(), "lr": 2e-5},
    {"params": embed.parameters(), "lr": 2e-5},
    {"params": predictor.parameters(), "lr": 2e-5}
])


# writer = SummaryWriter(f"/home/slavic/pomoika/copy_{time.time()}")

device = torch.device("cuda")
BS = 30
TOPK = 10


DataType = namedtuple("DataType", ["x", "y", "length"])
DataTypeWithMemory = Tuple[DataType, Tensor, Tensor]


class SeqDataFilterImpl(SlidingWindowFilter[DataType]):

    def __init__(self):
        super().__init__(BS, padding=BS // 5)

    def filter_data(self, data: DataType, i1: int, i2: int, i1_pad: int, i2_pad: int) -> DataType:
        pad_x = data.x[:, i1: i2]
        y = data.y[:, i1: i2]

        return DataType(pad_x, y, i2_pad - i1_pad)


class MemUpMemoryImpl(MemUpMemory):

    def __init__(self, embed: nn.Embedding, mem_tr: TorchRecurrentTransformer):
        super().__init__()
        self.mem_tr = mem_tr
        self.embed = embed

    def forward(self, data: DataType, state: State) -> Tuple[Tensor, State]:
        x_embed = self.embed.forward(data.x.cuda())
        os = self.mem_tr.forward(x_embed, state)
        assert os.out.shape[1] >= data.y.shape[1]
        return os.out[:, os.out.shape[1] - data.y.shape[1]:], os.state


mem_acc = Accumulator(mem_transformer, decay=0.95)
pred_acc = Accumulator(predictor, decay=0.95)
embed_acc = Accumulator(embed, decay=0.95)


# memup_iter_eval = MemUpLossIterator[DataType](
#     rollout=2000,
#     memory=MemUpMemoryImpl(mem_transformer),
#     loss=EvalLoss(predictor, [
#         PearsonCorrMetric()
#     ]),
#     data_filter=SeqDataFilterImpl(),
#     info_update=[
#         IncrementStep()
#     ]
# )


memup_iter_with_extra_targets = MemUpLossIterator[DataType](
    rollout=2,
    memory=MemUpMemoryImpl(embed, mem_transformer),
    loss=PredictorLossWithContext(predictor, [
        LossModule(nn.CrossEntropyLoss(), "CE", 1.0),
    ], lambda data: data.y),
    data_filter=SeqDataFilterImpl(),
    info_update=[
        IncrementStep(),
        NStepUpdate(ContextPreprocessor(MemUpMemoryImpl(embed_acc.get_module(), mem_acc.get_module()), SeqDataFilterImpl()), 200),
        NStepUpdate(ErrorPreprocessor(pred_acc.get_module(), nn.CrossEntropyLoss(reduction="none"), lambda data: data.y), 200),
        NStepUpdate(TargetsSampler(TOPK, lambda data: data.y), 2, offset=1)
    ]
)


# @torch.no_grad()
# def evaluate(train_loader):
#     text2, target2, coords2 = next(iter(train_loader))
#     state2 = torch.zeros(target2.shape[0], 50, bert.config.hidden_size, device=device)
#     info = {}
#     _, _, info, _ = memup_iter_eval.forward(DataType(text2, target2, coords2), state2, info)
#     return info


def train_one_epoch(memup_iter, train_loader, global_step):

    for x1, y1 in train_loader:
        print()

        # eval_res = evaluate(train_loader)
        # writer.add_scalar("eval/pearson corr coef 1", eval_res["pearson corr coef 1"], global_step)
        # writer.add_scalar("eval/pearson corr coef 2", eval_res["pearson corr coef 2"], global_step)
        # writer.add_scalar("eval/poisson errors", eval_res["poisson errors"], global_step)

        state = torch.zeros(x1.shape[0], 10, 256, device=device)
        done = False
        info = {}

        while not done:
            global_step += 1

            opt.zero_grad()
            loss, state, info, done = memup_iter.forward(DataType(x1, y1, x1.shape[1]), state, info)
            assert loss is not None
            loss.backward()
            opt.step()
            print(loss.item())

            # if global_step % 10 == 0:
            #     if "pearson_corr current" in info:
            #         writer.add_scalar("pearson_corr/current", info["pearson_corr current"], global_step)
            #         writer.add_scalar("poisson_nll/current", info["poisson_nll current"], global_step)
            #     writer.add_scalar("pearson_corr/selected", info["pearson_corr selected"], global_step)
            #     writer.add_scalar("poisson_nll/selected", info["poisson_nll selected"], global_step)
            #     writer.add_scalar("sum loss", info["sum loss"], global_step)

        mem_acc.accumulate()
        pred_acc.accumulate()
        embed_acc.accumulate()

    return global_step


global_step = 0
train_data = CopyTask(10000, 10, 100)
train_loader = DataLoader(train_data, shuffle=True, batch_size=64)

for i in range(1000):
    print("epoch", i)

    global_step = train_one_epoch(memup_iter_with_extra_targets, train_loader, global_step)

