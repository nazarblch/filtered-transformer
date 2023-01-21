from collections import namedtuple
from typing import Iterator, Tuple, List, Callable, Optional
import torch
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
import time
from gena_lm.modeling_bert import BertModel, BertForSequenceClassification
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from modules import Predictor, DataType, MemUpMemoryImpl, SeqDataFilterImpl, TailAccuracyMetric
from data import CopyTask
from datasets.enformer_h5 import EnformerDataset
from memup.accumulator import Accumulator
from memup.base import SeqDataFilter, MemUpMemory, MemUpLoss, MemUpLossIterator, State, Info, Done, InfoUpdate
from memup.data_filters import SlidingWindowFilter
from memup.loss import PredictorLossWithContext, LossModule, EvalLoss
from memup.preproc import ContextPreprocessor, NStepUpdate, IncrementStep, ErrorPreprocessor, TargetsSampler, \
    TailTargets
from metrics.accuracy import AccuracyMetric
from metrics.base import Metric
from metrics.pearson import PearsonCorrLoss, PearsonCorrMetric
from models.pos_encoding import EmbedWithPos
from models.transformers import BertRecurrentTransformer, RecurrentTransformerFromBert, \
    BertRecurrentTransformerWithTokenizer, TorchRecurrentTransformer, TorchRecurrentNN

mem_transformer = TorchRecurrentTransformer(128, 4, 3, 512, dropout=0.1).cuda()
embed = EmbedWithPos(10, 128, 5.0).cuda()
predictor = Predictor().cuda()

seq_length = 500
rollout = 50
state_length = 20
writer = SummaryWriter(f"/home/slavic/pomoika/copy_{seq_length}_{time.time()}")

train_loader = DataLoader(CopyTask(10000, 10, seq_length), shuffle=True, batch_size=128)
test_loader = DataLoader(CopyTask(1000, 10, seq_length), shuffle=False, batch_size=250)


opt = torch.optim.Adam([
    {"params": mem_transformer.parameters(), "lr": 5e-5},
    {"params": embed.parameters(), "lr": 5e-5},
    {"params": predictor.parameters(), "lr": 5e-5}
])

mem_acc = Accumulator(mem_transformer, decay=0.95)
pred_acc = Accumulator(predictor, decay=0.95)
embed_acc = Accumulator(embed, decay=0.95)


memup_iter = MemUpLossIterator[DataType](
    rollout=2,
    memory=MemUpMemoryImpl(embed, mem_transformer),
    loss=PredictorLossWithContext(predictor, [
        LossModule(nn.CrossEntropyLoss(), "CE", 1.0),
        LossModule(AccuracyMetric(), "TAcc", 0.0)
    ], lambda data: data.y),
    data_filter=SeqDataFilterImpl(rollout),
    info_update=[
        IncrementStep(),
        NStepUpdate(ContextPreprocessor(MemUpMemoryImpl(embed_acc.get_module(), mem_acc.get_module()), SeqDataFilterImpl(rollout)), 200),
        NStepUpdate(ErrorPreprocessor(pred_acc.get_module(), nn.CrossEntropyLoss(reduction="none"), lambda data: data.y), 200),
        NStepUpdate(TargetsSampler(10, lambda data: data.y), 4, offset=3)
        # NStepUpdate(TailTargets(10, lambda data: data.y), 200)
    ]
)


memup_iter_eval = MemUpLossIterator[DataType](
    rollout=200,
    memory=MemUpMemoryImpl(embed, mem_transformer),
    loss=EvalLoss(predictor, [TailAccuracyMetric()], lambda data: data.y),
    data_filter=SeqDataFilterImpl(rollout),
    info_update=[
        IncrementStep()
    ]
)


@torch.no_grad()
def eval(i):

    print("evaluate")
    mem_transformer.eval()
    predictor.eval()

    for x, y in test_loader:
        state2 = torch.zeros(x.shape[0], state_length, 128).cuda()
        info = {}
        _, _, info, _ = memup_iter_eval.forward(DataType(x, y, x.shape[1]), state2, info)

        print(info["metrics"])
        for name, val in info["metrics"].items():
            writer.add_scalar(f"eval/{name}", val, i)

    mem_transformer.train()
    predictor.train()


for i in range(1000):
    print("epoch", i)
    eval(i)

    last_info = {}

    for x, y in train_loader:
        print()

        state = torch.zeros(x.shape[0], state_length, 128).cuda()
        T = x.shape[1]
        data = DataType(x, y, T)

        done = False
        info = {}

        while not done:

            opt.zero_grad()

            loss, state, info, done = memup_iter.forward(DataType(x, y, T), state, info)
            last_info = info['losses']

            loss.backward()
            opt.step()

        mem_acc.accumulate()
        pred_acc.accumulate()
        embed_acc.accumulate()

        print(last_info)
        for name, val in last_info.items():
            writer.add_scalar(f"train/{name}", val, i)

