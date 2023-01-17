import time
from collections import namedtuple
from copy import deepcopy
from typing import Tuple, List

import numpy as np
import torch
from gena_lm.modeling_bert import BertForSequenceClassification, BertModel, BertEncoder
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, BertConfig
from tqdm import tqdm

from datasets.gena import HumanDataset, HumanDataset2
from filter_model.base import FilterModel, FilteredRecurrentTransformer, NStepFilterObject
from filter_model.chunk_filter import BertChunkFilter
from filter_model.seq_filter import DictSeqFilterBidirectional, DictSeqFilter
from memup.base import MemUpMemory, State, MemUpLoss, Info, MemUpLossIterator
from memup.data_filters import SlidingWindowFilter
from memup.preproc import IncrementStep
from metrics.accuracy import AccuracyMetric
from models.transformers import TransformerClassifier, BertClassifier, BertRecurrentTransformer, BertRecurrentLSTM, \
    BertRecurrentTransformerWithTokenizer

torch.cuda.set_device("cuda:0")

tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
bert_model: BertModel = BertForSequenceClassification.from_pretrained('AIRI-Institute/gena-lm-bert-base').bert
mem_transformer = BertRecurrentTransformerWithTokenizer(bert_model, tokenizer, 300, 4, 3, bert_model.config.hidden_size * 2).cuda()
head = BertClassifier(2, bert_model.config, 4, 2, bert_model.config.hidden_size).cuda()

opt = torch.optim.Adam([
    {"params": mem_transformer.bert.parameters(), "lr": 4e-6},
    {"params": head.parameters(), "lr": 2e-5},
    {"params": mem_transformer.encoder.parameters(), "lr": 2e-5},
])

dataset = HumanDataset2(
    [
        "/home/slavic/PycharmProjects/len_16000/fold_1.csv",
        "/home/slavic/PycharmProjects/len_16000/fold_2.csv",
        "/home/slavic/PycharmProjects/len_16000/fold_3.csv",
        "/home/slavic/PycharmProjects/len_16000/fold_4.csv",
        "/home/slavic/PycharmProjects/len_16000/fold_5.csv",
     ]
)

print("data")

train_data, test_data = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])


train_loader = DataLoader(train_data, shuffle=True, batch_size=32)
test_loader = DataLoader(test_data, shuffle=False, batch_size=256)

writer = SummaryWriter(f"/home/slavic/pomoika/gena_16000_seq_tr_{time.time()}")
device = torch.device("cuda")
BS = 1000
DataType = namedtuple("DataType", ["text", "target"])
DataTypeWithMemory = Tuple[DataType, Tensor, Tensor]


class SeqDataFilterImpl(SlidingWindowFilter[DataType]):

    def __init__(self):
        super().__init__(BS, padding=BS // 5)

    def filter_data(self, data: DataType, i1: int, i2: int, i1_pad: int, i2_pad: int) -> DataType:
        pad_text = [t[i1_pad:i2_pad] for t in data.text]

        return DataType(pad_text, data.target)


class MemUpMemoryImpl(MemUpMemory):

    def __init__(self, mem_tr: BertRecurrentTransformerWithTokenizer):
        super().__init__()
        self.mem_tr = mem_tr

    def forward(self, data: DataType, state: State) -> Tuple[Tensor, State]:
        os = self.mem_tr.forward(data.text, state)
        return None, os.state


class MemUpLossImpl(MemUpLoss):

    def __init__(self):
        super().__init__()

    def loss(self, data, state, target):
        target = target.cuda()
        target = torch.cat([target] * len(data), 0)
        pred = head.forward(state)
        loss = nn.CrossEntropyLoss()(pred, target)
        acc = AccuracyMetric()(pred, target)
        return loss, acc

    def forward(self, data: List[DataTypeWithMemory], info: Info) -> Tensor:
        target = data[-1][0].target
        s0 = torch.cat([d[2] for d in data], 0)

        loss, acc = self.loss(data, s0, target)
        info["loss"] = loss.item()
        info["acc"] = acc

        return loss


class MemUpEval(MemUpLoss):

    def __init__(self):
        super().__init__()

    def loss(self, data, state, target):
        target = target.cuda()
        pred = head.forward(state)
        loss = nn.CrossEntropyLoss()(pred, target)
        acc = AccuracyMetric()(pred, target)
        return loss, acc

    def forward(self, data: List[DataTypeWithMemory], info: Info) -> Tensor:
        target = data[-1][0].target
        s0 = data[-1][2]
        loss, acc = self.loss(data, s0, target)
        info["loss"] = loss.item()
        info["acc"] = acc

        return loss


memup_iter = MemUpLossIterator[DataType](
    rollout=2,
    memory=MemUpMemoryImpl(mem_transformer),
    loss=MemUpLossImpl(),
    data_filter=SeqDataFilterImpl(),
    info_update=[
        IncrementStep()
    ]
)

memup_iter_eval = MemUpLossIterator[DataType](
    rollout=2000,
    memory=MemUpMemoryImpl(mem_transformer),
    loss=MemUpEval(),
    data_filter=SeqDataFilterImpl(),
    info_update=[
        IncrementStep()
    ]
)


@torch.no_grad()
def evaluate(global_step):
    print("evaluate")
    mem_transformer.eval()
    head.eval()

    n = 0
    acc = []
    for data2 in test_loader:
        state2 = torch.zeros(data2["label"].shape[0], 50, bert_model.config.hidden_size, device=device)
        data2 = DataType(data2["text"], data2["label"])
        info = {}
        _, _, info, _ = memup_iter_eval.forward(data2, state2, info)
        writer.add_scalar("eval/loss", info["loss"], global_step + n)
        writer.add_scalar("eval/acc", info["acc"], global_step + n)
        acc.append(info["acc"])
        n += 1
        print(n)
        if n > 30:
            break

    writer.add_scalar("eval/acc_mean", sum(acc) / len(acc), global_step + n)

    mem_transformer.train()
    head.train()

def train_one_epoch(memup_iter, train_loader, global_step):

    for data1 in train_loader:
        print()

        if global_step % 1000 == 0 and global_step > 0:
            evaluate(global_step)

        state = torch.zeros(data1["label"].shape[0], 50, bert_model.config.hidden_size, device=device)
        data1 = DataType(data1["text"], data1["label"])
        done = False
        info = {}

        while not done:
            global_step += 1

            opt.zero_grad()
            loss, state, info, done = memup_iter.forward(data1, state, info)

            loss.backward()
            opt.step()

        print(global_step, info["loss"], info["acc"])
        writer.add_scalar("loss", info["loss"], global_step)
        writer.add_scalar("acc", info["acc"], global_step)

    return global_step


global_step = 0

for i in range(1000):
    print("epoch", i)
    global_step = train_one_epoch(memup_iter, train_loader, global_step)

    torch.save({
        "mem": mem_transformer.state_dict(),
        "pred": head.state_dict()
    }, "/home/slavic/PycharmProjects/promoter.pt")


