import time
from typing import Iterable, Iterator

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, BertModel, BertForSequenceClassification

from datasets.add import AddTask
from datasets.gena import HumanDataset
from filter_model.base import FilteredRecurrentTransformer, NStepFilterObject, FilterModel
from filter_model.chunk_filter import ChunkFilter
from filter_model.seq_filter import SeqFilter, DictSeqFilter
from metrics.accuracy import AccuracyMetric
from datasets.pmnist import PermMNISTTaskGenerator
from models.pos_encoding import LinearEmbedWithPos
from models.transformers import RecurrentTransformer, TransformerClassifier, BertRecurrentTransformer

torch.cuda.set_device("cuda:1")

def inf_loader(loader: DataLoader):
    while True:
        for data in loader:
            yield data


tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
bert_model: BertModel = BertForSequenceClassification.from_pretrained('AIRI-Institute/gena-lm-bert-base').bert

data = HumanDataset(
    "/home/nazar/GENA_LM/downstream_tasks/promoter_prediction/hg38_len_2000_promoters_dataset.csv",
    tokenizer
)

train_data, test_data = torch.utils.data.random_split(data, [int(len(data) * 0.8), len(data) - int(len(data) * 0.8)])

train_loader = inf_loader(DataLoader(train_data, shuffle=True, batch_size=32))
test_loader = inf_loader(DataLoader(test_data, shuffle=True, batch_size=256))


def make_batch(loader: Iterator[DataLoader]):
    X, m, y = next(loader)
    X, m, y = X.cuda(), m.cuda(), y.cuda()
    return {'input_ids': X, 'attention_mask': m}, y

filter_model: FilterModel = DictSeqFilter(
    size=40,
    key='input_ids'
).cuda()

rec_transformer = FilteredRecurrentTransformer(
    BertRecurrentTransformer(bert_model, num_layers=2, dim_feedforward=512),
    filter_model,
    embedding=None,
    rollout=2
).cuda()

predictor = TransformerClassifier(2, bert_model.config.hidden_size, 4, 1, 512).cuda()

writer = SummaryWriter(f"/home/nazar/pomoika/gena_2000_{time.time()}")

opt = torch.optim.Adam(
    [{'params': rec_transformer.transformer.encoder.parameters(), 'lr': 2e-5},
     {'params': rec_transformer.transformer.bert.parameters(), 'lr': 2e-6},
     {'params': predictor.parameters(), 'lr': 2e-5}
     ]
)

scheduler = StepLR(opt, step_size=200, gamma=0.99)

for i in range(30000):
    print("iter", i)
    rec_transformer.train()
    predictor.train()
    X, Y = make_batch(train_loader)
    s0 = torch.zeros(X['input_ids'].shape[0], 30, bert_model.config.hidden_size).cuda()
    t0 = time.time()

    states_generator = rec_transformer.forward(X, s0)

    for _ in range(5):
        opt.zero_grad()
        s = next(states_generator)
        if s is None:
            break
        pred = predictor(s)
        loss1 = nn.CrossEntropyLoss()(pred, Y)

        s = next(states_generator)
        pred = predictor(s)
        loss2 = nn.CrossEntropyLoss()(pred, Y)

        (loss1 + loss2).backward()
        opt.step()

    scheduler.step()

    print("iter time", time.time() - t0)
    print("train loss", loss2.item())
    writer.add_scalar("train loss", loss2.item(), i)

    if i % 20 == 0:
        rec_transformer.eval()
        predictor.eval()
        with torch.no_grad():
            X, Y = make_batch(test_loader)
            s0 = torch.zeros(X['input_ids'].shape[0], 30, bert_model.config.hidden_size).cuda()
            *_, last_state = rec_transformer.forward(X, s0)
            pred = predictor(last_state)
            loss = nn.CrossEntropyLoss()(pred, Y)
            acc = AccuracyMetric()(pred, Y)
            print("loss:", loss.item(), "acc:", acc)
            writer.add_scalar("test loss", loss.item(), i)
            writer.add_scalar("test acc", acc, i)
