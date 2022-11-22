import time
from copy import deepcopy

import numpy as np
import torch
from gena_lm.modeling_bert import BertForSequenceClassification, BertModel, BertEncoder
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, BertConfig
import pandas as pd
from tqdm import tqdm

from datasets.gena import HumanDataset
from filter_model.base import FilterModel, FilteredRecurrentTransformer
from filter_model.seq_filter import DictSeqFilterBidirectional
from models.transformers import TransformerClassifier, BertClassifier, BertRecurrentTransformer

torch.cuda.set_device("cuda:0")

tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
bert_model: BertModel = BertForSequenceClassification.from_pretrained('AIRI-Institute/gena-lm-bert-base').bert

rec_transformer = FilteredRecurrentTransformer(
    BertRecurrentTransformer(bert_model, 4, 2, bert_model.config.hidden_size),
    DictSeqFilterBidirectional(
        size=50,
        key='input_ids'
    ),
    embedding=None,
    rollout=2
).cuda()
head = BertClassifier(2, bert_model.config, 4, 2, bert_model.config.hidden_size).cuda()

opt = torch.optim.Adam([
    {"params": rec_transformer.transformer.bert.parameters(), "lr": 4e-6},
    {"params": head.parameters(), "lr": 4e-5},
    {"params": rec_transformer.transformer.encoder.parameters(), "lr": 4e-5},
])
scheduler = StepLR(opt, step_size=200, gamma=0.95)

data = HumanDataset(
    "/home/nazar/PycharmProjects/GENA_LM/downstream_tasks/promoter_prediction/hg38_len_2000_promoters_dataset.csv",
    tokenizer
)

train_data, test_data = torch.utils.data.random_split(data, [int(len(data) * 0.8), len(data) - int(len(data) * 0.8)])

train_loader = DataLoader(train_data, shuffle=True, batch_size=64)
test_loader = DataLoader(test_data, shuffle=False, batch_size=256)

writer = SummaryWriter(f"/home/nazar/pomoika/gena_2000_seq_tr_{time.time()}")
step = 0

for epoch_num in range(100):

    rec_transformer.train()
    head.train()

    for X, m, y in tqdm(train_loader):

        X, m, y = X.cuda(), m.cuda(), y.cuda()
        X = rec_transformer.transformer.bert.embeddings(X).detach()
        B = X.shape[0]

        s0 = torch.zeros(X.shape[0], 30, bert_model.config.hidden_size, device=X.device)
        gen = rec_transformer.forward({"input_ids": X, "attention_mask": m}, s0)

        for s in gen:
            pred_1 = head(s)

            batch_loss = nn.CrossEntropyLoss()(pred_1, y.repeat(pred_1.shape[0] // B))
            opt.zero_grad()
            batch_loss.backward()
            opt.step()

        scheduler.step()

        acc = (pred_1[pred_1.shape[0] - B:].argmax(dim=1) == y).sum().item() / X.shape[0]
        writer.add_scalar("train acc", acc, step)
        writer.add_scalar("train loss", batch_loss.item(), step)

        step += 1

    with torch.no_grad():

        rec_transformer.eval()
        head.eval()

        test_acc = 0

        for X, m, y in test_loader:
            X, m, y = X.cuda(), m.cuda(), y.cuda()
            X = rec_transformer.transformer.bert.embeddings(X)
            B = X.shape[0]
            s0 = torch.zeros(X.shape[0], 30, bert_model.config.hidden_size, device=X.device)
            *_, s = rec_transformer.forward({"input_ids": X, "attention_mask": m}, s0)
            pred_1 = head(s)
            acc = (pred_1[pred_1.shape[0] - B:].argmax(dim=1) == y).sum().item()
            test_acc += acc

        writer.add_scalar("test acc", test_acc / len(test_data), epoch_num)