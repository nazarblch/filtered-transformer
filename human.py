import time
from copy import deepcopy

import numpy as np
import torch
from gena_lm.modeling_bert import BertForSequenceClassification, BertModel, BertEncoder
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, BertConfig
import pandas as pd
from tqdm import tqdm

from datasets.gena import HumanDataset, HumanDataset2
from filter_model.base import FilterModel, FilteredRecurrentTransformer, NStepFilterObject
from filter_model.chunk_filter import BertChunkFilter
from filter_model.seq_filter import DictSeqFilterBidirectional, DictSeqFilter
from models.transformers import TransformerClassifier, BertClassifier, BertRecurrentTransformer, BertRecurrentLSTM

torch.cuda.set_device("cuda:0")

tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
bert_model: BertModel = BertForSequenceClassification.from_pretrained('AIRI-Institute/gena-lm-bert-base').bert

# filter_model = BertChunkFilter(bert_model, 40, 2)

rec_transformer = FilteredRecurrentTransformer(
    BertRecurrentLSTM(bert_model, 3, bert_model.config.hidden_size),
    # NStepFilterObject(6)(filter_model),
    DictSeqFilter(200, "input_ids"),
    embedding=None,
    rollout=2
).cuda()
head = BertClassifier(2, bert_model.config, 4, 2, bert_model.config.hidden_size).cuda()

opt = torch.optim.Adam([
    {"params": rec_transformer.transformer.bert.parameters(), "lr": 4e-6},
    {"params": head.parameters(), "lr": 2e-5},
    {"params": rec_transformer.transformer.encoder.parameters(), "lr": 2e-5},
    {"params": rec_transformer.state_filter.parameters(), "lr": 2e-5},
    # {"params": filter_model.encoder.parameters(), "lr": 1e-5},
    # {"params": filter_model.head.parameters(), "lr": 1e-5},
])
# scheduler = StepLR(opt, step_size=200, gamma=0.96)

data = HumanDataset2(
    [
        "/home/nazar/PycharmProjects/GENA_LM/data/len_16000/len_16000/fold_1.csv",
        "/home/nazar/PycharmProjects/GENA_LM/data/len_16000/len_16000/fold_2.csv",
        "/home/nazar/PycharmProjects/GENA_LM/data/len_16000/len_16000/fold_3.csv",
        "/home/nazar/PycharmProjects/GENA_LM/data/len_16000/len_16000/fold_4.csv",
        "/home/nazar/PycharmProjects/GENA_LM/data/len_16000/len_16000/fold_5.csv",
     ],
    tokenizer
)

print("data")

train_data, test_data = torch.utils.data.random_split(data, [int(len(data) * 0.8), len(data) - int(len(data) * 0.8)])

def custom_collate(data):
    inputs = tokenizer([d['text'] for d in data], max_length=4000, truncation=True)
    labels = [d['label'] for d in data]
    inputs["input_ids"] = pad_sequence([torch.tensor(t) for t in inputs["input_ids"]], batch_first=True)
    inputs["attention_mask"] = pad_sequence([torch.tensor(t) for t in inputs["attention_mask"]], batch_first=True)
    labels = torch.tensor(labels)
    return inputs["input_ids"], inputs["attention_mask"], labels


train_loader = DataLoader(train_data, shuffle=True, batch_size=32, collate_fn=custom_collate)
test_loader = DataLoader(test_data, shuffle=False, batch_size=256, collate_fn=custom_collate)

writer = SummaryWriter(f"/home/nazar/pomoika/gena_2000_seq_tr_{time.time()}")
step = 0

for epoch_num in range(100):

    rec_transformer.train()
    head.train()

    for X, m, y in tqdm(train_loader):

        X, m, y = X.cuda(), m.cuda(), y.cuda()
        # X = rec_transformer.transformer.bert.embeddings(X).detach()
        B = X.shape[0]

        s0 = torch.zeros(X.shape[0], 30, bert_model.config.hidden_size, device=X.device)
        gen = rec_transformer.forward({"input_ids": X, "attention_mask": m}, s0)

        for s in gen:
            pred_1 = head(s)

            batch_loss = nn.CrossEntropyLoss()(pred_1, y.repeat(pred_1.shape[0] // B))
            opt.zero_grad()
            batch_loss.backward()
            opt.step()

        # scheduler.step()

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
            # X = rec_transformer.transformer.bert.embeddings(X)
            B = X.shape[0]
            s0 = torch.zeros(X.shape[0], 30, bert_model.config.hidden_size, device=X.device)
            *_, s = rec_transformer.forward({"input_ids": X, "attention_mask": m}, s0)
            pred_1 = head(s)
            acc = (pred_1[pred_1.shape[0] - B:].argmax(dim=1) == y).sum().item()
            test_acc += acc

        writer.add_scalar("test acc", test_acc / len(test_data), epoch_num)