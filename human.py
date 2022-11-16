import numpy as np
import torch
from gena_lm.modeling_bert import BertForSequenceClassification, BertModel
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm

from datasets.gena import HumanDataset

torch.cuda.set_device("cuda:1")

class BertClassifier(nn.Module):

    def __init__(self, bert, dropout=0.1):

        super(BertClassifier, self).__init__()

        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        output = self.bert(input_ids=input_id, attention_mask=mask)
        print(type(output))
        print(output['last_hidden_state'].shape)
        dropout_output = self.dropout(output['pooler_output'])
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
model: BertModel = BertForSequenceClassification.from_pretrained('AIRI-Institute/gena-lm-bert-base').bert
cls = BertClassifier(model).cuda()
opt = torch.optim.Adam([
    {"params": cls.bert.parameters(), "lr": 1e-5},
    {"params": cls.linear.parameters(), "lr": 1e-4},
])

data = HumanDataset(
    "/home/nazar/GENA_LM/downstream_tasks/promoter_prediction/hg38_len_2000_promoters_dataset.csv",
    tokenizer
)

loader = DataLoader(data, shuffle=True, batch_size=24)

for epoch_num in range(100):

    total_acc_train = 0
    total_loss_train = 0

    for X, m, y in tqdm(loader):

        X, m, y = X.cuda(), m.cuda(), y.cuda()

        pred = cls(X, m)

        batch_loss = nn.CrossEntropyLoss()(pred, y)
        total_loss_train += batch_loss.item()

        acc = (pred.argmax(dim=1) == y).sum().item()
        total_acc_train += acc

        cls.zero_grad()
        batch_loss.backward()
        opt.step()

    print(f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(data): .3f} \
                    | Train Accuracy: {total_acc_train / len(data): .3f}')