from typing import List

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd


class HumanDataset(Dataset):

    def __init__(self, path, tokenizer):
        df = pd.read_csv(path)
        self.labels = [label for label in df['promoter_presence']]
        self.texts = tokenizer([
            text for text in df['sequence']
        ], return_tensors="pt", padding='max_length', max_length=400)
        self.ids = np.asarray(self.texts['input_ids'])
        self.att = self.texts['attention_mask']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ids[idx], self.att[idx], self.labels[idx]


class HumanDataset2(Dataset):

    def __init__(self, folds: List[str], tokenizer):
        self.labels = []
        self.texts = []
        self.tokenizer = tokenizer

        for path in folds:
            df = pd.read_csv(path)
            print(path)
            self.labels.extend([int(label) for label in df['promoter_presence']])
            self.texts.extend([text for text in df['sequence']])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # tok = self.tokenizer(text, return_tensors="pt", padding='max_length', max_length=2800)
        # ids = np.asarray(tok['input_ids'][0])
        # att = tok['attention_mask'][0]
        return {'text': text, 'label': self.labels[idx]}