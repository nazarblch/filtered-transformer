from typing import List
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd


class Promoters(Dataset):

    def __init__(self, folds: List[str]):
        self.labels = []
        self.texts = []

        for path in folds:
            df = pd.read_csv(path)
            print(path)
            self.labels.extend([int(label) for label in df['promoter_presence']])
            self.texts.extend([text for text in df['sequence']])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        return text, self.labels[idx]


class PromotersTest(Dataset):

    def __init__(self, folds: List[str]):
        self.labels = []
        self.texts = []

        for path in folds:
            df = pd.read_csv(path)
            print(path)

            self.texts.extend([text for text in df['sequence']])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        return text, self.labels[idx]