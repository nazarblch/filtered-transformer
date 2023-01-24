import torch
import random
from torch.utils.data import Dataset


class AddTask(Dataset):
    def __init__(self, size):
        self.size = size
        self.X, self.Y = self.gen_batch(size)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return self.X[i], self.Y[i]

    def gen_batch(self, B):
        Y = torch.randn(B, 1)
        X = torch.randn(B, self.size, 1)
        X0 = torch.zeros(B, self.size, 1)
        for i in range(B):
            p1 = random.randint(0, self.size - 2)
            p2 = random.randint(p1, self.size - 1)
            X0[i, p1] = 1
            X0[i, p2] = 1
            Y[i] = X[i, p1] + X[i, p2]

        X = torch.cat([X, X0], -1)

        return X, Y
