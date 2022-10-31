import random

import torch


class LocationDetect:
    def __init__(self, size):
        self.loc = random.randint(0, size-1)
        self.size = size

    def gen_batch(self, B):
        Y = torch.randn(B, 1)
        X = torch.zeros(B, self.size, 1)
        X[:, self.loc] = Y

        return X, Y


class AddTask:
    def __init__(self, size):
        self.size = size

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
