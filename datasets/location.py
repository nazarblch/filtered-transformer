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

