import numpy as np
from torch.utils.data import Dataset
import torch


class CopyTask(Dataset):

    def __init__(self, n: int, head_size: int = 10, middle_size: int = 100):
        self.head_size = head_size
        self.middle_size = middle_size
        self.trajs = np.asarray([self.gen_trajectory() for _ in range(n)], dtype=object)

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, i: int):
        return self.trajs[i][0], self.trajs[i][1]

    def gen_trajectory(self):
        head = np.random.randint(0, 8, self.head_size)
        middle = np.zeros(self.middle_size, dtype=np.int64) + 8
        tail = np.zeros(self.head_size, dtype=np.int64) + 9

        x = np.concatenate([head, middle, tail])
        y = np.concatenate([head * 0 + 8, middle, head])

        return torch.from_numpy(x), torch.from_numpy(y)
