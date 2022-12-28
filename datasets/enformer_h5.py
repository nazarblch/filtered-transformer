import h5py
import numpy as np
from torch.utils.data import Dataset
from typing import List


class EnformerDataset(Dataset):

    def __init__(self, folds: List[str]):
        self.targets = []
        self.texts = []
        self.coords = []

        for path in folds:
            print(path)
            with h5py.File("/home/nazar/PycharmProjects/enformer/train_0.h5", "r") as f:
                for key in f.keys():
                    self.targets.append(f[key]["target"][()])
                    self.coords.append(f[key]["coordinates"][()])
                    self.texts.append(f[key]["seq"][()].decode('UTF-8'))

        self.targets = np.stack(self.targets)
        # sd = np.std(self.targets, axis=0) + 1e-3
        # m = np.mean(self.targets, axis=0)
        # for i in range(self.targets.shape[0]):
        #     self.targets[i] = (self.targets[i] - m) / sd
        # print(np.max(self.targets))
        self.coords = np.stack(self.coords)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        return text, self.targets[idx], self.coords[idx]