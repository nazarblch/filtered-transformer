import h5py
import numpy as np
from torch.utils.data import Dataset
from typing import List
import torch 


class EnformerDataset(Dataset):

    PAD = (196608 - 128 * 896) // 2 
    TG_COUNT = 896

    def __init__(self, path: str):
        self.h5_file = h5py.File(path, "r")
        self.h5_keys = np.asarray(list(self.h5_file.keys()))
        self.coords = torch.arange(EnformerDataset.TG_COUNT) * 128 + 64 + EnformerDataset.PAD 

    def __len__(self):
        return len(self.h5_keys)

    def __getitem__(self, idx):
        key = self.h5_keys[idx]
        f = self.h5_file
        return f[key]["seq"][()].decode('UTF-8'), f[key]["target"][()], self.coords