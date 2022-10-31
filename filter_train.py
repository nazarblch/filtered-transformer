from itertools import chain

import torch
from torch import Tensor, nn
from hfilter.filter_tree import FilterConstructorTree, KeysModel, ValuesModel, StateTransform, ValuesToFilterModel, \
    HierarchicalFilter
from models import FloatTransformer
from pmnist import PermMNISTTaskGenerator
import numpy as np


class KeysModelImpl(KeysModel):
    def __init__(self):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128)
        )

    def _model_forward(self, x: Tensor, filter_data: Tensor) -> Tensor:
        return filter_data.bmm(self.embed(x).mean(1)[:, :, None])[:, :, 0]


class ValuesModelImpl(ValuesModel):
    def forward(self, x: Tensor, values_data: Tensor, keys_sample: Tensor) -> Tensor:
        return keys_sample[:, None, :].bmm(values_data)


class StateTransformImpl(StateTransform):

    def __init__(self, dim1: int, dim2: int):
        super().__init__()
        self.net = nn.Sequential(
            FloatTransformer(dim1, 128),
            nn.Linear(128, dim2)
        )

    def forward(self, x: Tensor, keys_sample: Tensor):
        B, N = keys_sample.shape
        k = keys_sample[:, None, :].expand(B, x.shape[1], N)
        return self.net(torch.cat([x, k], dim=-1))


class ValuesToFilterModelImpl(ValuesToFilterModel):

    def __init__(self):
        super().__init__()
        self.transformer = nn.Transformer(128, 4, 3, 3, 128, batch_first=True)
        self.head = nn.Linear(128, 2)

    def _model_forward(self, x: Tensor, v: Tensor) -> Tensor:
        x1 = self.transformer(v, x)
        return self.head(x1)


if __name__ == "__main__":

    ft = FilterConstructorTree(5, 4, 128, StateTransformImpl(128 + 5, 128), KeysModelImpl(), ValuesModelImpl()).cuda()
    hf = HierarchicalFilter(1, 128, 28 * 4, ValuesToFilterModelImpl()).cuda()

    data_transform = nn.Transformer(128, 4, 3, 3, 128 * 4, batch_first=True).cuda()
    predictor = nn.Linear(128, 10).cuda()

    gen = PermMNISTTaskGenerator(is_train=True, padding=False)

    opt = torch.optim.Adam(
        chain(ft.parameters(), hf.parameters(), data_transform.parameters(), predictor.parameters()),
        lr=1e-5
    )

    for i in range(10000):
        batch = [gen.gen_trajectory() for _ in range(16)]
        X = torch.from_numpy(np.stack([b["x"] for b in batch])).view(16, 28*28, 1).cuda()
        Y = torch.from_numpy(np.stack([b["s"] for b in batch])).view(16).cuda()
        s0 = torch.ones(16, 10, 128).cuda()
        s = s0
        print("new batch")
        for _ in range(100):
            print("detach")
            opt.zero_grad()
            for _ in range(10):
                v_list = ft.forward(s)
                fd = hf.forward(v_list, X)[:, 0:100]
                print(fd.shape[1])
                s = data_transform(fd, s)
            pred = predictor(s.mean(1))
            loss = nn.CrossEntropyLoss()(pred, Y)
            loss.backward()
            print(loss.item())
            opt.step()
            s = s.detach()
