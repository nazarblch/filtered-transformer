from collections import namedtuple
from typing import List, Callable

import torch
from torch import nn, Tensor
from memup.base import MemUpLoss, SDWithMemory, Info, SD
from metrics.base import Metric

LossModule = namedtuple("LossItem", ["module", "name", "coefficient"])


class MemUpLossFromPredictor(MemUpLoss):

    def __init__(self,
                 predictor: nn.Module,
                 loss_modules: List[LossModule],
                 get_target: Callable[[SD], Tensor]):
        super().__init__()
        self.predictor = predictor
        self.loss_modules = loss_modules
        self.get_target = get_target

    def loss(self, state, out, target):
        target = target.cuda()
        N = state.shape[0] // target.shape[0]
        out = torch.cat([out] * N, 0)
        target = torch.cat([target] * N, 0)
        pred = self.predictor(out, state)

        losses = {}
        sum_loss = 0

        for m in self.loss_modules:
            loss_item = m.module(pred, target)
            sum_loss = sum_loss + loss_item * m.coefficient
            losses[m.name] = loss_item.item()

        return sum_loss, losses

    def forward(self, data: List[SDWithMemory], info: Info) -> Tensor:
        out, target = torch.cat([d[1] for d in data], 1), torch.cat([self.get_target(d[0]) for d in data], 1)
        s0 = torch.cat([d[2] for d in data], 0)
        assert out.shape[1] == target.shape[1]

        loss = 0

        if out.shape[1] > 0:
            loss, losses = self.loss(s0, out, target)
            for name, l in losses.items():
                info[f"{name} current"] = l

        assert "context_selected" in info and "context_target" in info
        context = info["context_selected"].cuda()
        context_target = info["context_target"]
        assert context.shape[1] == context_target.shape[1]
        loss2, losses = self.loss(s0, context, context_target)
        loss = loss + loss2
        for name, l in losses.items():
            info[f"{name} selected"] = l

        info["sum loss"] = loss.item()

        return loss