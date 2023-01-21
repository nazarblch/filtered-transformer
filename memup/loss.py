from collections import namedtuple
from typing import List, Callable, Optional

import torch
from torch import nn, Tensor
from memup.base import MemUpLoss, SDWithMemory, Info, SD
from memup.preproc import select_by_index
from metrics.base import Metric

LossModule = namedtuple("LossModule", ["module", "name", "coefficient"])


class PredictorLoss(MemUpLoss):

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

        B, T = pred.shape[0], pred.shape[1]

        for m in self.loss_modules:
            loss_item = m.module(pred.view(B * T, *pred.shape[2:]), target.view(B * T, *target.shape[2:]))
            sum_loss = sum_loss + loss_item * m.coefficient
            losses[m.name] = loss_item.item()

        return sum_loss, losses

    def forward(self, data: List[SDWithMemory], info: Info) -> Optional[Tensor]:
        out, target = torch.cat([d[1] for d in data], 1), torch.cat([self.get_target(d[0]) for d in data], 1)
        s0 = torch.cat([d[2] for d in data], 0)
        assert out.shape[1] == target.shape[1]

        info["losses"] = {}
        count = sum([int(d[1].shape[1] > 0) for d in data])
        loss = None

        if count > 1:
            sample_size = out.shape[1] // count
            index = torch.multinomial(
                torch.ones(out.shape[0], out.shape[1], device=out.device) / out.shape[1],
                sample_size,
                replacement=False)
            out, target = select_by_index(index, out), select_by_index(index.cpu(), target)
            assert out.shape[1] == sample_size

        if count > 0:
            loss, losses = self.loss(s0, out, target)
            info["losses"]["sum loss"] = loss.item()
            for name, l in losses.items():
                info["losses"][name] = l

        return loss


class PredictorLossStateOnly(MemUpLoss):

    def __init__(self,
                 predictor: nn.Module,
                 loss_modules: List[LossModule],
                 get_target: Callable[[SD], Tensor]):

        super().__init__()
        self.predictor = predictor
        self.loss_modules = loss_modules
        self.get_target = get_target

    def loss(self, state, target):
        target = target.cuda()
        N = state.shape[0] // target.shape[0]
        target = torch.cat([target] * N, 0)
        pred = self.predictor(state)

        losses = {}
        sum_loss = 0

        for m in self.loss_modules:
            loss_item = m.module(pred, target)
            sum_loss = sum_loss + loss_item * m.coefficient
            losses[m.name] = loss_item.item()

        return sum_loss, losses

    def forward(self, data: List[SDWithMemory], info: Info) -> Optional[Tensor]:
        target = self.get_target(data[-1][0])
        s0 = torch.cat([d[2] for d in data], 0)

        loss, losses = self.loss(s0, target)
        info["losses"] = {}
        for name, l in losses.items():
            info["losses"][name] = l

        info["losses"]["sum loss"] = loss.item()

        return loss


class PredictorLossWithContext(PredictorLoss):

    def __init__(self,
                 predictor: nn.Module,
                 loss_modules: List[LossModule],
                 get_target: Callable[[SD], Tensor],
                 context_selected_key="context_selected",
                 context_target_key="context_target"):

        super().__init__(predictor, loss_modules, get_target)
        self.context_selected_key = context_selected_key
        self.context_target_key = context_target_key

    def forward(self, data: List[SDWithMemory], info: Info) -> Tensor:
        loss1 = super().forward(data, info)
        s0 = torch.cat([d[2] for d in data], 0)

        assert self.context_selected_key in info and self.context_target_key in info
        context = info[self.context_selected_key].cuda()
        context_target = info[self.context_target_key]
        assert context.shape[1] == context_target.shape[1]
        loss, losses = self.loss(s0, context, context_target)
        if loss1 is not None:
            loss = (loss + loss1) / 2
        for name, l in losses.items():
            info["losses"][f"{name} selected"] = l

        info["losses"]["sum loss"] = loss.item()

        return loss


class EvalLoss(MemUpLoss):

    def __init__(self, predictor: nn.Module, metrics: List[Metric], get_target: Callable[[SD], Tensor]):
        super().__init__()
        self.predictor = predictor
        self.metrics = metrics
        self.get_target = get_target

    @torch.no_grad()
    def forward(self, data: List[SDWithMemory], info: Info) -> None:

        targets = []
        pred_collection = []

        for d, o, s in data:
            pred = self.predictor.forward(o, s)
            pred_collection.append(pred.cpu())
            targets.append(self.get_target(d))

        predictions = torch.cat(pred_collection, 1)
        targets = torch.cat(targets, 1)
        assert predictions.shape[1] == targets.shape[1]

        info["metrics"] = {}

        for m in self.metrics:
            val = m(predictions, targets)
            info["metrics"][f"{m.name} tmp"] = val

        out_collection = [o for d, o, s in data]
        _, _, last_state = data[-1]
        predictions = self.predictor.forward(torch.cat(out_collection, 1), last_state).cpu()

        assert predictions.shape[1] == targets.shape[1]
        for m in self.metrics:
            val = m(predictions, targets)
            info["metrics"][f"{m.name} last state"] = val


class EvalLossStateOnly(MemUpLoss):

    def __init__(self, predictor: nn.Module, metrics: List[Metric]):
        super().__init__()
        self.predictor = predictor
        self.metrics = metrics

    @torch.no_grad()
    def forward(self, data: List[SDWithMemory], info: Info) -> None:

        targets = data[-1][0].target
        s0 = data[-1][2]
        predictions = self.predictor(s0).cpu()
        info["metrics"] = {}

        for m in self.metrics:
            val = m(predictions, targets)
            info["metrics"][m.name] = val