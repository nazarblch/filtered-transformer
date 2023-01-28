from collections import namedtuple
from typing import List, Callable, Optional

import torch
from torch import nn, Tensor
from memup.base import MemUpLoss, SDWithMemory, Info, SD, DataCollector, DataCollectorAppend, DataCollectorReplace
from memup.preproc import select_by_index
from metrics.base import Metric

LossModule = namedtuple("LossModule", ["module", "name", "coefficient"])
TOS = namedtuple("TOS", ["target", "out", "state"])
TS = namedtuple("TS", ["target", "state"])
PT = namedtuple("PT", ["prediction", "target"])


class PredictorLoss(MemUpLoss):

    def __init__(self,
                 predictor: nn.Module,
                 loss_modules: List[LossModule]):

        super().__init__()
        self.predictor = predictor
        self.loss_modules = loss_modules

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

    def forward(self, collector: DataCollectorAppend[SD, TOS], info: Info) -> Optional[Tensor]:
        target_seq, out_seq, state_seq = collector.result()
        out, target = torch.cat(out_seq, 1), torch.cat(target_seq, 1)
        s0 = torch.cat(state_seq, 0)
        assert out.shape[1] == target.shape[1]

        info["losses"] = {}
        count = sum([int(o.shape[1] > 0) for o in out_seq])
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
                 loss_modules: List[LossModule]):

        super().__init__()
        self.predictor = predictor
        self.loss_modules = loss_modules

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

    def forward(self, collector: DataCollectorAppend[SD, TS], info: Info) -> Optional[Tensor]:
        target_seq, state_seq = collector.result()
        target = target_seq[-1]
        s0 = torch.cat(state_seq, 0)

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
                 cur_step_loss_coef: float,
                 context_selected_key="context_selected",
                 context_target_key="context_target"):

        super().__init__(predictor, loss_modules)
        self.context_selected_key = context_selected_key
        self.context_target_key = context_target_key
        self.cur_step_loss_coef = cur_step_loss_coef

    def forward(self, collector: DataCollectorAppend[SD, TOS], info: Info) -> Tensor:
        loss1 = super().forward(collector, info)
        _, out_seq, state_seq = collector.result()
        s0 = torch.cat(state_seq, 0)

        assert self.context_selected_key in info and self.context_target_key in info
        context = info[self.context_selected_key].cuda()

        context_target = info[self.context_target_key]
        assert context.shape[1] == context_target.shape[1]
        loss, losses = self.loss(s0, context, context_target)
        if loss1 is not None:
            loss = (loss + loss1 * self.cur_step_loss_coef) / 2
        for name, l in losses.items():
            info["losses"][f"{name} selected"] = l

        info["losses"]["sum loss"] = loss.item()

        return loss


class EvalLoss(MemUpLoss):

    def __init__(self, metrics: List[Metric]):
        super().__init__()
        self.metrics = metrics

    @torch.no_grad()
    def forward(self, collector: DataCollectorAppend[SD, PT], info: Info) -> None:
        pred_seq, target_seq = collector.result()
        targets = torch.cat(target_seq, 1)
        predictions = torch.cat(pred_seq, 1)

        assert predictions.shape[1] == targets.shape[1]

        info["metrics"] = {}

        for m in self.metrics:
            val = m(predictions, targets)
            info["metrics"][m.name] = val


class EvalLossWithMask(EvalLoss):

    @torch.no_grad()
    def forward(self, collector: DataCollectorAppend[SD, PT], info: Info) -> None:
        pred_seq, target_seq = collector.result()
        targets = torch.cat(target_seq, 1)
        predictions = torch.cat(pred_seq, 1)

        assert "mask" in info

        mask = info["mask"]
        predictions, targets = predictions[mask], targets[mask]
        assert predictions.shape[0] == targets.shape[0]

        info["metrics"] = {}

        for m in self.metrics:
            val = m(predictions, targets.cuda())
            info["metrics"][m.name] = val


class EvalLossStateOnly(MemUpLoss):

    def __init__(self, predictor: nn.Module, metrics: List[Metric]):
        super().__init__()
        self.predictor = predictor
        self.metrics = metrics

    @torch.no_grad()
    def forward(self, collector: DataCollectorReplace[SD, TS], info: Info) -> None:
        target_seq, state_seq = collector.result()
        targets = target_seq[-1]
        s0 = state_seq[-1]
        predictions = self.predictor(s0).cpu()
        info["metrics"] = {}

        for m in self.metrics:
            val = m(predictions, targets)
            info["metrics"][m.name] = val