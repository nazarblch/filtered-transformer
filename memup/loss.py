from collections import namedtuple
from typing import List, Callable, Optional

import torch
from torch import nn, Tensor

from data_filters.top_errors import InputTarget, InputTargetMask
from memup.base import MemUpLoss, SDWithMemory, Info, SD, DataCollector, DataCollectorAppend, DataCollectorReplace
from memup.preproc import select_by_index
from metrics.base import Metric

LossModule = namedtuple("LossModule", ["module", "name", "coefficient"])
TOS = namedtuple("TOS", ["target", "out", "state"])
TOSM = namedtuple("TOS", ["target", "out", "state", "mask"])
TS = namedtuple("TS", ["target", "state"])
PT = namedtuple("PT", ["prediction", "target"])


class PredictorLoss(MemUpLoss):

    def __init__(self,
                 predictor: nn.Module,
                 loss_modules: List[LossModule]):

        super().__init__()
        self.predictor = predictor
        self.loss_modules = loss_modules

    def loss(self, state, out, target, mask):
        target = target.cuda()
        mask = mask.cuda()
        N = state.shape[0] // target.shape[0]
        out = torch.cat([out] * N, 0)
        target = torch.cat([target] * N, 0)
        mask = torch.cat([mask] * N, 0)
        pred = self.predictor(out, state, mask)

        losses = {}
        sum_loss = 0

        # B, T = pred.shape[0], pred.shape[1]

        for m in self.loss_modules:
            loss_item = m.module(pred[mask], target[mask])
            sum_loss = sum_loss + loss_item * m.coefficient
            losses[m.name] = loss_item.item()

        return sum_loss, losses

    def forward(self, collector: DataCollectorAppend[SD, TOSM], info: Info, last_state=None) -> Optional[Tensor]:
        if len(collector.collection) == 0:
            return None
        target_seq, out_seq, state_seq, mask_seq = collector.result()
        state_seq = list(state_seq) + [last_state]

        count = sum([int(o is not None and o.shape[1] > 0) for o in out_seq])
        info["losses"] = {}
        if count == 0:
            return None

        out = torch.cat(list(filter(lambda o: o is not None and o.shape[1] > 0, out_seq)), 1) 
        target = torch.cat(list(filter(lambda o: o is not None and o.shape[1] > 0, target_seq)), 1)
        mask = torch.cat(list(filter(lambda o: o is not None and o.shape[1] > 0, mask_seq)), 1)
        s0 = torch.cat(state_seq, 0)
        assert out.shape[1] == target.shape[1]
        assert out.shape[1] == mask.shape[1]

        loss = None

        if count > 1:
            sample_size = out.shape[1] // count
            index = torch.multinomial(
                torch.ones(out.shape[0], out.shape[1], device=out.device) * mask.type(torch.int32) / out.shape[1] + 1e-8,
                sample_size,
                replacement=False)
            out, target, mask = select_by_index(index, out), select_by_index(index, target), select_by_index(index, mask)
            assert out.shape[1] == sample_size
            # assert sample_size == 14
            

        if count > 0:
            loss, losses = self.loss(s0, out, target, mask)
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

    def forward(self, collector: DataCollectorAppend[SD, TOSM], info: Info, selected_data: InputTargetMask = None, last_state=None) -> Tensor:
        loss1 = super().forward(collector, info, last_state)
        _, _, state_seq, _ = collector.result()
        s0 = torch.cat(state_seq, 0)

        assert (self.context_selected_key in info and self.context_target_key in info) or selected_data is not None
        context = info[self.context_selected_key].cuda() if selected_data is None else selected_data.input.cuda()

        context_target = info[self.context_target_key] if selected_data is None else selected_data.target
        assert context.shape[1] == context_target.shape[1]
        # mask = torch.ones(context.shape[:2], device=context.device, dtype=torch.bool)
        mask = selected_data.mask
        loss, losses = self.loss(s0, context, context_target, mask)

        if loss1 is not None and not torch.isnan(loss1):
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
        info["predictions"] = predictions

        for m in self.metrics:
            val = m(predictions, targets)
            info["metrics"][m.name] = val