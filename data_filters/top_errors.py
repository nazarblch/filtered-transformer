import random
from abc import ABC
from collections import namedtuple
from typing import Callable, List, Tuple
import torch
from torch import nn, Tensor
from data_filters.sliding_window import SlidingWindowFilterTuple
from memup.base import SeqDataFilter, SD, State, Info, Done
from memup.preproc import IncrementStep, select_by_index

InputTarget = namedtuple("InputTarget", ["input", "target", "length"])
InputTargetMask = namedtuple("InputTarget", ["input", "target", "mask", "length"])


class TopErrorsFilter(SeqDataFilter[InputTarget]):

    def __init__(self, rollout: int, count: Tuple[int, int], predictor: nn.Module, metric: nn.Module, is_random=False):
        super().__init__()
        self.predictor = predictor
        self.metric = metric
        self.rollout = rollout
        self.window_filter = SlidingWindowFilterTuple[InputTarget](rollout, padding=0, skip_fields={"target", "length"})
        self.count = count
        self.is_random = is_random

    def sample_data(self, context: Tensor, errors: Tensor, target: Tensor, info: Info):
        count = random.randint(*self.count)
        assert torch.all(errors >= 0)
        if self.is_random:
            probs = errors / errors.sum(dim=1, keepdim=True)
            index = torch.multinomial(probs, count, replacement=False)
        else:
            index = torch.topk(errors, count, dim=1).indices

        info["selected_index"] = index
        mask = torch.ones(index.shape[:2], device=index.device, dtype=torch.bool)

        return InputTargetMask(select_by_index(index, context), select_by_index(index, target), mask, self.count)

    @torch.no_grad()
    def forward(self, data: InputTarget, state: State, info: Info, *args) -> Tuple[InputTarget, Done]:
        done = False
        info2 = {}
        increment_step = IncrementStep()
        predictions = []

        while not done:
            info2 = increment_step.forward(data, state, info2)
            filtered_data, done = self.window_filter.forward(data, state, info2)
            mask = torch.ones(filtered_data.input.shape[:2], dtype=torch.bool).cuda()
            pred = self.predictor(filtered_data.input.cuda(), state, mask).cpu()
            predictions.append(pred)

        predictions = torch.cat(predictions, 1)
        target = data.target
        assert predictions.shape[1] == target.shape[1]

        B, T = predictions.shape[0], predictions.shape[1]
        errors = self.metric(
            predictions.view(B * T, *predictions.shape[2:]),
            target.view(B * T, *target.shape[2:])
        )

        if len(errors.shape) > 1:
            errors = errors.mean(dim=list(range(1, len(errors.shape))))

        errors = errors.view(B, T)

        if not torch.all(errors >= 0):
           errors = errors - errors.min()

        return self.sample_data(data.input, errors, target, info), False
    

class TopErrorsFilterWithMask:

    def __init__(self, count: Tuple[int, int], predictor: nn.Module, metric: nn.Module, is_random=False):
        super().__init__()
        self.predictor = predictor
        self.metric = metric
        self.count = count
        self.is_random = is_random

    def sample_data(self, context: Tensor, errors: Tensor, target: Tensor, mask: Tensor, info: Info):
        count = random.randint(*self.count)
        assert torch.all(errors >= 0)
        if self.is_random:
            errors[errors.sum(dim=1) < 1e-8] += 1e-8
            probs = errors / errors.sum(dim=1, keepdim=True)
            index = torch.multinomial(probs, count, replacement=False)
        else:
            index = torch.topk(errors, count, dim=1).indices

        info["selected_index"] = index

        return InputTargetMask(select_by_index(index, context).cuda(), 
                               select_by_index(index, target).cuda(), 
                               select_by_index(index, mask).cuda(), 
                               self.count)

    @torch.no_grad()
    def forward(self, data: List[InputTargetMask], state: State, info: Info, *args) -> InputTargetMask:
        predictions = []
        state = state.cuda()

        for dk in data:
            pred = self.predictor(dk.input.cuda(), state, dk.mask.cuda()).cpu()
            predictions.append(pred)

        predictions = torch.cat(predictions, 1)
        target = torch.cat([dk.target for dk in data], 1)
        assert predictions.shape[1] == target.shape[1]

        B, T = predictions.shape[0], predictions.shape[1]
        errors = self.metric(
            predictions.view(B * T, *predictions.shape[2:]),
            target.view(B * T, *target.shape[2:])
        )

        if len(errors.shape) > 1:
            errors = errors.mean(dim=list(range(1, len(errors.shape))))

        errors = errors.view(B, T)

        if not torch.all(errors >= 0):
           errors = errors - errors.min()

        mask = torch.cat([dk.mask for dk in data], 1)
        input = torch.cat([dk.input for dk in data], 1)
        errors = errors * mask.type(torch.int32)

        return self.sample_data(input, errors, target, mask, info)

