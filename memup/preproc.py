from typing import Generic, Dict, Any, Callable
from memup.base import SeqDataFilter, SD, MemUpMemory, State, InfoUpdate, Info
import torch
from torch import nn, Tensor


class ContextPreprocessor(InfoUpdate[SD]):
    def __init__(self, mem_transformer: MemUpMemory[SD], seq_filter: SeqDataFilter[SD], key="context", last_state_key="last_state"):
        self.mem_transformer = mem_transformer
        self.seq_filter = seq_filter
        self.key = key
        self.last_state_key = last_state_key
        self.increment_step = IncrementStep()

    @torch.no_grad()
    def forward(self, data: SD, state: State, info: Info, *args) -> Info:
        state = state * 0
        done = False
        out_collection = []
        info2 = {}
        print("process context")

        while not done:
            info2 = self.increment_step.forward(data, state, info2)
            filtered_data, done = self.seq_filter.forward(data, state, info2)
            o, state = self.mem_transformer.forward(filtered_data, state)
            out_collection.append(o.cpu())

        info[self.key] = out_collection
        info[self.last_state_key] = state.cpu()

        return info


class NStepUpdate(InfoUpdate[SD]):

    def forward(self, data: SD, state: State, info: Info, *args) -> Info:
        if (info[self.step_key] + self.offset) % self.n == 0:
            info = self.update.forward(data, state, info)
        return info

    def __init__(self, update: InfoUpdate[SD], n: int, step_key="step", offset=0):
        self.n = n
        self.offset = offset
        self.update = update
        self.step_key = step_key


class IncrementStep(InfoUpdate[SD]):

    def forward(self, data: SD, state: State, info: Info, *args) -> Info:
        if self.step_key not in info:
            info[self.step_key] = 0
        else:
            info[self.step_key] += 1

        return info

    def __init__(self, step_key="step"):
        self.step_key = step_key


class ErrorPreprocessor(InfoUpdate[SD]):
    def __init__(self, predictor: nn.Module, metric: nn.Module, get_target: Callable[[SD], Tensor],
                 key="context", last_state_key="last_state", errors_key="errors"):
        self.predictor = predictor
        self.metric = metric
        self.get_target = get_target
        self.key = key
        self.last_state_key = last_state_key
        self.errors_key = errors_key

    @torch.no_grad()
    def forward(self, data: SD, state: State, info: Info, *args) -> Info:
        print("compute errors")
        state = info[self.last_state_key].cuda()
        out_collection = []
        for out in info[self.key]:
            if out.shape[1] > 0:
                out_collection.append(out.cuda())

        predictions = self.predictor.forward(torch.cat(out_collection, 1), state).cpu()
        target = self.get_target(data)
        assert predictions.shape[1] == target.shape[1]
        errors = self.metric(predictions, target).mean(dim=2)
        info[self.errors_key] = errors

        return info


class TargetsSampler(InfoUpdate[SD]):
    def __init__(self, count: int, get_target: Callable[[SD], Tensor],
                 context_key="context", errors_key="errors",
                 selected_context_key="context_selected", selected_target_key="context_target"):
        self.get_target = get_target
        self.errors_key = errors_key
        self.context_key = context_key
        self.count = count
        self.selected_context_key = selected_context_key
        self.selected_target_key = selected_target_key

    def sample_data(self, context: Tensor, errors: Tensor, target: Tensor):
        count = self.count
        assert torch.all(errors >= 0)
        probs = errors / errors.sum(dim=1, keepdim=True)
        sample = torch.multinomial(probs, count, replacement=False)
        index = sample[:, :, None]
        B = context.shape[0]

        selected_errors = torch.gather(errors, 1, index[:, :, 0]).reshape(B, count)
        print("errors:", errors.mean(), "selected_errors:", selected_errors.mean().item())

        return torch.gather(context, 1, index.expand(-1, -1, context.shape[-1])).reshape(B, count, context.shape[-1]), \
               torch.gather(target, 1, index.expand(-1, -1, target.shape[-1])).reshape(B, count, target.shape[-1])

    @torch.no_grad()
    def forward(self, data: SD, state: State, info: Info, *args) -> Info:

        info[self.selected_context_key], info[self.selected_target_key] = self.sample_data(
            torch.cat(info[self.context_key], dim=1),
            info[self.errors_key],
            self.get_target(data)
        )

        return info






