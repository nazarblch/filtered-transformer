import math
from typing import Callable, List
import torch
from torch import nn, Tensor, LongTensor


class StochasticMultinomialTensor:
    def __init__(self, logit: Tensor):
        self.logit = logit

    def sample(self):
        logit: Tensor = self.logit
        dist = torch.distributions.Multinomial(logits=logit)
        stoch = dist.sample()
        probs = logit.softmax(-1)
        stoch = stoch + probs - probs.detach()
        return stoch


class StochasticMultinomialTensorFiltered:
    def __init__(self, logit: Tensor, mask: Tensor):
        self.probs = logit.softmax(-1) * mask
        self.probs = self.probs / self.probs.sum(-1, keepdim=True)

    def sample(self):
        dist = torch.distributions.Multinomial(probs=self.probs)
        stoch = dist.sample()
        probs = self.probs
        stoch = stoch + probs - probs.detach()
        return stoch