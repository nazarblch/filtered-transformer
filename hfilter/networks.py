import math
import random
from typing import Callable, List
import torch
from torch import nn, Tensor, LongTensor


class StochasticMultinomialTensor:
    def __init__(self, logit: Tensor):
        self.logit = logit
        self.stoch = None
        self.rnd = torch.distributions.Bernoulli(probs=0.8)

    def sample(self):
        logit: Tensor = self.logit #if self.rnd.sample().item() > 0.5 else torch.ones_like(self.logit)
        dist = torch.distributions.Multinomial(logits=logit)
        stoch = dist.sample().detach()
        # probs = logit.softmax(-1)
        # stoch = stoch + probs - probs.detach()
        self.stoch = stoch
        return stoch

    def add_probs(self, logit_update: Tensor):
        probs = logit_update.exp() / self.logit.exp().sum(-1).repeat_interleave(self.logit.shape[-1], dim=0)[self.stoch.max(1)[0].view(-1) > 0.5]
        stoch = self.stoch.transpose(1, 2).reshape(-1, self.stoch.shape[1])[self.stoch.max(1)[0].view(-1) > 0.5]
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