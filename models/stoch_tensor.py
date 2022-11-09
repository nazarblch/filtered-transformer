import torch
from torch import nn, Tensor, LongTensor


class StochasticMultinomialTensor:
    def __init__(self, logit: Tensor):
        self.logit = logit
        self.stoch = None

    def sample(self):
        mask = torch.distributions.Bernoulli(torch.ones_like(self.logit[:, :, 0]) * 0.5).sample()[:, :, None]
        logit: Tensor = self.logit
        dist = torch.distributions.Multinomial(logits=logit)
        dist2 = torch.distributions.Multinomial(logits=torch.ones_like(self.logit))
        stoch = dist.sample().detach() + dist2.sample() * mask
        self.stoch = stoch
        return stoch

    def make_diff_sample(self, logit_update: Tensor):
        norm = self.logit.exp().sum(-1).repeat_interleave(self.logit.shape[-1], dim=0)[self.stoch.max(1)[0].view(-1) > 0.5]
        probs = logit_update.exp() / norm
        stoch = self.stoch.transpose(1, 2).reshape(-1, self.stoch.shape[1])[self.stoch.max(1)[0].view(-1) > 0.5]
        stoch = stoch + probs - probs.detach()
        return stoch
