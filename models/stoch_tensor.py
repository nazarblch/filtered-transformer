import torch
from torch import nn, Tensor, LongTensor


class StochasticMultinomialTensor:
    def __init__(self, logit: Tensor):
        self.logit = logit / 300
        self.stoch = None

    def sample(self):
        mask = torch.distributions.Bernoulli(torch.ones_like(self.logit[:, :, 0]) * 0.1).sample()[:, :, None]
        logit: Tensor = self.logit
        # print(logit.min().item(), logit.max().item(), self.logit.exp().sum(-1).min().item())
        dist = torch.distributions.Multinomial(logits=logit)
        dist2 = torch.distributions.Multinomial(logits=torch.ones_like(self.logit))
        stoch = dist.sample().detach() * (1 - mask) + dist2.sample() * mask
        self.stoch = stoch
        return stoch

    def make_diff_sample(self, logit_update: Tensor):
        norm = self.logit.clip(-30, 30).exp().sum(-1).repeat_interleave(self.logit.shape[-1], dim=0)[self.stoch.max(1)[0].view(-1) > 0.5]
        probs = (logit_update / 300).clip(-30, 30).exp() / (norm + 1e-8)
        stoch = self.stoch.transpose(1, 2).reshape(-1, self.stoch.shape[1])[self.stoch.max(1)[0].view(-1) > 0.5]
        stoch = stoch + probs - probs.detach()
        return stoch


class StochasticBinaryTensor:
    def __init__(self, logit: Tensor):
        self.logit = logit
        assert logit.shape[-1] == 2

    def sample(self):
        # mask = torch.distributions.Bernoulli(torch.ones_like(self.logit[:, 0]) * 0.1).sample()[:, :, None]
        logit: Tensor = self.logit
        dist = torch.distributions.Multinomial(logits=logit)
        # dist2 = torch.distributions.Multinomial(logits=torch.ones_like(self.logit))
        stoch = dist.sample().detach()
        # self.stoch = stoch
        probs = logit.softmax(-1)
        return stoch + probs - probs.detach()

