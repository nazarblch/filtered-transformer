import torch
from torch import nn, Tensor
from metrics.base import Metric


class PearsonCorrLoss(nn.Module):
    def forward(self, x, y, dim=1):
        x_centered = x - x.mean(dim=dim, keepdim=True)
        y_centered = y - y.mean(dim=dim, keepdim=True)
        return nn.functional.cosine_similarity(x_centered, y_centered, dim=dim).mean()


class PearsonCorrMetric(Metric, PearsonCorrLoss):

    def __init__(self):
        super().__init__("PearsonCorr")

    @torch.no_grad()
    def __call__(self, x: Tensor, y: Tensor) -> float:
        return super().forward(x, y).item()