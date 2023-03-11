import torch
from torch import nn, Tensor
from metrics.base import Metric
from torchmetrics import Metric as torch_metric
from typing import Optional
import torch


class PearsonCorrLoss(nn.Module):
    def forward(self, x, y, dim=1):
        x_centered = x - x.mean(dim=dim, keepdim=True)
        y_centered = y - y.mean(dim=dim, keepdim=True)
        return -nn.functional.cosine_similarity(x_centered, y_centered, dim=dim).mean()


class PearsonCorrMetric(Metric, PearsonCorrLoss):

    def __init__(self):
        super().__init__("PearsonCorr")

    @torch.no_grad()
    def __call__(self, x: Tensor, y: Tensor) -> float:
        return super().forward(x, y).item()


class MeanPearsonCorrCoefPerChannel(torch_metric):
    is_differentiable: Optional[bool] = False
    full_state_update:bool = False
    higher_is_better: Optional[bool] = True
    def __init__(self, n_channels:int, dist_sync_on_step=False):
        """Calculates the mean pearson correlation across channels aggregated over regions"""
        super().__init__(dist_sync_on_step=dist_sync_on_step, full_state_update=False)
        self.reduce_dims=(0, 1)
        self.add_state("product", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("true", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("true_squared", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("pred", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("pred_squared", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("count", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        self.product += torch.sum(preds * target, dim=self.reduce_dims)
        self.true += torch.sum(target, dim=self.reduce_dims)
        self.true_squared += torch.sum(torch.square(target), dim=self.reduce_dims)
        self.pred += torch.sum(preds, dim=self.reduce_dims)
        self.pred_squared += torch.sum(torch.square(preds), dim=self.reduce_dims)
        self.count += torch.sum(torch.ones_like(target), dim=self.reduce_dims)

    @torch.no_grad()
    def compute(self):
        true_mean = self.true / self.count
        pred_mean = self.pred / self.count

        covariance = (self.product
                    - true_mean * self.pred
                    - pred_mean * self.true
                    + self.count * true_mean * pred_mean)

        true_var = self.true_squared - self.count * torch.square(true_mean)
        pred_var = self.pred_squared - self.count * torch.square(pred_mean)
        tp_var = torch.sqrt(true_var) * torch.sqrt(pred_var)
        correlation = covariance / tp_var
        return correlation