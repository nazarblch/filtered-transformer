import torch
from torch import Tensor
from metrics.base import Metric


class MSEMetric(Metric):

    def __init__(self):
        super().__init__("MSE")

    @torch.no_grad()
    def __call__(self, pred: Tensor, target: Tensor) -> float:
        return torch.nn.MSELoss()(pred.reshape(-1), target.reshape(-1)).item()