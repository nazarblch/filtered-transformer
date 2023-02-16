from abc import abstractmethod, ABC
import torch
from sklearn.metrics import f1_score
from torch import Tensor
from metrics.base import Metric


class F1Metric(Metric):

    def __init__(self):
        super().__init__("F1")

    @torch.no_grad()
    def __call__(self, logits: Tensor, labels: Tensor) -> float:
        return f1_score(logits.argmax(-1).reshape(-1).cpu().numpy(), labels.reshape(-1).cpu().numpy())


