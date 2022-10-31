from abc import abstractmethod, ABC
import torch
from sklearn.metrics import accuracy_score
from torch import Tensor


class Metric(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def __call__(self, *args, **kwargs) -> float:
        pass


class AccuracyMetric(Metric):

    def __init__(self):
        super().__init__("Accuracy")

    @torch.no_grad()
    def __call__(self, logits: Tensor, labels: Tensor) -> float:
        return accuracy_score(logits.argmax(-1).reshape(-1).cpu().numpy(), labels.reshape(-1).cpu().numpy())


