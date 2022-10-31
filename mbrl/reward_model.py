import random
from abc import ABC, abstractmethod
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from torch import Tensor, nn
import numpy as np
import torch


class RewardModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, state: Tensor, context: Tensor, context_change: Tensor) -> Tensor: pass

class PEstimator:
    def __init__(self):
        self.X = np.empty((10000, 5), dtype=np.float32)
        self.gm = GaussianMixture(n_components=20, random_state=0,  covariance_type="full")
        # params = {"bandwidth": np.logspace(-1, 1, 20)}
        # self.grid = GridSearchCV(KernelDensity(), params)
        self.pos = 0
        self.is_full = False
        self.model = None

    def add_element(self, x: np.ndarray):
        self.X[self.pos, :] = x
        self.pos += 1
        if self.pos >= 10000:
            self.is_full = True
            self.pos = 0

    def fit(self):

        if self.is_full:
            # self.grid.fit(self.X)
            self.model = self.gm.fit(self.X)
        else:
            # self.grid.fit(self.X[:self.pos])
            self.model = self.gm.fit(self.X[:self.pos])

    def log_prob(self, x: Tensor):
        return self.model.score_samples(x)


class Reward(RewardModel):

    def __init__(self, context_prob: float):
        super().__init__()
        self.context_prob = context_prob
        self.p_estimator = PEstimator()

    def forward(self, state: Tensor, context: Tensor, context_change: Tensor) -> Tensor:
        dist = torch.distributions.multivariate_normal.MultivariateNormal(
            context, torch.eye(context.shape[-1], device=context.device)
        )
        for i in range(context.shape[0]):
            for _ in range(10):
                j = random.randint(0, context.shape[1]-1)
                self.p_estimator.add_element(context[i, j].detach().cpu().numpy())

        log_pxc = -(state - context).pow(2).sum(-1)[:, :, None] / 2
        log_pc = torch.tensor(np.log(self.context_prob), device=context.device)
        int_change = context_change.squeeze().type(torch.int32)
        selected_context = context[int_change == 1]
        if self.p_estimator.model is not None:
            log_pc_selected = torch.from_numpy(self.p_estimator.log_prob(selected_context.detach().cpu().numpy())).cuda()
            log_pc = torch.zeros(context_change.shape, dtype=torch.float32, device=context.device)
            log_pc[int_change == 1] = log_pc_selected[:, None].type(torch.float32)
            # print("p", log_pc_selected.exp())

        reward = log_pxc + context_change * log_pc.detach() * 2

        return reward

