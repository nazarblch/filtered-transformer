import random
from abc import ABC, abstractmethod
from itertools import chain
from typing import List
import numpy as np
from torch import Tensor, nn
import torch

from hfilter.networks import StochasticMultinomialTensor
from mbrl.reward_model import Reward, RewardModel
from models import FloatTransformer


def gen_sequence(context_set, size):
    seq = []
    c_num = random.randint(0, len(context_set) - 1)
    c = context_set[c_num]
    c_seq = []
    for _ in range(size):
        if random.randint(0, 100) > 90:
            c_num = random.randint(0, len(context_set) - 1)
            c = context_set[c_num]
        seq.append(np.random.multivariate_normal(c, np.eye(5)).astype(np.float32))
        c_seq.append(c_num)

    return np.stack(seq), c_seq


def disable_gradients(network):
    # Disable calculations of gradients.
    for param in network.parameters():
        param.requires_grad = False


def enable_gradients(network):
    # Disable calculations of gradients.
    for param in network.parameters():
        param.requires_grad = True


class StochasticBinaryTensor:
    def __init__(self, logit: Tensor):
        self.logit = logit

    def sample(self):
        logit = self.logit
        dist = torch.distributions.Bernoulli(logits=logit)
        stoch = dist.sample().detach()
        log_prob = dist.log_prob(stoch)
        stoch = stoch + dist.probs - dist.probs.detach()
        return stoch, log_prob, logit



class ValueModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, state: Tensor, prev_context: Tensor) -> Tensor: pass


class ContextPredictor(nn.Module, ABC):
    @abstractmethod
    def forward(self, state: Tensor, prev_context: Tensor) -> Tensor: pass


class ContextChangeModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, state: Tensor, prev_context: Tensor) -> StochasticBinaryTensor: pass


def compute_return(
                reward: torch.Tensor,
                value: torch.Tensor,
                discount: torch.Tensor,
                bootstrap: torch.Tensor,
                lambda_: float
            ):
    """
    Compute the discounted reward for a batch of data.
    reward, value, and discount are all shape [horizon - 1, batch, 1] (last element is cut off)
    Bootstrap is [batch, 1]
    """
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    target = reward + discount * next_values * (1 - lambda_)
    timesteps = list(range(reward.shape[0] - 1, -1, -1))
    outputs = []
    accumulated_reward = bootstrap
    for t in timesteps:
        inp = target[t]
        discount_factor = discount[t]
        accumulated_reward = inp + discount_factor * lambda_ * accumulated_reward
        outputs.append(accumulated_reward)
    returns = torch.flip(torch.stack(outputs), [0])
    return returns


def compute_val_target(rewards: Tensor, g: float):
    T = rewards.shape[0]
    timesteps = list(range(T - 1, -1, -1))
    accumulated_reward = 0
    outputs = []

    for t in timesteps:
        accumulated_reward = rewards[t] + g * accumulated_reward
        outputs.append(accumulated_reward)

    return torch.flip(torch.stack(outputs), [0])




class Trainer(nn.Module):
    def __init__(self, reward_model: RewardModel, value_model: ValueModel, predictor: ContextPredictor,
                 change_model: ContextChangeModel):
        super().__init__()
        self.reward_model = reward_model
        self.value_model = value_model
        self.predictor = predictor
        self.change_model = change_model
        self.gamma = 0.95
        self.lambda_ = 0.9

    def context_ids_to_bin(self, context: Tensor):
        ct = torch.ones_like(context[:, 0])
        new_c_list = [ct]
        for t in range(1, context.shape[1]):
            ct = torch.zeros_like(context[:, 0])
            ct[context[:, t] != context[:, t-1]] = 1
            new_c_list.append(ct)
        contexts = torch.stack(new_c_list, 1)
        return contexts[:, :, None]

    def predict(self, states: Tensor, true_context: Tensor = None):
        B, L = states.shape[0], states.shape[1]
        if true_context is not None:
            change_context = self.context_ids_to_bin(true_context).type(torch.float32)
            log_p = torch.zeros_like(change_context)
            change_prob = change_context
        else:
            change_context = self.change_model(states).sample()
        change_context[:, 0, 0] = 1.0
        # h = self.predictor(states, change_context)

        ct = 0
        new_c_list = []
        for t in range(L):
            ct = self.predictor(states[:, t]) * change_context[:, t, 0][:, None] + change_context[:, t, 1][:, None] * ct
            new_c_list.append(ct)

        contexts = torch.stack(new_c_list, 1)
        # contexts = h
        rewards = self.reward_model(states, contexts, change_context[:, :, 0][:, :, None])

        return change_context, contexts, rewards

    def actor_loss(self, states: Tensor, true_context: Tensor = None):
        actor_entropy_scale = 0.0
        change_context, contexts, rewards = self.predict(states, true_context)
        reward = rewards.transpose(0, 1) / 100
        values = self.value_model(states, contexts).transpose(0, 1)
        # log_p = log_p.transpose(0, 1)
        # gamma = torch.ones_like(reward) * self.gamma

        # lambda_returns = compute_return(reward[:-1], values[:-1], gamma[:-1], bootstrap=values[-1], lambda_=self.lambda_)

        # discount_arr = torch.cat([torch.ones_like(gamma[:1]), gamma[1:]])
        # discount = torch.cumprod(discount_arr[:-1], 0)
        actor_loss = -torch.sum(torch.mean(reward + values * 0.1, dim=1))
        return actor_loss, reward, contexts, change_context

    def value_loss(self, states, contexts, reward):
        values = self.value_model(states, contexts.detach()).transpose(0, 1)
        with torch.no_grad():
            value_target = compute_val_target(reward, self.gamma).detach()

        value_loss = nn.MSELoss()(values, value_target)
        return value_loss


if __name__ == "__main__":

    class Value(ValueModel):

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(10, 100),
                nn.ReLU(),
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Linear(100, 1)
            )

        def forward(self, state: Tensor, context: Tensor) -> Tensor:
            return self.net(torch.cat([state, context], -1))

    class Pred(ContextPredictor):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(5, 100),
                nn.ReLU(),
                nn.Linear(100, 5)
            )

        def forward(self, state: Tensor) -> Tensor:
            return self.net(state)

    class Change(ContextChangeModel):

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                FloatTransformer(5, 100),
                nn.ReLU(),
                nn.Linear(100, 2)
            )

        def forward(self, state: Tensor) -> StochasticMultinomialTensor:
            return StochasticMultinomialTensor(self.net(state))


    context_set = []
    for _ in range(10):
        context_set.append(np.random.uniform(0, 100, 5).astype(np.float32))

    trainer = Trainer(Reward(0.02), Value(), Pred(), Change()).cuda()
    opt = torch.optim.Adam([
        {'params': trainer.predictor.parameters(), 'lr': 1e-4},
        {'params': trainer.change_model.parameters(), 'lr': 1e-5},
     ])
    val_opt = torch.optim.Adam(trainer.value_model.parameters(), lr=1e-4)

    sequences = []
    ref_c = []
    N = 500
    for _ in range(N):
        seq, c_seq = gen_sequence(context_set, 50)
        sequences.append(seq)
        ref_c.append(c_seq)

    dataset = np.stack(sequences)
    targets = np.stack(ref_c)
    B = 100

    BB = 10000
    data_buffer = np.empty((BB, 50, 5), dtype=dataset.dtype)
    context_buffer = np.empty((BB, 50, 5), dtype=dataset.dtype)
    reward_buffer = np.empty((BB, 50, 1), dtype=dataset.dtype)
    buffer_pos = 0
    buffer_is_full = False

    for i in range(100000):

        b1 = (i * B) % N
        b2 = (i * B + B) % N
        if b1 >= b2 or b1 > N - B:
            continue

        opt.zero_grad()
        batch = torch.from_numpy(dataset[b1:b2]).cuda()
        c_batch = torch.from_numpy(targets[b1:b2]).cuda()
        # change_context, pred_context, r, log_p = trainer.predict(batch)
        # loss = -r.sum() + 1 * log_p.sum()
        loss, reward, contexts, change_context = trainer.actor_loss(batch)
        data_buffer[buffer_pos: buffer_pos + B] = dataset[b1:b2]
        context_buffer[buffer_pos: buffer_pos + B] = contexts.detach().cpu().numpy()
        reward_buffer[buffer_pos: buffer_pos + B] = reward.transpose(0, 1).detach().cpu().numpy()
        buffer_pos += B
        if buffer_pos >= BB:
            buffer_pos = 0
            buffer_is_full = True

        loss = loss / 10
        print(i, loss.item())
        if i % 10 == 0 and i > 1000:
            trainer.reward_model.p_estimator.fit()
        if i % 21 == 0 and i > 1000:
            print(b1)
            print(change_context[0, :, 0].flatten().type(torch.int32).cpu().numpy().tolist())
            print(ref_c[b1])
        if i > 1000:
            loss.backward()
            opt.step()

        if buffer_is_full:
            val_opt.zero_grad()
            b1 = random.randint(0, BB - B - 1)
            b2 = b1 + B

            data = torch.from_numpy(data_buffer[b1:b2]).cuda()
            context = torch.from_numpy(context_buffer[b1:b2]).cuda()
            reward = torch.from_numpy(reward_buffer[b1:b2]).cuda().transpose(0, 1)

            val_loss = trainer.value_loss(data, context, reward)
            print("v loss", val_loss.item())
            val_loss.backward()
            val_opt.step()









