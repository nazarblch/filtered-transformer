import random
from typing import List, Any, Dict
import numpy as np
import gym
import torch
from gym import spaces
from torch import nn, optim
from torch.distributions import Distribution, MultivariateNormal
from sklearn.mixture import GaussianMixture


class PEstimator:
    def __init__(self):
        self.X = np.empty((10000, 5), dtype=np.float32)
        self.gm = GaussianMixture(n_components=50, random_state=0)
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
            self.model = self.gm.fit(self.X)
        else:
            self.model = self.gm.fit(self.X[:self.pos])

    def log_prob(self, x):
        return self.model.score_samples(x[np.newaxis, ])[0]


class ContextEnv(gym.Env):

    def __init__(self,
                 sequences: List[np.ndarray],
                 ref_context: List[List[int]],
                 context_count: int):

        super(ContextEnv, self).__init__()
        self.action_space = spaces.Discrete(context_count)
        self.observation_space = spaces.Dict({
            "context": spaces.Box(0, 10, (5,), dtype=np.float32),
            "data": spaces.Box(-100, 100, (5,), dtype=np.float32),
        })
        self.sequences = sequences
        self.cur_sequence = None
        self.cur_ref_context = None
        self.cur_context_number = -1
        self.step_count = 0
        self.cur_obs = None
        self.ref_context = ref_context
        self.context_prob = np.ones(context_count, dtype=np.float32) / context_count

        self.context_embed = nn.Embedding(context_count, 5).cuda()
        self.opt = optim.Adam(self.context_embed.parameters(), lr=0.0005)
        self.train_data = np.empty((10000, 5), dtype=np.float32)
        self.train_labels = np.empty(10000, dtype=np.int)
        self.train_pos = 0
        self.is_full = False
        self.context_hist = np.ones(context_count, dtype=np.float32) * 100

    def step(self, action: np.ndarray):
        self.step_count += 1

        next_context_number = action
        log_pc = np.log(self.context_prob[next_context_number])

        with torch.no_grad():
            next_context = self.context_embed(torch.tensor(next_context_number).cuda()).cpu()
            dist = MultivariateNormal(next_context, torch.eye(5))
            log_pxc = dist.log_prob(torch.from_numpy(self.cur_obs))

        change_context = (next_context_number != self.cur_context_number) or (self.step_count == 1)
        reward = log_pxc + int(change_context) * log_pc
        reward = reward.item()
        done = self.step_count == (len(self.cur_sequence) - 1)
        info = {"c_pred": next_context_number, "c_ref": self.cur_ref_context[self.step_count - 1]}

        self.train_data[self.train_pos] = self.cur_obs
        self.train_labels[self.train_pos] = next_context_number
        self.train_pos += 1
        self.context_hist[next_context_number] += 1
        if self.train_pos == self.train_data.shape[0]:
            self.train_pos = 0
            self.is_full = True

        self.cur_context_number = next_context_number
        self.cur_obs = self.cur_sequence[self.step_count]
        observation_dict = {
            "context": next_context.numpy(),
            "data": self.cur_obs
        }

        return observation_dict, reward, done, info

    def reset(self):
        num = random.randint(0, len(self.sequences) - 1)
        self.cur_sequence = self.sequences[num]
        self.cur_ref_context = self.ref_context[num]
        self.cur_context_number = 0
        self.step_count = 0
        observation = self.cur_sequence[self.step_count]
        self.cur_obs = observation

        with torch.no_grad():
            observation_dict = {
                "context": self.context_embed(torch.tensor(0).cuda()).cpu().numpy() * 0,
                "data": observation
            }

            return observation_dict

    def train(self):
        n = self.train_data.shape[0] if self.is_full else self.train_pos
        indices = np.random.randint(0, n, 100)
        x = self.context_embed(torch.from_numpy(self.train_labels[indices]).cuda())
        y = torch.from_numpy(self.train_data[indices]).cuda()
        self.opt.zero_grad()
        nn.MSELoss()(x, y).backward()
        self.opt.step()

        new_context_prob = self.context_hist / np.sum(self.context_hist)
        self.context_prob = new_context_prob



class ContextEnvWithPredictor(gym.Env):

    def __init__(self,
                 sequences: List[np.ndarray],
                 ref_context: List[List[int]],
                 predictor: nn.Module,
                 context_prob: float):

        super(ContextEnvWithPredictor, self).__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict({
            "context": spaces.Box(-100, 100, (5,), dtype=np.float32),
            "data": spaces.Box(-100, 100, (5,), dtype=np.float32),
            "data_f": spaces.Box(-100, 100, (5,), dtype=np.float32),
            "data_p": spaces.Box(-100, 100, (5,), dtype=np.float32),
        })
        self.sequences = sequences
        self.cur_sequence = None
        self.cur_ref_context = None
        self.cur_context = None
        self.step_count = 0
        self.cur_obs = None
        self.ref_context = ref_context
        self.context_prob = context_prob
        self.predictor = predictor
        self.p_estimator = PEstimator()

    def step(self, action: np.ndarray):
        self.step_count += 1

        change_context = action == 1 or (self.step_count == 1)

        if change_context:
            with torch.no_grad():
                obs_torch = torch.cat([
                    torch.from_numpy(self.cur_obs["data"]),
                    torch.from_numpy(self.cur_obs["data_f"]),
                    torch.from_numpy(self.cur_obs["data_p"]),
                ], dim=-1)

                self.cur_context = self.cpu_predictor(obs_torch[None, ]).view(-1).numpy()

        self.p_estimator.add_element(self.cur_context)

        # print(self.step_count, self.cur_context)
        dist = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.from_numpy(self.cur_context), torch.eye(5))

        log_pxc = dist.log_prob(torch.from_numpy(self.cur_obs["data"]))
        log_pc = np.log(self.context_prob)
        if self.p_estimator.model is not None and change_context:
            log_pc = self.p_estimator.log_prob(self.cur_context)

        reward = log_pxc + int(change_context) * log_pc
        reward = reward.item()
        done = self.step_count == (len(self.cur_sequence) - 1)
        info = {"c_change": int(change_context), "c_ref": self.cur_ref_context[self.step_count - 1]}

        self.cur_obs = {
            "context": self.cur_context,
            "data": self.cur_sequence[self.step_count],
            "data_f": self.cur_sequence[self.step_count + 1] if not done
            else np.zeros_like(self.cur_sequence[self.step_count]),
            "data_p": self.cur_sequence[self.step_count - 1]
        }

        return self.cur_obs, reward, done, info

    def reset(self):
        self.cpu_predictor = self.predictor
        num = random.randint(0, len(self.sequences) - 1)
        self.cur_sequence = self.sequences[num]
        self.cur_ref_context = self.ref_context[num]
        self.step_count = 0
        self.cur_context = None
        self.cur_obs = {
            "context": np.zeros(5, dtype=np.float32),
            "data": self.cur_sequence[self.step_count],
            "data_f": self.cur_sequence[self.step_count + 1],
            "data_p": np.zeros_like(self.cur_sequence[self.step_count]),

        }

        return self.cur_obs


class FullyContinuousContextEnv(gym.Env):

    def __init__(self,
                 sequences: List[np.ndarray],
                 ref_context: List[List[int]],
                 context_prob: float):

        super(FullyContinuousContextEnv, self).__init__()
        self.action_space = spaces.Box(-100, 100, (6,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "context": spaces.Box(0, 10, (5,), dtype=np.float32),
            "data": spaces.Box(-100, 100, (5,), dtype=np.float32),
        })
        self.sequences = sequences
        self.cur_sequence = None
        self.cur_ref_context = None
        self.cur_context = None
        self.step_count = 0
        self.cur_obs = None
        self.ref_context = ref_context
        self.context_prob = context_prob
        self.p_estimator = PEstimator()

    def step(self, action: np.ndarray):
        self.step_count += 1

        change_prob = 1.0 / (1.0 + np.exp(-action[0]))

        change_context = np.random.uniform(0, 1) <= change_prob or (self.step_count == 1)
        if change_context:
            self.cur_context = action[1:]

        self.p_estimator.add_element(self.cur_context)

        # print(self.step_count, self.cur_context)
        dist = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.from_numpy(self.cur_context), torch.eye(5))

        log_pxc = dist.log_prob(torch.from_numpy(self.cur_obs["data"]))
        log_pc = np.log(self.context_prob)
        if self.p_estimator.model is not None and change_context:
            log_pc = self.p_estimator.log_prob(self.cur_context)

        reward = log_pxc + int(change_context) * log_pc
        reward = reward.item()
        done = self.step_count == (len(self.cur_sequence) - 1)
        info = {"c_change": change_context, "c_ref": self.cur_ref_context[self.step_count - 1]}

        self.cur_obs = {
            "context": self.cur_context,
            "data": self.cur_sequence[self.step_count]
        }

        return self.cur_obs, reward, done, info

    def reset(self):
        num = random.randint(0, len(self.sequences) - 1)
        self.cur_sequence = self.sequences[num]
        self.cur_ref_context = self.ref_context[num]
        self.step_count = 0
        self.cur_context = None
        self.cur_obs = {
            "context": np.zeros(5, dtype=np.float32),
            "data": self.cur_sequence[self.step_count]
        }

        return self.cur_obs