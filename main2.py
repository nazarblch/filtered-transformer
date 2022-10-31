from typing import List

from ray.rllib.agents import Trainer
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution import ParallelRollouts, synchronous_parallel_sample
from ray.rllib.models import ModelV2
from ray.rllib.utils import override
from ray.tune import register_env
import ray
import ray.rllib.agents.ppo as ppo
import gym
import numpy as np
import random
import torch
from ray.tune import register_env
from ray.util.client import ray
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from torch import nn
from context_env import ContextEnv, ContextEnvWithPredictor, FullyContinuousContextEnv
from ray.rllib.examples.models.rnn_model import TorchRNNModel


def gen_sequence(context_set, size):
    seq = []
    c_num = random.randint(0, len(context_set) - 1)
    c = context_set[c_num]
    c_seq = []
    for _ in range(size):
        if random.randint(0, 100) > 95:
            c_num = random.randint(0, len(context_set) - 1)
            c = context_set[c_num]
        seq.append(np.random.multivariate_normal(c, np.eye(5)).astype(np.float32))
        c_seq.append(c_num)

    return np.stack(seq), c_seq


def custom_eval_function(algorithm: Trainer, eval_workers: WorkerSet):
    """Example of a custom evaluation function.
    Args:
        algorithm: Algorithm class to evaluate.
        eval_workers: Evaluation WorkerSet.
    Returns:
        metrics: Evaluation metrics dict.
    """

    samples = synchronous_parallel_sample(
        worker_set=eval_workers, max_env_steps=200
    )

    episodes = samples.split_by_episode()

    for e in episodes:
        pred = [info["c_change"] for info in e['infos']]
        ref = [info["c_ref"] for info in e['infos']]
        print(pred)
        print(ref)
        print()


class MyModel(TorchRNNModel):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        policy_model_config,
        fc_size=64,
        lstm_state_size=256,
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.fc1 = nn.Sequential(
            nn.Linear(self.obs_size, self.fc_size),
            nn.ReLU(),
            nn.Linear(self.fc_size, self.fc_size)
        )
        self.action_branch = nn.Sequential(
            nn.Linear(self.lstm_state_size, self.lstm_state_size),
            nn.ReLU(),
            nn.Linear(self.lstm_state_size, self.lstm_state_size),
            nn.ReLU(),
            nn.Linear(self.lstm_state_size, num_outputs)
        )

        self.value_branch = nn.Sequential(
            nn.Linear(self.lstm_state_size, self.lstm_state_size),
            nn.ReLU(),
            nn.Linear(self.lstm_state_size, self.lstm_state_size),
            nn.ReLU(),
            nn.Linear(self.lstm_state_size, 1)
        )

    @override(ModelV2)
    def get_initial_state(self):
        h = [
            self.fc1[0].weight.new(1, self.lstm_state_size).zero_().squeeze(0),
            self.fc1[0].weight.new(1, self.lstm_state_size).zero_().squeeze(0),
        ]
        return h


if __name__ == '__main__':

    context_set = []
    for _ in range(10):
        context_set.append(np.random.uniform(0, 10, 5).astype(np.float32))

    sequences = []
    ref_c = []
    for _ in range(1000):
        seq, c_seq = gen_sequence(context_set, 200)
        sequences.append(seq)
        ref_c.append(c_seq)

    ray.init()
    register_env("ctx_e", lambda _: FullyContinuousContextEnv(sequences, ref_c, 1 / 100))

    trainer = ppo.PPOTrainer(
        env="ctx_e",
        config={
            "gamma": 0.95,
            "num_gpus": 1,
            "num_workers": 20,
            "num_envs_per_worker": 1,
            "rollout_fragment_length": 1000,
            "sgd_minibatch_size": 1000,
            "train_batch_size": 20000,
            "evaluation_num_workers": 5,
            "model": {
                "custom_model": MyModel,
                "max_seq_len": 32
            },
            "framework": "torch",
        }

    )

    for i in range(500):
        results = trainer.train()
        print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}")
        if i % 10 == 0:
            print(trainer.evaluate())
            custom_eval_function(trainer, trainer.evaluation_workers)






