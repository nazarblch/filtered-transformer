from typing import List

import gym
import numpy as np
import random

import ray
import torch
from ray.rllib import RolloutWorker
from ray.rllib.agents.dqn import dqn
from ray.rllib.agents.ppo import ppo
from ray.rllib.agents.sac import RNNSACTrainer
from ray.rllib.agents.sac.rnnsac_torch_model import RNNSACTorchModel
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution import synchronous_parallel_sample
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.tune import register_env
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from torch import nn
from context_env import ContextEnv, ContextEnvWithPredictor
from main2 import MyModel, custom_eval_function


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


def make_train_batch(eval_workers: WorkerSet, observation_space):

    samples = synchronous_parallel_sample(
        worker_set=eval_workers, max_env_steps=400
    )

    samples["obs_orig"] = restore_original_dimensions(
        samples["obs"], observation_space, "numpy"
    )

    episodes = samples.split_by_episode()
    keys = samples["obs_orig"].keys()

    for e in episodes:
        actions = e["actions"].reshape(-1)
        data = {}
        for k in keys:
            data[k] = e["obs_orig"][k]

        num = -1
        context_numbers = []
        new_data = []

        for i in range(len(actions)):
            if actions[i] == 1 or i == 0:
                num += 1
            context_numbers.append(num)
            new_data.append({k: data[k][i].reshape(-1) for k in keys})

        context_numbers = np.asarray(context_numbers)
        data = np.asarray(new_data)
        targets = []
        for t in range(len(data)):
            t_num = context_numbers[t]
            pos = np.arange(len(data))[context_numbers == t_num]
            gamma = 0.95
            dist = np.abs(pos - t)
            p = (np.ones_like(dist) * gamma) ** (dist)
            p = p / np.sum(p)
            target = np.random.choice(data[context_numbers == t_num], p=p)
            targets.append(target)

        yield data.tolist(), targets


def predictor_loss(data, targets, predictor):
    x = np.stack([np.concatenate([d["data"], d["data_f"], d["data_p"]]) for d in data])
    y = np.stack([d["data"] for d in targets])
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    y_pred = predictor(x)
    return nn.MSELoss()(y_pred, y)


class SimplePredictor(nn.Module):
    def __init__(self, context_set: List[np.ndarray]):
        super().__init__()
        self.context_set = [torch.from_numpy(c).cuda() for c in context_set]

    def forward(self, x):
        d = 100000
        c_res = None
        for c in self.context_set:
            di = (x.view(-1) - c).pow(2).sum().sqrt()
            if di < d:
                d = di
                c_res = c

        return c_res


if __name__ == '__main__':

    context_set = []
    for _ in range(10):
        context_set.append(np.random.uniform(0, 10, 5).astype(np.float32))

    sequences = []
    ref_c = []
    for _ in range(200):
        seq, c_seq = gen_sequence(context_set, 200)
        sequences.append(seq)
        ref_c.append(c_seq)

    predictor = nn.Sequential(
        nn.Linear(15, 64),
        nn.ReLU(inplace=True),
        nn.Linear(64, 64),
        nn.ReLU(inplace=True),
        nn.Linear(64, 5)
    )
    opt = torch.optim.Adam(predictor.parameters(), lr=0.0005)
    # predictor = SimplePredictor(context_set)

    # env = ContextEnvWithPredictor(sequences, ref_c, predictor, 1 / 20)
    # env.reset()
    ray.init()
    register_env("ctx_e", lambda _: ContextEnvWithPredictor(sequences, ref_c, predictor, 1 / 20))
    #
    # policy_kwargs = dict(net_arch=[128, 128, 128])
    #
    # model = DQN("MultiInputPolicy", env, verbose=1, policy_kwargs=policy_kwargs, device="cuda",
    #             gamma=0.95, learning_starts=10000)

    trainer = dqn.DQNTrainer(
        env="ctx_e",
        config={
            "gamma": 0.95,
            "num_gpus": 1,
            "num_workers": 10,
            "num_envs_per_worker": 1,
            "rollout_fragment_length": 400,
            # "sgd_minibatch_size": 1000,
            "train_batch_size": 4000,
            "evaluation_num_workers": 10,
            "batch_mode": "complete_episodes",
            # "model": {
            #     "custom_model": RNNSACTorchModel,
            #     "max_seq_len": 32
            # },
            "framework": "torch",
        }

    )

    for iter in range(3000):
        print("train iter", iter)
        # env.reset()
        # if iter == 0:
        #     model.learn(total_timesteps=100, log_interval=4)
        # else:
        #     model.learn(total_timesteps=100, log_interval=4, reset_num_timesteps=False)
        results = trainer.train()
        print(f"avg. reward={results['episode_reward_mean']}")
        if iter % 10 == 0:
            # print(trainer.evaluate())
            custom_eval_function(trainer, trainer.evaluation_workers)
        print("train predictor")
        # predictor = predictor.cuda()
        # if iter > 1 and iter % 10 == 0:
        #     env.p_estimator.fit()
        def update_env(w: RolloutWorker):
            if w.env is not None:
                w.env.predictor = predictor

        def update_pc(w: RolloutWorker):
            if w.env is not None:
                if w.env.p_estimator.is_full:
                    w.env.p_estimator.fit()

        for _ in range(4):
            for x, y in make_train_batch(trainer.evaluation_workers, trainer.get_policy().observation_space):
                opt.zero_grad()
                loss = predictor_loss(x, y, predictor)
                loss.backward()
                print("loss", loss.item())
                opt.step()
            trainer.evaluation_workers.foreach_worker(update_env)
        trainer.workers.foreach_worker(update_env)

        if iter % 10 == 0:
            trainer.evaluation_workers.foreach_worker(update_pc)
            trainer.workers.foreach_worker(update_pc)

        # if iter % 10 == 0:
            # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=3)
            # print(mean_reward, std_reward)

            # for _ in range(10):
            #     obs = env.reset()
            #     actions = []
            #     ref_actions = []
            #     for i in range(199):
            #         action, _states = model.predict(obs, deterministic=True)
            #         obs, rewards, dones, info = env.step(action)
            #         actions.append(action)
            #         ref_actions.append(info["c_ref"])
            #
            #     print(actions)
            #     print(ref_actions)
            #     print()

