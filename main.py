import gym
import numpy as np
import random

import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy

from context_env import ContextEnv
from metric import Confusion


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

    env = ContextEnv(sequences, ref_c, 20)
    env.reset()

    policy_kwargs = dict(net_arch=[128, 128, 128, 128])

    model = DQN("MultiInputPolicy", env, verbose=1, policy_kwargs=policy_kwargs, device="cuda",
                gamma=0.97, learning_starts=10000)

    for iter in range(2000):
        print("train iter", iter)
        env.reset()
        if iter == 0:
            model.learn(total_timesteps=50, log_interval=4)
        else:
            model.learn(total_timesteps=50, log_interval=4, reset_num_timesteps=False)
        print("train predictor")

        for _ in range(50):
            env.train()

        if iter % 100 == 0:
            mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=3)
            print(mean_reward, std_reward)


    for _ in range(10):
        obs = env.reset()
        actions = []
        actions_change = []
        ref_actions = []
        for i in range(199):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            if i == 0 or action != actions[-1]:
                actions_change.append(1)
            else:
                actions_change.append(0)
            actions.append(action)
            ref_actions.append(info["c_ref"])

        print(actions_change)
        print(ref_actions)
        print()

