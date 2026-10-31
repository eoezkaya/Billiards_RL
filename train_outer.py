#!/usr/bin/env python3

import os
import re
import sys
from collections import deque

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

NUM_ENVS = 4
LEARNING_STARTS = 1000

_PARAM_ORDER = [
    "learning_rate",
    "buffer_size",
    "batch_size",
    "gamma",
    "tau",
    "exploration_fraction",
    "exploration_initial_eps",
    "exploration_final_eps",
]


class RewardTrackingWrapper(VecEnv):
    """Wraps vectorized env to track episode rewards and print stats."""

    def __init__(self, vec_env):
        self.env = vec_env
        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space
        self.num_envs = vec_env.num_envs
        self.single_observation_space = getattr(
            vec_env, "single_observation_space", vec_env.observation_space
        )
        self.single_action_space = getattr(
            vec_env, "single_action_space", vec_env.action_space
        )

        self.episode_rewards = deque(maxlen=100)
        self.episode_count = 0
        self.env_rewards = [0.0] * vec_env.num_envs
        self.last_print = 0

    def reset(self):
        self.env_rewards = [0.0] * self.num_envs
        return self.env.reset()

    def step_async(self, actions):
        self.env.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.env.step_wait()

        # Track rewards per environment
        for i, (reward, done) in enumerate(zip(rewards, dones)):
            self.env_rewards[i] += reward
            if done:
                self.episode_rewards.append(self.env_rewards[i])
                self.episode_count += 1
                self.env_rewards[i] = 0.0

        # Print every 1000 episodes
        if self.episode_count - self.last_print >= 1000 and len(self.episode_rewards) > 0:
            self.last_print = self.episode_count
            mean_rew = np.mean(list(self.episode_rewards))
            sys.stdout.write(f"\n{'='*70}\n")
            sys.stdout.write(
                f"MEAN REWARD - Episodes: {self.episode_count:,} | Mean (last 100): {mean_rew:.4f}\n"
            )
            sys.stdout.write(f"{'='*70}\n\n")
            sys.stdout.flush()

        return obs, rewards, dones, infos

    def close(self):
        return self.env.close()

    def env_is_wrapped(self, wrapper_type):
        return self.env.env_is_wrapped(wrapper_type)

    def env_method(self, method_name, *args, **kwargs):
        return self.env.env_method(method_name, *args, **kwargs)

    def get_attr(self, attr_name, indices=None):
        return self.env.get_attr(attr_name, indices)

    def set_attr(self, attr_name, value, indices=None):
        return self.env.set_attr(attr_name, value, indices)

    def __getattr__(self, name):
        return getattr(self.env, name)


def _coerce_value(value_str: str):
    value_str = value_str.strip()
    if not value_str:
        raise ValueError("Empty value in dv_outer.txt")

    try:
        if "." in value_str or "e" in value_str.lower():
            return float(value_str)
        return int(value_str)
    except ValueError:
        return value_str


def load_design_vector(file_path: str = "dv_outer.txt") -> dict:
    defaults = {
        "learning_rate": 1e-3,
        "buffer_size": 50000,
        "batch_size": 32,
        "gamma": 0.99,
        "tau": 1.0,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "total_timesteps": 500000,
    }

    resolved_path = os.path.abspath(os.path.expanduser(file_path))
    if not os.path.exists(resolved_path):
        print(
            f"[Config Warning] Config file '{file_path}' not found. Using default hyperparameters."
        )
        return defaults

    parsed_params = {}
    numeric_tokens = []

    try:
        with open(resolved_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if "=" in line:
                    parts = line.split("=", 1)
                elif " " in line:
                    parts = line.split(None, 1)
                else:
                    parts = [None, line]

                if len(parts) == 2 and parts[0] is not None:
                    key = parts[0].strip()
                    val_str = parts[1].strip()
                    parsed_params[key] = _coerce_value(val_str)
                    continue

                numeric_tokens.extend(re.split(r"[\s,]+", line))
    except OSError as exc:
        raise OSError(f"Could not read config file '{resolved_path}': {exc}") from exc

    if not parsed_params:
        numeric_values = [
            _coerce_value(token) for token in numeric_tokens if token.strip()
        ]
        if len(numeric_values) != len(_PARAM_ORDER):
            raise ValueError(
                f"Expected {len(_PARAM_ORDER)} hyperparameters but got {len(numeric_values)} "
                f"from '{resolved_path}'."
            )
        parsed_params = {
            name: value for name, value in zip(_PARAM_ORDER, numeric_values)
        }

    hp = {**defaults, **parsed_params}
    print(f"[Config] Loaded hyperparameters from {resolved_path}:")
    for k, v in hp.items():
        print(f"  {k}: {v}")

    return hp


def make_env(rank: int = 0, seed: int = 42):
    def _init():
        from RLEnv_outer import RLBilliard_outer

        env = RLBilliard_outer(verbose=False)
        env.reset()
        return env

    return _init


def main():
    print(f"\n========================================")
    print(f"Training DQN Billiards Pocket Selection Agent")
    print(f"========================================\n")

    hp = load_design_vector("dv_outer.txt")

    model_dir = "./models/dqn_billiards_outer/"
    os.makedirs(model_dir, exist_ok=True)

    num_envs = NUM_ENVS

    if num_envs > 1:
        base_env = SubprocVecEnv([make_env(rank=i) for i in range(num_envs)])
    else:
        base_env = DummyVecEnv([make_env(rank=0)])

    # Wrap environment BEFORE creating the DQN model
    vec_env = RewardTrackingWrapper(base_env)

    model = DQN(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=float(hp["learning_rate"]),
        buffer_size=int(hp["buffer_size"]),
        batch_size=int(hp["batch_size"]),
        gamma=float(hp["gamma"]),
        tau=float(hp["tau"]),
        exploration_fraction=float(hp["exploration_fraction"]),
        exploration_initial_eps=float(hp["exploration_initial_eps"]),
        exploration_final_eps=float(hp["exploration_final_eps"]),
        learning_starts=LEARNING_STARTS,
        verbose=0,
    )

    total_timesteps = int(hp["total_timesteps"])
    print(
        f"\nTraining DQN with {num_envs} parallel environments for"
        f" {total_timesteps} timesteps..."
    )

    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    final_model_path = os.path.join(model_dir, "dqn_billiards_outer_final")
    model.save(final_model_path)
    print("\nTraining finished.")


if __name__ == "__main__":
    main()