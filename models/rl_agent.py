# rl_agent.py
# models/rl_agent.py

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.env_context import EnvContext
from ray.tune.registry import register_env
import numpy as np
import os

import asyncio
import websockets
import json

class ReservoirEnv:
    """
    Custom reinforcement learning environment for a reservoir system.
    This is a placeholder. Replace logic to reflect real reward structure.
    """

    def __init__(self, config: EnvContext):
        self.config = config
        self.state = np.array([0.5, 0.5])  # [pressure_level, production_rate]
        self.step_count = 0
        self.max_steps = 50

    def reset(self):
        self.state = np.array([0.5, 0.5])
        self.step_count = 0
        return self.state

    def step(self, action):
        """
        Placeholder step logic. Replace with:
        - Real pressure/saturation transitions
        - BHP feedback
        """
        self.step_count += 1
        delta = np.clip(action[0], -0.1, 0.1)
        self.state[0] = np.clip(self.state[0] + delta, 0, 1)
        reward = -abs(self.state[0] - 0.8)  # reward closer to target pressure
        done = self.step_count >= self.max_steps
        return self.state, reward, done, {}

    @property
    def observation_space(self):
        from gym.spaces import Box
        return Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

    @property
    def action_space(self):
        from gym.spaces import Box
        return Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)


def make_env(env_config: EnvContext):
    return ReservoirEnv(env_config)

register_env("ReservoirEnv-v0", make_env)


class RLAgent:
    """
    Wrapper around Ray RLlib PPOTrainer for live training.
    """

    def __init__(self, env_name: str = "ReservoirEnv-v0", checkpoint_dir="checkpoints"):
        ray.init(ignore_reinit_error=True, include_dashboard=False)
        self.env_name = env_name
        self.checkpoint_dir = checkpoint_dir
        self.agent = PPOTrainer(env=env_name, config={
            "framework": "torch",
            "env_config": {},
            "num_workers": 0,
        })

    def train(self, episodes=10) -> list:
        rewards = []
        for _ in range(episodes):
            result = self.agent.train()
            rewards.append(result["episode_reward_mean"])
        return rewards
    await stream_action_to_websocket(action.tolist())


    def get_action(self, state: np.ndarray) -> float:
        return self.agent.compute_single_action(state)

    def save(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        return self.agent.save(self.checkpoint_dir)

    def load(self, checkpoint_path: str = None):
        if not checkpoint_path:
            files = os.listdir(self.checkpoint_dir)
            if not files:
                raise FileNotFoundError("No checkpoints found.")
            checkpoint_path = os.path.join(self.checkpoint_dir, sorted(files)[-1])
        self.agent.restore(checkpoint_path)


async def stream_action_to_websocket(action, host="localhost", port=6789):
    try:
        async with websockets.connect(f"ws://{host}:{port}") as ws:
            await ws.send(json.dumps({"rl_decision": action}))
    except Exception as e:
        print(f"[WebSocket Stream Error] {e}")

