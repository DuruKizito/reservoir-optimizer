# rl_agent.py
# models/rl_agent.py

import numpy as np
import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import torch
import asyncio
import websockets
import json

class ReservoirEnv(gym.Env):
    """
    Custom reinforcement learning environment for a reservoir system.
    Now supports gas, rock, fluid properties and optional fields.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.gas = self.config.get('gas', {})
        self.rock = self.config.get('rock', {})
        self.fluid = self.config.get('fluid', {})
        self.state = np.array([
            self.config.get('pressure_level', 0.5),
            self.config.get('production_rate', 0.5),
            self.gas.get('gor', 0),
            self.gas.get('gas sg', 0)
        ], dtype=np.float32)
        self.step_count = 0
        self.max_steps = 50
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self):
        self.state = np.array([
            self.config.get('pressure_level', 0.5),
            self.config.get('production_rate', 0.5),
            self.gas.get('gor', 0),
            self.gas.get('gas sg', 0)
        ], dtype=np.float32)
        self.step_count = 0
        return self.state

    def step(self, action):
        self.step_count += 1
        delta = np.clip(action[0], -0.1, 0.1)
        self.state[0] = np.clip(self.state[0] + delta, 0, 1)
        # Reward uses gas properties if available
        reward = -abs(self.state[0] - 0.8)
        if self.gas.get('gor', None) is not None:
            reward += 0.1 * self.gas['gor'] / 1000.0
        done = self.step_count >= self.max_steps
        info = {}
        return self.state, reward, done, info

    def render(self, mode="human"):
        pass  # Optional: implement visualization

# Optional: check environment compliance
# check_env(ReservoirEnv())

class RLAgent:
    """
    Wrapper around Stable Baselines3 PPO for live training.
    Accepts gas, rock, fluid config.
    """
    def __init__(self, env_name: str = None, checkpoint_dir="checkpoints", config=None):
        self.env = ReservoirEnv(config=config)
        self.checkpoint_dir = checkpoint_dir
        self.model = PPO("MlpPolicy", self.env, verbose=0, device="auto")

    def train(self, episodes=10) -> list:
        timesteps = episodes * self.env.max_steps
        self.model.learn(total_timesteps=timesteps)
        # Dummy rewards
        rewards = [-abs(self.env.state[0] - 0.8) for _ in range(episodes)]
        return rewards

    def get_action(self, state: np.ndarray) -> float:
        action, _ = self.model.predict(state, deterministic=True)
        return action

    def save(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, "ppo_reservoir")
        self.model.save(path)
        return path

    def load(self, checkpoint_path: str = None):
        if not checkpoint_path:
            path = os.path.join(self.checkpoint_dir, "ppo_reservoir.zip")
            if not os.path.exists(path):
                raise FileNotFoundError("No checkpoints found.")
            checkpoint_path = path
        self.model = PPO.load(checkpoint_path, env=self.env)


async def stream_action_to_websocket(action, host="localhost", port=6789):
    try:
        async with websockets.connect(f"ws://{host}:{port}") as ws:
            await ws.send(json.dumps({"rl_decision": action.tolist()}))
    except Exception as e:
        print(f"[WebSocket Stream Error] {e}")

