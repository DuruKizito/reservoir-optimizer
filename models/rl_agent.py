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
    This is a placeholder. Replace logic to reflect real reward structure.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.state = np.array([0.5, 0.5], dtype=np.float32)  # [pressure_level, production_rate]
        self.step_count = 0
        self.max_steps = 50
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self):
        self.state = np.array([0.5, 0.5], dtype=np.float32)
        self.step_count = 0
        return self.state

    def step(self, action):
        self.step_count += 1
        delta = np.clip(action[0], -0.1, 0.1)
        self.state[0] = np.clip(self.state[0] + delta, 0, 1)
        reward = -abs(self.state[0] - 0.8)  # reward closer to target pressure
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
    """
    def __init__(self, env_name: str = None, checkpoint_dir="checkpoints"):
        self.env = ReservoirEnv()
        self.checkpoint_dir = checkpoint_dir
        self.model = PPO("MlpPolicy", self.env, verbose=0, device="auto")

    def train(self, episodes=10) -> list:
        timesteps = episodes * 50  # 50 steps per episode
        self.model.learn(total_timesteps=timesteps)
        # SB3 does not return reward per episode by default, so we return empty or custom logic
        return []

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

