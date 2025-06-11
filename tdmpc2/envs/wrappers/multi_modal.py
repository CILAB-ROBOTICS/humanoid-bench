from collections import deque

import gymnasium as gym
import numpy as np
import torch


class MultimodalWrapper(gym.Wrapper):
    """
    Wrapper for pixel observations. Compatible with DMControl environments.
    """

    def __init__(self, cfg, env, num_frames=3, render_size=64):
        super().__init__(env)
        self.cfg = cfg
        self.env = env
        self._frames = deque([], maxlen=num_frames)
        self._render_size = render_size

    @property
    def observation_space(self):
        obs_space = self.env.observation_space
        if type(obs_space) == gym.spaces.Dict:
            for k in obs_space.keys():
                if k.endswith("_eye"):
                    obs_space[k] = gym.spaces.Box(
                        low=0,
                        high=255,
                        shape=(3, self._render_size, self._render_size),
                        dtype=np.uint8,
                    )
        return obs_space

    def _process_obs(self, obs):
        if type(obs) == dict:
            for k in obs.keys():
                if k.endswith("_eye"):
                    image = obs[k].permute(2, 0, 1).unsqueeze(0)  # -> (1, 3, 256, 256)
                    image_resized = torch.nn.functional.interpolate(
                        image,
                        size=(self._render_size, self._render_size),
                        mode='bilinear',
                         align_corners=False
                    )
                    obs[k] = image_resized.squeeze(0)  # -> (3, 64, 64)
        return obs

    def reset(self, options=None):
        obs, info = self.env.reset(options=options)
        return self._process_obs(obs), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return self._process_obs(obs), reward, done, truncated, info
