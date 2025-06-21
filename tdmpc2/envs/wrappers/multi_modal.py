from collections import deque

import gymnasium as gym
import numpy as np
import torch


class MultimodalWrapper(gym.Wrapper):
    """
    Wrapper for pixel observations. Compatible with DMControl environments.
    """

    def __init__(
        self, cfg, env,
        image_frames=3, image_size=64,
        tactile_frames=10,
    ):
        super().__init__(env)
        self.cfg = cfg
        self.env = env
        self._frames = {
            'image': deque([], maxlen=image_frames),
            'tactile': deque([], maxlen=tactile_frames),
        }
        self._image_size = image_size

    @property
    def observation_space(self):
        obs_space = self.env.observation_space
        if type(obs_space) == gym.spaces.Dict:
            for k in obs_space.keys():
                if k.endswith("_eye"):  # image
                    obs_space[k] = gym.spaces.Box(
                        low=0,
                        high=255,
                        shape=(self._frames['image'].maxlen * 3, self._image_size, self._image_size),
                        dtype=np.uint8,
                    )
                elif k.startswith("tactile"):  # tactile
                    obs_shape = obs_space[k].shape
                    obs_space[k] = gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self._frames['tactile'].maxlen * obs_shape[0],),
                        dtype=np.float64,
                    )
        return obs_space

    def _process_obs(self, obs):
        new_obs = {}
        if type(obs) == dict:
            for k in obs.keys():
                if k.endswith("_eye"):  # image -> resize -> stacking
                    image = obs[k].permute(2, 0, 1).unsqueeze(0)  # -> (1, 3, 256, 256)
                    image_resized = torch.nn.functional.interpolate(
                        image,
                        size=(self._image_size, self._image_size),
                        mode='bilinear',
                         align_corners=False
                    )
                    frame = image_resized.squeeze(0)  # -> (3, 64, 64)
                    self._frames['image'].append(frame)
                    new_obs[k] = torch.concatenate(list(self._frames['image']), dim=0)
                elif k.startswith("tactile"):  # tactile -> stacking
                    self._frames['tactile'].append(obs[k])
                    new_obs[k] = torch.concatenate(list(self._frames['tactile']), dim=0)
                else:
                    new_obs[k] = obs[k]

        return new_obs

    def reset(self, options=None):
        obs, info = self.env.reset(options=options)

        max_frames = max([v.maxlen for v in self._frames.values()])
        for _ in range(max_frames-1):
            self._process_obs(obs)

        return self._process_obs(obs), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return self._process_obs(obs), reward, done, truncated, info
