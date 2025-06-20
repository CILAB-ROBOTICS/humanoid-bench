import os
import sys
from omegaconf import MissingMandatoryValue

import numpy as np
import gymnasium as gym

from tdmpc2.envs.wrappers.time_limit import TimeLimit


class HumanoidWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        if sys.platform != "darwin" and "MUJOCO_GL" not in os.environ:
            os.environ["MUJOCO_GL"] = "egl"
        if "SLURM_STEP_GPUS" in os.environ:
            os.environ["EGL_DEVICE_ID"] = os.environ["SLURM_STEP_GPUS"]
            print(f"EGL_DEVICE_ID set to {os.environ['SLURM_STEP_GPUS']}")
        if "SLURM_JOB_GPUS" in os.environ:
            os.environ["EGL_DEVICE_ID"] = os.environ["SLURM_JOB_GPUS"]
            print(f"EGL_DEVICE_ID set to {os.environ['SLURM_JOB_GPUS']}")

        super().__init__(env)
        self.env = env
        self.cfg = cfg

    @staticmethod
    def _process_obs(obs):
        if isinstance(obs, dict):
            for key, value in obs.items():
                obs[key] = value.astype(np.float32)
        else:
            obs = obs.astype(np.float32)
        return obs

    def reset(self, options=None):
        obs, info = self.env.reset(options=options)
        return self._process_obs(obs), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action.copy())
        return self._process_obs(obs), reward, done, truncated, info

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, *args, **kwargs):
        return self.env.render()


def make_env(cfg):
    """
    Make Humanoid environment.
    """
    if not cfg.task.startswith("humanoid_"):
        raise ValueError("Unknown task:", cfg.task)
    import humanoid_bench

    policy_path = cfg.get("policy_path", None)
    mean_path = cfg.get("mean_path", None)
    var_path = cfg.get("var_path", None)
    policy_type = cfg.get("policy_type", None)
    small_obs = cfg.get("small_obs", None)
    if small_obs is not None:
        small_obs = str(small_obs)
    tactile_info = cfg.get("tactile_info", None)
    if tactile_info is not None:
        tactile_info = str(tactile_info)
    condition_dim = None
    if cfg.instruct:
        try:
            if cfg.modality == "vector":
                condition_dim = 3
            elif cfg.modality == "embed":
                condition_dim = 768
            else:
                raise ValueError("Unknown condition modality:", cfg.modality)
        except MissingMandatoryValue:
            raise ValueError("Condition modality not specified in config.")

    print("small obs start:", small_obs)

    env = gym.make(
        cfg.task.removeprefix("humanoid_"),
        policy_path=policy_path,
        mean_path=mean_path,
        var_path=var_path,
        policy_type=policy_type,
        small_obs=small_obs,
        obs_wrapper='true' if cfg.obs == "multi-modal" else None,
        sensors=cfg.sensors if cfg.obs == "multi-modal" else None,
        tactile_info=tactile_info,
        condition_dim=condition_dim,
    )
    env = HumanoidWrapper(env, cfg)
    env.max_episode_steps = env.get_wrapper_attr("_max_episode_steps")
    return env
