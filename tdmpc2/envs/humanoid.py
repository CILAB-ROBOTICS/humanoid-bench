import os
import sys
from functools import partial
from typing import Any

import numpy as np
import gymnasium as gym
from dm_control.mujoco.testing.image_utils import humanoid
from gymnasium.vector import VectorEnv
from gymnasium.wrappers import PassiveEnvChecker, OrderEnforcing
from sympy.physics.vector.printing import params
from tdmpc2.envs import TensorWrapper

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

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action.copy())
        obs = obs.astype(np.float32)
        return obs, reward, done, truncated, info

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, *args, **kwargs):
        return self.env.render()

    def get_wrapper_attr(self, name: str) -> Any:
        """
        Get the attribute of the wrapped environment.
        """
        if hasattr(self.env.spec, name):
            return getattr(self.env.spec, name)
        else:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")


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
    task = cfg.get("task", None)
    n_envs = cfg.get("n_envs", 1)
    vec_env = cfg.get("vec_env", False)

    # policy_path = getattr(cfg, "policy_path", None)
    # mean_path = getattr(cfg, "mean_path", None)
    # var_path = getattr(cfg, "var_path", None)
    # policy_type = getattr(cfg, "policy_type", None)
    # small_obs = getattr(cfg, "small_obs", None)
    # task = getattr(cfg, "task", None)
    # n_envs = getattr(cfg, "n_envs", 1)

    # partial function for HumanoidEnv
    humanoid_wrapper_fn = partial(
        HumanoidWrapper,
        cfg=cfg,
    )

    if small_obs is not None:
        small_obs = str(small_obs)

    if vec_env:
        env = gym.make_vec(
            task.removeprefix("humanoid_"),
            # autoreset=True,
            num_envs=n_envs,
            policy_path=policy_path,
            mean_path=mean_path,
            var_path=var_path,
            policy_type=policy_type,
            small_obs=small_obs,
            wrappers=[PassiveEnvChecker, OrderEnforcing, humanoid_wrapper_fn],
        )
    else:
        env = gym.make(
            task.removeprefix("humanoid_"),
            policy_path=policy_path,
            mean_path=mean_path,
            var_path=var_path,
            policy_type=policy_type,
            small_obs=small_obs,
        )

    env = HumanoidWrapper(env, cfg)
    env.max_episode_steps = env.get_wrapper_attr("max_episode_steps")
    return env
