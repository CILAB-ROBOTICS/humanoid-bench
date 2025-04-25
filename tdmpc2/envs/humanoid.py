import os
import sys

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
                obs[key] = value.flatten().astype(np.float32)
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

    print("small obs start:", small_obs)

    if cfg.tactile:
        assert cfg.obs == "multi-modal", "The observation must be multi-modal in config."

    env = gym.make(
        cfg.task.removeprefix("humanoid_"),
        policy_path=policy_path,
        mean_path=mean_path,
        var_path=var_path,
        policy_type=policy_type,
        small_obs=small_obs,
        obs_wrapper='true' if cfg.tactile else None,
        sensors="proprio,tactile" if cfg.tactile else None,
    )
    env = HumanoidWrapper(env, cfg)
    env.max_episode_steps = env.get_wrapper_attr("_max_episode_steps")
    return env
