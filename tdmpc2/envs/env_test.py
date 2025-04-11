import argparse
import hydra
import torch
from omegaconf import OmegaConf, DictConfig

from tdmpc2.envs import make_env

@hydra.main(config_name="config", config_path="..")
def main(cfg: dict):
    print(cfg)

    env = make_env(cfg)


    obs = env.reset()
    print(obs[0].shape)
    print(env.action_space.shape)

    for i in range(10):
        action = env.action_space.sample()
        action = torch.Tensor(action)
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i}: Obs: {obs.shape}, Reward: {reward.shape}, Done: {done.shape}")



if __name__ == '__main__':
    main()
