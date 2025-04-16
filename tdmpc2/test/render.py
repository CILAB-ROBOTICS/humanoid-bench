import os
import sys
from os.path import dirname

if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"

os.environ["LAZY_LEGACY_OP"] = "0"
import warnings
warnings.filterwarnings("ignore")

import torch
import cv2
import numpy as np

import hydra
from termcolor import colored

from tdmpc2.common.parser import parse_cfg
from tdmpc2.common.seed import set_seed
from tdmpc2.envs import make_env, make_eval_env
from tdmpc2.trainer.offline_trainer import OfflineTrainer
from tdmpc2.trainer.online_trainer import OnlineTrainer
from tdmpc2.common.logger import Logger

torch.backends.cudnn.benchmark = True


@hydra.main(config_name="config", config_path="..")
def train(cfg: dict):
    """
    Script for training single-task / multi-task TD-MPC2 agents.
    """
    cfg.task = "humanoid_h1hand-window-v0"

    assert cfg.steps > 0, "Must train for at least 1 step."
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    print(colored("Work dir:", "yellow", attrs=["bold"]), cfg.work_dir)

    env = make_eval_env(cfg)
    env.reset()

    # Prepare video writer
    video_path = os.path.join(dirname(__file__), "recorded_video.mp4")
    fps = 30
    image = env.render()
    height, width, _ = image.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for i in range(100):
        env.step(env.rand_act())
        frame = env.render()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to: {video_path}")


if __name__ == "__main__":
    train()
