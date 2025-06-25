import os
import sys
import shutil
from os.path import dirname
import collections

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("QtAgg")  # Use TkAgg backend for interactive plotting


if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"

os.environ["LAZY_LEGACY_OP"] = "0"

import hydra
from termcolor import colored

from tdmpc2.common.parser import parse_cfg
from tdmpc2.envs import make_env


@hydra.main(config_name="config", config_path="../tdmpc2")
def collect_tactile(cfg: dict):
    cfg.task = 'humanoid_h1touchdualarm-tactiletest-v0'
    cfg.sensors = 'proprio/tactile'
    cfg = parse_cfg(cfg)

    env = make_env(cfg)
    env.reset()

    tactile_buffer = collections.deque(maxlen=100)  # (최근 100 스텝)
    num_sensors = None

    # 실시간 matplotlib 설정
    plt.ion()
    fig, ax = plt.subplots()
    lines = []

    def update_tactile_lineplot(buffer):
        nonlocal lines, num_sensors

        data = np.array(buffer)  # shape: (T, N)
        ax.clear()
        ax.set_title("Tactile Sensor Time Series (last 100 steps)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Sensor Value")
        ax.grid(True)

        if num_sensors is None:
            num_sensors = data.shape[1]

        for i in range(num_sensors):
            ax.plot(data[:, i], label=f"{i}")

        # ax.set_ylim(0, 0.05)  # 값 범위 조정 (필요시 변경)
        ax.set_xlim(0, 100)  # x축 범위 조정
        # draw at top
        ax.legend(loc='upper left', fontsize='small')
        #
        plt.draw()
        plt.pause(0.001)

    for i_epoch in range(10):  # 10 에피소드 동안 수집
        env.reset()

        for i in range(40):
            image = env.render()
            cv2.imshow("env", image)
            cv2.waitKey(1)

            action = torch.tensor(env.action_space.sample(), dtype=torch.float32)
            obs, reward, done, truncated, info = env.step(action)

            tactile = obs['tactile']
            if isinstance(tactile, torch.Tensor):
                tactile = tactile.cpu().numpy()
            elif not isinstance(tactile, np.ndarray):
                tactile = np.array(tactile)

            if tactile.ndim != 1:
                tactile = tactile.flatten()  # 예외 처리

            tactile_buffer.append(tactile)
            print(f"Step {i}: reward = {reward:.3f}, tactile = {tactile[:5]}...")

            if len(tactile_buffer) > 1:
                update_tactile_lineplot(tactile_buffer)

    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    collect_tactile()
