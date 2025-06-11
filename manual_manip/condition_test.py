import gymnasium as gym
from gymnasium import register
import cv2
from sympy.strategies.branch import condition

from tdmpc2.common.sampler import Condition, ConditionSampler


class DummyConfig:
    def __init__(self, instruct_path, modality):
        self.instruct_path = instruct_path
        self.modality = modality

if __name__ == "__main__":
    register(
        id="temp-v0",
        entry_point="humanoid_bench.env:HumanoidEnv",
        max_episode_steps=1000,
        kwargs={
            "robot": "h1dualarm",
            "control": "pos",
            "task": "rub",
        },
    )


    sampler = ConditionSampler(DummyConfig(
        instruct_path="../instruct_rl/instruct/bert-base-uncased/strmid.csv",
        modality="vector"
    ))

    env = gym.make("temp-v0")

    condition = sampler.sample()
    ob, _ = env.reset(options={'condition': condition})

    print(f"ob_space = {env.observation_space}, ob = {ob.shape}")
    print(f"ac_space = {env.action_space.shape}")
    # env.render()
    while True:
        action = env.action_space.sample()
        ob, rew, terminated, truncated, info = env.step(action)
        print(f"ob_space = {env.observation_space}, ob = {ob.shape}, info = {info}")
        image = env.render()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("mujoco", image)
        cv2.waitKey(1)

        if terminated or truncated:
            env.reset()

        # break
    env.close()

