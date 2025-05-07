from copy import deepcopy

import numpy as np


class Statistics:
    def __init__(self):
        self.reset()

    def append(self, info: dict):
        for k, v in info.items():
            if k not in self.items:
                self.items[k] = []
            self.items[k].append(v)

    def summarize(self):
        summary = dict()

        for k, v in self.items.items():
            summary[k] = np.array(v).mean(axis=0)

        return summary

    def pop(self, step: int = None, episode_reward: float = None):
        summary = deepcopy(self.summarize())

        if step is not None:
            summary["step"] = step
        if episode_reward is not None:
            summary["episode_reward"] = episode_reward

        self.reset()
        return summary

    def reset(self):
        self.items = dict()

