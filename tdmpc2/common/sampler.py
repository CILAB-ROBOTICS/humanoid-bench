import random

import pandas as pd



class ConditionEnum:
    Strength = 0
    Direction = 1
    Speed = 2


class Condition:
    def __init__(self, condition_type: ConditionEnum, value: float):
        self.condition_type = condition_type
        self.value = value

    def __repr__(self):
        return f"Condition(type={self.condition_type}, value={self.value})"

class ConditionSet:
    conditions: list[Condition]

    def __init__(self, conditions: list[Condition]):
        self.conditions = conditions

    def __repr__(self):
        return f"ConditionSet(conditions={self.conditions})"

class ConditionSampler:

    def __init__(self, cfg):
        self.cfg = cfg

        self.condition_sets: list

        self.load(cfg.instruct_path)

    def load(self, csv_path: str) -> None:
        """
        Load condition sets from a CSV file.
        Each row in the CSV file represents a condition set.
        """
        df = pd.read_csv(csv_path)

        self.condition_sets = []

        for _, row in df.iterrows():
            conditions = []

            for condition_i in range(3):
                condition_type = condition_i
                value = row[f"condition_{condition_i}"]

                if value != -1:
                    conditions.append(Condition(condition_type, value))

            self.condition_sets.append(ConditionSet(conditions=conditions))
        print(f"Loaded {len(self.condition_sets)} condition sets from {csv_path}")

    def sample(self, n: int = 1) -> Condition:
        """
        Sample a condition set from the loaded condition sets.
        """
        if len(self.condition_sets) == 0:
            raise ValueError("No condition sets loaded.")

        sampled_sets = random.sample(self.condition_sets, n)
        return sampled_sets

    def __repr__(self):
        return f"ConditionSampler(condition_sets={len(self.condition_sets)} sets)"