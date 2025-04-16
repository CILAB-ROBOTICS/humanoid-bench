import random
import numpy as np
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
    embed: np.ndarray
    modality: str

    def __init__(self, conditions: list[Condition],
                 embed: np.ndarray = None,
                 modality: str = "vector"):
        self.conditions = conditions
        self.embed = embed
        self.modality = modality

    def get_feature_size(self) -> int:
        # get the maximum value in ConditionEnum
        return ConditionEnum.Speed + 1

    def get_feature(self):
        if self.modality == "embed":
            return self.embed
        elif self.modality == "vector":
            feature = [-1] * self.get_feature_size()
            for condition in self.conditions:
                feature[condition.condition_type] = condition.value
            feature = np.array(feature, dtype=np.float32)
            return feature

    def __repr__(self):
        return f"ConditionSet(conditions={self.conditions})"

class ConditionSampler:

    def __init__(self, cfg):
        self.cfg = cfg

        self.condition_sets: list
        self.load(cfg.instruct_path)

    @property
    def modality(self):
        return self.cfg.modality

    def load(self, csv_path: str) -> None:
        """
        Load condition sets from a CSV file.
        Each row in the CSV file represents a condition set.
        """
        df = pd.read_csv(csv_path)

        self.condition_sets = []

        for _, row in df.iterrows():
            conditions = list()

            for condition_i in range(3):
                condition_type = condition_i
                value = row[f"condition_{condition_i}"]

                if value != -1:
                    conditions.append(Condition(condition_type, value))

            # get the value starts with embed_*
            embeds = [
                col for col in df.columns if col.startswith("embed_")
            ]
            embed_vals = np.array(row[embeds].values, dtype=np.float32)

            self.condition_sets.append(ConditionSet(conditions=conditions,
                                                    modality=self.modality,
                                                    embed=embed_vals))
        print(f"Loaded {len(self.condition_sets)} condition sets from {csv_path}")

    def sample(self, n: int = 1) -> ConditionSet:
        """
        Sample a condition set from the loaded condition sets.
        """
        if len(self.condition_sets) == 0:
            raise ValueError("No condition sets loaded.")

        sampled_set = random.sample(self.condition_sets, n)[0]

        return sampled_set

    def __repr__(self):
        return f"ConditionSampler(condition_sets={len(self.condition_sets)} sets, modality={self.modality})"