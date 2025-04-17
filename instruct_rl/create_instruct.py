from enum import IntEnum
import time

import torch
import pandas as pd
import random
import numpy as np
import itertools
import os
import json
from copy import deepcopy
from transformers import BertTokenizer, BertModel
from collections import deque


class ConditionFeature(IntEnum):
    strength = 0
    direction = 1
    frequency = 2


class Config:
    def __init__(self):
        self.max_length = 128
        self.train_ratio = 0.8
        self.num_scenario = 1
        self.similar_words = True
        self.prompt_path = "instruct_template.json"  # ✨ 경로 수정 필요




def prompt_combination(
        data, scenario_combination, current_prompt, evaluation_values, num_scenario, similar
):
    """
    프롬프트 조합하는 코드

    Args:
        data (_type_): _description_
        scenario_combination (_type_): _description_
        current_prompt (_type_): _description_
        evaluation_values (_type_): _description_
        num_scenario (_type_): _description_

    Returns:
        _type_: _description_
    """
    # TODO: 타입 선언 추가하기

    if not scenario_combination:
        lsttext = current_prompt + "."
        return {lsttext: evaluation_values}

    final_prompts = {}
    scenario_key = scenario_combination.popleft()

    scenario = data["scenarios"].get(scenario_key)
    if not scenario:
        return final_prompts

    prompt = scenario.get("prompt", "")
    feature = scenario.get("feature", "")

    for key, value in data[feature].items():
        if similar:
            similar_words = value["similar"]
        else:
            similar_words = [value["similar"][0]]

        for similar_word in similar_words:
            formatted_prompt = prompt.format(**{feature: similar_word})
            if current_prompt == "":
                updated_prompt = formatted_prompt
            else:
                updated_prompt = current_prompt + ", " + formatted_prompt

            updated_evaluation_values = deepcopy(evaluation_values)
            updated_evaluation_values[feature] = value["value"]
            sub_prompts = prompt_combination(
                data=data,
                scenario_combination=scenario_combination.copy(),
                current_prompt=updated_prompt,
                evaluation_values=updated_evaluation_values,
                num_scenario=num_scenario,
                similar=similar,
            )
            if sub_prompts and len(list(sub_prompts.values())[0]) == num_scenario:
                final_prompts.update(sub_prompts)

    return final_prompts

def match_prompt_eval(config, combinations):
    with open(config.prompt_path, "r") as file:
        data = json.load(file)

    text = ""
    eval_list = {}
    prompt_list = {}
    prompt_list.update(
        prompt_combination(
            data=data,
            scenario_combination=deque(combinations),
            current_prompt=text,
            evaluation_values=eval_list,
            num_scenario=config.num_scenario,
            similar=config.similar_words,
        )
    )

    return prompt_list


def dict_to_csv(prompt_eval):
    instructs = list(prompt_eval.keys())
    conditions = []
    conditions_values = []
    example_values = [-1] * 3

    for v in prompt_eval.values():
        condition, values = list(v.keys()), list(v.values())
        values = [int(x) if str(x).isdigit() else x for x in values]

        # FIXED: use .value to get int from enum
        condition_str = "".join(
            sorted([str(ConditionFeature[cond].value) for cond in condition], key=int)
        )

        example_value = deepcopy(example_values)
        for idx in range(len(condition)):
            example_value[ConditionFeature[condition[idx]].value] = values[idx]
        print(example_value)

        conditions.append(int(condition_str))
        conditions_values.append(example_value)

    print(conditions_values)
    conditions = np.array(conditions)
    conditions_values = np.array(conditions_values)

    prompt_eval_csv = pd.DataFrame(instructs, columns=["instruction"])
    prompt_eval_csv["condition_enum"] = conditions
    print(f"prompt_eval_csv shape: {conditions_values.shape}")
    for i in range(conditions_values.shape[1]):
        prompt_eval_csv[f"condition_{i}"] = conditions_values[:, i]

    return prompt_eval_csv


def make_data(config, random_list, combination, pretrained_model, tokenizer):
    prompt_eval = match_prompt_eval(config, combination)
    prompt_eval_csv = dict_to_csv(prompt_eval)

    text_samples = list(prompt_eval_csv["instruction"])
    input_text = text_samples

    start_time = time.time()
    encoded_inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding="max_length",
        max_length=config.max_length,
        truncation=True,
    )
    end_time = time.time()
    print(f"tokenizer: {end_time - start_time:.4f} sec")

    with torch.no_grad():
        outputs = pretrained_model(**encoded_inputs).last_hidden_state
    end_time2 = time.time()
    print(f"pretrained_model: {end_time2 - end_time:.4f} sec")
    print(f"whole_process: {end_time2 - start_time:.4f} sec")
    print(f"output shape: {outputs.shape}")

    cls_outputs = outputs[:, 0, :]
    print(f"CLS shape: {cls_outputs.shape}")


    cls_outputs_np = cls_outputs.cpu().numpy()
    embedding_df = pd.DataFrame(cls_outputs_np, columns=[f"embed_{i}" for i in range(cls_outputs_np.shape[1])])
    prompt_eval_csv = pd.concat([prompt_eval_csv, embedding_df], axis=1)

    return prompt_eval_csv, random_list


def main():
    config = Config()
    models = ["bert-base-uncased"]
    start_scenario = 1
    end_scenario = 4
    num_scenario = config.num_scenario
    random_list = None

    for model_name in models:
        print(f"Loading model: {model_name}")
        tokenizer = BertTokenizer.from_pretrained(model_name)
        pretrained_model = BertModel.from_pretrained(model_name)
        pretrained_model.eval()

        scenario_keys = [str(i) for i in range(start_scenario, end_scenario)]
        combinations = list(itertools.combinations(scenario_keys, num_scenario))

        for combination in combinations:
            print(f"combination: {combination}")

            prompt_eval_csv, random_list = make_data(config, random_list, combination, pretrained_model, tokenizer)

            save_dir = f"instruct/{model_name}"
            os.makedirs(save_dir, exist_ok=True)

            # subtract -1 from the combination
            combination = [str(int(x) - 1) for x in combination]

            save_path = os.path.join(
                save_dir,
                f"scn-{num_scenario}_se-{''.join(sorted(combination, key=int))}.csv"
            )

            # 우선 train 컬럼을 전부 True로 초기화
            prompt_eval_csv["train"] = True

            # 그룹화하고 그룹별로 20%를 False로 설정
            grouped = prompt_eval_csv.groupby(["condition_0", "condition_1", "condition_2"])

            for _, group_indices in grouped.groups.items():
                group_indices = list(group_indices)
                n_test = max(1, int(len(group_indices) * 0.2))  # 그룹마다 최소 1개는 테스트
                test_indices = np.random.choice(group_indices, size=n_test, replace=False)
                prompt_eval_csv.loc[test_indices, "train"] = False

            prompt_eval_csv.to_csv(save_path, index=False)
            print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
