import os
import sys
from os.path import basename, dirname

from tdmpc2.common.sampler import ConditionSampler

if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"

import pandas as pd
import logging
import cv2
import hydra
import imageio
from tqdm import tqdm
from termcolor import colored

from tdmpc2.common.parser import parse_cfg
from tdmpc2.common.seed import set_seed
from tdmpc2.envs import make_env
from tdmpc2.tdmpc2 import TDMPC2

logger = logging.getLogger(basename(__file__))


@hydra.main(config_name="config", config_path=".")
def evaluate(cfg: dict):

    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    logger.info(colored(f"Task: {cfg.task}", "blue", attrs=["bold"]))

    # Make environment
    env = make_env(cfg)

    # Load agent
    agent = TDMPC2(cfg)

    # load the agent checkpoint
    # agent.load(cfg.checkpoint) # TODO checkpoint 로드 하는 부분 작성하기

    # Evaluate
    logger.info(f"Evaluating agent on {cfg.task}")

    if cfg.instruct:
        instruct_dir = os.path.join(dirname(__file__), "..", "instruct_rl", "instruct", "bert-base-uncased")
        cfg.instruct_path = os.path.abspath(os.path.join(instruct_dir, f"{cfg.instruct}.csv"))
    else:
        cfg.instruct_path = None

    cond_sampler = ConditionSampler(cfg) if cfg.instruct_path else None
    logger.info(f"Condition sampler: {cond_sampler}")


    if cfg.save_video:
        video_dir = os.path.join(cfg.work_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)

    eval_logs = list()

    conditions = cond_sampler.to_list() if cond_sampler else [None]

    for i, condition in enumerate(conditions):

        for i in tqdm(range(cfg.eval_episodes), desc=f"Evaluating {i} / {len(conditions)} condition"):
            obs, done, ep_reward, t = env.reset(options={'condition': condition})[0], False, 0, 0
            step_logs = list()

            frames = list()
            if cfg.save_video:
                frames.append(env.render())

            while not done:

                action = agent.act(obs, t0=t == 0)
                obs, reward, done, truncated, info = env.step(action)
                ep_reward += reward
                t += 1

                if cfg.save_video:
                    frames.append(env.render())

                if cfg.render:
                    image = env.render()
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    cv2.imshow("mujoco", image)
                    cv2.waitKey(1)

                # prefix info_ to the info dict
                info_prefix = {"info_" + k: v for k, v in info.items()}

                if condition is not None:
                    condition_prefix = {"cond_" + k: v for k, v in condition.to_dict().items()}
                else:
                    condition_prefix = {}

                step_logs.append({
                    'task': cfg.task,
                    'repeat': i,
                    **condition_prefix,
                    'step': t,
                    'reward': float(reward),
                    'ep_reward': float(ep_reward),
                    'done': bool(done),
                    'truncated': bool(truncated),
                    **info_prefix,
                })


            if cfg.save_video:
                imageio.mimsave(os.path.join(video_dir, f"{i}.mp4"), frames, fps=15)

            step_df = pd.DataFrame(step_logs)
            info_columns = [col for col in step_df.columns if col.startswith("info_")]
            cond_columns = [col for col in step_df.columns if col.startswith("cond_")]

            eval_logs += step_logs

            eval_df = pd.DataFrame(eval_logs)
            eval_df.to_csv(os.path.join(cfg.work_dir, f"step.csv"), index=False)


            # eval
            eval_summary_df = eval_df.groupby(["task", "repeat", *cond_columns]).agg(
                {
                    "step": "last",
                    "reward": "sum",
                    "done": "last",
                    "truncated": "last",
                    **{col: "mean" for col in info_columns},
                }
            ).reset_index()
            eval_summary_df.to_csv(os.path.join(cfg.work_dir, f"episode.csv"), index=False)

            # print the last series of the eval_summary_df
            last_series = eval_summary_df.iloc[-1]

            cond_strs = [
                f"{col}: {last_series[col]:<6.3f}" for col in cond_columns
            ]
            cond_str = " ".join(cond_strs)

            info_strs = [
                f"{col}: {last_series[col]:<6.3f}" for col in info_columns
            ]
            info_str = " ".join(info_strs)

            logger.info(
                f"iter: {last_series['repeat']:<4} "
                f"{cond_str} "
                f"ep_length: {last_series['step']:<4.1f} "
                f"ep_reward: {last_series['reward']:<4.1f} "
                f"done: {last_series['done']} "
                f"truncated: {last_series['truncated']} "
                f"{info_str}"
            )

if __name__ == "__main__":
    evaluate()
