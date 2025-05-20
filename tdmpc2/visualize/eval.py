import os
import sys
from os.path import basename, dirname

from tdmpc2.common.sampler import ConditionSampler
from tdmpc2.utils.checkpoint import find_last_checkpoint, find_best_checkpoint

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


def wrap_env_for_rendering_tactile(env):
    from humanoid_bench.wrappers import TactileInfoWrapper
    from tdmpc2.visualize.const import IMAGE_PATH, VERTICES
    import numpy as np
    from matplotlib import colors
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    def subdivide_quad(corners, H, W):
        """
        Divide a quad (4 corners) into H×W smaller quads (cells).
        corners: list of four (x,y) in TL, TR, BR, BL order
        H, W: number of rows and cols
        return: list of H*W cells, each a list of 4 (x,y)
        """
        TL, TR, BR, BL = [np.array(p) for p in corners]
        # interpolate along left and right edges
        left_pts = [TL + (BL - TL) * (i / H) for i in range(H + 1)]
        right_pts = [TR + (BR - TR) * (i / H) for i in range(H + 1)]
        cells = []
        for i in range(H):
            topL, botL = left_pts[i], left_pts[i + 1]
            topR, botR = right_pts[i], right_pts[i + 1]
            for j in range(W):
                u0, u1 = j / W, (j + 1) / W
                tl = topL + (topR - topL) * u0
                tr = topL + (topR - topL) * u1
                bl = botL + (botR - botL) * u0
                br = botL + (botR - botL) * u1
                cells.append([tl, tr, br, bl])
        return cells

    class TactileHeatmapInfoWrapper(TactileInfoWrapper):
        def step(self, action):
            obs, rew, terminated, truncated, info = self.task.step(action)
            raw_tactile = self.get_tactile_obs()
            info.update({'tactile': raw_tactile})
            info.update({'tactile_heatmap': self._visualize_tactile(raw_tactile)})
            return obs, rew, terminated, truncated, info

        @staticmethod
        def _visualize_tactile(data: dict):
            """
            For each tactile channel, render the PatchCollection on top
            of the static hand image, capture as an RGB ndarray, and store
            into info['heatmap_<channel>'].
            """
            heatmap_imgs = {}
            channel_names = ["x_tangent", "y_tangent", "normal"]
            # load static background once
            bg = plt.imread(IMAGE_PATH)
            h_img, w_img = bg.shape[:2]

            for idx, ch in enumerate(channel_names):
                # build patches + colors
                patches, facecolors = [], []
                for name, sensor in data.items():
                    H, W = sensor.shape[1], sensor.shape[2]
                    verts = VERTICES[name]
                    if len(verts) != 4:
                        xs, ys = zip(*verts)
                        quad = [(min(xs),min(ys)), (max(xs),min(ys)),
                                (max(xs),max(ys)), (min(xs),max(ys))]
                    else:
                        quad = verts

                    cells = subdivide_quad(quad, H, W)
                    patches += [Polygon(c, closed=True) for c in cells]

                    vals = np.abs(sensor[idx]).flatten()
                    norm = colors.Normalize(vmin=0, vmax=vals.max() or 1.0)
                    cmap = plt.get_cmap('jet')
                    facecolors += list(cmap(norm(vals)))

                # create a figure & axis with exact size of the image
                fig = plt.figure(frameon=False, figsize=(w_img/100, h_img/100), dpi=100)
                ax  = fig.add_axes([0,0,1,1])
                ax.imshow(bg)
                coll = PatchCollection(patches, match_original=True)
                coll.set_facecolor(facecolors)
                ax.add_collection(coll)
                ax.set_xlim(0, w_img)
                ax.set_ylim(h_img, 0)
                ax.axis('off')

                # render and grab as ndarray
                fig.canvas.draw()
                buf = fig.canvas.tostring_rgb()
                W_fig, H_fig = fig.canvas.get_width_height()
                img = np.frombuffer(buf, dtype='uint8').reshape(H_fig, W_fig, 3)
                heatmap_imgs[f"heatmap_{ch}"] = img
                plt.close(fig)

            return heatmap_imgs

    env.unwrapped.task = TactileHeatmapInfoWrapper(env.unwrapped.task)

    return env


@hydra.main(config_name="config", config_path="../")
def evaluate(cfg: dict):

    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)


    logger.info(f"cfg.eval_dir: {cfg.eval_dir}")


    cfg.checkpoint = find_best_checkpoint(cfg.model_dir) if cfg.load_best else find_last_checkpoint(cfg.model_dir)

    assert cfg.checkpoint is not None, "Checkpoint not found. Please check the checkpoint path."
    logger.info(f"cfg.checkpoint: {cfg.checkpoint}")

    # Make environment
    env = make_env(cfg)

    # Wrap for rendering tactile
    env = wrap_env_for_rendering_tactile(env)

    agent = TDMPC2(cfg)
    agent.load(cfg.checkpoint)

    # Evaluate
    logger.info(f"Evaluating agent on {cfg.task}")


    if cfg.eval_instruct is not False:
        cfg.instruct = cfg.eval_instruct

    if cfg.instruct:
        instruct_dir = os.path.join(dirname(__file__), "..", "instruct_rl", "instruct", "bert-base-uncased")
        cfg.instruct_path = os.path.abspath(os.path.join(instruct_dir, f"{cfg.instruct}.csv"))
    else:
        cfg.instruct_path = None

    cond_sampler = ConditionSampler(cfg) if cfg.instruct_path else None
    logger.info(f"Condition sampler: {cond_sampler}")


    if cfg.save_video:
        video_dir = os.path.join(cfg.eval_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)

    eval_logs = list()

    conditions = cond_sampler.to_list() if cond_sampler else [None]

    for i, condition in enumerate(conditions):

        for i in tqdm(range(cfg.eval_episodes), desc=f"Evaluating {i} / {len(conditions)} condition"):
            obs, done, ep_reward, t = env.reset(options={'condition': condition})[0], False, 0, 0
            step_logs = list()

            frames = list()
            tactile_frames = {}
            if cfg.save_video:
                frames.append(env.render())

            while not done:

                action = agent.act(obs, t0=t == 0)
                obs, reward, done, truncated, info = env.step(action)
                ep_reward += reward
                t += 1

                if cfg.save_video:
                    frames.append(env.render())
                    for key, value in info['tactile_heatmap'].items():
                        if key not in tactile_frames:
                            tactile_frames[key] = []
                        tactile_frames[key].append(value)

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
                imageio.mimsave(os.path.join(video_dir, f"{i}.mp4"), frames, fps=30)
                for key, value in tactile_frames.items():
                    imageio.mimsave(
                        os.path.join(video_dir, f"{key}-{i}.mp4"), value, fps=15
                    )

            step_df = pd.DataFrame(step_logs)
            info_columns = [col for col in step_df.columns if col.startswith("info_")]
            cond_columns = [col for col in step_df.columns if col.startswith("cond_")]

            eval_logs += step_logs

            eval_df = pd.DataFrame(eval_logs)
            eval_df.to_csv(os.path.join(cfg.eval_dir, f"step.csv"), index=False)

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
            eval_summary_df.to_csv(os.path.join(cfg.eval_dir, f"episode.csv"), index=False)

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
