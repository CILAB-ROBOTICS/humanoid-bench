import numpy as np
import pandas as pd
import json
import os
from os.path import join, abspath
from os import makedirs
from PIL import Image
from teleop.specs.tactile import split_tactile_data
import cv2
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import glob


def process_episode(args):
    episode_path, input_dir, output_root = args

    rel_path = episode_path.replace(input_dir, '').strip('/')
    out_path = join(output_root, rel_path)
    npz_path = join(out_path, f'{rel_path.replace("/", "_")}.npz')
    video_path = join(out_path, f'{rel_path.replace("/", "_")}.mp4')


    data_json_path = join(episode_path, 'data.json')
    if not os.path.exists(data_json_path):
        return f'Skipped (no json): {episode_path}'

    # ✅ Skip if both .npz and .mp4 already exist
    if os.path.exists(npz_path) and os.path.exists(video_path):
        return f'Skipped (already exists): {rel_path}'

    try:
        with open(data_json_path, 'r') as f:
            content = json.load(f)
    except json.JSONDecodeError as e:
        return f'Skipped (invalid json): {data_json_path} - {str(e)}'

    info = content['info']
    data = content['data']

    proprio_rows, action_rows, rgb_rows = [], [], []
    tactile_rows = dict()

    for row in data:
        joint_row, action_row = dict(), dict()

        for joint_set_key, joint_set_val in info['joint_names'].items():
            states = row['states']
            if (joint_set_key not in states) or (states[joint_set_key]) is None:
                continue

            qpos_vals = states[joint_set_key]['qpos']
            qvel_vals = states[joint_set_key]['qvel']
            for joint_name, qpos_val, qvel_val in zip(joint_set_val, qpos_vals, qvel_vals):
                joint_row[f'state.{joint_set_key}.{joint_name}.qpos'] = qpos_val
                joint_row[f'state.{joint_set_key}.{joint_name}.qvel'] = qvel_val

                action_row[f'action.{joint_set_key}.{joint_name}.qpos'] = qpos_val
                action_row[f'action.{joint_set_key}.{joint_name}.qvel'] = qvel_val

        tactiles = row['tactiles']
        for tactile_set_key, tactile_set_val in tactiles.items():
            tactile_path = abspath(join(episode_path, tactile_set_val))
            tactile_npy = np.load(tactile_path)
            tactile_row = tactile_npy.flatten()
            tactile_dict = split_tactile_data(tactile_row)
            tactile_dict = {f'tactile.{tactile_set_key}.{k}': v for k, v in tactile_dict.items()}
            for key, value in tactile_dict.items():
                tactile_rows.setdefault(key, []).append(value)

        camera = row['colors']['color_0']
        rgb_path = abspath(join(episode_path, camera))
        rgb_image = Image.open(rgb_path).convert('RGB')
        rgb_array = np.array(rgb_image)
        rgb_rows.append(rgb_array)

        proprio_rows.append(joint_row)
        action_rows.append(action_row)

    rel_path = episode_path.replace(input_dir, '').strip('/')
    out_path = join(output_root, rel_path)
    makedirs(out_path, exist_ok=True)

    proprio_df = pd.DataFrame(proprio_rows)
    action_df = pd.DataFrame(action_rows)

    with open(join(out_path, 'proprio.txt'), 'w') as f:
        f.write('\n'.join(proprio_df.columns.tolist()))
    with open(join(out_path, 'action.txt'), 'w') as f:
        f.write('\n'.join(action_df.columns.tolist()))

    tactile_rows = {k: np.stack(v) for k, v in tactile_rows.items()}

    # Save video
    video_path = join(out_path, f'{rel_path.replace("/", "_")}.mp4')
    height, width, _ = rgb_rows[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
    for frame in rgb_rows:
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video_writer.release()

    npz_dict = {
        'proprio': proprio_df.to_numpy(),
        'action': action_df.to_numpy(),
        **tactile_rows
    }

    npz_path = join(out_path, f'{rel_path.replace("/", "_")}.npz')
    np.savez(npz_path, **npz_dict)

    return f'✅ Done: {rel_path}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Root directory containing episodes')
    parser.add_argument('--output_dir', required=True, help='Directory to save processed outputs')
    parser.add_argument('--n_proc', type=int, default=min(cpu_count(), 16), help='Number of parallel processes')
    args = parser.parse_args()

    input_dir = abspath(args.input_dir)
    output_dir = abspath(args.output_dir)
    makedirs(output_dir, exist_ok=True)

    print(f'Input directory: {input_dir}, Output directory: {output_dir}')
    episode_dirs = glob.glob(join(input_dir, '**/episode_*'), recursive=True)
    episode_dirs = [d for d in episode_dirs if os.path.isdir(d)]

    print(f'Found {len(episode_dirs)} episode directories')

    tasks = [(ep_path, input_dir, output_dir) for ep_path in episode_dirs]

    with Pool(processes=args.n_proc) as pool:
        for result in tqdm(pool.imap_unordered(process_episode, tasks), total=len(tasks), desc="Processing episodes"):
            if result:
                print(result)
