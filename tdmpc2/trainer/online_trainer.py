from time import time
import numpy as np
import torch
from numpy.ma.core import indices
from tensordict.tensordict import TensorDict
from tdmpc2.trainer.base import Trainer


class OnlineTrainer(Trainer):
    """Trainer class for multi-env online TD-MPC2 training with manual per-env reset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step = 0
        self._ep_idx = 0
        self._start_time = time()

    def common_metrics(self):
        return dict(
            step=self._step,
            episode=self._ep_idx,
            total_time=time() - self._start_time,
        )

    def eval(self):
        """Evaluate a TD-MPC2 agent with manual per-env reset."""
        n_envs = self.env.num_envs
        ep_rewards = [[] for _ in range(n_envs)]
        ep_successes = []
        current_rewards = torch.zeros(n_envs)
        obs = self.env.reset()[0]
        episodes_finished = 0

        t0_flags = torch.ones(n_envs, dtype=torch.bool, device=obs.device)

        if self.cfg.save_video:
            self.logger.video.init(self.env, enabled=True)

        while episodes_finished < self.cfg.eval_episodes:
            action = self.agent.act(obs, t0=t0_flags, eval_mode=True)
            next_obs, reward, done, truncated, info = self.env.step(action)
            final = np.logical_or(done, truncated)

            current_rewards += reward

            for i in range(n_envs):
                if final[i]:
                    ep_rewards[i].append(current_rewards[i])
                    # ep_successes.append(info["success"][i])
                    current_rewards[i] = 0
                    episodes_finished += 1
                    # obs_i = self.env.reset(indices=[i])[0]
                    # obs[i] = obs_i[0] if isinstance(obs_i, (list, tuple)) else obs_i

                    t0_flags[i] = True
                else:
                    t0_flags[i] = False

            obs = next_obs
            if self.cfg.save_video:
                self.logger.video.record(self.env)

        if self.cfg.save_video:
            self.logger.video.save(self._step, key="results/video")

        flat_rewards = [r for env_r in ep_rewards for r in env_r]
        return dict(
            episode_reward=np.nanmean(flat_rewards),
            # episode_success=np.nanmean(ep_successes),
        )

    def to_td(self, obs, action=None, reward=None):
        if isinstance(obs, dict):
            obs = TensorDict(obs, batch_size=obs["state"].shape[:-1], device="cpu")
        else:
            obs = obs.cpu()
        if action is None:
            action = torch.full_like(self.env.rand_act(), float("nan"))
        if reward is None:
            reward = torch.full((obs.shape[0],), float("nan"))
        td = TensorDict(
            dict(
                obs=obs,
                action=action,
                reward=reward,
            ),
            batch_size=[obs.shape[0]],
        )
        return td

    def train(self):
        """Train a TD-MPC2 agent with manual per-env reset."""
        train_metrics = {}
        obs = self.env.reset()[0]

        self._tds = [self.to_td(obs)]
        t0_flags = torch.ones(self.env.num_envs, dtype=torch.bool, device=obs.device)  # 시작 시 모두 True

        while self._step <= self.cfg.steps:
            if self._step % self.cfg.eval_freq == 0:
                eval_metrics = self.eval()
                eval_metrics.update(self.common_metrics())
                self.logger.log(eval_metrics, "eval")

            if self._step > self.cfg.seed_steps:
                action = self.agent.act(obs, t0=t0_flags)
            else:
                action = self.env.rand_act()

            next_obs, reward, done, truncated, info = self.env.step(action)
            final = np.logical_or(done, truncated)
            self._tds.append(self.to_td(next_obs, action, reward))

            for i in range(self.env.num_envs):
                if final[i]:
                    # log this environment's episode

                    # log this environment's episode
                    if len(self._tds) > 1:
                        rewards = torch.stack([td["reward"] for td in self._tds[1:]], dim=0)  # [T, B]
                        episode_reward = rewards[:, i].sum()
                        episode_length = rewards.shape[0]
                    else:
                        episode_reward = torch.tensor(0.0)
                        episode_length = 0

                    # episode_reward = rewards[:, i].sum()
                    # success = info["success"][i]
                    self._ep_idx += 1

                    train_metrics.update(
                        episode_reward=episode_reward,
                        # episode_success=torch.tensor(success, dtype=torch.float32),
                    )
                    train_metrics.update(self.common_metrics())

                    self.logger.log(train_metrics, "train")
                    self.logger.log({
                        "return": episode_reward,
                        "episode_length": len(self._tds[1:]),
                        # "success": success,
                        "step": self._step,
                        "success_subtasks": info.get("success_subtasks", None),
                    }, "results")

                    td_i = torch.cat([td[i].unsqueeze(0) for td in self._tds], dim=0)
                    self._ep_idx = self.buffer.add(td_i)

                    # reset env i
                    # obs_i = self.env.reset(indices=[i])[0]
                    # obs[i] = obs_i[0] if isinstance(obs_i, (list, tuple)) else obs_i
                    self._tds = [self.to_td(obs)]  # restart tracking

                    t0_flags[i] = True  # reset flag for this env
                else:
                    t0_flags[i] = False

            obs = next_obs

            # Agent update
            if self._step >= self.cfg.seed_steps:
                if self._step == self.cfg.seed_steps:
                    print("Pretraining agent on seed data...")
                    num_updates = self.cfg.seed_steps
                else:
                    num_updates = 1
                for _ in range(num_updates):
                    _train_metrics = self.agent.update(self.buffer)
                train_metrics.update(_train_metrics)

            self._step += self.env.num_envs

        self.logger.finish(self.agent)
