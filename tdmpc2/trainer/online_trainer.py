from time import time
import numpy as np
import torch
from tensordict.tensordict import TensorDict

from tdmpc2.envs import make_eval_env
from tdmpc2.trainer.base import Trainer


class OnlineTrainer(Trainer):
    """Trainer class for multi-env online TD-MPC2 training with manual per-env reset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step = 0
        self._ep_idx = 0
        self._start_time = time()

        self._init_tds()

    def _init_tds(self):
        self._tds = dict()
        for i in range(self.env.num_envs):
            self._tds[i] = list()

    def common_metrics(self):
        return dict(
            step=self._step,
            episode=self._ep_idx,
            total_time=time() - self._start_time,
        )

    def eval(self):
        """Evaluate a TD-MPC2 agent with manual per-env reset."""


        if not hasattr(self, "env_eval") or self.env_eval is None:
            self.env_eval = make_eval_env(self.cfg)
        self.env_eval.reset()

        if self.cfg.save_video:
            self.logger.video.init(self.env_eval, enabled=True)

        """Evaluate a TD-MPC2 agent."""
        ep_rewards, ep_successes = [], []
        for i in range(self.cfg.eval_episodes):

            condition = self.cond_sampler.sample() if self.cond_sampler else None
            obs, done, ep_reward, t = self.env.reset(options={'condition': condition})[0], False, 0, 0
            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=(i == 0))

            while not done:
                t0 = torch.tensor([t == 0], dtype=torch.bool, device=obs.device)

                action = self.agent.act(obs.unsqueeze(0), t0=t0.unsqueeze(0), eval_mode=True)
                obs, reward, done, truncated, info = self.env_eval.step(action.squeeze(0))
                done = done or truncated
                ep_reward += reward
                t += 1
                if self.cfg.save_video:
                    self.logger.video.record(self.env_eval)
            ep_rewards.append(ep_reward)
            ep_successes.append(info["success"])
            if self.cfg.save_video:
                self.logger.video.save(self._step, key='results/video')
        return dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_success=np.nanmean(ep_successes),
        )

    def to_td(self, obs, action=None, reward=None):

        if isinstance(obs, dict):
            obs = TensorDict(obs, batch_size=obs["state"].shape[:-1], device="cpu")
        else:
            obs = obs.cpu()
        if action is None:
            action = self.env.rand_act()[0]
            action = torch.full_like(action, float("nan")).unsqueeze(0)
        if reward is None:
            reward = torch.full((1,), float("nan"))

        td = TensorDict(
            dict(
                obs=obs,
                action=action,
                reward=reward,
            ),
            batch_size=[obs.shape[0]],
        )
        return td

    def add_td(self, env_id, td):
        """Add a TD to the buffer."""
        self._tds[env_id].append(td)

    def get_tds(self, env_id):
        """Get TDs for a specific environment."""
        return self._tds[env_id]

    def reset_tds(self, env_id):
        """Reset TDs for a specific environment."""
        self._tds[env_id] = list()

    def train(self):
        """Train a TD-MPC2 agent with manual per-env reset."""
        train_metrics = {}

        for i in range(self.env.num_envs):
            self.add_td(i, self.to_td(obs[i].unsqueeze(0)))

        t0_flags = torch.ones(self.env.num_envs, dtype=torch.bool, device=obs.device)  # 시작 시 모두 True

        condition = self.cond_sampler.sample() if self.cond_sampler else None
        obs = self.env.reset(options={'condition': condition})[0]

        train_metrics, done, eval_next = {}, True, True

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

            for i in range(self.env.num_envs):
                self.add_td(i, self.to_td(next_obs[i].unsqueeze(0), action[i].unsqueeze(0), reward[i].unsqueeze(0)))

            for i in range(self.env.num_envs):
                if final[i]:
                    # log this environment's episode

                    rewards = torch.stack([td["reward"] for td in self.get_tds(i)[1:]], dim=0)  # [T, B]
                    episode_reward = rewards.sum()
                    # success = info["success"][i].item() # TODO vecenv에 맞춰서 변경

                    train_metrics.update(
                        episode_reward=episode_reward,
                        # episode_success=torch.tensor(success, dtype=torch.float32),  # TODO vecenv에 맞춰서 변경
                    )
                    train_metrics.update(self.common_metrics())

                    self.logger.log(train_metrics, "train")
                    self.logger.log({
                        "return": episode_reward,
                        "episode_length": len(self.get_tds(i)[1:]),
                        # "success": success,  # TODO vecenv에 맞춰서 변경
                        "step": self._step,
                        "success_subtasks": info.get("success_subtasks", None),
                    }, "results")

                    # concatenate all TDs for this env
                    tds = torch.cat(self.get_tds(i), dim=0)  # [T, B]
                    self._ep_idx = self.buffer.add(tds)

                    self.reset_tds(i)  # reset TDs for this env
                    self.add_td(i, self.to_td(obs[i].unsqueeze(0)))  # add new TD for this env

                    t0_flags[i] = True  # reset flag for this env
                else:
                    t0_flags[i] = False

            obs = next_obs

            # TODO reset 할 때 새로운 task로 샘플링하는 코드 추가
            # condition = self.cond_sampler.sample() if self.cond_sampler else None
            # obs = self.env.reset(options={'condition': condition})[0]

            # Agent update
            if self._step >= self.cfg.seed_steps:
                if self._step == self.cfg.seed_steps:
                    print("Pretraining agent on seed data...")
                    num_updates = self.cfg.seed_steps
                else:
                    num_updates = self.env.num_envs // 4
                for _ in range(num_updates):
                    _train_metrics = self.agent.update(self.buffer)
                train_metrics.update(_train_metrics)

            self._step += self.env.num_envs

        self.logger.finish(self.agent)
