import numpy as np
import torch
import torch.nn.functional as F

from tdmpc2.common import math
from tdmpc2.common.scale import RunningScale
from tdmpc2.common.world_model import WorldModel


class TDMPC2:
    """
    TD-MPC2 agent. Implements training + inference.
    Can be used for both single-task and multi-task experiments,
    and supports both state and pixel observations.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model = WorldModel(cfg).to(self.device)
        self.optim = torch.optim.Adam(
            [
                {
                    "params": self.model._encoder.parameters(),
                    "lr": self.cfg.lr * self.cfg.enc_lr_scale,
                },
                {"params": self.model._dynamics.parameters()},
                {"params": self.model._reward.parameters()},
                {"params": self.model._Qs.parameters()},
                {
                    "params": self.model._task_emb.parameters()
                    if self.cfg.multitask
                    else []
                },
            ],
            lr=self.cfg.lr,
        )
        self.pi_optim = torch.optim.Adam(
            self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5
        )
        self.model.eval()
        self.scale = RunningScale(cfg)
        self.cfg.iterations += 2 * int(
            cfg.action_dim >= 20
        )  # Heuristic for large action spaces
        self.discount = (
            torch.tensor(
                [self._get_discount(ep_len) for ep_len in cfg.episode_lengths],
                device="cuda",
            )
            if self.cfg.multitask
            else self._get_discount(cfg.episode_length)
        )

    def _get_discount(self, episode_length):
        """
        Returns discount factor for a given episode length.
        Simple heuristic that scales discount linearly with episode length.
        Default values should work well for most tasks, but can be changed as needed.

        Args:
                episode_length (int): Length of the episode. Assumes episodes are of fixed length.

        Returns:
                float: Discount factor for the task.
        """
        frac = episode_length / self.cfg.discount_denom
        return min(
            max((frac - 1) / (frac), self.cfg.discount_min), self.cfg.discount_max
        )

    def save(self, fp):
        """
        Save state dict of the agent to filepath.

        Args:
                fp (str): Filepath to save state dict to.
        """
        torch.save({"model": self.model.state_dict()}, fp)

    def load(self, fp):
        """
        Load a saved state dict from filepath (or dictionary) into current agent.

        Args:
                fp (str or dict): Filepath or state dict to load.
        """
        state_dict = fp if isinstance(fp, dict) else torch.load(fp)
        self.model.load_state_dict(state_dict["model"])

    @torch.no_grad()
    def act(self, obs, t0, eval_mode=False, task=None):
        """
        Select an action by planning in the latent space of the world model.

        Args:
                obs (torch.Tensor): Observation from the environment.
                t0 (bool): Whether this is the first observation in the episode.
                eval_mode (bool): Whether to use the mean of the action distribution.
                task (int): Task index (only used for multi-task experiments).

        Returns:
                torch.Tensor: Action to take in the environment.
        """
        obs = obs.to(self.device, non_blocking=True)
        if task is not None:
            task = torch.tensor([task], device=self.device)
        z = self.model.encode(obs, task)

        if self.cfg.mpc:
            a = self.plan(z, t0=t0, eval_mode=eval_mode, task=task)
        else:
            a = self.model.pi(z, task)[int(not eval_mode)][0]
        return a.cpu()

    @torch.no_grad()
    def _estimate_value(self, z, actions, task):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1

        for t in range(self.cfg.horizon):
            reward = math.two_hot_inv(self.model.reward(z, actions[:, t], task), self.cfg)
            z = self.model.next(z, actions[:, t], task)
            G += discount * reward
            discount *= (
                self.discount[torch.tensor(task)]
                if self.cfg.multitask
                else self.discount
            )
        return G + discount * self.model.Q(
            z, self.model.pi(z, task)[1], task, return_type="avg"
        )

    @torch.no_grad()
    def plan(self, z, t0, eval_mode=False, task=None):
        """
        Plan a sequence of actions using the learned world model.

        Args:
                z (torch.Tensor): Latent state from which to plan.
                t0 (bool): Whether this is the first observation in the episode.
                eval_mode (bool): Whether to use the mean of the action distribution.
                task (Torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
                torch.Tensor: Action to take in the environment.
        """
        # Sample policy trajectories
        if self.cfg.num_pi_trajs > 0:

            n_envs = z.shape[0]

            pi_actions = torch.empty(
                n_envs,
                self.cfg.horizon,
                self.cfg.num_pi_trajs,
                self.cfg.action_dim,
                device=self.device,
            )

            # (4, 512) -> (4, self.cfg.num_pi_trajs, 512)
            _z = z.unsqueeze(1).repeat(1, self.cfg.num_pi_trajs, 1)

            for t in range(self.cfg.horizon - 1):
                pi_actions[:, t, :, :] = self.model.pi(_z, task)[1]
                _z = self.model.next(_z, pi_actions[:, t, :, :], task)

            # print('pi_actions', pi_actions.shape, 'pi', self.model.pi(_z, task)[1].shape)
            pi_actions[:, -1, :, :] = self.model.pi(_z, task)[1]

        z = z.unsqueeze(1).repeat(1, self.cfg.num_samples, 1)

        mean = torch.zeros(n_envs, self.cfg.horizon, self.cfg.action_dim, device=self.device)
        std = self.cfg.max_std * torch.ones(
            n_envs,
            self.cfg.horizon,
            self.cfg.action_dim,
            device=self.device
        )

        if hasattr(self, "_prev_mean"):
            B, H, D = mean.shape
            shifted = self._prev_mean[:, 1:, :]  # [B, H-1, D]
            mask = (~t0).view(B, 1, 1)  # [B, 1, 1]
            mean[:, :-1] = torch.where(mask, shifted, mean[:, :-1])

        actions = torch.empty(
            n_envs,
            self.cfg.horizon,
            self.cfg.num_samples,
            self.cfg.action_dim,
            device=self.device,
        )

        if self.cfg.num_pi_trajs > 0:
            actions[:, :, :self.cfg.num_pi_trajs, :] = pi_actions

        # Iterate MPPI
        for _ in range(self.cfg.iterations):
            # Sample actions

            actions[:, :, self.cfg.num_pi_trajs:] = (
                mean.unsqueeze(2)
                + std.unsqueeze(2)
                * torch.randn(
                    n_envs,
                    self.cfg.horizon,
                    self.cfg.num_samples - self.cfg.num_pi_trajs,
                    self.cfg.action_dim,
                    device=std.device,
                )
            ).clamp(-1, 1)


            if self.cfg.multitask:
                actions = actions * self.model._action_masks[task]

            value = self._estimate_value(z, actions, task).nan_to_num_(0)
            value = value.squeeze(-1)

            # elite_idxs: [B, top_k]
            elite_idxs = torch.topk(value, self.cfg.num_elites, dim=1).indices  # [4, 64]

            # elite_value: gather along candidate dim (dim=1)
            elite_value = torch.gather(value, dim=1, index=elite_idxs) # [4, 64]
            elite_value = elite_value.unsqueeze(-1)  # [4, 1, 64]

            # actions: [B, A, C, D]
            # expand elite_idxs for gather: [B, 1, top_k, 1] → broadcast to [B, A, top_k, D]
            expanded_idxs = elite_idxs[:, None, :, None].expand(-1, actions.size(1), -1,
                                                                actions.size(3))  # [4, 3, 64, 4]

            # elite_actions: gather along candidate dim (dim=2)
            elite_actions = torch.gather(actions, dim=2, index=expanded_idxs)  # [4, 3, 64, 4]

            # elite_value torch.Size([4, 64]) elite_actions torch.Size([4, 3, 64, 4])
            # Update parameters
            # elite_value: [4, 64]
            max_value = elite_value.max(dim=1)[0]  # [4, 1]
            max_value = max_value.unsqueeze(1)  # [4, 1, 1]

            # elite_value: [4, 64, 1] - max_value: [4, 1] -> broadcast to [4, 64, 1]
            score = torch.exp(self.cfg.temperature * (elite_value - max_value))
            score /= score.sum(dim=1, keepdim=True)

            mean = torch.sum(score.unsqueeze(1) * elite_actions, dim=2) / (
                score.sum(dim=1, keepdim=True) + 1e-9
            )

            std = torch.sqrt(
                torch.sum(
                    score.unsqueeze(1) * (elite_actions - mean.unsqueeze(2)) ** 2, dim=2
                )
                / (score.sum(dim=1, keepdim=True) + 1e-9)
            ).clamp_(self.cfg.min_std, self.cfg.max_std)

            if self.cfg.multitask:
                mean = mean * self.model._action_masks[task]
                std = std * self.model._action_masks[task]

        score = score.squeeze(-1)  # [4, 64]
        sampled_idxs = torch.multinomial(score, num_samples=1)  # [4, 1]
        sampled_idxs = sampled_idxs.unsqueeze(1).expand(-1, 3, -1).unsqueeze(-1).expand(-1, -1, -1, 61)  # [4, 3, 1, 61]
        actions = torch.gather(elite_actions, dim=2, index=sampled_idxs).squeeze(2)  # [4, 3, 61]

        self._prev_mean = mean

        a, std = actions[:, 0], std[:, 0]

        if not eval_mode:
            a += std * torch.randn(self.cfg.action_dim, device=std.device)

        return a.clamp_(-1, 1)

    def update_pi(self, zs, task):
        """
        Update policy using a sequence of latent states.

        Args:
                zs (torch.Tensor): Sequence of latent states.
                task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
                float: Loss of the policy update.
        """
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)
        _, pis, log_pis, _ = self.model.pi(zs, task)
        qs = self.model.Q(zs, pis, task, return_type="avg")
        self.scale.update(qs[0])
        qs = self.scale(qs)

        # Loss is a weighted sum of Q-values
        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1, 2)) * rho).mean()
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model._pi.parameters(), self.cfg.grad_clip_norm
        )
        self.pi_optim.step()
        self.model.track_q_grad(True)

        return pi_loss.item()

    @torch.no_grad()
    def _td_target(self, next_z, reward, task):
        """
        Compute the TD-target from a reward and the observation at the following time step.

        Args:
                next_z (torch.Tensor): Latent state at the following time step.
                reward (torch.Tensor): Reward at the current time step.
                task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
                torch.Tensor: TD-target.
        """
        pi = self.model.pi(next_z, task)[1]
        discount = (
            self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
        )
        return reward + discount * self.model.Q(
            next_z, pi, task, return_type="min", target=True
        )

    def update(self, buffer):
        """
        Main update function. Corresponds to one iteration of model learning.

        Args:
                buffer (common.buffer.Buffer): Replay buffer.

        Returns:
                dict: Dictionary of training statistics.
        """
        obs, action, reward, task = buffer.sample()

        # Compute targets
        with torch.no_grad():
            next_z = self.model.encode(obs[1:], task)
            td_targets = self._td_target(next_z, reward, task)

        # Prepare for update
        self.optim.zero_grad(set_to_none=True)
        self.model.train()

        # Latent rollout
        zs = torch.empty(
            self.cfg.horizon + 1,
            self.cfg.batch_size,
            self.cfg.latent_dim,
            device=self.device,
        )
        z = self.model.encode(obs[0], task)
        zs[0] = z
        consistency_loss = 0
        for t in range(self.cfg.horizon):
            z = self.model.next(z, action[t], task)
            consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho**t
            zs[t + 1] = z

        # Predictions
        _zs = zs[:-1]
        qs = self.model.Q(_zs, action, task, return_type="all")
        reward_preds = self.model.reward(_zs, action, task)

        # Compute losses
        reward_loss, value_loss = 0, 0
        for t in range(self.cfg.horizon):
            reward_loss += (
                math.soft_ce(reward_preds[t], reward[t], self.cfg).mean()
                * self.cfg.rho**t
            )
            for q in range(self.cfg.num_q):
                value_loss += (
                    math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean()
                    * self.cfg.rho**t
                )
        consistency_loss *= 1 / self.cfg.horizon
        reward_loss *= 1 / self.cfg.horizon
        value_loss *= 1 / (self.cfg.horizon * self.cfg.num_q)
        total_loss = (
            self.cfg.consistency_coef * consistency_loss
            + self.cfg.reward_coef * reward_loss
            + self.cfg.value_coef * value_loss
        )

        # Update model
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.grad_clip_norm
        )
        self.optim.step()

        # Update policy
        pi_loss = self.update_pi(zs.detach(), task)

        # Update target Q-functions
        self.model.soft_update_target_Q()

        # Return training statistics
        self.model.eval()
        return {
            "consistency_loss": float(consistency_loss.mean().item()),
            "reward_loss": float(reward_loss.mean().item()),
            "value_loss": float(value_loss.mean().item()),
            "pi_loss": pi_loss,
            "total_loss": float(total_loss.mean().item()),
            "grad_norm": float(grad_norm),
            "pi_scale": float(self.scale.value),
        }
