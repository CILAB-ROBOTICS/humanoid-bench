import os

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium.spaces import Box

from humanoid_bench.tasks import Task
from humanoid_bench.mjx.flax_to_torch import TorchModel, TorchPolicy

from dm_control.utils import rewards

class Polishing(Task):
    qpos0_robot = {
        "h1": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0.7 0 1 1 0 0 0",
        "h1hand": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 1.57 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.57 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.7 0 1 1 0 0 0",
        "h1touch": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 1.57 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.57 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.7 0 1 1 0 0 0",
        "g1": "0 0 0.75 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1.57 0 0 0 0 0 0 0 0 0 0 0 1.57 0 0 0 0 0 0 0 0.7 0 1 1 0 0 0"
    }
    dof = 0
    max_episode_steps = 500
    camera_name = "cam_tabletop"
    # Below args are only used for reaching-based hierarchical control
    htarget_low = np.array([0, -1, 0.8])
    htarget_high = np.array([2.0, 1, 1.2])

    success_bar = 700

    def __init__(
        self,
        robot=None,
        env=None,
        **kwargs,
    ):
        super().__init__(robot, env, **kwargs)

        if env is None:
            return

        self.reward_dict = {
            "hand_dist": 0.1,
            "success": 1000,
            "terminate": True,
        }

        #self.goal = np.array([1.0, 0.0, 1.0])

        if robot.__class__.__name__ == "G1":
            global _STAND_HEIGHT
            _STAND_HEIGHT = 1.28

    @property
    def observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.robot.dof * 2 - 1 + 12,),
            dtype=np.float64,
        )

    def get_obs(self):
        position = self._env.data.qpos.flat.copy()[: self.robot.dof]
        velocity = self._env.data.qvel.flat.copy()[: self.robot.dof - 1]
        right_hand = self.robot.right_hand_position()
        left_hand = self.robot.left_hand_position()
        vase = self._env.named.data.xpos["vase"][:3]
        dofadr = self._env.named.model.body_dofadr["vase"]
        vase_vel = self._env.data.qvel.flat.copy()[dofadr : dofadr + 3]

        return np.concatenate((position, velocity, right_hand, left_hand, vase, vase_vel))

    def hand_dist(self):
        left_hand = self.robot.left_hand_position()
        right_hand = self.robot.right_hand_position()
        vase = self._env.named.data.xpos["vase"][:3]

        left_hand_dist = np.sqrt(np.square(left_hand - vase).sum())
        right_hand_dist = np.sqrt(np.square(right_hand - vase).sum())
        
        return left_hand_dist, right_hand_dist

    def get_reward(self):

        left_hand_dist, right_hand_dist = self.hand_dist()

        small_control = rewards.tolerance(
            self.robot.actuator_forces(),
            margin=10,
            value_at_margin=0,
            sigmoid="quadratic",
        ).mean()

        small_control = (4 + small_control) / 5

        dofadr = self._env.named.model.body_dofadr["vase"]
        vase_vel_norm = np.linalg.norm(self._env.data.qvel.flat.copy()[dofadr : dofadr + 3])
        
        vase_stability_reward = rewards.tolerance(
            vase_vel_norm,
            bounds=(0.0, 0.1),
            margin=1.0,
            sigmoid="linear",
        )

        left_hand_vel = np.linalg.norm(self.robot.left_hand_velocity()[:2])
        right_hand_vel = np.linalg.norm(self.robot.right_hand_velocity()[:2])

        left_hand_proximity = rewards.tolerance(
            left_hand_dist, bounds=(0, 0.05), margin=1.0
        )
        right_hand_proximity = rewards.tolerance(
            right_hand_dist, bounds=(0, 0.05), margin=1.0
        )

        left_rubbing = rewards.tolerance(
            left_hand_vel, bounds=(0.5, 0.5), margin=0.5, sigmoid="linear"
        )
        right_rubbing = rewards.tolerance(
            right_hand_vel, bounds=(0.5, 0.5), margin=0.5, sigmoid="linear"
        )
        
        left_contact = 1 if left_hand_dist < 0.08 else 0
        right_contact = 1 if right_hand_dist < 0.08 else 0

        left_total = left_hand_proximity + left_contact * left_rubbing
        right_total = right_hand_proximity + right_contact * right_rubbing
        vase_contact_total_reward = (left_total + right_total) / 2 

        reward = small_control + vase_contact_total_reward + vase_stability_reward
        
        info = {
            "small_control": small_control,
            "vase_contact_total_reward": vase_contact_total_reward,
            "left_proximity_reward": left_hand_proximity,
            "right_proximity_reward": right_hand_proximity,
            "left_rubbing_reward": left_rubbing,
            "right_rubbing_reward": right_rubbing,
            "left_contact": left_contact,
            "right_contact": right_contact,
            "vase_stability_reward": vase_stability_reward,
        }

        return reward, info

    def get_terminated(self):
        terminated = False
        left_hand_dist, right_hand_dist = self.hand_dist()

        if left_hand_dist > 1.5 or right_hand_dist > 1.5:
            terminated = True

        return terminated, {}

    def reset_model(self):
        return self.get_obs()

    def render(self):
        return self._env.mujoco_renderer.render(
            self._env.render_mode, self._env.camera_id, self._env.camera_name
        )
