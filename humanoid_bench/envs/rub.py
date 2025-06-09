from cgitb import small

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium.spaces import Box
from dm_control.utils import rewards

from humanoid_bench.tasks import Task
from instruct_rl.create_instruct import ConditionFeature

_STAND_HEIGHT = 1.65
_MIN_FORCE = 100.0
_MAX_FORCE = 2000.0


def _is_body_descendant(model, body_id, target_name):
    cur = body_id
    while True:
        name = model.body(cur).name

        if name == target_name:
            return True

        parent = model.body_parentid[cur]
        if parent <= 0:
            return False
        cur = parent


class Rub(Task):
    qpos0_robot = {
        "h1hand": """
            0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            """,
        "h1touch": """
            0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            """,
        "g1": """
            0 0 0.75
            1 0 0 0
            0 0 0 0 0 0
            0 0 0 0 0 0
            0
            0 0 0 0 -1.57
            0 0 0 0 0 0 0
            0 0 0 0 1.57
            0 0 0 0 0 0 0
        """
    }
    frame_skip = 10
    camera_name = "cam_hand_visible"

    success_bar = 650

    def __init__(self, robot=None, env=None, **kwargs):
        super().__init__(robot, env, **kwargs)
        if robot.__class__.__name__ == "G1":
            global _STAND_HEIGHT
            _STAND_HEIGHT = 1.28

    @property
    def observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.robot.dof * 2 - 1 + self.dof * 2,),
            dtype=np.float64,
        )

    def get_reward(self):
        small_control = self._compute_small_control_reward() * 0.05
        hand_window_proximity_reward = self._compute_hand_window_proximity_reward() * 1000
        rubbing_reward = self._compute_rubbing_reward() * 0.15

        if self._env.condition is not None:
            pressure_reward, pressure_info = self._compute_pressure_reward(self._env.condition)
        else:
            pressure_reward, pressure_info = 0.0, {}

        reward = (small_control + hand_window_proximity_reward + rubbing_reward
                  + pressure_reward)

        return reward, {
            "small_control": small_control,
            "hand_window_proximity": hand_window_proximity_reward,
            "rubbing": rubbing_reward,
            "pressure": pressure_reward,
            "window_contact_filter": self._check_window_contact(),
            **pressure_info,
        }

    def _get_window_pane_cid_with(self, body_names=[]):
        window_pane_id = self._env.named.data.geom_xpos.axes.row.names.index("window_pane_collision")

        cids = []
        for cid in range(self._env.data.ncon):
            c = self._env.data.contact[cid]
            geom1, geom2 = c.geom1, c.geom2

            if geom1 != window_pane_id and geom2 != window_pane_id:
                continue

            if len(body_names) == 0:
                cids.append(cid)

            other_geom = geom2 if geom1 == window_pane_id else geom1
            body_id = self._env.model.geom_bodyid[other_geom]

            for body_name in body_names:
                if _is_body_descendant(self._env.model, body_id, body_name):
                    cids.append(cid)
                    break
        return cids

    def _compute_stand_reward(self):
        standing = rewards.tolerance(
            self.robot.head_height(),
            bounds=(_STAND_HEIGHT, float("inf")),
            margin=_STAND_HEIGHT / 4,
        )
        upright = rewards.tolerance(
            self.robot.torso_upright(),
            bounds=(0.9, float("inf")),
            sigmoid="linear",
            margin=1.9,
            value_at_margin=0,
        )
        return standing * upright

    def _compute_small_control_reward(self):
        ctrl = self.robot.actuator_forces()
        reward = rewards.tolerance(ctrl, margin=10, value_at_margin=0, sigmoid="quadratic").mean()
        return (4 + reward) / 5

    def _compute_hand_window_proximity_reward(self):
        ldist = np.linalg.norm(
            self._env.named.data.site_xpos["left_hand"] - self._env.named.data.geom_xpos["window_pane_collision"]
        )
        rdist = np.linalg.norm(
            self._env.named.data.site_xpos["right_hand"] - self._env.named.data.geom_xpos["window_pane_collision"]
        )
        return min([
            rewards.tolerance(ldist, bounds=(0, 0.1), margin=0.5),
            rewards.tolerance(rdist, bounds=(0, 0.1), margin=0.5),
        ])

    def _compute_head_window_distance_reward(self):
        return rewards.tolerance(
            np.linalg.norm(self._env.named.data.site_xpos["head"] - self.head_pos0),
            bounds=(0.4, 0.4),
            margin=0.1,
        )

    def _compute_rubbing_reward(self):
        lvel = np.linalg.norm(self.robot.left_hand_velocity()[:2])
        rvel = np.linalg.norm(self.robot.right_hand_velocity()[:2])
        return rewards.tolerance(
            max(lvel, rvel),
            bounds=(0.5, 0.5),
            margin=0.5,
            sigmoid="linear",
        )

    def _compute_pressure_reward(self, condition):
        strength_cond = condition.get_strength()

        # rescale [0, 1] to [_MIN_FORCE, _MAX_FORCE]
        # strength_target = strength_cond.value * (_MAX_FORCE - _MIN_FORCE) + _MIN_FORCE
        strength_target = strength_cond.value
        strength_target_denormalized = strength_cond.value * (_MAX_FORCE - _MIN_FORCE) + _MIN_FORCE
        # calculate current maximum strength
        strengths = []
        for cid in self._get_window_pane_cid_with(body_names=["left_hand", "right_hand"]):
            contact_force = np.zeros(6)
            mujoco.mj_contactForce(self._env.model, self._env.data, cid, contact_force)
            strengths.append(contact_force[0])

        is_window_contact = self._check_window_contact()

        strength_current = max(strengths) if len(strengths) > 0 else 0.0
        strength_current_denormalized = strength_current
        strength_current = (strength_current_denormalized - _MIN_FORCE) / (_MAX_FORCE - _MIN_FORCE)

        reward = rewards.tolerance(
            strength_current,
            bounds=(strength_target, strength_target),
            margin=0.2,
            sigmoid="linear",
        )
        reward = reward if is_window_contact else 0.0

        # update info
        info = {
            'strength_condition': strength_cond.value,
            'strength_reward': reward,
            'strength_current': strength_current,
            'strength_current_denorm': strength_current_denormalized,
            'strength_target': strength_target,
            'strength_target_denorm': strength_target_denormalized,
            'strength_error': abs(strength_target - strength_current),
            'strength_min': _MIN_FORCE,
            'strength_max': _MAX_FORCE,
        }

        return reward, info

    def _check_window_contact(self):
        window_pane_id = self._env.named.data.geom_xpos.axes.row.names.index("window_pane_collision")
        return any(window_pane_id in pair for pair in self._env.data.contact.geom)

    def get_terminated(self):
        if self._env.data.qpos[2] < 0.58:
            return True, {}
        return False, {}

    def reset_model(self):
        self.head_pos0 = np.copy(self._env.named.data.site_xpos["head"])
        return super().reset_model()
