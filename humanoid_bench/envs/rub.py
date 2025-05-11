import numpy as np
import mujoco
import gymnasium as gym
from gymnasium.spaces import Box
from dm_control.utils import rewards

from humanoid_bench.tasks import Task
from instruct_rl.create_instruct import ConditionFeature

_STAND_HEIGHT = 1.65
_MIN_FORCE = 0.0
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
        stand_reward = self._compute_stand_reward()
        small_control = self._compute_small_control_reward()
        hand_window_proximity_reward = self._compute_hand_window_proximity_reward()
        head_window_distance_reward = self._compute_head_window_distance_reward()
        rubbing_reward = self._compute_rubbing_reward()
        pressure_reward, pressure_info = self._compute_pressure_reward(self._env.condition)
        direction_reward, direction_info = self._compute_direction_reward(self._env.condition)
        frequency_reward, frequency_info = self._compute_frequency_reward(self._env.condition)
        window_contact_filter = self._check_window_contact()

        manipulation_reward = (
            0.05 * (stand_reward * small_control * head_window_distance_reward)
            + 0.15 * rubbing_reward
            + 0.15 * hand_window_proximity_reward
            + 0.5 * pressure_reward
            + 0.1 * direction_reward
            + 0.05 * frequency_reward
        )

        window_contact_total_reward = window_contact_filter * hand_window_proximity_reward
        reward = 0.7 * manipulation_reward + 0.3 * window_contact_total_reward

        return reward, {
            "stand_reward": stand_reward,
            "small_control": small_control,
            "rubbing_reward": rubbing_reward,
            "hand_window_proximity_reward": hand_window_proximity_reward,
            "pressure_reward": pressure_reward,
            "direction_reward": direction_reward,
            "frequency_reward": frequency_reward,
            "window_contact_filter": window_contact_filter,
            "window_contact_total_reward": window_contact_total_reward,
            **pressure_info,
            **direction_info,
            **frequency_info,
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
            rewards.tolerance(ldist, bounds=(0, 0.05), margin=0.2),
            rewards.tolerance(rdist, bounds=(0, 0.05), margin=0.2),
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
        if condition is not None:
            modality = condition.modality
            if modality == "embed":
                feature = [-1] * condition.get_feature_size()
                for condition in condition.conditions:
                    feature[condition.condition_type] = condition.value
                feature = np.array(feature, dtype=np.float32)
            elif modality == "vector":
                feature = condition.get_feature()
            else:
                raise ValueError(
                    "There is no such modality. "
                    "Please refer to the 'ConditionSet' class in 'tdmpc2/common/sampler.py'."
                )
            target_str = feature[ConditionFeature.strength]
        else:
            target_str = 1.0

        # calculate current maximum strength
        current_str = []
        for cid in self._get_window_pane_cid_with(body_names=["left_hand", "right_hand"]):
            contact_force = np.zeros(6)
            mujoco.mj_contactForce(self._env.model, self._env.data, cid, contact_force)
            current_str.append(contact_force[0])  # vertical force
        current_str = max(current_str) if len(current_str) > 0 else 0.0

        # normalize strength
        clipped_str = min(max(current_str, _MIN_FORCE), _MAX_FORCE)
        normalized_str = (clipped_str - _MIN_FORCE) / (_MAX_FORCE - _MIN_FORCE)

        # rewarding
        reward = 0
        curr_gap = abs(target_str - normalized_str)
        if self.prev_gap:
            reward += self.prev_gap - curr_gap
        self.prev_gap = curr_gap

        # update max/min strength
        if self.max_strength < current_str:
            self.max_strength = current_str
        elif self.min_strength > current_str:
            self.min_strength = current_str

        # update info
        info = {
            'target_str': target_str,
            'normalized_str': normalized_str,
            'current_str': current_str,
            'max_strength': self.max_strength,
            'min_strength': self.min_strength,
        }

        return reward, info

    def _compute_direction_reward(self, condition):
        # TODO: implement
        return 0, {}

    def _compute_frequency_reward(self, condition):
        # TODO: implement
        return 0, {}

    def _check_window_contact(self):
        window_pane_id = self._env.named.data.geom_xpos.axes.row.names.index("window_pane_collision")
        return any(window_pane_id in pair for pair in self._env.data.contact.geom)

    def get_terminated(self):
        if self._env.data.qpos[2] < 0.58:
            return True, {}
        return False, {}

    def reset_model(self):
        self.head_pos0 = np.copy(self._env.named.data.site_xpos["head"])
        self.max_strength = -np.inf
        self.min_strength = np.inf
        self.prev_gap = None
        return super().reset_model()
