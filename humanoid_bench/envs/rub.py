import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from dm_control.utils import rewards

from humanoid_bench.tasks import Task

_STAND_HEIGHT = 1.65


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
        self.window_pane_id = self._env.named.data.geom_xpos.axes.row.names.index(
            "window_pane_collision"
        )

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
        stand_reward = standing * upright
        small_control = rewards.tolerance(
            self.robot.actuator_forces(),
            margin=10,
            value_at_margin=0,
            sigmoid="quadratic",
        ).mean()
        small_control = (4 + small_control) / 5

        left_hand_window_distance = np.linalg.norm(
            self._env.named.data.site_xpos["left_hand"] - self._env.named.data.geom_xpos["window_pane_collision"]
        )
        right_hand_window_distance = np.linalg.norm(
            self._env.named.data.site_xpos["right_hand"] - self._env.named.data.geom_xpos["window_pane_collision"]
        )
        hand_window_proximity_reward = min(
            [
                rewards.tolerance(left_hand_window_distance, bounds=(0, 0.05), margin=0.2),
                rewards.tolerance(right_hand_window_distance, bounds=(0, 0.05), margin=0.2),
            ]
        )

        left_hand_vel = np.linalg.norm(self.robot.left_hand_velocity()[:2])
        right_hand_vel = np.linalg.norm(self.robot.right_hand_velocity()[:2])
        rubbing_reward = rewards.tolerance(
            max(left_hand_vel, right_hand_vel),
            bounds=(0.5, 0.5),
            margin=0.5,
            sigmoid="linear",
        )

        head_window_distance_reward = rewards.tolerance(
            np.linalg.norm(self._env.named.data.site_xpos["head"] - self.head_pos0),
            bounds=(0.4, 0.4),
            margin=0.1,
        )

        manipulation_reward = (
            0.2 * (stand_reward * small_control * head_window_distance_reward)
            + 0.4 * rubbing_reward
            + 0.4 * hand_window_proximity_reward
        )

        window_contact_filter = 0
        for pair in self._env.data.contact.geom:
            if self.window_pane_id in pair:
                window_contact_filter = 1
                break
        window_contact_total_reward = window_contact_filter * hand_window_proximity_reward
        reward = 0.5 * manipulation_reward + 0.5 * window_contact_total_reward

        return reward, {
            "stand_reward": stand_reward,
            "small_control": small_control,
            "rubbing_reward": rubbing_reward,
            "hand_window_proximity_reward": hand_window_proximity_reward,
            "window_contact_filter": window_contact_filter,
            "window_contact_total_reward": window_contact_total_reward,
        }

    def get_terminated(self):
        if self._env.data.qpos[2] < 0.58:
            return True, {}
        # if self._env.named.data.xpos["window_wiping_tool"][2] < 0.58:
        #     return True, {}
        return False, {}

    def reset_model(self):
        self.head_pos0 = np.copy(self._env.named.data.site_xpos["head"])
        return super().reset_model()
