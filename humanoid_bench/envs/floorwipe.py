import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from dm_control.utils import rewards

from humanoid_bench.tasks import Task

_STAND_HEIGHT = 1.65


class FloorWipe(Task):
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
    camera_name = "cam_hurdle"
    dof = 15

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
            shape=(self.robot.dof * 2 - 2 + self.dof * 2 - 2,),
            dtype=np.float64,
        )

    def get_reward(self):
        self.floor_pane_id = self._env.named.data.geom_xpos.axes.row.names.index(
            "floor"
        )
        self.window_wipe_id = self._env.named.data.geom_xpos.axes.row.names.index(
            "window_wipe_collision"
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
        small_control = (4 + small_control) / 5 * 0.1

        ground_y = self._env.named.data.geom_xpos["floor", "y"]

        floor_contact_reward = np.min(
            [
                rewards.tolerance(
                    self._env.named.data.site_xpos[site_name, "y"],
                    bounds=(ground_y, ground_y + 0.4),
                    margin=0.4,
                    sigmoid="linear",
                )
                for site_name in [
                    "wipe_contact_site_a",
                    "wipe_contact_site_b",
                    "wipe_contact_site_c",
                    "wipe_contact_site_d",
                    "wipe_contact_site_e",
                ]
            ]
        )

        floor_contact_filter = 0
        for pair in self._env.data.contact.geom:
            if (
                self.floor_pane_id in pair and self.window_wipe_id in pair
            ):  # if has hand
                floor_contact_filter = 1
                break

        #
        moving_wipe_reward = rewards.tolerance(
            abs(self._env.named.data.sensordata["window_wiping_tool_subtreelinvel"][2]),
            bounds=(0.3, 0.7),
            margin=0.5,
        )

        manipulation_reward = (
            0.2 * (stand_reward * small_control)
        ) * 0.1
        moving_wipe_reward = floor_contact_filter * moving_wipe_reward * 5
        reward = manipulation_reward + moving_wipe_reward

        return reward, {
            "manipulation_reward": manipulation_reward,
            "moving_wipe_reward": moving_wipe_reward,
        }

    def get_terminated(self):
        self.floor_pane_id = self._env.named.data.geom_xpos.axes.row.names.index(
            "floor"
        )

        if self._env.data.qpos[2] < 0.58:
            return True, {}
        return False, {}

    def reset_model(self):
        self.head_pos0 = np.copy(self._env.named.data.site_xpos["head"])
        return super().reset_model()
