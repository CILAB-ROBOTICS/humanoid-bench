import numpy as np
from gymnasium.spaces import Box
from dm_control.utils import rewards

from humanoid_bench.tasks import Task



class Rolling(Task):
    qpos0_robot = {
        "h1": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0.7 0 1 1 0 0 0",
        "h1hand": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 1.57 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.57 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.7 0 1 1 0 0 0",
        "h1touch": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 1.57 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.57 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.7 0 1 1 0 0 0",
        "g1": "0 0 0.75 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1.57 0 0 0 0 0 0 0 0 0 0 0 1.57 0 0 0 0 0 0 0 0.7 0 1 1 0 0 0"
    }
    dof = 7
    max_episode_steps = 500
    camera_name = "cam_tabletop"

    def __init__(
        self,
        robot=None,
        env=None,
        **kwargs,
    ):
        super().__init__(robot, env, **kwargs)

        if env is None:
            return

        if robot.__class__.__name__ == "G1":
            global _STAND_HEIGHT
            _STAND_HEIGHT = 1.28

    @property
    def observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.robot.dof * 2 - 1 + 15,),
            dtype=np.float64,
        )

    def get_obs(self):
        position = self._env.data.qpos.flat.copy()[: self.robot.dof]
        velocity = self._env.data.qvel.flat.copy()[: self.robot.dof - 1]
        left_hand = self.robot.left_hand_position()
        right_hand = self.robot.right_hand_position()

        left_handle = self._env.named.data.geom_xpos["roller_handle_left"]
        right_handle = self._env.named.data.geom_xpos["roller_handle_right"]
        roller_pos = self._env.named.data.geom_xpos["roller"]

        concatenated = np.concatenate((position, velocity, left_hand, right_hand,
                               roller_pos, left_handle, right_handle))

        return concatenated


    def get_reward(self):
        if not hasattr(self, "left_hand_contact_id"):
            self.right_hand_contact_id = self._env.named.data.xpos.axes.row.names.index("right_hand")
            self.left_hand_contact_id = self._env.named.data.xpos.axes.row.names.index("left_hand")
            self.right_roller_handle_id = self._env.named.data.geom_xpos.axes.row.names.index("roller_handle_right")
            self.left_roller_handle_id = self._env.named.data.geom_xpos.axes.row.names.index("roller_handle_left")

        left_hand_tool_distance = np.linalg.norm(
            self._env.named.data.site_xpos["left_hand"]
            - self._env.named.data.geom_xpos["roller_handle_left"]
        )
        right_hand_tool_distance = np.linalg.norm(
            self._env.named.data.site_xpos["right_hand"]
            - self._env.named.data.geom_xpos["roller_handle_right"]
        )

        left_hand_reward = rewards.tolerance(left_hand_tool_distance, bounds=(0, 0.0), margin=0.5)
        right_hand_reward = rewards.tolerance(
            right_hand_tool_distance, bounds=(0, 0.0), margin=0.5
        )
        hand_tool_proximity_reward = left_hand_reward * right_hand_reward


        left_hand_contact_filter = False
        right_hand_contact_filter = False

        for pair in self._env.data.contact.geom:

            if self.right_hand_contact_id in pair and self.right_roller_handle_id in pair:
                right_hand_contact_filter = True

            if self.left_hand_contact_id in pair and self.left_roller_handle_id in pair:
                left_hand_contact_filter = True

            if left_hand_contact_filter or right_hand_contact_filter:
                break

        contact_filter = left_hand_contact_filter or right_hand_contact_filter

        moving_tool_reward = rewards.tolerance(
            abs(self._env.named.data.sensordata["roller_tool_subtreelinvel"][0]),
            bounds=(0.4, 0.5),
            margin=0.2,
        )

        hand_tool_proximity_reward = hand_tool_proximity_reward
        moving_tool_reward = moving_tool_reward * 1

        reward = hand_tool_proximity_reward + moving_tool_reward

        info = {
            "hand_tool_proximity": hand_tool_proximity_reward,
            "moving_tool": moving_tool_reward,
            "contact_filter": contact_filter,
        }

        return reward, info

    def get_terminated(self):
        terminated = False

        if self._env.named.data.xpos["roller"][2] < 0.58:
            return True, {}

        return terminated, {}


