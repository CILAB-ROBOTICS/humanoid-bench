import numpy as np
from gymnasium.spaces import Box
from dm_control.utils import rewards

from humanoid_bench.tasks import Task

class Iron(Task):
    qpos0_robot = {
        "h1": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0.7 0 1 1 0 0 0",
        "h1hand": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 1.57 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.57 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.7 0 1 1 0 0 0",
        "h1touch": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 1.57 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.57 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.7 0 1 1 0 0 0",
        "g1": "0 0 0.75 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1.57 0 0 0 0 0 0 0 0 0 0 0 1.57 0 0 0 0 0 0 0 0.7 0 1 1 0 0 0"
    }
    dof = 7
    max_episode_steps = 500
    camera_name = "cam_tabletop"

    htarget_low = np.array([0, -1, 0.8])
    htarget_high = np.array([2.0, 1, 1.2])

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
            shape=(self.robot.dof * 2 - 1 + 12,),
            dtype=np.float64,
        )

    def get_obs(self):
        position = self._env.data.qpos.flat.copy()[: self.robot.dof]
        velocity = self._env.data.qvel.flat.copy()[: self.robot.dof - 1]
        left_hand = self.robot.left_hand_position()
        right_hand = self.robot.right_hand_position()
        iron_pos = self._env.named.data.geom_xpos["iron"]
        iron_handle_pos = self._env.named.data.geom_xpos["iron_handle"]

        concatenated = np.concatenate((position, velocity, left_hand, right_hand,
                               iron_pos, iron_handle_pos))

        return concatenated

    def hand_dist(self):
        #left_hand = self.robot.left_hand_position()
        right_hand = self.robot.right_hand_position()
        #iron = self._env.named.data.geom_xpos["iron"]
        iron_handle = self._env.named.data.geom_xpos["iron_handle"]
        fabric = self._env.named.data.geom_xpos["fabric"]

        #left_hand_dist = np.sqrt(np.square(left_hand - iron_handle).sum())
        right_hand_iron_handle_dist = np.sqrt(np.square(right_hand - iron_handle).sum())
        right_hand_fabric_dist = np.sqrt(np.square(right_hand - fabric).sum())
        
        return right_hand_iron_handle_dist, right_hand_fabric_dist

    def get_reward(self):

        right_hand_iron_handle_dist, right_hand_fabric_dist = self.hand_dist()
        
        small_control = rewards.tolerance(
            self.robot.actuator_forces(),
            margin=10,
            value_at_margin=0,
            sigmoid="quadratic",
        ).mean()

        small_control = (4 + small_control) / 5

        right_hand_vel = np.linalg.norm(self.robot.right_hand_velocity()[:2])

        right_hand_proximity = rewards.tolerance(
            right_hand_iron_handle_dist, bounds=(0, 0.05), margin=1.0
        )

        right_rubbing = rewards.tolerance(
            right_hand_vel, bounds=(0.5, 0.5), margin=0.5, sigmoid="linear"
        )
        
        right_iron_contact = 1 if right_hand_iron_handle_dist < 0.08 else 0
        right_fabric_contact = 1 if right_hand_fabric_dist < 0.08 else 0

        right_total = right_hand_proximity + right_iron_contact * right_fabric_contact * right_rubbing

        reward = small_control + right_total

        info = {
            "right_hand_proximity":right_hand_proximity,
            "right_iron_contact":right_iron_contact,
            "right_fabric_contact":right_fabric_contact,
            "right_rubbing":right_rubbing,
            "right_total": right_total,
        }

        return reward, info

    def get_terminated(self):

        terminated = False
        right_hand_iron_handle_dist, right_hand_fabric_dist = self.hand_dist()

        if right_hand_iron_handle_dist > 1.0:
            return True, {}

        return terminated, {}