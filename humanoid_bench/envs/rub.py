import cv2
import numpy as np
import mujoco
from gymnasium.spaces import Box
from dm_control.utils import rewards
from sympy.strategies.branch import condition

from humanoid_bench.tasks import Task

_STAND_HEIGHT = 1.65
_MIN_FORCE = 0.0
_MAX_FORCE = 600.0


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


    def __init__(self, robot=None, env=None, **kwargs):
        super().__init__(robot, env, **kwargs)
        if robot.__class__.__name__ == "G1":
            global _STAND_HEIGHT
            _STAND_HEIGHT = 1.28

        self.curr_progress = 0.0

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
        hand_window_proximity_reward = self._compute_hand_window_proximity_reward()
        rubbing_reward = self._compute_rubbing_reward() * 0.15

        if self._env.condition is not None:
            pressure_reward, pressure_info = self._compute_pressure_reward(self._env.condition)

            if abs(self._env.condition.get_strength().value - pressure_info['strength_current']) < 0.05 and \
                    self._check_window_contact():
                self.curr_progress = min(1.0, self.curr_progress + 0.01)

        else:
            pressure_reward, pressure_info = 0.0, {}

        reward = (small_control + hand_window_proximity_reward + rubbing_reward
                  + pressure_reward)

        info =  {
            "small_control": small_control,
            "hand_window_proximity": hand_window_proximity_reward,
            "rubbing": rubbing_reward,
            "pressure": pressure_reward,
            "window_contact_filter": self._check_window_contact(),
            **pressure_info,
            'progress': self.curr_progress,
        }

        self.latest_info = info

        return reward, info
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
        # clip strength_current to [_MIN_FORCE, _MAX_FORCE]

        strength_clipped = np.clip(strength_current_denormalized, _MIN_FORCE, _MAX_FORCE)
        strength_current = (strength_clipped - _MIN_FORCE) / (_MAX_FORCE - _MIN_FORCE)

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
        self.curr_progress = 0.0
        return super().reset_model()

    def _draw_progress_bar(self, img):
        height, width, _ = img.shape

        # --- 설정값 ---
        padding = int(width * 0.05)  # 이미지 왼쪽 여백
        bar_width = int(width * 0.03)  # 프로그레스바 너비
        bar_top = int(height * 0.15)  # 바의 상단 위치
        bar_bottom = int(height * 0.85)  # 바의 하단 위치
        bar_height = bar_bottom - bar_top  # 전체 바 높이

        # --- 흰색 테두리 ---
        border_color = (255, 255, 255)
        border_thickness = 1
        cv2.rectangle(
            img,
            (padding, bar_top),
            (padding + bar_width, bar_bottom),
            border_color,
            thickness=border_thickness
        )

        # --- 녹색 프로그레스 바 ---
        fill_height = int(bar_height * self.curr_progress)
        fill_top = bar_bottom - fill_height
        fill_color = (0, 255, 0)
        cv2.rectangle(
            img,
            (padding + border_thickness, fill_top),
            (padding + bar_width - border_thickness, bar_bottom),
            fill_color,
            thickness=-1
        )

        # --- 백분율 숫자 ---
        percent = int(self.curr_progress * 100)
        text = f"{percent}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = padding + bar_width + 10
        text_y = fill_top + text_size[1] // 2

        cv2.putText(
            img,
            text,
            (text_x, max(text_y, bar_top + text_size[1])),  # 상단 초과 방지
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA
        )

        return img

    def _draw_force_bar(self, img):
        height, width, _ = img.shape

        if not hasattr(self, 'latest_info'):
            return img
        if 'strength_current' not in self.latest_info:
            return img

        strength_current = self.latest_info['strength_current']
        strength_target = self.latest_info['strength_target']

        # --- 설정값 ---
        padding = int(width * 0.05)  # 오른쪽 여백
        bar_width = int(width * 0.03)
        bar_top = int(height * 0.15)
        bar_bottom = int(height * 0.85)
        bar_height = bar_bottom - bar_top
        bar_left = width - padding - bar_width
        bar_right = width - padding

        # --- 흰색 테두리 ---
        border_color = (255, 255, 255)
        border_thickness = 1
        cv2.rectangle(
            img,
            (bar_left, bar_top),
            (bar_right, bar_bottom),
            border_color,
            thickness=border_thickness
        )

        # --- 빨간색 current 프로그레스 바 ---
        fill_height = int(bar_height * strength_current)
        fill_top = bar_bottom - fill_height
        fill_color = (0, 0, 255)  # 빨간색
        cv2.rectangle(
            img,
            (bar_left + border_thickness, fill_top),
            (bar_right - border_thickness, bar_bottom),
            fill_color,
            thickness=-1
        )

        # --- 흰색 target 선 ---
        target_y = int(bar_bottom - bar_height * strength_target)
        cv2.line(
            img,
            (bar_left, target_y),
            (bar_right, target_y),
            (255, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA
        )

        # --- 백분율 숫자 ---
        percent = int(strength_current * 100)
        text = f"{percent}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = bar_left - text_size[0] - 5  # 막대 왼쪽에 위치
        text_y = fill_top + text_size[1] // 2
        text_y = max(text_y, bar_top + text_size[1])  # 상단 초과 방지

        cv2.putText(
            img,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA
        )

        return img


    def render(self):
        img = self._env.mujoco_renderer.render(
            self._env.render_mode, self._env.camera_id, self._env.camera_name
        )

        img = img.copy()

        if self._env.condition is not None:
            img = self._draw_progress_bar(img)
            img = self._draw_force_bar(img)


        return img