import torch
import numpy as np
from gym import spaces
from .task_base import BaseTask
from ..termination_conditions import MyTerminationV1
from ..reward_functions import MyRewardFunctionV1
from ..utils.adaptor_v1 import NetworkAdaptorV1


def _rot_world_to_body_xy(dx, dy, yaw):
    """
    用 yaw 做 2D 机体系旋转（机头为 +x，右侧为 +y）
    body = R(-yaw) * world
    """
    c, s = np.cos(yaw), np.sin(yaw)
    bx =  c * dx + s * dy
    by = -s * dx + c * dy
    return bx, by

class MyCombatTaskV1(BaseTask):
    def __init__(self, config):
        super().__init__(config)
        self.reward_functions = [MyRewardFunctionV1(self.config)]
        self.termination_conditions = [MyTerminationV1(self.config)]
        self.adaptor = NetworkAdaptorV1(self.config)
        self.truncated = False
        self.use_baseline = getattr(self.config, 'use_baseline', False)

    @property
    def num_agents(self) -> int:
        return 2 if not self.use_baseline else 1

    def load_variables(self):
        self.state_var = [

        ]
        self.action_var = [

        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(15,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.Box(
            low=np.array([0.4, -1, -1, -1], dtype=np.float32),
            high=np.array([1.0, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

    def get_obs(self, env, agent_id):
        agent = env.agents[agent_id]
        my_state = agent.my_state
        enm_state = agent.enemy_state

        # 解包
        mx, my, mz = float(my_state[0]), float(my_state[1]), float(my_state[2])
        mr, mp, myaw = float(my_state[3]), float(my_state[4]), float(my_state[5])
        mvx, mvy, mvz = float(my_state[6]), float(my_state[7]), float(my_state[8])
        my_hp = float(my_state[12])

        ex, ey, ez = float(enm_state[0]), float(enm_state[1]), float(enm_state[2])
        evx, evy, evz = float(enm_state[6]), float(enm_state[7]), float(enm_state[8])
        enm_hp = float(enm_state[12])

        # 相对量（世界系）
        dx, dy, dz = (ex - mx), (ey - my), (ez - mz)
        dvx, dvy, dvz = (evx - mvx), (evy - mvy), (evz - mvz)
        ego_vbx, ego_vby = _rot_world_to_body_xy(mvx, mvy, myaw)
        ego_vbz = mvz
        delta_vbx, _delta_vby = _rot_world_to_body_xy(dvx, dvy, myaw)
        ego_vc = float(np.hypot(mvx, mvy))  # world horizontal speed magnitude
        Rh = float(np.hypot(dx, dy))  # horizontal distance
        R3 = float(np.sqrt(dx * dx + dy * dy + dz * dz))  # 3D distance
        ego_feature = np.array([mx, my, mz, mvx, mvy, mvz], dtype=np.float32)
        enm_feature = np.array([ex, ey, ez, evx, evy, evz], dtype=np.float32)
        AO, TA, R, side_flag = self.get2d_AO_TA_R_ue(ego_feature, enm_feature, return_side=True)
        AO = float(np.clip(AO, 0.0, np.pi))
        TA = float(np.clip(TA, 0.0, np.pi))

        roll_s, roll_c = float(np.sin(mr)), float(np.cos(mr))
        pitch_s, pitch_c = float(np.sin(mp)), float(np.cos(mp))

        h_scale = 500
        v_scale = 30
        # [0] ego altitude
        ego_alt_n = mz / h_scale

        # [5-7] ego v_body_x/y/z
        ego_vbx_n = ego_vbx / v_scale
        ego_vby_n = ego_vby / v_scale
        ego_vbz_n = ego_vbz / v_scale

        # [8] ego_vc
        ego_vc_n = ego_vc / v_scale

        # [9] delta_v_body_x
        delta_vbx_n = delta_vbx / v_scale

        # [10] delta_altitude
        delta_alt_n = dz / h_scale

        # [11-12] ego_AO / ego_TA
        # 方案A（推荐）：归一到 [0,1]，避免 observation_space 还要设到 pi
        ego_AO_n = AO / np.pi
        ego_TA_n = TA / np.pi


        # [13] relative distance（用 3D 距离 or 水平距离，你没写清，我用 3D）
        rel_dist_n = R3 / 1000

        obs = np.array([
            ego_alt_n,  # 0
            roll_s,  # 1
            roll_c,  # 2
            pitch_s,  # 3
            pitch_c,  # 4
            ego_vbx_n,  # 5
            ego_vby_n,  # 6
            ego_vbz_n,  # 7
            ego_vc_n,  # 8
            delta_vbx_n,  # 9
            delta_alt_n,  # 10
            ego_AO_n,  # 11
            ego_TA_n,  # 12
            rel_dist_n,  # 13
            side_flag  # 14  (-1/0/1)
        ], dtype=np.float32)

        obs = np.clip(obs, env.observation_space.low, env.observation_space.high)
        return obs

    def normalize_action(self, env, agent_id, action):
        if self.truncated:
            truncation = 1.0
        else:
            truncation = 0.0
        full_pack = np.append(action, truncation)
        return full_pack

    def reset(self, env):
        self.truncated = False
        return super().reset(env)


    def step(self, env):
        """
        这是task独有的step函数，这里应该不需要多做处理
        """
        pass

    def get_reward(self, env, agent_id, info=...):
        return super().get_reward(env, agent_id, info=info)

    def get2d_AO_TA_R_ue(self, ego_feature, enm_feature, return_side=False):
        ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz = ego_feature
        ego_v = np.linalg.norm([ego_vx, ego_vy])
        enm_x, enm_y, enm_z, enm_vx, enm_vy, enm_vz = enm_feature
        enm_v = np.linalg.norm([enm_vx, enm_vy])
        delta_x, delta_y, delta_z = enm_x - ego_x, enm_y - ego_y, enm_z - ego_z
        R = np.linalg.norm([delta_x, delta_y]) # 避免除0

        proj_dist = delta_x * ego_vx + delta_y * ego_vy
        ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))

        proj_dist = delta_x * enm_vx + delta_y * enm_vy
        ego_TA = np.arccos(np.clip(proj_dist / (R * enm_v + 1e-8), -1, 1))

        if not return_side:
            return ego_AO, ego_TA, R
        else:
            cross_z = ego_vx * delta_y - ego_vy * delta_x
            side_flag = float(np.sign(cross_z)) if abs(cross_z) > 1e-6 else 0.0
            return ego_AO, ego_TA, R, side_flag

