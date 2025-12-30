import torch
import numpy as np
from gym import spaces
from .task_base import BaseTask
from ..termination_conditions import MyTerminationV1
from ..reward_functions import MyRewardFunctionV1
from ..utils.adaptor_v1 import NetworkAdaptorV1

R_MAX = 300.0   # m
H_MAX = 150.0   # m
V_MAX = 40.0    # m/s
EPS   = 1e-6

def _rot_world_to_body_xy(dx, dy, yaw):
    """
    用 yaw 做 2D 机体系旋转（机头为 +x，右侧为 +y）
    body = R(-yaw) * world
    """
    c, s = np.cos(yaw), np.sin(yaw)
    bx =  c * dx + s * dy
    by = -s * dx + c * dy
    return bx, by

def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))

def _clip11(x):
    return float(np.clip(x, -1.0, 1.0))

def _bearing(dx, dy):
    return float(np.arctan2(dy, dx))  # (-pi, pi]

def _angle_wrap_pi(a):
    # wrap to (-pi, pi]
    return float((a + np.pi) % (2*np.pi) - np.pi)

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
        self.observation_space = spaces.Box(low=-1, high=1., shape=(21,))

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

        # 机体系（用 yaw 旋转）
        rel_bx, rel_by = _rot_world_to_body_xy(dx, dy, myaw)
        rel_vbx, rel_vby = _rot_world_to_body_xy(dvx, dvy, myaw)

        # 距离、速度
        Rh = float(np.hypot(dx, dy))
        my_spd = float(np.hypot(mvx, mvy))
        enm_spd = float(np.hypot(evx, evy))

        dist_feat = _clip01(Rh / max(R_MAX, EPS))
        rel_height = _clip11(dz / max(H_MAX, EPS))
        rel_pos_body_x = _clip11(rel_bx / max(R_MAX, EPS))
        rel_pos_body_y = _clip11(rel_by / max(R_MAX, EPS))
        rel_vel_body_x = _clip11(rel_vbx / max(V_MAX, EPS))
        rel_vel_body_y = _clip11(rel_vby / max(V_MAX, EPS))
        my_spd_n = _clip01(my_spd / max(V_MAX, EPS))
        enm_spd_n = _clip01(enm_spd / max(V_MAX, EPS))

        # 自机高度/垂直速度（配合奖励）
        my_alt_n = _clip01(mz / max(H_MAX, EPS))
        my_vz_n = _clip11(mvz / max(15, EPS))  # VZ_MAX 你自己定义一个常量

        # AO/TA/R_h：复用几何函数
        ego_feature = np.array([mx, my, mz, mvx, mvy, mvz], dtype=np.float32)
        enm_feature = np.array([ex, ey, ez, evx, evy, evz], dtype=np.float32)
        AO, TA, _Rh = self.get2d_AO_TA_R_ue(ego_feature, enm_feature)
        AO = float(np.clip(AO, 0.0, np.pi))
        TA = float(np.clip(TA, 0.0, np.pi))
        facing_cos = float(np.cos(AO))
        aspect_cos = float(np.cos(TA))  # 新增

        # 闭合率：径向速度 / V_MAX
        if Rh > 1e-3:
            Vr = (dx * dvx + dy * dvy) / Rh
        else:
            Vr = 0.0
        closure = _clip11(Vr / max(V_MAX, EPS))

        # LOS 角速率
        if not hasattr(env, "_obs_cache"):
            env._obs_cache = {}
        if agent_id not in env._obs_cache:
            env._obs_cache[agent_id] = {}
        prev_bearing = env._obs_cache[agent_id].get("bearing", _bearing(dx, dy))
        curr_bearing = _bearing(dx, dy)
        dpsi = _angle_wrap_pi(curr_bearing - prev_bearing)
        env._obs_cache[agent_id]["bearing"] = curr_bearing
        los_rate = _clip11(dpsi / np.pi)

        # 姿态 sin/cos
        roll_s, roll_c = float(np.sin(mr)), float(np.cos(mr))
        pitch_s, pitch_c = float(np.sin(mp)), float(np.cos(mp))

        # HP 归一化
        if my_hp > 1.0 or enm_hp > 1.0:
            my_hp_n = _clip01(my_hp / 100.0)
            enm_hp_n = _clip01(enm_hp / 100.0)
        else:
            my_hp_n, enm_hp_n = _clip01(my_hp), _clip01(enm_hp)

        # 侧向标志
        cross_z = mvx * dy - mvy * dx
        side_flag = float(np.sign(cross_z))

        obs = np.array([
            rel_pos_body_x,  # 0
            rel_pos_body_y,  # 1
            rel_height,  # 2
            dist_feat,  # 3
            rel_vel_body_x,  # 4
            rel_vel_body_y,  # 5
            my_spd_n,  # 6
            enm_spd_n,  # 7
            my_alt_n,  # 8  (new)
            my_vz_n,  # 9  (new)
            facing_cos,  # 10
            aspect_cos,  # 11 (new)
            closure,  # 12
            los_rate,  # 13
            roll_s,  # 14
            roll_c,  # 15
            pitch_s,  # 16
            pitch_c,  # 17
            my_hp_n,  # 18
            enm_hp_n,  # 19
            side_flag  # 20
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
            side_flag = np.sign(np.cross([ego_vx, ego_vy], [delta_x, delta_y]))
            return ego_AO, ego_TA, R, side_flag

