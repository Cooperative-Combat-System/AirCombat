import torch
import numpy as np
from gym import spaces
from .task_base import BaseTask
from ..termination_conditions import MyTerminationV1
from ..reward_functions import MyRewardFunctionV1
from ..utils.adaptor_v1 import NetworkAdaptorV1



class MyCombatTaskV1(BaseTask):
    def __init__(self, config):
        super().__init__(config)
        self.reward_functions = [MyRewardFunctionV1(self.config)]
        self.termination_conditions = [MyTerminationV1(self.config)]
        self.adaptor = NetworkAdaptorV1(self.config)
        self.truncated = False

    @property
    def num_agents(self) -> int:
        return 2

    def load_variables(self):
        self.state_var = [

        ]
        self.action_var = [

        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-1, high=1., shape=(18,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.Box(
            low=np.array([0.4, -1, -1, -1], dtype=np.float32),
            high=np.array([1.0, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

    # def get_obs(self, env, agent_id):
    #     my_state = env.agents[agent_id].my_state
    #     enemy_state = env.agents[agent_id].enemy_state
    #
    #     my_pos = np.array(my_state[0:3])
    #     my_rot = np.array(my_state[3:6])
    #     my_vel = np.array(my_state[6:9])
    #     my_forward = np.array(my_state[9:12])
    #     my_hp = my_state[12]
    #
    #     enm_pos = np.array(enemy_state[0:3])
    #     enm_rot = np.array(enemy_state[3:6])
    #     enm_vel = np.array(enemy_state[6:9])
    #     enm_forward = np.array(enemy_state[9:12])
    #     enm_hp = enemy_state[12]
    #
    #     norm_obs = np.zeros(18)
    #     norm_obs[0] = enm_pos[2] / 100
    #     norm_obs[1] = np.sin(my_rot[0])  # roll_sin
    #     norm_obs[2] = np.cos(my_rot[0])  # roll_cos
    #     norm_obs[3] = np.sin(my_rot[1])  # pitch_sin
    #     norm_obs[4] = np.cos(my_rot[1])  # pitch_cos
    #     norm_obs[5:8] = my_vel  # vx, vy, vz（单位化）
    #     norm_obs[8] = my_hp  # ego_hp
    #     norm_obs[9] = enm_hp  # enm_hp
    #
    #     ego_feature = np.concatenate([my_pos, my_vel])
    #     enm_feature = np.concatenate([enm_pos, enm_vel])
    #     AO, TA, R, side_flag = self.get2d_AO_TA_R_ue(ego_feature, enm_feature, return_side=True)
    #     norm_obs[10] = (enm_vel[0] - my_vel[0])  # delta_vx
    #     norm_obs[11] = (enm_pos[2] - my_pos[2]) / 100  # delta_height
    #     norm_obs[12] = AO
    #     norm_obs[13] = TA
    #     norm_obs[14] = R / 1000  # 距离单位化
    #     norm_obs[15] = side_flag
    #
    #     norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
    #     return norm_obs

    def get_obs(self, env, agent_id):
        ego = env.agents[agent_id]
        my = ego.my_state  # [x,y,z, yaw,pitch,roll, vx,vy,vz, fx,fy,fz, hp]
        enm = ego.enemy_state

        # --- 取各字段 ---
        my_pos = np.asarray(my[0:3], dtype=np.float32)
        my_rot = np.asarray(my[3:6], dtype=np.float32)  # yaw,pitch,roll（弧度）
        my_vel = np.asarray(my[6:9], dtype=np.float32)
        my_hp = float(my[12])

        enm_pos = np.asarray(enm[0:3], dtype=np.float32)
        enm_vel = np.asarray(enm[6:9], dtype=np.float32)
        enm_hp = float(enm[12])

        # --- 姿态旋转矩阵（世界->机体）: 以自机为参照 ---
        yaw, pitch, roll = my_rot
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        # ZYX (yaw-pitch-roll) 机体->世界，故世界->机体用转置
        R_w2b = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ], dtype=np.float32).T  # 转置得到世界->机体

        # --- 相对量（机体系）---
        r_rel_w = enm_pos - my_pos
        v_rel_w = enm_vel - my_vel
        r_rel_b = R_w2b @ r_rel_w
        v_rel_b = R_w2b @ v_rel_w

        R_h = np.hypot(r_rel_w[0], r_rel_w[1])  # 水平距离 (m)
        R_3 = np.linalg.norm(r_rel_w) + 1e-8  # 3D 距离 (m)
        r_hat_w = r_rel_w / R_3

        # AO（自机速度 vs LOS(ego->enemy)）
        v_e = np.linalg.norm(my_vel[:2]) + 1e-8
        dot_e = (my_vel[0] * r_hat_w[0] + my_vel[1] * r_hat_w[1])
        crs_e = (my_vel[0] * r_hat_w[1] - my_vel[1] * r_hat_w[0])
        AO = np.arctan2(abs(crs_e), dot_e)  # 0~pi
        cos_AO = np.cos(AO)

        # TA（敌机速度 vs LOS(enemy->ego)）
        v_n = np.linalg.norm(enm_vel[:2]) + 1e-8
        rxn, ryn = -r_hat_w[0], -r_hat_w[1]
        dot_n = (enm_vel[0] * rxn + enm_vel[1] * ryn)
        crs_n = (enm_vel[0] * ryn - enm_vel[1] * rxn)
        TA = np.arctan2(abs(crs_n), dot_n)
        cos_TA = np.cos(TA)

        # 闭合率（沿 LOS 的接近速度, m/s，接近为正）
        Vc = -(v_rel_w[0] * r_hat_w[0] + v_rel_w[1] * r_hat_w[1])

        # 侧向率（垂直 LOS 的相对速度模）
        Vlat = abs(v_rel_w[0] * r_hat_w[1] - v_rel_w[1] * r_hat_w[0])

        # 低空与高度差
        delta_h = (enm_pos[2] - my_pos[2]) / 100.0

        # 观测拼装（示例 18 维，可按空间限制删减）
        obs = np.array([
            # 机体系相对位置/速度（裁剪/归一见下）
            r_rel_b[0] / 1000.0, r_rel_b[1] / 1000.0, r_rel_b[2] / 1000.0,
            v_rel_b[0] / 100.0, v_rel_b[1] / 100.0, v_rel_b[2] / 100.0,

            # 姿态（自机）
            np.sin(roll), np.cos(roll),
            np.sin(pitch), np.cos(pitch),

            # 几何量
            cos_AO, cos_TA,
            R_h / 1000.0,  # km
            Vc / 100.0, Vlat / 100.0,  # 归一

            # 生命值
            my_hp, enm_hp,

            # 高度差
            delta_h,
        ], dtype=np.float32)
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
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

