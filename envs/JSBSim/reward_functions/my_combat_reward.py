import numpy as np
from .reward_function_base import BaseRewardFunction

class MyRewardFunction(BaseRewardFunction):

    def __init__(self, config):
        super().__init__(config)
        self.safe_altitude = 100  # 安全高度上限（米）
        self.danger_altitude = 90  # 危险高度（米）
        self.Kv = 10.0  # 垂直速度惩罚系数
        self.target_dist = 1000  # 期望作战距离（米）
        self.last_hp = {}  # 缓存上一次血量

    def get_reward(self, task, env, agent_id):
        ego = env.agents[agent_id]
        enm = ego.enemies[0]

        # 如果观测为空或为特殊信息包，直接返回零奖励
        if not ego.last_raw_obs or not enm.last_raw_obs:
            return self._process(0.0, agent_id, {})

        ego_obs = ego.last_raw_obs[0] if isinstance(ego.last_raw_obs, list) else ego.last_raw_obs
        enm_obs = enm.last_raw_obs[0] if isinstance(enm.last_raw_obs, list) else enm.last_raw_obs

        # 校验观测字段有效性
        if not isinstance(ego_obs, dict) or "position" not in ego_obs or "line_v" not in ego_obs:
            return self._process(0.0, agent_id, {})
        if not isinstance(enm_obs, dict) or "position" not in enm_obs or "line_v" not in enm_obs:
            return self._process(0.0, agent_id, {})

        curr_hp = ego_obs.get("hp", 1.0)
        curr_enm_hp = enm_obs.get("hp", 1.0)

        # 初始化 hp 缓存
        if agent_id not in self.last_hp:
            self.last_hp[agent_id] = curr_hp
        if enm.agent_id not in self.last_hp:
            self.last_hp[enm.agent_id] = curr_enm_hp

        last_hp = self.last_hp[agent_id]
        last_enm_hp = self.last_hp[enm.agent_id]

        delta_enm_hp = last_enm_hp - curr_enm_hp
        delta_ego_hp = last_hp - curr_hp

        # ========= 1. Event Reward =========
        event_reward = 0.0
        event_reward += delta_enm_hp * 100  # 攻击敌人奖励
        event_reward -= delta_ego_hp * 100  # 被攻击惩罚

        # ========= 2. Altitude Reward =========
        ego_z = ego_obs["position"][2]
        ego_vz = ego_obs["line_v"][2]
        Pv, PH = 0.0, 0.0

        if ego_z <= self.safe_altitude:
            Pv = -np.clip((ego_vz / self.Kv) * ((self.safe_altitude - ego_z) / self.safe_altitude), 0.0, 1.0)
        if ego_z <= self.danger_altitude:
            PH = -1.0  # 强惩罚

        altitude_reward = Pv + PH

        # ========= 3. Posture Reward =========
        ego_feature = np.concatenate([ego_obs["position"], ego_obs["line_v"]])
        enm_feature = np.concatenate([enm_obs["position"], enm_obs["line_v"]])
        AO, TA, R = task.get2d_AO_TA_R_ue(ego_feature, enm_feature)

        orientation_reward = 1 / (1 + AO)  # 面向奖励
        range_reward = max(-0.0001 * (R * 10 - self.target_dist) ** 2 + 1, 0)  # 距离奖励
        posture_reward = orientation_reward * range_reward

        # ========= 4. 总奖励加权 =========
        total_reward = (
            1.0 * event_reward +
            0.3 * altitude_reward +
            0.5 * posture_reward
        )

        # ========= 更新缓存 =========
        self.last_hp[agent_id] = curr_hp
        self.last_hp[enm.agent_id] = curr_enm_hp

        return self._process(total_reward, agent_id, (event_reward,altitude_reward,posture_reward))


