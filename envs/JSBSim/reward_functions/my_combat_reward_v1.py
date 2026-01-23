import numpy as np
from .reward_function_base import BaseRewardFunction


class MyRewardFunctionV1(BaseRewardFunction):
    """
    狗斗版 reward：
    - HP 事件 + kill/death
    - 姿态+距离 shapings (AO + R_3d)
    - OCZ 尾随优势区
    - WEZ 黏斗 (in_wez + streak)
    - 远距离接近/脱离 shaping
    - 低空惩罚（权重降低）
    - 软边界惩罚
    - no_engagement 额外惩罚（由 termination 写入 env.noeng_penalty）
    """

    def __init__(self, config):
        super().__init__(config)

        # ===== 基础参数 =====
        self.safe_altitude   = getattr(config, "safe_altitude", 400.0)   # m
        self.danger_altitude = getattr(config, "danger_altitude", 350.0) # m
        self.hp_scale        = getattr(config, "hp_scale", 100.0)
        self.Kv              = getattr(config, "Kv", 0.2)
        self.orientation_fn = self.get_orientation_function()
        self.range_fn = self.get_range_funtion()


    def get_reward(self, task, env, agent_id):
        ego = env.agents[agent_id]
        posture_reward = 0
        ego_feature = np.hstack([ego.get_position(),
                                 ego.get_velocity()])
        for enm in env.agents[agent_id].enemies:
            enm_feature = np.hstack([enm.get_position(),
                                     enm.get_velocity()])
            AO, TA, R = task.get2d_AO_TA_R_ue(ego_feature, enm_feature)
            orientation_reward = self.orientation_fn(AO, TA)
            range_reward = self.range_fn(R / 100)
            posture_reward += orientation_reward * range_reward

        altitude_reward = 0
        ego_z = env.agents[agent_id].get_position()[-1] / 100
        ego_vz = env.agents[agent_id].get_velocity()[-1] / 30
        Pv = 0.
        if ego_z <= self.safe_altitude:
            Pv = -np.clip(ego_vz / self.Kv * (self.safe_altitude - ego_z) / self.safe_altitude, 0., 1.)
        PH = 0.
        if ego_z <= self.danger_altitude:
            PH = np.clip(ego_z / self.danger_altitude, 0., 1.) - 1. - 1.
        altitude_reward = Pv + PH

        event_reward = 0
        if ego.get_result() == 1:
            event_reward = 200
        elif ego.get_result() == -1:
            event_reward = -200
        elif ego.get_result() == 0:
            event_reward = 0
        total = event_reward + posture_reward + altitude_reward
        return self._process(
            total, agent_id,
            (
                float(event_reward),
                float(posture_reward),
                float(altitude_reward),
            )
        )

    def get_orientation_function(self):
        return lambda AO, TA: 1 / (50 * AO / np.pi + 2) + 1 / 2 \
            + min((np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi), 0.) + 0.5


    def get_range_funtion(self):
        return lambda R: 1 * (R < 5) + (R >= 5) * np.clip(-0.032 * R**2 + 0.284 * R + 0.38, 0, 1) + np.clip(np.exp(-0.16 * R), 0, 0.2)
