import numpy as np
from .reward_function_base import BaseRewardFunction

def _safe_scalar(x, lo=-10.0, hi=10.0):
    x = float(np.nan_to_num(x, nan=0.0, posinf=hi, neginf=lo))
    return max(min(x, hi), lo)

def _soft_band_penalty_piecewise(x_abs: float, a: float, b: float) -> float:
    """返回 [0, -1]：a<b，内区0，过渡带线性到-1，越界-1。"""
    if x_abs <= a:
        return 0.0
    elif x_abs >= b:
        return -1.0
    else:
        return -(x_abs - a) / max(b - a, 1e-6)



class MyRewardFunctionV1(BaseRewardFunction):

    def __init__(self, config):
        super().__init__(config)
        self.safe_altitude = 90  # 安全高度上限（米）
        self.danger_altitude = 60  # 危险高度（米）
        self.hp_scale = 100.0
        self.Kv = 10.0  # 垂直速度惩罚系数
        self.target_dist = 60  # 期望作战距离（米）
        self._last_hp = {}
        self._last_enm_hp = {}
        self.sigma_R = 30
        self.kill_bonus = 4.0
        self.death_pen  = -4.0
        # 软边界（与终止条件对齐 |x|、|y|<=150）
        self.arena_half_size   = getattr(config, "arena_half_size", 150.0)  # m
        self.soft_band_start   = getattr(config, "soft_band_start", 140.0)  # m，开始“感觉到边界”
        self.soft_band_k       = getattr(config, "soft_band_k", 1.0)        # 斜率
        self.soft_band_weight  = getattr(config, "soft_band_weight", 0.5)   # 惩罚权重（可小）

        # 姿态/低空权重（总和不必=1，最后有总裁剪）
        self.w_posture  = getattr(config, "w_posture", 2.0)
        self.w_altitude = getattr(config, "w_altitude", 0.3)

    def _reset_if_first_step(self, env):
        if getattr(env, "current_step", 0) == 1:
            self._last_hp.clear()
            self._last_enm_hp.clear()

    def get_reward(self, task, env, agent_id):
        self._reset_if_first_step(env)

        ego = env.agents[agent_id]

        ego_obs = ego.my_state
        enm_obs = ego.enemy_state

        my_hp = float(ego_obs[12])
        enm_hp = float(enm_obs[12])

        my_alt = float(ego_obs[2])
        my_vz = float(ego_obs[8])

        # 用几何函数拿 AO & 水平距离（与18维观测一致）
        ego_feature = np.concatenate([ego_obs[0:3], ego_obs[6:9]], dtype=np.float32)  # [x,y,z,vx,vy,vz]
        enm_feature = np.concatenate([enm_obs[0:3], enm_obs[6:9]], dtype=np.float32)
        AO, TA, R_h = task.get2d_AO_TA_R_ue(ego_feature, enm_feature)  # AO: rad, R_h: m
        AO = float(np.clip(np.nan_to_num(AO), 0.0, np.pi))
        R_h = float(np.clip(np.nan_to_num(R_h), 0.0, 1e6))

        # ========= 1) 事件奖励（HP 差分）=========
        if agent_id not in self._last_hp:
            self._last_hp[agent_id] = my_hp
        if agent_id not in self._last_enm_hp:
            self._last_enm_hp[agent_id] = enm_hp

        d_enm = max(self._last_enm_hp[agent_id] - enm_hp, 0.0)  # 我方造成伤害
        d_ego = max(self._last_hp[agent_id] - my_hp, 0.0)  # 我方受伤

        event_reward = self.hp_scale * (+ d_enm - d_ego)
        # ========= 2) 姿态/距离塑形 =========
        # cosAO 越接近 1 越好；距离用高斯围绕 target_dist
        cos_AO = np.cos(AO)
        dr = (R_h - self.target_dist) / max(self.sigma_R, 1e-6)
        range_reward = float(np.exp(-0.5 * dr * dr))  # (0,1]
        posture_reward = float(cos_AO) * range_reward

        # ========= 3) 低空安全塑形 =========
        altitude_reward = 0.0
        if my_alt <= self.safe_altitude:
            depth = (self.safe_altitude - my_alt) / max(self.safe_altitude, 1.0)  # 0~1
            climb = max(my_vz, 0.0) / max(self.Kv, 1e-6)  # 低空上升奖励
            sink = max(-my_vz, 0.0) / max(self.Kv, 1e-6)  # 低空下降惩罚
            altitude_reward += +0.5 * depth * climb - 0.8 * depth * sink
        if my_alt <= self.danger_altitude:
            altitude_reward += -1.0

        # ========= 4) 终局奖励（由 HP 到 0 触发）=========
        terminal_bonus = 0.0
        if enm_hp <= 0.0 and my_hp > 0.0:
            terminal_bonus += self.kill_bonus
        if my_hp <= 0.0 and enm_hp > 0.0:
            terminal_bonus += self.death_pen

        # ========= 5) 软边界惩罚（靠近 |x|,|y|=L 前就罚）=========
        L = float(self.arena_half_size)
        a = float(self.soft_band_start)
        b = L
        k = float(self.soft_band_k)
        x_abs = abs(float(ego_obs[0]))
        y_abs = abs(float(ego_obs[1]))
        soft_oob = (
                _soft_band_penalty_piecewise(x_abs, self.soft_band_start, self.arena_half_size) +
                _soft_band_penalty_piecewise(y_abs, self.soft_band_start, self.arena_half_size)
        )
        soft_oob *= float(self.soft_band_weight)  # 例如先设 0.1~0.2
        soft_oob = float(np.clip(soft_oob, -1.0, 0.0))  # 保证在 [-1, 0]

        total = (
                float(event_reward)
                + self.w_posture * float(posture_reward)
                + self.w_altitude * float(altitude_reward)
                + float(terminal_bonus)
                + float(soft_oob)
        )
        total = _safe_scalar(total, -10.0, 10.0)

        # ========= 8) 更新缓存 =========
        self._last_hp[agent_id] = my_hp
        self._last_enm_hp[agent_id] = enm_hp

        components = {
            "total": float(total),
            "event": float(event_reward),
            "posture": float(posture_reward),
            "altitude": float(altitude_reward),
            "terminal": float(terminal_bonus),
            "soft_oob": float(soft_oob),
        }

        try:
            env.log_step_components(agent_id, components)
        except Exception:
            pass

        # 返回各分量，便于 Logger 展示
        return self._process(
            total, agent_id,
            (
                float(event_reward),
                float(posture_reward),
                float(altitude_reward),
                float(terminal_bonus),
                float(soft_oob),
            )
        )


