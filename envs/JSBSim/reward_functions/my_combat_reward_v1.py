import numpy as np
from .reward_function_base import BaseRewardFunction

EPS = 1e-6

def _safe_scalar(x, lo=-10.0, hi=10.0):
    x = float(np.nan_to_num(x, nan=0.0, posinf=hi, neginf=lo))
    return max(min(x, hi), lo)

def _soft_band_penalty_piecewise(x_abs, a, b):
    """
    分段线性到平滑混合：a 内惩罚=0；[a,b]从0线性到-1；>b 固定为-1。
    你也可以换成 sigmoid 版，关键是“逐步变重、且饱和到 -1”。
    """
    if x_abs <= a:
        return 0.0
    if x_abs >= b:
        return -1.0
    t = (x_abs - a) / max(b - a, EPS)
    return -t  # [-1,0]

class MyRewardFunctionV1(BaseRewardFunction):

    def __init__(self, config):
        super().__init__(config)
        self.safe_altitude = 50  # 安全高度上限（米）
        self.danger_altitude = 40  # 危险高度（米）
        self.hp_scale = 100.0
        self.Kv = 10.0  # 垂直速度惩罚系数
        self.target_dist = 60  # 期望作战距离（米）
        self._last_hp = {}
        self._last_enm_hp = {}
        self.sigma_R = 30
        self.kill_bonus = 4.0
        self.death_pen  = -4.0
        # 软边界（与终止条件对齐 |x|、|y|<=150）
        self.arena_half_size   = getattr(config, "arena_half_size", 500.0)  # m
        self.soft_band_start   = getattr(config, "soft_band_start", 480.0)  # m，开始“感觉到边界”
        self.soft_band_k       = getattr(config, "soft_band_k", 1.0)        # 斜率
        self.soft_band_weight  = getattr(config, "soft_band_weight", 0.5)   # 惩罚权重（可小）

        # 姿态/低空权重（总和不必=1，最后有总裁剪）
        self.w_posture  = getattr(config, "w_posture", 2.0)
        self.w_altitude = getattr(config, "w_altitude", 0.3)
        self.w_trend = getattr(config, "w_trend", 0.10)  # AO 变小的趋势奖励
        self.w_turn = getattr(config, "w_turn", 0.05)  # 交战圈内的机动/转弯塑形

        # 缓存
        self._last_AO = {}
        self._last_bearing = {}

    def _reset_if_first_step(self, env):
        if getattr(env, "current_step", 0) == 1:
            self._last_hp.clear()
            self._last_enm_hp.clear()

    def get_reward(self, task, env, agent_id):
        self._reset_if_first_step(env)

        ego = env.agents[agent_id]
        ego_obs = ego.my_state
        enm_obs = ego.enemy_state

        # 解包
        my_hp = float(ego_obs[12])
        enm_hp = float(enm_obs[12])
        my_alt = float(ego_obs[2])
        my_vz = float(ego_obs[8])

        # AO/TA/Rh
        ego_feature = np.concatenate([ego_obs[0:3], ego_obs[6:9]], dtype=np.float32)
        enm_feature = np.concatenate([enm_obs[0:3], enm_obs[6:9]], dtype=np.float32)
        AO, TA, R_h = task.get2d_AO_TA_R_ue(ego_feature, enm_feature)  # AO(rad), R_h(m)
        AO = float(np.clip(np.nan_to_num(AO), 0.0, np.pi))
        R_h = float(np.clip(np.nan_to_num(R_h), 0.0, 1e6))

        # 1) 事件奖励（HP 差分，限幅）
        if agent_id not in self._last_hp:
            self._last_hp[agent_id] = my_hp
        if agent_id not in self._last_enm_hp:
            self._last_enm_hp[agent_id] = enm_hp

        d_enm_raw = self._last_enm_hp[agent_id] - enm_hp  # 我方造成伤害
        d_ego_raw = self._last_hp[agent_id] - my_hp  # 我方受伤
        # 步级限幅（避免脉冲过大）
        d_enm = float(np.clip(d_enm_raw, 0.0, 0.05))
        d_ego = float(np.clip(d_ego_raw, 0.0, 0.05))
        event_reward = self.hp_scale * (d_enm - d_ego)

        # 2) 姿态/距离塑形（狗斗核心）
        cos_AO = np.cos(AO)
        dr = (R_h - self.target_dist) / max(self.sigma_R, 1e-6)
        range_reward = float(np.exp(-0.5 * dr * dr))  # (0,1]
        posture_reward = float(cos_AO) * range_reward  # [-1,1]*[0,1]

        # ---- 2.1) AO 趋势奖励（让 AO 逐步变小）
        last_AO = self._last_AO.get(agent_id, AO)
        dAO = float(np.clip(last_AO - AO, -0.2, 0.2))  # AO 下降为正，小限幅
        trend_reward = dAO

        # ---- 2.2) 交战圈内机动塑形（可用 los_rate 或滚转程度；这里给简单版）
        turn_reward = 0.0
        if R_h <= 150.0:  # 进入狗斗圈，鼓励一定的“机动”
            roll = float(ego_obs[3])
            turn_reward = 0.5 * (np.sin(roll) ** 2)  # 0~0.5，横滚越大越鼓励（可按需替换成 |los_rate| 范围目标）

        # 3) 低空安全塑形
        altitude_reward = 0.0
        if my_alt <= self.safe_altitude:
            depth = (self.safe_altitude - my_alt) / max(self.safe_altitude, 1.0)
            climb = max(my_vz, 0.0) / max(self.Kv, 1e-6)
            sink = max(-my_vz, 0.0) / max(self.Kv, 1e-6)
            altitude_reward += +0.5 * depth * climb - 0.8 * depth * sink
        if my_alt <= self.danger_altitude:
            altitude_reward += -1.0

        # 4) 终局奖励
        terminal_bonus = 0.0
        if enm_hp <= 0.0 and my_hp > 0.0:
            terminal_bonus += self.kill_bonus
        if my_hp <= 0.0 and enm_hp > 0.0:
            terminal_bonus += self.death_pen

        # 5) 软边界
        x_abs = abs(float(ego_obs[0]))
        y_abs = abs(float(ego_obs[1]))
        soft_oob = (
                _soft_band_penalty_piecewise(x_abs, self.soft_band_start, self.arena_half_size) +
                _soft_band_penalty_piecewise(y_abs, self.soft_band_start, self.arena_half_size)
        )
        soft_oob *= float(self.soft_band_weight)  # [-0.2, 0]

        total = (
                float(event_reward)
                + self.w_posture * float(posture_reward)
                + self.w_altitude * float(altitude_reward)
                + self.w_trend * float(trend_reward)
                + self.w_turn * float(turn_reward)
                + float(terminal_bonus)
                + float(soft_oob)
        )
        total = _safe_scalar(total, -10.0, 10.0)

        # 缓存更新
        self._last_hp[agent_id] = my_hp
        self._last_enm_hp[agent_id] = enm_hp
        self._last_AO[agent_id] = AO

        # （可选）把分量喂到 env 日志（供 wandb 聚合）
        components = {
            "total": float(total),
            "event": float(event_reward),
            "posture": float(posture_reward),
            "trend_AO": float(trend_reward),
            "turning": float(turn_reward),
            "altitude": float(altitude_reward),
            "terminal": float(terminal_bonus),
            "soft_oob": float(soft_oob),
            "R_h": float(R_h),
            "AO": float(AO),
        }
        try:
            env.log_step_components(agent_id, components)
        except Exception:
            pass

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