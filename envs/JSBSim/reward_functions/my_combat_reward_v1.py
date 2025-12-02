import numpy as np
from .reward_function_base import BaseRewardFunction

# ---------- 小工具 ----------
def _safe_scalar(x, lo, hi):
    try:
        if np.isnan(x) or np.isinf(x):
            return lo
    except Exception:
        pass
    return float(np.clip(x, lo, hi))

def _soft_band_penalty_piecewise(x_abs, band_start, band_end):
    """
    在 [band_start, band_end] 内线性增长、到达 band_end 处约 -1；band_end 之外直接 -1。
    """
    if x_abs <= band_start:
        return 0.0
    if x_abs >= band_end:
        return -1.0
    ratio = (x_abs - band_start) / max(band_end - band_start, 1e-6)
    return -ratio  # ∈ [-1, 0]

# ---------- 奖励函数 ----------
class MyRewardFunctionV1(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        # —— 安全与几何参数 ——
        self.safe_altitude   = getattr(config, "safe_altitude", 50.0)  # m
        self.danger_altitude = getattr(config, "danger_altitude", 40.0) # m
        self.hp_scale = getattr(config, "hp_scale", 100.0)
        self.Kv       = getattr(config, "Kv", 10.0)

        # 作战圈/目标距离（与你给的一致）
        self.target_dist = getattr(config, "target_dist", 60.0)  # m
        self.sigma_R     = getattr(config, "sigma_R", 25.0)      # m（高斯宽度）

        # WEZ（2°×2°×60m）
        self.wez_max_range = getattr(config, "wez_max_range", 80.0)
        self.wez_hdeg      = np.deg2rad(getattr(config, "wez_hdeg", 5.0))  # 水平偏角阈值（AO）
        self.wez_vdeg      = np.deg2rad(getattr(config, "wez_vdeg", 5.0))  # 垂直偏角阈值（elev_off）

        # 软边界
        self.arena_half_size   = getattr(config, "arena_half_size", 500.0)
        self.soft_band_start   = getattr(config, "soft_band_start", 480.0)
        self.soft_band_k       = getattr(config, "soft_band_k", 1.0)
        self.soft_band_weight  = getattr(config, "soft_band_weight", 0.5)

        # 姿态/低空/圈内机动权重
        self.w_posture  = getattr(config, "w_posture", 2.0)
        self.w_altitude = getattr(config, "w_altitude", 0.1)
        self.w_trend    = getattr(config, "w_trend",   0.10)  # AO 变小趋势
        self.w_turn     = getattr(config, "w_turn",    0.05)  # 圈内机动

        # 黏斗相关权重
        self.w_stick      = getattr(config, "w_stick", 0.30)   # WEZ 连续驻留奖励（每步）
        self.w_approach   = getattr(config, "w_approach", 0.30) # 圈外接近奖励
        self.w_disengage  = getattr(config, "w_disengage", 0.05) # 脱离趋势惩罚
        self.stick_max    = getattr(config, "stick_max",  80)   # streak 上限（步）

        # 终局
        self.kill_bonus = getattr(config, "kill_bonus",  4.0)
        self.death_pen  = getattr(config, "death_pen",  -4.0)

        # 缓存
        self._last_hp       = {}
        self._last_enm_hp   = {}
        self._last_AO       = {}
        self._last_Rh       = {}
        self._wez_streak    = {}

    def _reset_if_first_step(self, env):
        if getattr(env, "current_step", 0) == 1:
            self._last_hp.clear()
            self._last_enm_hp.clear()
            self._last_AO.clear()
            self._last_Rh.clear()
            self._wez_streak.clear()

    def get_reward(self, task, env, agent_id):
        self._reset_if_first_step(env)

        extra_noeng_penalty = 0.0
        if hasattr(env, "noeng_penalty"):
            # 用 pop 取出来，防止下一步重复扣
            extra_noeng_penalty = float(env.noeng_penalty.pop(agent_id, 0.0))

        ego = env.agents[agent_id]
        ego_obs = ego.my_state
        enm_obs = ego.enemy_state

        # 解析（与你的索引约定保持一致）
        my_hp  = float(ego_obs[12])
        enm_hp = float(enm_obs[12])
        my_alt = float(ego_obs[2])
        enm_alt= float(enm_obs[2])
        my_vz  = float(ego_obs[8])

        # 相对几何（你已有方法：AO(rad), TA, R_h(m)）
        ego_feature = np.concatenate([ego_obs[0:3], ego_obs[6:9]]).astype(np.float32)
        enm_feature = np.concatenate([enm_obs[0:3], enm_obs[6:9]]).astype(np.float32)
        AO, TA, R_h = task.get2d_AO_TA_R_ue(ego_feature, enm_feature)
        AO  = float(np.clip(np.nan_to_num(AO), 0.0, np.pi))
        R_h = float(np.clip(np.nan_to_num(R_h), 0.0, 1e6))

        # 垂直偏角（用高度差/水平距近似）
        dz = float(enm_alt - my_alt)
        elev_off = float(np.arctan2(abs(dz), max(R_h, 1e-6)))  # rad
        R_3d = float(np.hypot(R_h, dz))

        # ---------------- 1) 事件奖励（伤害差分，限幅） ----------------
        if agent_id not in self._last_hp:
            self._last_hp[agent_id] = my_hp
        if agent_id not in self._last_enm_hp:
            self._last_enm_hp[agent_id] = enm_hp

        d_enm_raw = self._last_enm_hp[agent_id] - enm_hp   # 我方对敌伤害
        d_ego_raw = self._last_hp[agent_id]    - my_hp     # 我方受伤
        d_enm = float(np.clip(d_enm_raw, 0.0, 0.05))
        d_ego = float(np.clip(d_ego_raw, 0.0, 0.05))
        event_reward = self.hp_scale * (d_enm - d_ego)

        # ---------------- 2) 姿态/距离塑形（盘旋黏斗核心） ----------------
        # 距离在 target_dist 附近的高斯形状
        dr = (R_3d - self.target_dist) / max(self.sigma_R, 1e-6)
        range_reward = float(np.exp(-0.5 * dr * dr))  # (0,1]

        # 水平对准（AO 越小越好）× 距离高斯
        posture_reward = float(np.cos(AO)) * range_reward  # [-1,1] * (0,1]

        # AO 趋势（更对准）
        last_AO = self._last_AO.get(agent_id, AO)
        dAO = float(np.clip(last_AO - AO, -0.2, 0.2))  # AO 下降为正
        trend_reward = dAO

        # 圈内机动塑形（≤150 m 覆盖“狗斗圈”）
        turn_reward = 0.0
        if R_3d <= 150.0:
            roll = float(ego_obs[3])  # 你之前就把 3 当作 roll
            turn_reward = 0.5 * (np.sin(roll) ** 2)  # 0~0.5，替代方案：|LOS_rate|

        # ---------------- 3) WEZ（2°×2°×60m）+ 黏斗驻留 ----------------
        in_wez = (R_3d <= self.wez_max_range) and (AO <= self.wez_hdeg) and (elev_off <= self.wez_vdeg)
        streak = self._wez_streak.get(agent_id, 0)
        if in_wez:
            streak = min(streak + 1, int(self.stick_max))
        else:
            streak = 0
        self._wez_streak[agent_id] = streak
        # 驻留奖励：在 WEZ 内每步追加小额奖励，连续越久越多（上限保护）
        stick_reward = self.w_stick * (streak / max(self.stick_max, 1))

        # ---------------- 4) 脱离/接近趋势塑形 ----------------
        last_Rh = self._last_Rh.get(agent_id, R_h)
        dR_h = R_h - last_Rh  # >0 变远，<0 变近（水平面）
        # 在圈外（>200 m）且朝目标靠近 → 鼓励
        approach_reward = 0.0
        if R_h > 200.0 and dR_h < 0.0:
            approach_reward = self.w_approach * float(np.clip(-dR_h / 20.0, 0.0, 1.0))
        # 脱离趋势惩罚：在 200~800 m 内如果持续变远，给轻惩罚
        disengage_pen = 0.0
        if 200.0 < R_h < 800.0 and dR_h > 0.0:
            disengage_pen = -self.w_disengage * float(np.clip(dR_h / 20.0, 0.0, 1.0))

        # ---------------- 5) 低空安全塑形 ----------------
        altitude_reward = 0.0
        if my_alt <= self.safe_altitude:
            depth = (self.safe_altitude - my_alt) / max(self.safe_altitude, 1.0)
            climb = max(my_vz, 0.0) / max(self.Kv, 1e-6)
            sink  = max(-my_vz, 0.0) / max(self.Kv, 1e-6)
            altitude_reward += +0.5 * depth * climb - 0.8 * depth * sink
        if my_alt <= self.danger_altitude:
            altitude_reward += -1.0

        # ---------------- 6) 终局奖励 ----------------
        terminal_bonus = 0.0
        if enm_hp <= 0.0 and my_hp > 0.0:
            terminal_bonus += self.kill_bonus
        if my_hp <= 0.0 and enm_hp > 0.0:
            terminal_bonus += self.death_pen

        # ---------------- 7) 软边界 ----------------
        x_abs = abs(float(ego_obs[0]))
        y_abs = abs(float(ego_obs[1]))
        soft_oob = (
            _soft_band_penalty_piecewise(x_abs, self.soft_band_start, self.arena_half_size) +
            _soft_band_penalty_piecewise(y_abs, self.soft_band_start, self.arena_half_size)
        ) * float(self.soft_band_weight)

        # ---------------- 汇总（含裁剪） ----------------
        total = (
            float(event_reward)
            + self.w_posture * float(posture_reward)
            + self.w_altitude * float(altitude_reward)
            + self.w_trend   * float(trend_reward)
            + self.w_turn    * float(turn_reward)
            + float(stick_reward)
            + float(approach_reward)
            + float(disengage_pen)
            + float(terminal_bonus)
            + float(soft_oob)
            + float(extra_noeng_penalty)
        )
        total = _safe_scalar(total, -10.0, 10.0)

        # 缓存更新
        self._last_hp[agent_id]     = my_hp
        self._last_enm_hp[agent_id] = enm_hp
        self._last_AO[agent_id]     = AO
        self._last_Rh[agent_id]     = R_h

        # 日志
        components = {
            "total": float(total),
            "event": float(event_reward),
            "posture": float(posture_reward),
            "trend_AO": float(trend_reward),
            "turning": float(turn_reward),
            "stick": float(stick_reward),
            "approach": float(approach_reward),
            "disengage": float(disengage_pen),
            "altitude": float(altitude_reward),
            "terminal": float(terminal_bonus),
            "soft_oob": float(soft_oob),
            "R_h": float(R_h),
            "R_3d": float(R_3d),
            "AO": float(AO),
            "elev_off": float(elev_off),
            "in_wez": float(1.0 if in_wez else 0.0),
            "wez_streak": float(streak),
            "noeng_penalty": float(extra_noeng_penalty),
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
