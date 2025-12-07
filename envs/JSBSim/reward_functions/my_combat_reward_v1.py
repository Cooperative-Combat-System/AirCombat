import numpy as np
from .reward_function_base import BaseRewardFunction

# ---------- 小工具 ----------
def _safe_scalar(x, lo, hi):
    try:
        if np.isnan(x) or np.isinf(x):
            return float(lo)
    except Exception:
        return float(lo)
    return float(np.clip(x, lo, hi))

def _soft_band_penalty_piecewise(x_abs, band_start, band_end):
    """
    |x| <= band_start: 0
    band_start < |x| < band_end: 线性到 -1
    |x| >= band_end: -1
    """
    if x_abs <= band_start:
        return 0.0
    if x_abs >= band_end:
        return -1.0
    ratio = (x_abs - band_start) / max(band_end - band_start, 1e-6)
    return -ratio


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
        self.safe_altitude   = getattr(config, "safe_altitude", 50.0)   # m
        self.danger_altitude = getattr(config, "danger_altitude", 40.0) # m
        self.hp_scale        = getattr(config, "hp_scale", 100.0)
        self.Kv              = getattr(config, "Kv", 10.0)

        # 作战圈/目标距离
        self.target_dist = getattr(config, "target_dist", 60.0)  # m
        self.sigma_R     = getattr(config, "sigma_R", 25.0)      # m，高斯宽度

        # ===== WEZ（训练期先放宽，后面你可以在配置里收紧）=====
        self.wez_max_range = getattr(config, "wez_max_range", 80.0)           # m
        self.wez_hdeg      = np.deg2rad(getattr(config, "wez_hdeg", 5.0))     # 水平 AO 阈值
        self.wez_vdeg      = np.deg2rad(getattr(config, "wez_vdeg", 5.0))     # 垂直偏角阈值

        # ===== OCZ 尾随优势区（R + AO 窗口）=====
        # 距离窗 [ocz_R_min, ocz_R_max]，角度窗 AO_deg ∈ [0, ocz_A_deg]
        self.ocz_R_min  = getattr(config, "ocz_R_min", 40.0)
        self.ocz_R_max  = getattr(config, "ocz_R_max", 120.0)
        self.ocz_A_deg  = getattr(config, "ocz_A_deg", 30.0)
        self.ocz_k_R    = getattr(config, "ocz_k_R", 10.0)   # 平滑宽度
        self.ocz_k_A    = getattr(config, "ocz_k_A", 5.0)
        self.w_ocz      = getattr(config, "w_ocz",   1.0)

        # ===== 狗斗能量/速度塑形 =====
        self.corner_v      = getattr(config, "corner_v", 180.0)  # 你环境里大致拐弯速度（m/s）
        self.corner_v_tol  = getattr(config, "corner_v_tol", 0.3) # 相对误差容忍
        self.w_energy      = getattr(config, "w_energy", 0.2)
        self.dogfight_Rmax = getattr(config, "dogfight_Rmax", 200.0)  # R_h<Rmax 时才考虑角速度/能量

        # ===== 软边界（与终止条件对齐）=====
        self.arena_half_size   = getattr(config, "arena_half_size", 500.0)
        self.soft_band_start   = getattr(config, "soft_band_start", 480.0)
        self.soft_band_k       = getattr(config, "soft_band_k", 1.0)
        self.soft_band_weight  = getattr(config, "soft_band_weight", 0.5)

        # ===== 权重：姿态/低空/趋势/机动 =====
        self.w_posture  = getattr(config, "w_posture", 2.0)
        self.w_altitude = getattr(config, "w_altitude", 0.15)   # 降一点，避免抢梯度
        self.w_trend    = getattr(config, "w_trend",   0.10)    # AO 变小趋势
        self.w_turn     = getattr(config, "w_turn",    0.05)    # 圈内机动

        # ===== 黏斗相关权重 =====
        self.w_stick     = getattr(config, "w_stick",    0.25)  # WEZ 连续驻留
        self.w_approach  = getattr(config, "w_approach", 0.30)  # 圈外接近
        self.w_disengage = getattr(config, "w_disengage",0.05)  # 脱离惩罚
        self.stick_max   = getattr(config, "stick_max",  80)    # streak 上限（步）

        # ===== 终局奖励 =====
        self.kill_bonus = getattr(config, "kill_bonus",  4.0)
        self.death_pen  = getattr(config, "death_pen",  -4.0)

        # ===== 缓存 =====
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

    def _ocz_window(self, x, a, b, k):
        """平滑的 [a,b] 窗口函数，内部≈1，两端平滑到0"""
        # 用两个 sigmoid 差实现
        return (
            1.0 / (1.0 + np.exp((x - a) / max(k, 1e-6)))
            - 1.0 / (1.0 + np.exp((x - b) / max(k, 1e-6)))
        )

    def get_reward(self, task, env, agent_id):
        self._reset_if_first_step(env)

        # ==== 取 no_engagement 罚分（如果终止里打了标记）====
        extra_noeng_penalty = 0.0
        if hasattr(env, "noeng_penalty"):
            extra_noeng_penalty = float(env.noeng_penalty.pop(agent_id, 0.0))

        ego = env.agents[agent_id]
        ego_obs = ego.my_state
        enm_obs = ego.enemy_state

        # ===== 解包状态（按你原来的索引）=====
        my_hp  = float(ego_obs[12])
        enm_hp = float(enm_obs[12])
        my_alt = float(ego_obs[2])
        enm_alt= float(enm_obs[2])
        my_vz  = float(ego_obs[8])

        # 自身速度（水平面）
        mvx, mvy = float(ego_obs[6]), float(ego_obs[7])
        my_spd   = float(np.hypot(mvx, mvy))

        # ===== 相对几何：AO / TA / R_h =====
        ego_feature = np.concatenate([ego_obs[0:3], ego_obs[6:9]]).astype(np.float32)
        enm_feature = np.concatenate([enm_obs[0:3], enm_obs[6:9]]).astype(np.float32)
        AO, TA, R_h = task.get2d_AO_TA_R_ue(ego_feature, enm_feature)  # AO(rad), TA(rad), R_h(m)
        AO  = float(np.clip(np.nan_to_num(AO), 0.0, np.pi))
        R_h = float(np.clip(np.nan_to_num(R_h), 0.0, 1e6))

        # 垂直偏角：高度差 / 水平距离
        dz = float(enm_alt - my_alt)
        elev_off = float(np.arctan2(abs(dz), max(R_h, 1e-6)))  # rad
        R_3d = float(np.hypot(R_h, dz))
        AO_deg = AO * 180.0 / np.pi

        # ===== 1) 事件奖励：HP 差分 =====
        if agent_id not in self._last_hp:
            self._last_hp[agent_id] = my_hp
        if agent_id not in self._last_enm_hp:
            self._last_enm_hp[agent_id] = enm_hp

        d_enm_raw = self._last_enm_hp[agent_id] - enm_hp    # 我打敌
        d_ego_raw = self._last_hp[agent_id]    - my_hp      # 我被打
        d_enm = float(np.clip(d_enm_raw, 0.0, 0.05))
        d_ego = float(np.clip(d_ego_raw, 0.0, 0.05))
        event_reward = self.hp_scale * (d_enm - d_ego)

        # ===== 2) 姿态 + 距离塑形（狗斗基础）=====
        # 距离靠近 target_dist 的高斯
        dr = (R_3d - self.target_dist) / max(self.sigma_R, 1e-6)
        range_reward = float(np.exp(-0.5 * dr * dr))  # (0,1]

        # AO 小 + 距离合适
        posture_reward = float(np.cos(AO)) * range_reward  # [-1,1] * (0,1]

        # AO 趋势（更对准）
        last_AO = self._last_AO.get(agent_id, AO)
        dAO = float(np.clip(last_AO - AO, -0.2, 0.2))  # AO 变小为正
        trend_reward = dAO

        # 圈内机动塑形：≤150m 鼓励横滚
        turn_reward = 0.0
        if R_3d <= 150.0:
            roll = float(ego_obs[3])
            turn_reward = 0.5 * (np.sin(roll) ** 2)   # 0~0.5

        # ===== 3) OCZ 尾随优势区 =====
        # 距离窗 + 角度窗
        phi_R = self._ocz_window(R_h,
                                 self.ocz_R_min,
                                 self.ocz_R_max,
                                 self.ocz_k_R)
        phi_A = self._ocz_window(AO_deg,
                                 0.0,
                                 self.ocz_A_deg,
                                 self.ocz_k_A)
        ocz_reward = self.w_ocz * 2.0 * phi_R * phi_A  # 最大 ~2*w_ocz

        # ===== 4) WEZ + 黏斗 =====
        in_wez = (R_3d <= self.wez_max_range) and (AO <= self.wez_hdeg) and (elev_off <= self.wez_vdeg)
        streak = self._wez_streak.get(agent_id, 0)
        if in_wez:
            streak = min(streak + 1, int(self.stick_max))
        else:
            streak = 0
        self._wez_streak[agent_id] = streak

        in_wez_reward = 0.02 if in_wez else 0.0
        stick_reward  = self.w_stick * (streak / max(self.stick_max, 1))

        # ===== 5) 接近 / 脱离 塑形 =====
        last_Rh = self._last_Rh.get(agent_id, R_h)
        dR_h    = R_h - last_Rh  # >0 变远，<0 变近
        self._last_Rh[agent_id] = R_h

        approach_reward = 0.0
        if R_h > 200.0 and dR_h < 0.0:
            approach_reward = self.w_approach * float(np.clip(-dR_h / 20.0, 0.0, 1.0))

        disengage_pen = 0.0
        if 200.0 < R_h < 800.0 and dR_h > 0.0:
            disengage_pen = -self.w_disengage * float(np.clip(dR_h / 20.0, 0.0, 1.0))

        # ===== 6) 狗斗能量/速度塑形（靠近 corner speed）=====
        energy_reward = 0.0
        if R_h <= self.dogfight_Rmax:
            v_err = (my_spd - self.corner_v) / max(self.corner_v, 1e-6)
            # 在 |v_err| < corner_v_tol 时有较高奖励
            energy_shape = float(np.exp(-0.5 * (v_err ** 2) / (self.corner_v_tol ** 2)))
            energy_reward = self.w_energy * energy_shape

        # ===== 7) 低空安全 =====
        altitude_reward = 0.0
        if my_alt <= self.safe_altitude:
            depth = (self.safe_altitude - my_alt) / max(self.safe_altitude, 1.0)
            climb = max(my_vz, 0.0) / max(self.Kv, 1e-6)
            sink  = max(-my_vz, 0.0) / max(self.Kv, 1e-6)
            altitude_reward += +0.2 * depth * climb - 0.4 * depth * sink
        if my_alt <= self.danger_altitude:
            altitude_reward += -0.5

        # ===== 8) 终局奖励 =====
        terminal_bonus = 0.0
        if enm_hp <= 0.0 and my_hp > 0.0:
            terminal_bonus += self.kill_bonus
        if my_hp <= 0.0 and enm_hp > 0.0:
            terminal_bonus += self.death_pen

        # ===== 9) 软边界 =====
        x_abs = abs(float(ego_obs[0]))
        y_abs = abs(float(ego_obs[1]))
        soft_oob = (
            _soft_band_penalty_piecewise(x_abs, self.soft_band_start, self.arena_half_size) +
            _soft_band_penalty_piecewise(y_abs, self.soft_band_start, self.arena_half_size)
        ) * float(self.soft_band_weight)

        # ===== 汇总 + 裁剪 =====
        total = (
            float(event_reward)
            + self.w_posture * float(posture_reward)
            + self.w_altitude * float(altitude_reward)
            + self.w_trend   * float(trend_reward)
            + self.w_turn    * float(turn_reward)
            + float(ocz_reward)
            + float(in_wez_reward)
            + float(stick_reward)
            + float(approach_reward)
            + float(disengage_pen)
            + float(energy_reward)
            + float(terminal_bonus)
            + float(soft_oob)
            + float(extra_noeng_penalty)
        )
        total = _safe_scalar(total, -10.0, 10.0)

        # ===== 缓存更新 =====
        self._last_hp[agent_id]     = my_hp
        self._last_enm_hp[agent_id] = enm_hp
        self._last_AO[agent_id]     = AO

        # ===== 日志 =====
        components = {
            "total": float(total),
            "event": float(event_reward),
            "posture": float(posture_reward),
            "trend_AO": float(trend_reward),
            "turning": float(turn_reward),
            "altitude": float(altitude_reward),
            "terminal": float(terminal_bonus),
            "soft_oob": float(soft_oob),
            "ocz": float(ocz_reward),
            "in_wez": float(1.0 if in_wez else 0.0),
            "wez_streak": float(streak),
            "stick": float(stick_reward),
            "approach": float(approach_reward),
            "disengage": float(disengage_pen),
            "energy": float(energy_reward),
            "noeng_penalty": float(extra_noeng_penalty),
            "R_h": float(R_h),
            "R_3d": float(R_3d),
            "AO": float(AO),
            "AO_deg": float(AO_deg),
            "elev_off": float(elev_off),
            "my_spd": float(my_spd),
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
