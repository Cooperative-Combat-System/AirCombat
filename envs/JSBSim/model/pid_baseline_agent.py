import numpy as np
from ..utils.pid_controller import PID, PIDController

class DogFightController(PIDController):
    """
    基于你现有 PIDController 的狗斗版本：
      - 模式：INTERCEPT / ATTACK
      - 高度：不强制等高敌机，而是守战术高度带（相对初始高度）
      - 修复关键：把 body(u,v,w) -> world(vx,vy,vz) 再喂给 AltitudeHold
    """

    # --- 战术速度 ---
    V_TGT_INT   = 22.0
    V_TGT_ATK   = 20.0
    V_MIN       = 12.0
    V_SLOW      = 16.0
    V_FAST      = 28.0

    # --- 战术高度带（相对初始高度 h0） ---
    H_BAND_LOW  = 50.0
    H_BAND_HIGH = 200.0
    H_KEEP_BIAS = 0.0

    # --- 模式阈值 ---
    ATTACK_DIST  = 250.0
    ATTACK_ANGLE = np.deg2rad(60.0)   # 前半球

    # --- 高度/对准融合 ---
    H_BAND     = 20.0
    W_ALT_MIN  = 0.15
    W_ALT_MAX  = 0.75

    # --- 近距防穿越（可选） ---
    CLOSE_X    = 30.0
    CLOSE_GAIN = 0.6

    # --- 横向参数（模式化） ---
    BANK_K_INT   = 2.0
    BANK_K_ATK   = 3.2
    BANK_MAX_INT = np.deg2rad(60.0)
    BANK_MAX_ATK = np.deg2rad(80.0)
    BEHIND_BANK  = np.deg2rad(70.0)

    # --- 掉头更快（对狗斗很重要） ---
    PHI_DOT_MAX_INT = np.deg2rad(60.0)
    PHI_DOT_MAX_ATK = np.deg2rad(100.0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # 复用你 PIDController 的 PID/AltitudeHold/限幅等
        self.h0 = None

    def reset(self, my=None, enemy=None):
        super().reset(my, enemy)
        if my is not None:
            self.h0 = float(my[2]) * self.pos_scale
        else:
            self.h0 = None

    def _decide_mode(self, r_b, dist):
        los_angle = np.arccos(
            np.clip(r_b[0] / (np.linalg.norm(r_b) + 1e-6), -1.0, 1.0)
        )
        in_front = (los_angle <= self.ATTACK_ANGLE)
        close_enough = (dist <= self.ATTACK_DIST * self.pos_scale)
        return "ATTACK" if (in_front and close_enough) else "INTERCEPT"

    def step(self, my, enemy, dt):
        # ========= 0) 位置尺度 =========
        my_pos = np.asarray(my[:3], dtype=np.float64) * self.pos_scale
        en_pos = np.asarray(enemy[:3], dtype=np.float64) * self.pos_scale

        # ========= 1) 姿态/旋转 =========
        euler = np.asarray(my[3:6], dtype=np.float64)  # 假设 roll,pitch,yaw (rad)
        R_wb = self._world_from_body(euler)            # 复用父类
        R_bw = R_wb.T

        # 相对向量
        r_w = en_pos - my_pos
        r_b = R_bw @ r_w
        dist = float(np.linalg.norm(r_w))

        # ========= 2) 速度：body -> world（修复螺旋上升关键）=========
        v_b = np.asarray(my[6:9], dtype=np.float64)  # body u,v,w
        v_w = R_wb @ v_b                              # world vx,vy,vz (z-up)
        speed = float(np.linalg.norm(v_b))
        vxvy  = float(np.linalg.norm(v_w[:2]))
        vz    = float(v_w[2])                         # world vertical speed (up positive)

        roll  = float(my[3])
        theta = float(my[4])

        behind = (r_b[0] < 0.0)

        # ========= 3) 模式 =========
        mode = self._decide_mode(r_b, dist)

        # ========= 4) 战术高度带（不追敌机高度）=========
        h = float(my_pos[2])
        if self.h0 is None:
            self.h0 = h

        h_ref_base = self.h0 + self.H_KEEP_BIAS
        h_ref = float(np.clip(h_ref_base, self.H_BAND_LOW, self.H_BAND_HIGH))
        h_ref = float(np.clip(h_ref, self.h_floor, self.h_ceil))

        theta_alt, e_h = self.alt_hold.step(
            h=h, h_ref=h_ref,
            vxvy=vxvy, vz=vz,
            theta=theta, roll=roll,
            dt=dt
        )
        e_pitch_alt = theta_alt - theta

        # ========= 5) 对准误差（body）=========
        eps = 1e-6
        e_pitch_align = np.arctan2(-r_b[2], r_b[0] + eps)
        e_yaw_align   = np.arctan2( r_b[1], r_b[0] + eps)

        # ========= 6) 俯仰融合：高度 vs 对准 + 能量保护 =========
        if abs(e_h) > self.H_BAND * self.pos_scale:
            w_alt = self.W_ALT_MAX
        else:
            scale = np.clip(abs(e_h) / (self.H_BAND * self.pos_scale), 0.0, 1.0)
            w_alt = self.W_ALT_MIN + (self.W_ALT_MAX - self.W_ALT_MIN) * scale

        if mode == "ATTACK":
            w_alt *= 0.5  # 攻击时更重视对准

        # 能量不足：压头、减小高度占比
        if speed < self.V_SLOW:
            w_alt *= 0.7
            e_pitch_align -= np.deg2rad(3.0)

        e_pitch = (1.0 - w_alt) * e_pitch_align + w_alt * e_pitch_alt
        pitch_lim = np.deg2rad(22.0 if mode == "ATTACK" else 18.0)
        e_pitch = float(np.clip(e_pitch, -pitch_lim, pitch_lim))

        # ========= 7) 横向：bank-to-turn + behind 掉头 =========
        if behind:
            phi_cmd = np.sign(e_yaw_align) * self.BEHIND_BANK
            phi_dot_max = self.PHI_DOT_MAX_ATK
            bank_cap = self.BANK_MAX_ATK
        else:
            if mode == "ATTACK":
                bank_k, bank_cap, phi_dot_max = self.BANK_K_ATK, self.BANK_MAX_ATK, self.PHI_DOT_MAX_ATK
            else:
                bank_k, bank_cap, phi_dot_max = self.BANK_K_INT, self.BANK_MAX_INT, self.PHI_DOT_MAX_INT
            phi_cmd = float(np.clip(bank_k * e_yaw_align, -bank_cap, bank_cap))

        # 复用父类的“phi_cmd 平滑”逻辑，但用我们当前的 phi_dot_max
        dphi = np.clip(phi_cmd - self.phi_cmd_prev, -phi_dot_max * dt, phi_dot_max * dt)
        phi_cmd = self.phi_cmd_prev + dphi
        self.phi_cmd_prev = phi_cmd

        e_roll = float(phi_cmd - roll)

        # ========= 8) 油门：按模式给不同 V_TGT =========
        self.V_TGT = self.V_TGT_ATK if mode == "ATTACK" else self.V_TGT_INT
        e_v = float(self.V_TGT - speed)
        thr = float(self.pid_thr(e_v, dt))

        # 前馈（保持你原本做法）
        gamma_cmd = theta_alt - self.alt_hold.alpha_ref
        thr += self.THR_FF_G * abs(gamma_cmd)
        thr += self.THR_FF_B * (1.0 - np.cos(abs(roll)))
        thr = float(np.clip(thr, 0.4, 1.0))

        if speed < self.V_MIN:
            thr = 1.0
            e_pitch -= np.deg2rad(5.0)

        # # 近距防穿越（可选）
        # if r_b[0] < self.CLOSE_X * self.pos_scale:
        #     k = (self.CLOSE_X * self.pos_scale - max(r_b[0], 0.0)) / (self.CLOSE_X * self.pos_scale)
        #     thr = float(np.clip(thr - self.CLOSE_GAIN * k, 0.4, 1.0))
        #     e_pitch = float(e_pitch + self.CLOSE_GAIN * 0.2 * k)

        # ========= 9) PID 输出：复用父类 yaw 协调（更稳）=========
        pit_cmd = float(self.pid_pitch(e_pitch, dt))
        rol_cmd = float(self.pid_roll(e_roll, dt))

        # yaw：对准 + 协调转弯（沿用你父类那套更稳）
        v_side = float(v_b[1])
        yaw_coord = np.clip(-0.08 * v_side, -0.3, 0.3)
        yaw_align = float(self.pid_yaw(e_yaw_align, dt))
        if behind:
            yaw_align *= 0.6
        yaw_cmd = float(np.clip(yaw_align + yaw_coord, -1.0, 1.0))

        # ========= 10) 变化率限幅：沿用父类 last_u =========
        u = np.array([thr, pit_cmd, rol_cmd, yaw_cmd], dtype=np.float32)

        du = np.array([self.DLIM_THR, self.DLIM_ATT, self.DLIM_ATT, self.DLIM_YAW], dtype=np.float32)
        u = np.clip(u, self.last_u - du, self.last_u + du)

        u[0]  = float(np.clip(u[0], 0.4, 1.0))
        u[1:] = np.clip(u[1:], -1.0, 1.0)
        self.last_u[:] = u

        return u
