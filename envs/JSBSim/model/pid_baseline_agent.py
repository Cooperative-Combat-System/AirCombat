import numpy as np
from ..utils.pid_controller import PID, AltitudeHold
from scipy.spatial.transform import Rotation as R

EPS = 1e-6
def wrap_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class DogFightController:
    """
        输出动作: [throttle(0.4~1.0), pitch(-1~1), roll(-1~1), yaw(-1~1)]

        关键增强点（相对你旧版）：
          1) 用姿态把 body u/v/w 转到 world，得到 vxvy & vz_world（高度环稳定很多）
          2) 转弯升力补偿加到 pitch（bank大时不再明显掉高度）
          3) behind 掉头分阶段 + phi_cmd 平滑（避免硬切绕圈）
          4) yaw = 对准 + 协调转弯（抑制侧滑，提高左右一致性）
        """

    # —— 速度 / 高度参数 —— #
    V_TGT = 20.0  # 期望空速（按你的量级）
    V_MIN = 10.0  # 低速保护阈值
    VZ_MAX = 10.0  # 爬升/下降率上限 (m/s)

    # —— 高度-对准权重相关 —— #
    H_BAND = 5.0
    H_ALT_FOCUS = 10.0
    W_ALT_MIN = 0.2
    W_ALT_MAX = 0.7

    # —— 变化率限幅 —— #
    DLIM_THR = 0.10
    DLIM_ATT = 0.55
    DLIM_YAW = 1.0

    # —— 油门前馈 —— #
    THR_FF_G = 0.35
    THR_FF_B = 0.20

    # —— bank-to-turn 参数 —— #
    BANK_K = 5.0
    BANK_MAX = np.deg2rad(80.0)
    BEHIND_BANK = np.deg2rad(60.0)

    # —— 新增：phi_cmd 平滑 —— #
    PHI_DOT_MAX = np.deg2rad(150.0)  # bank目标变化率上限(rad/s)

    # ===== 新增：锁盒参数（按平台 2×2×60）=====
    LOCK_X_ON = 90.0  # 进入锁盒模式的前向距离阈值（m）
    LOCK_X_OFF = 120.0  # 退出锁盒模式阈值（m）(滞回防抖)
    BOX_X_MAX = 60.0  # 平台盒子长度
    BOX_HALF_Y = 5.0  # 盒子半宽  (2m)
    BOX_HALF_Z = 5.0  # 盒子半高  (2m)

    # 锁盒控制强度（核心）
    KY_LAT = 0.65  # y(横向) -> bank 的强度（tanh 输入系数）
    KZ_LAT = 0.90  # z(竖向) -> pitch 的强度（tanh 输入系数）
    LOCK_PITCH_MAX = np.deg2rad(10.0)  # 锁盒阶段 pitch 修正限幅（别太大，容易抖）

    # 速度方向对齐 LOS（flight-path alignment）
    K_CHI = 0.8  # 航迹角误差 -> bank 修正（rad -> rad）
    K_GAMMA = 0.9  # 爬升角误差 -> pitch 修正（rad -> rad）

    # 近距离降速（驻留）
    V_HOLD = 16.0  # 进入盒子附近（<60m）希望降到的速度
    V_HOLD2 = 14.0  # 更近（<30m）再降一点，增加驻留

    def __init__(self,
                 v_min=12.0,
                 h_bias=0.0,
                 h_floor=10.0,
                 h_ceil=500.0,
                 # 如果你的位置单位是“10m”，建议 pos_scale=10.0；如果你已换算成米就设1.0
                 pos_scale=1.0):
        self.v_min = v_min
        self.h_bias = h_bias
        self.h_floor = h_floor
        self.h_ceil = h_ceil
        self.pos_scale = float(pos_scale)

        # 速度 / 姿态 PID
        self.pid_thr = PID(
            kp=0.30, ki=0.06, kd=0.02,
            out_min=0.4, out_max=1.0,
            i_min=-0.3, i_max=0.3, d_lp=0.5
        )
        self.pid_pitch = PID(
            kp=1.10, ki=0.00, kd=0.22,
            out_min=-1.0, out_max=1.0, d_lp=0.5
        )
        self.pid_yaw = PID(
            kp=3.00, ki=0.00, kd=0.40,
            out_min=-1.0, out_max=1.0, d_lp=0.5
        )
        self.pid_roll = PID(
            kp=3.0, ki=0.00, kd=0.45,
            out_min=-1.0, out_max=1.0, d_lp=0.5
        )

        # 高度保持模块
        self.alt_hold = AltitudeHold(
            kh_p=0.8, kh_i=0.08,
            vz_max=self.VZ_MAX,
            alpha_ref_deg=5.0,
            alpha_lim_deg=12.0,
            pitch_bnk_deg=12.0,
            bnk_ref_deg=60.0,
            kd_vz=1.2
        )

        self.last_u = np.array([0.7, 0.0, 0.0, 0.0], dtype=np.float32)
        self.phi_cmd_prev = 0.0
        self._lock_mode = False
        self.chi_cmd_prev = 0.0
        self.gamma_cmd_prev = 0.0
        self.V_cmd_prev = self.V_TGT

    def reset(self, my=None, enemy=None):
        self.pid_thr.reset()
        self.pid_pitch.reset()
        self.pid_yaw.reset()
        self.pid_roll.reset()
        self.alt_hold.reset()
        self.last_u[:] = [0.7, 0.0, 0.0, 0.0]
        self.phi_cmd_prev = 0.0

    def _world_from_body(self, euler_xyz):
        """
        返回 R_wb: body->world
        注意：这里用 SciPy 的 from_euler('xyz')；若你发现左右不对称仍存在，
        建议把这里替换成你平台明确的旋转矩阵定义。
        """
        R_wb = R.from_euler('xyz', euler_xyz).as_matrix()
        return R_wb

    def step(self, my, enemy, dt):
        # =========================
        # 0) 统一位置尺度
        # =========================
        my_pos = np.asarray(my[:3], dtype=np.float64) * self.pos_scale
        en_pos = np.asarray(enemy[:3], dtype=np.float64) * self.pos_scale

        # =========================
        # 1) 姿态与相对几何
        # =========================
        roll = float(my[3])
        theta = float(my[4])
        yaw = float(my[5])

        euler = np.asarray([roll, theta, yaw], dtype=np.float64)
        R_wb = self._world_from_body(euler)  # body->world
        R_bw = R_wb.T  # world->body

        r_w = en_pos - my_pos
        r_b = R_bw @ r_w

        x_fwd = float(r_b[0])
        y_lat = float(r_b[1])
        z_lat = float(r_b[2])

        # =========================
        # 2) ✅ 速度：world 系 (vx,vy,vz)
        # =========================
        v_w = np.asarray(my[6:9], dtype=np.float64)  # world vx,vy,vz (z-up)
        speed = float(np.linalg.norm(v_w))

        vxvy = float(np.linalg.norm(v_w[:2]))
        vz = float(v_w[2])

        # 协调转弯/侧滑抑制需要 body v：把 world 速度转回 body
        v_b = R_bw @ v_w
        v_side = float(v_b[1])

        # =========================
        # 3) 航迹(chi,gamma) vs LOS(chi_los,gamma_los) 误差（world）
        # =========================
        chi = float(np.arctan2(v_w[1], v_w[0] + EPS))
        gamma = float(np.arctan2(v_w[2], vxvy + EPS))

        chi_los = float(np.arctan2(r_w[1], r_w[0] + EPS))
        gamma_los = float(np.arctan2(r_w[2], np.linalg.norm(r_w[:2]) + EPS))

        dchi = wrap_pi(chi_los - chi)
        dgamma = float(np.clip(gamma_los - gamma, -np.deg2rad(12.0), np.deg2rad(12.0)))

        # =========================
        # 4) 高度保持（AltitudeHold 输入需要 world vxvy/vz ✅）
        # =========================
        h_ref = float(np.clip(float(enemy[2]) + self.h_bias, self.h_floor, self.h_ceil)) * self.pos_scale
        h = float(my[2]) * self.pos_scale

        theta_alt, e_h = self.alt_hold.step(
            h=h, h_ref=h_ref,
            vxvy=vxvy, vz=vz,
            theta=theta, roll=roll,
            dt=dt
        )
        e_pitch_alt = float(theta_alt - theta)

        # 机头对准角（body LOS）
        e_pitch_align = float(np.arctan2(-r_b[2], r_b[0] + EPS))

        # =========================
        # 5) 进入/退出锁盒模式（滞回）
        # =========================
        if not self._lock_mode:
            if (x_fwd > 0.0) and (x_fwd < self.LOCK_X_ON):
                self._lock_mode = True
        else:
            if (x_fwd <= 0.0) or (x_fwd > self.LOCK_X_OFF):
                self._lock_mode = False

        # =========================
        # 6) 高度权重 w_alt（方案A：只在 close+lock 时降低）
        #    ✅ 已去掉 “|e_h|<=5 降权” 的规则
        # =========================
        eh = abs(e_h)

        # base：远距按高度误差渐进增加高度权重
        if eh > self.H_ALT_FOCUS * self.pos_scale:
            w_alt = 1.0
        else:
            scale = np.clip(eh / (self.H_BAND * self.pos_scale), 0.0, 1.0)
            w_alt = self.W_ALT_MIN + (self.W_ALT_MAX - self.W_ALT_MIN) * scale

        # 只在 close+lock 才降低高度权重（让位给盒子几何控制）
        close_and_lock = (self._lock_mode and (x_fwd > 0.0) and (x_fwd < self.BOX_X_MAX))
        if close_and_lock:
            w_alt = max(w_alt, 0.25)  # 保留高度托底，防慢慢掉高
            w_alt *= 0.55

        w_alt = float(np.clip(w_alt, 0.20, 1.0))

        # =========================
        # 7) 俯仰误差：高度 vs 对准 + 过载补偿 + 航迹对齐 + 锁盒
        # =========================
        e_pitch = (1.0 - w_alt) * e_pitch_align + w_alt * e_pitch_alt

        # 转弯升力/过载补偿（更抗转弯掉高）
        bank = abs(roll)
        n_req = 1.0 / max(np.cos(bank), 0.35)
        pitch_bank_bias = np.clip((n_req - 1.0) * np.deg2rad(7.0), 0.0, np.deg2rad(12.0))
        e_pitch += float(pitch_bank_bias)

        # 航迹爬升角对齐：把速度方向压向 LOS
        e_pitch += float(0.55 * self.K_GAMMA * dgamma)

        # 锁盒：用 z_lat 压到盒子中心（优先几何，而不是 e_pitch_align 抖动）
        if self._lock_mode and (x_fwd > 0.0):
            e_pitch_lock = - self.LOCK_PITCH_MAX * np.tanh(self.KZ_LAT * (z_lat / (self.BOX_HALF_Z + EPS)))
            e_pitch_lock_total = 0.85 * e_pitch_lock + 0.55 * (self.K_GAMMA * dgamma)

            # lock 时：主跟随 lock_total，少量高度托底，极少对准项
            e_pitch = 0.70 * e_pitch_lock_total + 0.25 * e_pitch_alt + 0.05 * e_pitch_align

        e_pitch = float(np.clip(e_pitch, -np.deg2rad(20.0), np.deg2rad(20.0)))

        # =========================
        # 8) 横向控制：phi_cmd（az + dchi + 锁盒 y_lat）
        #    ✅ 暂时不做 behind 专用控制
        # =========================
        bearing = float(np.arctan2((en_pos[1] - my_pos[1]), (en_pos[0] - my_pos[0]) + EPS))
        az = wrap_pi(bearing - yaw)
        az_abs = abs(az)

        # bank_cap：不要过分保守，否则“转不起来”
        if eh > 20.0 * self.pos_scale:
            bank_cap = np.deg2rad(45.0)
        else:
            bank_cap = self.BANK_MAX

        # 更像“拦截/追踪”的 bank 指令：heading误差 + 航迹误差
        phi_base = self.BANK_K * az + 0.85 * self.K_CHI * dchi
        phi_base = float(np.clip(phi_base, -bank_cap, bank_cap))

        if self._lock_mode and (x_fwd > 0.0):
            phi_lock = bank_cap * np.tanh(self.KY_LAT * (y_lat / (self.BOX_HALF_Y + EPS)))
            phi_fp = float(np.clip(self.K_CHI * dchi, -np.deg2rad(25.0), np.deg2rad(25.0)))
            phi_cmd = 0.60 * phi_lock + 0.30 * phi_fp + 0.10 * phi_base
        else:
            phi_cmd = phi_base

        phi_cmd = float(np.clip(phi_cmd, -bank_cap, bank_cap))

        # phi_cmd 平滑（转弯不够：优先增大 PHI_DOT_MAX）
        dphi = np.clip(phi_cmd - self.phi_cmd_prev,
                       -self.PHI_DOT_MAX * dt, self.PHI_DOT_MAX * dt)
        phi_cmd = self.phi_cmd_prev + dphi
        self.phi_cmd_prev = phi_cmd

        e_roll = float(phi_cmd - roll)

        # =========================
        # 9) 油门：锁盒近距降速驻留 + 前馈补偿
        # =========================
        V_ref = self.V_TGT
        if self._lock_mode and (x_fwd > 0.0):
            if x_fwd < 30.0:
                V_ref = self.V_HOLD2
            elif x_fwd < self.BOX_X_MAX:
                V_ref = self.V_HOLD

        e_v = float(V_ref - speed)
        thr = float(self.pid_thr(e_v, dt))

        gamma_cmd = float(theta_alt - self.alt_hold.alpha_ref)
        thr += float(self.THR_FF_G * abs(gamma_cmd))
        thr += float(self.THR_FF_B * (1.0 - np.cos(abs(roll))))
        thr = float(np.clip(thr, 0.4, 1.0))

        if speed < self.V_MIN:
            thr = 1.0
            e_pitch -= np.deg2rad(6.0)

        # =========================
        # 10) PID 输出
        # =========================
        pit_cmd = float(self.pid_pitch(e_pitch, dt))
        rol_cmd = float(self.pid_roll(e_roll, dt))

        # yaw：对准 + 协调转弯（侧滑来自 body v）
        yaw_coord = float(np.clip(-0.10 * v_side, -0.35, 0.35))
        yaw_align = float(self.pid_yaw(az, dt))
        yaw_cmd = float(np.clip(yaw_align + yaw_coord, -1.0, 1.0))

        u = np.array([thr, pit_cmd, rol_cmd, yaw_cmd], dtype=np.float32)

        # 变化率限幅
        du = np.array([self.DLIM_THR, self.DLIM_ATT, self.DLIM_ATT, self.DLIM_YAW], dtype=np.float32)
        u = np.clip(u, self.last_u - du, self.last_u + du)

        # 最终限幅
        u[0] = float(np.clip(u[0], 0.4, 1.0))
        u[1:] = np.clip(u[1:], -1.0, 1.0)

        self.last_u[:] = u
        return u
