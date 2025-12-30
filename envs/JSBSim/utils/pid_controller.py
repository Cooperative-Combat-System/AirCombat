import numpy as np
from scipy.spatial.transform import Rotation as R

TICK_HZ = 60.0
DT      = 1.0 / TICK_HZ

class PID:
    def __init__(self, kp, ki, kd, out_min=-1., out_max=1., i_min=-np.inf, i_max=np.inf, d_lp=0.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.i_min, self.i_max = i_min, i_max
        self.i = 0.0
        self.prev_x = 0.0
        self.prev_y = 0.0
        self.d_lp = d_lp  # 0=不用；(0,1) 为D项低通平滑系数

    def reset(self):
        self.i = 0.0
        self.prev_x = 0.0
        self.prev_y = 0.0

    def __call__(self, err, dt):
        # P
        p = self.kp * err
        # I (anti-windup)
        self.i = np.clip(self.i + self.ki * err * dt, self.i_min, self.i_max)
        # D（对误差做滤波微分）
        raw_d = (err - self.prev_x) / max(dt, 1e-6)
        d = self.d_lp * self.prev_y + (1 - self.d_lp) * raw_d if self.d_lp > 0 else raw_d
        self.prev_x, self.prev_y = err, d
        u = p + self.i + self.kd * d
        return float(np.clip(u, self.out_min, self.out_max))


class AltitudeHold:
    """
    高度保持：
      高度误差 -> 期望爬升率 vz_cmd -> 轨迹角 gamma_cmd -> 目标俯仰 theta_alt
    注意：这里的 vxvy / vz 都应当是 WORLD 系（水平速度模 & 垂直速度，上为正）
    """
    def __init__(self,
                 kh_p=0.35, kh_i=0.06,
                 vz_max=8.0,
                 alpha_ref_deg=4.0,
                 alpha_lim_deg=10.0,
                 pitch_bnk_deg=6.0,
                 bnk_ref_deg=60.0,
                 kd_vz=1.5):
        self.kh_p = kh_p
        self.kh_i = kh_i
        self.vz_max = vz_max
        self.kd_vz = kd_vz

        self.alpha_ref = np.deg2rad(alpha_ref_deg)       # 巡航迎角
        self.alpha_lim = np.deg2rad(alpha_lim_deg)

        self.pitch_bnk = np.deg2rad(pitch_bnk_deg)
        self.bnk_ref = np.deg2rad(bnk_ref_deg)

        self.h_int = 0.0

    def reset(self):
        self.h_int = 0.0

    def step(self, h, h_ref, vxvy, vz, theta, roll, dt):
        # 高度误差 + 积分
        e_h = h_ref - h
        self.h_int += e_h * dt

        # 垂直速度阻尼：往下掉(vz<0)就加爬升率
        vz_damp = -self.kd_vz * vz

        # 高度 PI + 垂直速度阻尼 -> 目标爬升率
        vz_cmd = self.kh_p * e_h + self.kh_i * self.h_int + vz_damp
        vz_cmd = np.clip(vz_cmd, -self.vz_max, self.vz_max)

        # 目标轨迹角 gamma_cmd
        if vxvy < 1e-6:
            gamma_cmd = 0.0
        else:
            gamma_cmd = np.arctan2(vz_cmd, vxvy)

        # 使用固定巡航迎角作为基础抬头（避免自然下坠）
        alpha_cmd = np.clip(self.alpha_ref, -self.alpha_lim, self.alpha_lim)

        # 转弯抬头补偿：|roll| 越大，需要额外抬头越多
        bank_gain = np.clip(abs(roll) / self.bnk_ref, 0.0, 1.5)
        bank_bias = self.pitch_bnk * bank_gain

        theta_alt = gamma_cmd + alpha_cmd + bank_bias
        return theta_alt, e_h


class PIDController:
    """
    输出动作: [throttle(0.4~1.0), pitch(-1~1), roll(-1~1), yaw(-1~1)]

    关键增强点（相对你旧版）：
      1) 用姿态把 body u/v/w 转到 world，得到 vxvy & vz_world（高度环稳定很多）
      2) 转弯升力补偿加到 pitch（bank大时不再明显掉高度）
      3) behind 掉头分阶段 + phi_cmd 平滑（避免硬切绕圈）
      4) yaw = 对准 + 协调转弯（抑制侧滑，提高左右一致性）
    """

    # —— 速度 / 高度参数 —— #
    V_MIN     = 10.0      # 低速保护阈值
    VZ_MAX    = 10.0      # 爬升/下降率上限 (m/s)

    # —— 高度-对准权重相关 —— #
    H_BAND        = 5.0
    H_ALT_FOCUS   = 10.0
    W_ALT_MIN     = 0.2
    W_ALT_MAX     = 0.7

    # —— 变化率限幅 —— #
    DLIM_THR  = 0.10
    DLIM_ATT  = 0.30
    DLIM_YAW  = 1.0

    # —— 油门前馈 —— #
    THR_FF_G  = 0.35
    THR_FF_B  = 0.20

    # —— bank-to-turn 参数 —— #
    BANK_K      = 3.0
    BANK_MAX    = np.deg2rad(80.0)
    BEHIND_BANK = np.deg2rad(70.0)

    # —— 新增：phi_cmd 平滑 —— #
    PHI_DOT_MAX = np.deg2rad(60.0)  # bank目标变化率上限(rad/s)

    def __init__(self,
                 v_min=12.0,
                 V_TGT=20.0,
                 h_bias=0.0,
                 h_floor=10.0,
                 h_ceil=500.0,
                 # 如果你的位置单位是“10m”，建议 pos_scale=10.0；如果你已换算成米就设1.0
                 pos_scale=1.0):
        self.v_min   = v_min
        self.V_TGT   = V_TGT
        self.h_bias  = h_bias
        self.h_floor = h_floor
        self.h_ceil  = h_ceil
        self.pos_scale = float(pos_scale)

        # 速度 / 姿态 PID
        self.pid_thr   = PID(
            kp=0.30, ki=0.06, kd=0.02,
            out_min=0.4, out_max=1.0,
            i_min=-0.3, i_max=0.3, d_lp=0.5
        )
        self.pid_pitch = PID(
            kp=1.10, ki=0.00, kd=0.22,
            out_min=-1.0, out_max=1.0, d_lp=0.5
        )
        self.pid_yaw   = PID(
            kp=3.00, ki=0.00, kd=0.40,
            out_min=-1.0, out_max=1.0, d_lp=0.5
        )
        self.pid_roll  = PID(
            kp=1.4, ki=0.00, kd=0.25,
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
        """
        my    : 自机状态向量
        enemy : 敌机状态向量
        dt    : 仿真步长
        """

        # =========================
        # 0) 统一位置尺度（如果你的位置单位是10m，就 pos_scale=10）
        # =========================
        my_pos = np.asarray(my[:3], dtype=np.float64) * self.pos_scale
        en_pos = np.asarray(enemy[:3], dtype=np.float64) * self.pos_scale

        # =========================
        # 1) 基本状态量、相对几何（r_b）
        # =========================
        euler = np.asarray(my[3:6], dtype=np.float64)  # roll,pitch,yaw (rad)

        R_wb = self._world_from_body(euler)            # body->world
        R_bw = R_wb.T                                  # world->body

        r_w  = en_pos - my_pos
        r_b  = R_bw @ r_w

        v_b  = np.asarray(my[6:9], dtype=np.float64)   # body u,v,w (m/s)
        v_w  = R_wb @ v_b                              # ✅ world velocity

        speed = float(np.linalg.norm(v_b))
        vxvy  = float(np.linalg.norm(v_w[:2]))         # ✅ world horizontal speed
        vz    = float(v_w[2])                          # ✅ world vertical speed (up positive)

        roll  = float(my[3])
        theta = float(my[4])

        # =========================
        # 2) 高度保持 -> theta_alt
        # =========================
        h_ref = float(np.clip(float(enemy[2]) + self.h_bias,
                              self.h_floor, self.h_ceil)) * self.pos_scale
        h     = float(my[2]) * self.pos_scale

        theta_alt, e_h = self.alt_hold.step(
            h=h, h_ref=h_ref,
            vxvy=vxvy, vz=vz,
            theta=theta, roll=roll,
            dt=dt
        )
        e_pitch_alt = theta_alt - theta

        # =========================
        # 3) 几何对准误差（俯仰 + 水平）
        # =========================
        eps = 1e-6
        e_pitch_align = np.arctan2(-r_b[2], r_b[0] + eps)
        e_yaw_align   = np.arctan2( r_b[1], r_b[0] + eps)

        behind = (r_b[0] < 0.0)

        # =========================
        # 4) 俯仰融合：高度 vs 对准
        # =========================
        if abs(e_h) > self.H_ALT_FOCUS * self.pos_scale:
            w_alt = 1.0
        else:
            scale = np.clip(abs(e_h) / (self.H_BAND * self.pos_scale), 0.0, 1.0)
            w_alt = self.W_ALT_MIN + (self.W_ALT_MAX - self.W_ALT_MIN) * scale

        if behind:
            w_alt = max(w_alt, 0.8)

        e_pitch = (1.0 - w_alt) * e_pitch_align + w_alt * e_pitch_alt
        e_pitch = np.clip(e_pitch, -np.deg2rad(18.0), np.deg2rad(18.0))

        # ✅ 转弯升力补偿（补到 pitch）
        bank = abs(roll)
        n_req = 1.0 / max(np.cos(bank), 0.35)
        pitch_bank_bias = np.clip((n_req - 1.0) * np.deg2rad(6.0), 0.0, np.deg2rad(10.0))
        e_pitch += pitch_bank_bias

        # =========================
        # 4.b) 横向控制：az -> 目标滚转角（分阶段掉头 + 平滑）
        # =========================
        az = e_yaw_align
        az_abs = abs(az)

        # 高度误差大时限制 bank（先保高度）
        bank_cap = np.deg2rad(25.0) if abs(e_h) > 8.0 * self.pos_scale else self.BANK_MAX

        if behind or az_abs > np.deg2rad(90.0):
            # Turn-back: bank 随 az_abs 增大（90->较大，180->最大）
            k = np.clip((az_abs - np.deg2rad(90.0)) / np.deg2rad(90.0), 0.0, 1.0)
            phi_mag = (0.6 + 0.4 * k) * self.BEHIND_BANK
            phi_cmd = np.sign(az) * phi_mag
        else:
            phi_cmd = self.BANK_K * az

        phi_cmd = float(np.clip(phi_cmd, -bank_cap, bank_cap))

        # ✅ phi_cmd 平滑（限制 bank 目标变化率）
        dphi = np.clip(phi_cmd - self.phi_cmd_prev,
                       -self.PHI_DOT_MAX * dt, self.PHI_DOT_MAX * dt)
        phi_cmd = self.phi_cmd_prev + dphi
        self.phi_cmd_prev = phi_cmd

        e_roll = phi_cmd - roll

        # =========================
        # 5) 油门控制：空速 PI + 前馈
        # =========================
        e_v  = self.V_TGT - speed
        thr  = self.pid_thr(e_v, dt)

        gamma_cmd = theta_alt - self.alt_hold.alpha_ref
        thr += self.THR_FF_G * abs(gamma_cmd)
        thr += self.THR_FF_B * (1.0 - np.cos(abs(roll)))
        thr = float(np.clip(thr, 0.4, 1.0))

        # 低速保护：直接满油门 + 压头
        if speed < self.V_MIN:
            thr = 1.0
            e_pitch -= np.deg2rad(5.0)

        # =========================
        # 7) PID 输出
        # =========================
        pit_cmd = self.pid_pitch(e_pitch, dt)
        rol_cmd = self.pid_roll(e_roll, dt)

        # ✅ yaw：对准 + 协调转弯（抑制侧滑）
        v_side = float(v_b[1])  # body v
        yaw_coord = np.clip(-0.08 * v_side, -0.3, 0.3)

        yaw_align = self.pid_yaw(az, dt)
        if behind:
            yaw_align *= 0.6

        yaw_cmd = float(np.clip(yaw_align + yaw_coord, -1.0, 1.0))

        u  = np.array([thr, pit_cmd, rol_cmd, yaw_cmd], dtype=np.float32)
        du = np.array([self.DLIM_THR, self.DLIM_ATT, self.DLIM_ATT, self.DLIM_YAW], dtype=np.float32)

        # 控制量变化率限幅
        u = np.clip(u, self.last_u - du, self.last_u + du)

        # 最终限幅
        u[0]  = float(np.clip(u[0], 0.4, 1.0))
        u[1:] = np.clip(u[1:], -1.0, 1.0)

        self.last_u[:] = u

        # 调试输出（可注释掉）
        # print(f"h={h:.2f}, h_ref={h_ref:.2f}, e_h={e_h:.2f}, w_alt={w_alt:.2f}, "
        #       f"az={az:.3f}, phi_cmd={phi_cmd:.3f}, roll={roll:.3f}, vz={vz:.2f}")

        return u
