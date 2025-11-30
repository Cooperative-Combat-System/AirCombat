import math
import numpy as np
from .termination_condition_base import BaseTerminationCondition

class MyTerminationV1(BaseTerminationCondition):
    def __init__(self, config):
        super().__init__(config)
        # 基础：步数/高度/边界
        self.max_steps       = getattr(config, "max_steps", 1000)
        self.altitude_limit  = getattr(config, "altitude_limit", 20.0)   # m（低于则坠毁）
        self.arena_size      = getattr(config, "arena_size", 550.0)      # |x|,|y| 硬边界

        # 新增：出界宽限（连续 N 步在界外才终止）
        self.oob_grace_steps = getattr(config, "oob_grace_steps", 60)    # 例如 60 步≈3s（dt=0.05）
        # 新增：无交战（长时间不靠近/不对准 判定为脱战平局）
        self.noeng_R_min     = getattr(config, "noeng_R_min", 250.0)     # 超过此距离才考虑“无交战”
        self.noeng_dR_eps    = getattr(config, "noeng_dR_eps", 2.0)      # 水平距离变化阈值 m/步（≈40m/s 用 dt 换算）
        self.noeng_dAO_eps   = getattr(config, "noeng_dAO_eps", 0.01)    # AO 变化阈值 rad/步（≈0.6°/s）
        self.noeng_patience  = getattr(config, "noeng_patience", 200)    # 连续步数容忍（≈10s）
        # 既防“刷平局”，又避免“你追我逃”无限拖延

        # 缓存
        self._last_Rh   = {}
        self._last_AO   = {}
        self._oob_cnt   = {}   # 我机出界计数
        self._enm_oob_cnt = {} # 敌机出界计数
        self._noeng_cnt = {}   # 无交战计数（按我机 perspective）

    def _init_if_first(self, env, agent_id):
        if getattr(env, "current_step", 0) <= 1:
            self._last_Rh.clear()
            self._last_AO.clear()
            self._oob_cnt.clear()
            self._enm_oob_cnt.clear()
            self._noeng_cnt.clear()
        # 初始化缺省值
        self._oob_cnt.setdefault(agent_id, 0)
        self._enm_oob_cnt.setdefault(agent_id, 0)
        self._noeng_cnt.setdefault(agent_id, 0)

    def get_termination(self, task, env, agent_id, info={}):
        agent = env.agents[agent_id]
        if agent.my_state is None:
            return False, False, info

        self._init_if_first(env, agent_id)

        my_state    = agent.my_state
        enemy_state = agent.enemy_state

        # 0) 数值异常保护
        if not np.all(np.isfinite(my_state)) or not np.all(np.isfinite(enemy_state)):
            self.log(f"[InvalidState] {agent_id} nan/inf detected")
            info["term_reason"] = "invalid_state"
            return True, False, info

        # 1) 超时
        if env.current_step >= self.max_steps:
            self.log(f"[Timeout] {agent_id} reached step limit: {env.current_step}")
            info["term_reason"] = "timeout"
            return True, False, info

        # 2) 高度终止（坠毁）
        my_alt = float(my_state[2])
        if my_alt < self.altitude_limit:
            self.log(f"[LowAltitude] {agent_id} crashed, z={my_alt:.2f}")
            info["term_reason"] = "crash_low_alt"
            return True, False, info

        # 3) 出界宽限（我方 & 敌方分别计数）
        my_x, my_y       = float(my_state[0]), float(my_state[1])
        enemy_x, enemy_y = float(enemy_state[0]), float(enemy_state[1])

        my_oob   = (abs(my_x) > self.arena_size) or (abs(my_y) > self.arena_size)
        enm_oob  = (abs(enemy_x) > self.arena_size) or (abs(enemy_y) > self.arena_size)

        self._oob_cnt[agent_id]     = self._oob_cnt.get(agent_id, 0) + (1 if my_oob else 0)
        self._enm_oob_cnt[agent_id] = self._enm_oob_cnt.get(agent_id, 0) + (1 if enm_oob else 0)

        # 同时出界到宽限 → 平局
        if self._oob_cnt[agent_id] >= self.oob_grace_steps and self._enm_oob_cnt[agent_id] >= self.oob_grace_steps:
            self.log(f"[BothOutOfBounds] {agent_id} draw after grace")
            info["term_reason"] = "both_oob"
            return True, False, info

        # 敌方出界到宽限 → 我方胜（敌人逃离/犯规）
        if self._enm_oob_cnt[agent_id] >= self.oob_grace_steps and self._oob_cnt[agent_id] < self.oob_grace_steps:
            self.log(f"[EnemyOutOfBounds] {agent_id} wins (enemy oob)")
            info["term_reason"] = "enemy_oob"
            return True, True, info

        # 我方出界到宽限 → 我方负
        if self._oob_cnt[agent_id] >= self.oob_grace_steps and self._enm_oob_cnt[agent_id] < self.oob_grace_steps:
            self.log(f"[OutOfBounds] {agent_id} lost (self oob)")
            info["term_reason"] = "self_oob"
            return True, False, info

        # 若回到圈内，计数清零（避免一次小抖动就被记很久）
        if not my_oob:
            self._oob_cnt[agent_id] = 0
        if not enm_oob:
            self._enm_oob_cnt[agent_id] = 0

        # 4) HP 终止/平局
        my_hp    = float(my_state[12])
        enemy_hp = float(enemy_state[12])
        my_dead    = (my_hp    <= 0.1)
        enemy_dead = (enemy_hp <= 0.1)

        if my_dead and enemy_dead:
            self.log(f"[MutualKill] {agent_id} both destroyed")
            info["term_reason"] = "mutual_kill"
            return True, False, info  # 平局
        if my_dead:
            self.log(f"[Shotdown] {agent_id} has 0 hp")
            info["term_reason"] = "self_dead"
            return True, False, info
        if enemy_dead:
            self.log(f"[MissionSuccess] {agent_id} eliminated all enemies")
            info["term_reason"] = "enemy_dead"
            return True, True, info

        # 5) 无交战/脱战判定（防止“越拉越远”的无限局）
        #    利用你已有的 AO / 水平距离，并用上一帧缓存估计 dR、dAO
        ego_feat = np.concatenate([my_state[0:3],  my_state[6:9]]).astype(np.float32)
        enm_feat = np.concatenate([enemy_state[0:3], enemy_state[6:9]]).astype(np.float32)
        AO, TA, R_h = task.get2d_AO_TA_R_ue(ego_feat, enm_feat)  # AO(rad), R_h(m)
        if AO is None or np.isnan(AO):  AO = 3.14159
        if R_h is None or np.isnan(R_h): R_h = 1e9
        AO  = float(np.clip(AO, 0.0, math.pi))
        R_h = float(np.clip(R_h, 0.0, 1e9))

        last_Rh = self._last_Rh.get(agent_id, R_h)
        last_AO = self._last_AO.get(agent_id, AO)
        dR      = R_h - last_Rh
        dAO     = AO  - last_AO

        # 条件：离得远（>noeng_R_min）且既不靠近也不对准（|dR| 小，|dAO| 小）
        if R_h > self.noeng_R_min and abs(dR) < self.noeng_dR_eps and abs(dAO) < self.noeng_dAO_eps:
            self._noeng_cnt[agent_id] = self._noeng_cnt.get(agent_id, 0) + 1
        else:
            self._noeng_cnt[agent_id] = 0

        if self._noeng_cnt[agent_id] >= self.noeng_patience:
            self.log(f"[NoEngagement] {agent_id} draw: R={R_h:.1f}, dR={dR:.2f}, dAO={dAO:.3f}")
            info["term_reason"] = "no_engagement"
            return True, False, info

        # 缓存更新
        self._last_Rh[agent_id] = R_h
        self._last_AO[agent_id] = AO

        return False, False, info
