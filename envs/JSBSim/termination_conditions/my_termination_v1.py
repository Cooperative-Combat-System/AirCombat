import math
import numpy as np
from .termination_condition_base import BaseTerminationCondition

def _bearing(dx, dy):
    # 世界系方位角
    return math.atan2(dy, dx)

def _angle_wrap_pi(a):
    # 把角度 wrap 到 [-pi, pi]
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


class MyTerminationV1(BaseTerminationCondition):
    def __init__(self, config):
        super().__init__(config)
        # 基础：步数/高度/边界
        self.max_steps      = getattr(config, "max_steps", 3000)
        self.altitude_limit = getattr(config, "altitude_limit", 30.0)
        self.arena_size     = getattr(config, "arena_size", 500.0)

        # 出界宽限
        self.oob_grace_steps = getattr(config, "oob_grace_steps", 60)

        # 无交战判定参数
        self.enable_noeng     = getattr(config, "enable_noeng", False)  # ★ 训练早期建议 False
        self.noeng_R_min      = getattr(config, "noeng_R_min", 250.0)
        self.noeng_dR_eps     = getattr(config, "noeng_dR_eps", 2.0)
        self.noeng_dAO_eps    = getattr(config, "noeng_dAO_eps", 0.01)
        self.noeng_patience   = getattr(config, "noeng_patience", 200)
        self.noeng_penalty_val = getattr(config, "noeng_penalty_value", 1.0)

        # 缓存
        self._last_Rh   = {}
        self._last_AO   = {}
        self._oob_cnt   = {}
        self._enm_oob_cnt = {}
        self._noeng_cnt = {}

    def _init_if_first(self, env, agent_id):
        if getattr(env, "current_step", 0) <= 1:
            self._last_Rh.clear()
            self._last_AO.clear()
            self._oob_cnt.clear()
            self._enm_oob_cnt.clear()
            self._noeng_cnt.clear()
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

        # 0) 数值异常
        if (not np.all(np.isfinite(my_state))) or (not np.all(np.isfinite(enemy_state))):
            self.log(f"[InvalidState] {agent_id} nan/inf detected")
            info["term_reason"] = "invalid_state"
            return True, False, info

        # 1) 超时
        if env.current_step >= self.max_steps:
            self.log(f"[Timeout] {agent_id} reached step limit: {env.current_step}")
            info["term_reason"] = "timeout"
            return True, False, info

        # 2) 高度终止
        my_alt = float(my_state[2])
        if my_alt < self.altitude_limit:
            self.log(f"[LowAltitude] {agent_id} crashed, z={my_alt:.2f}")
            info["term_reason"] = "crash_low_alt"
            return True, False, info

        # 3) 出界宽限（我 & 敌）
        my_x, my_y       = float(my_state[0]), float(my_state[1])
        enemy_x, enemy_y = float(enemy_state[0]), float(enemy_state[1])

        my_oob  = (abs(my_x) > self.arena_size) or (abs(my_y) > self.arena_size)
        enm_oob = (abs(enemy_x) > self.arena_size) or (abs(enemy_y) > self.arena_size)

        self._oob_cnt[agent_id]     += 1 if my_oob  else 0
        self._enm_oob_cnt[agent_id] += 1 if enm_oob else 0

        # 同时出界到宽限 → 平局
        if self._oob_cnt[agent_id] >= self.oob_grace_steps and \
           self._enm_oob_cnt[agent_id] >= self.oob_grace_steps:
            self.log(f"[BothOutOfBounds] {agent_id} draw after grace")
            info["term_reason"] = "both_oob"
            return True, False, info

        # 敌方出界到宽限 → 我胜
        if self._enm_oob_cnt[agent_id] >= self.oob_grace_steps and \
           self._oob_cnt[agent_id] < self.oob_grace_steps:
            self.log(f"[EnemyOutOfBounds] {agent_id} wins (enemy oob)")
            info["term_reason"] = "enemy_oob"
            return True, True, info

        # 我方出界到宽限 → 我负
        if self._oob_cnt[agent_id] >= self.oob_grace_steps and \
           self._enm_oob_cnt[agent_id] < self.oob_grace_steps:
            self.log(f"[OutOfBounds] {agent_id} lost (self oob)")
            info["term_reason"] = "self_oob"
            return True, False, info

        # 回到圈内则计数清零
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
            return True, False, info
        if my_dead:
            self.log(f"[Shotdown] {agent_id} has 0 hp")
            info["term_reason"] = "self_dead"
            return True, False, info
        if enemy_dead:
            self.log(f"[MissionSuccess] {agent_id} eliminated all enemies")
            info["term_reason"] = "enemy_dead"
            return True, True, info

        # 5) 无交战判定（可关闭）
        if self.enable_noeng:
            ego_feat = np.concatenate([my_state[0:3],  my_state[6:9]]).astype(np.float32)
            enm_feat = np.concatenate([enemy_state[0:3], enemy_state[6:9]]).astype(np.float32)
            AO, TA, R_h = task.get2d_AO_TA_R_ue(ego_feat, enm_feat)
            if AO is None or np.isnan(AO):  AO = math.pi
            if R_h is None or np.isnan(R_h): R_h = 1e9
            AO  = float(np.clip(AO, 0.0, math.pi))
            R_h = float(np.clip(R_h, 0.0, 1e9))

            last_Rh = self._last_Rh.get(agent_id, R_h)
            last_AO = self._last_AO.get(agent_id, AO)
            dR      = R_h - last_Rh
            dAO     = AO  - last_AO

            # 条件：离得远且几何几乎不变
            if R_h > self.noeng_R_min and \
               abs(dR) < self.noeng_dR_eps and \
               abs(dAO) < self.noeng_dAO_eps:
                self._noeng_cnt[agent_id] += 1
            else:
                self._noeng_cnt[agent_id] = 0

            if self._noeng_cnt[agent_id] >= self.noeng_patience:
                self.log(f"[NoEngagement] {agent_id} draw: R={R_h:.1f}, dR={dR:.2f}, dAO={dAO:.3f}")
                info["term_reason"] = "no_engagement"

                # 在 env 上打上额外惩罚标记
                if not hasattr(env, "noeng_penalty"):
                    env.noeng_penalty = {}
                env.noeng_penalty[agent_id] = -float(self.noeng_penalty_val)

                return True, False, info

            self._last_Rh[agent_id] = R_h
            self._last_AO[agent_id] = AO

        return False, False, info