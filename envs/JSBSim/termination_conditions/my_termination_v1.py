import math
import numpy as np
from .termination_condition_base import BaseTerminationCondition

class MyTerminationV1(BaseTerminationCondition):
    def __init__(self, config):
        super().__init__(config)
        # 基础：步数/高度/边界
        self.max_steps      = getattr(config, "max_steps", 1000)
        self.altitude_limit = getattr(config, "altitude_limit", 300.0)
        self.arena_size     = getattr(config, "arena_size", 5000.0)


    def get_termination(self, task, env, agent_id, info={}):
        agent = env.agents[agent_id]
        if agent.my_state is None:
            return False, False, info

        my_state    = agent.my_state
        enemy_state = agent.enemy_state

        # 1) 超时
        if env.current_step >= self.max_steps:
            self.log(f"[Timeout] {agent_id} reached step limit: {env.current_step}")
            info["term_reason"] = "timeout"
            return True, False, info

        # 2) 高度终止
        my_alt = float(my_state[2])
        if my_alt <= self.altitude_limit:
            self.log(f"[LowAltitude] {agent_id} crashed, z={my_alt:.2f}")
            info["term_reason"] = "crash_low_alt"
            return True, False, info

        # 3) 出界宽限（我 & 敌）
        my_x, my_y       = float(my_state[0]), float(my_state[1])
        enemy_x, enemy_y = float(enemy_state[0]), float(enemy_state[1])

        my_oob  = (abs(my_x) > self.arena_size) or (abs(my_y) > self.arena_size)
        enm_oob = (abs(enemy_x) > self.arena_size) or (abs(enemy_y) > self.arena_size)

        # 同时出界到宽限 → 平局
        if my_oob and enm_oob:
            self.log(f"[BothOutOfBounds] {agent_id} draw after grace")
            info["term_reason"] = "both_oob"
            return True, False, info

        # 敌方出界到宽限 → 我胜
        if enm_oob:
            self.log(f"[EnemyOutOfBounds] {agent_id} wins (enemy oob)")
            info["term_reason"] = "enemy_oob"
            return True, True, info

        # 我方出界到宽限 → 我负
        if my_oob:
            self.log(f"[OutOfBounds] {agent_id} lost (self oob)")
            info["term_reason"] = "self_oob"
            return True, False, info

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

        return False, False, info