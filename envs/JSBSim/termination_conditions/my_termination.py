import math
from .termination_condition_base import BaseTerminationCondition

class MyTermination(BaseTerminationCondition):
    def __init__(self, config):
        super().__init__(config)
        self.max_steps = 1000  # 直接写死最大步数
        self.altitude_limit = 50  # 高度单位为米，0 以下为坠毁

    def get_termination(self, task, env, agent_id, info={}):
        agent = env.agents[agent_id]
        if agent.last_raw_obs is None:
            return False, False, info

        raw_obs = agent.last_raw_obs[0] if isinstance(agent.last_raw_obs, list) else agent.last_raw_obs  # 取第一架飞机的观测数据

        # 1. 超时终止
        if env.current_step >= self.max_steps:
            self.log(f"[Timeout] {agent_id} reached step limit: {env.current_step}")
            return True, False, info

        # 2. 高度低于地面（z < 0）
        altitude = raw_obs.get("position", [0, 0, 0])[2]
        if altitude < self.altitude_limit:
            self.log(f"[LowAltitude] {agent_id} crashed, z={altitude}")
            return True, False, info

        # 3. 被击毁（hp <= 0）
        hp = raw_obs.get("hp", 1.0)
        if hp <= 0:
            self.log(f"[Shotdown] {agent_id} has 0 hp")
            return True, False, info

        # 4. 所有敌人 hp <= 0
        all_enemy_dead = True
        for enemy in agent.enemies:
            try:
                if enemy.last_raw_obs and (
                enemy.last_raw_obs[0]["hp"] if isinstance(enemy.last_raw_obs, list) else enemy.last_raw_obs["hp"]) > 0:
                    all_enemy_dead = False
                    break
            except:
                continue

        if all_enemy_dead:
            self.log(f"[MissionSuccess] {agent_id} eliminated all enemies")
            return True, True, info

        return False, False, info
