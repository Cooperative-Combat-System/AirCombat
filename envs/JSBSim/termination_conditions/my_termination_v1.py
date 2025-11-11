import math
from .termination_condition_base import BaseTerminationCondition

class MyTerminationV1(BaseTerminationCondition):
    def __init__(self, config):
        super().__init__(config)
        self.max_steps = 1000  # 直接写死最大步数
        self.altitude_limit = 50  # 高度单位为米，0 以下为坠毁
        self.arena_size = 300

    def get_termination(self, task, env, agent_id, info={}):
        agent = env.agents[agent_id]
        if agent.my_state is None:
            return False, False, info

        my_state = agent.my_state
        enemy_state = agent.enemy_state

        # 1. 超时终止
        if env.current_step >= self.max_steps:
            self.log(f"[Timeout] {agent_id} reached step limit: {env.current_step}")
            #print("timeout")
            return True, False, info

        # 2. 高度低于地面（z < 0）
        altitude =my_state[2]
        if altitude < self.altitude_limit:
            self.log(f"[LowAltitude] {agent_id} crashed, z={altitude}")
            #print("crashed")
            return True, False, info

        my_x = my_state[0]
        my_y = my_state[1]
        enemy_x = enemy_state[0]
        enemy_y = enemy_state[1]
        if abs(my_x) > self.arena_size or abs(my_y) > self.arena_size:
            self.log(f"[OutOfBounds] {agent_id} out of bounds")
            #print("out of bounds")
            return True, False, info

        if abs(enemy_x) > self.arena_size or abs(enemy_y) > self.arena_size:
            self.log(f"[EnemyOutOfBounds] {agent_id} out of bounds")
            #print("enemy out of bounds")
            return True, False, info


        # 3. 被击毁（hp <= 0）
        hp = my_state[12]
        if hp <= 0.1:
            self.log(f"[Shotdown] {agent_id} has 0 hp")
            #print("shotdown")
            return True, False, info

        # 4. 所有敌人 hp <= 0
        enemy_hp = enemy_state[12]

        if enemy_hp <= 0.1:
            self.log(f"[MissionSuccess] {agent_id} eliminated all enemies")
            #print("mission success")
            return True, True, info

        return False, False, info
