from typing import Dict, Any, Tuple
import numpy as np
from .env_base import BaseEnv
from ..core.my_agent_v1 import MyAirAgentSimV1
from ..tasks import MyCombatTaskV1
import random
from collections import defaultdict
from ..model.pid_baseline_agent import DogFightController

def generate_initial_state():
    my_initial_state = np.array([0, 0, 600, 0, 0, 0, 25, 0, 0, 0, 0, 0])
    enemy_initial_state = np.array([550, 0, 600, 0, 0, np.pi, -25, 0, 0, 0, 0, 0])
    return my_initial_state, enemy_initial_state

class AirCombatEnvV1(BaseEnv):
    def __init__(self, config_name: str):
        super().__init__(config_name)
        self.current_step = 0
        self._side_flag = False
        self.tick_per_action = 12
        self.agent_action_per_tick = {}

    def load_agent(self):
        self._air_agent = {}
        port_list = self.config.agent_ports
        my_initial_state, enemy_initial_state = generate_initial_state()

        # 生成 ego_ids 和 enm_ids 模拟 uid 命名规则，如 A00, B00
        self.ego_ids = [f"A{i:02d}" for i in range(1)]
        self.enm_ids = [f"B{i:02d}" for i in range(1)]
        all_agent_ids = self.ego_ids + self.enm_ids

        # 预创建 agent 并挂载初始状态
        for idx, (agent_id, port) in enumerate(zip(all_agent_ids, port_list)):
            init_state = np.concatenate((my_initial_state, enemy_initial_state)) if agent_id in self.ego_ids else np.concatenate((enemy_initial_state, my_initial_state))
            agent = MyAirAgentSimV1(agent_id, port, self.task.adaptor, init_state)
            self._air_agent[agent_id] = agent
            agent.initialize()

        # 设置伙伴和敌人
        for agent_id, agent in self._air_agent.items():
            agent.partners = [self._air_agent[pid] for pid in
                              (self.ego_ids if agent_id in self.ego_ids else self.enm_ids) if pid != agent_id]
            agent.enemies = [self._air_agent[eid] for eid in
                             (self.enm_ids if agent_id in self.ego_ids else self.ego_ids)]

        self.pid_baselines = {}
        if self.use_baseline:
            for eid in self.enm_ids:
                self.pid_baselines[eid] = DogFightController()

    def load(self):
        self.load_task()
        self.load_agent()
        self.seed()

    @property
    def agents(self) -> Dict[str, MyAirAgentSimV1]:
        return self._air_agent

    def reset(self) -> np.ndarray:
        #print('Resetting the environment.')
        self.current_step = 0
        self._side_flag = not self._side_flag
        self.task.adaptor.reconnect()
        self.reload_agent()
        self.task.reset(self)
        if self.use_baseline:
            for pid_agent in self.pid_baselines.values():
                pid_agent.reset()
        for agent in self.agents.values():
            agent.observe()
        obs = self.get_obs()
        return self._pack(obs)

    def reload_agent(self):
        my_initial_state, enemy_initial_state = generate_initial_state()
        if self._side_flag:
            my_initial_state, enemy_initial_state = enemy_initial_state, my_initial_state
        for agent_id, agent in self.agents.items():
            agent.reload(np.concatenate((my_initial_state, enemy_initial_state)) if agent_id in self.ego_ids else np.concatenate((enemy_initial_state, my_initial_state)))

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        dones = {}
        info = {}
        for agent_id, agent in self.agents.items():
            done, info = self.task.get_termination(self, agent_id, info)
            dones[agent_id] = [done]

        if any(v[0] for v in dones.values()):
            for k in dones:
                dones[k] = [True]
            self.task.truncated = True
        # 解包动作
        unpacked_action = self._unpack(action)
        if self.use_baseline:
            for eid, pid_agent in self.pid_baselines.items():
                my_state = self._air_agent[eid].my_state
                enemy_state = self._air_agent[eid].enemy_state
                pid_act = pid_agent.step(my_state, enemy_state, 1/60)
                unpacked_action[eid] = pid_act

        for agent_id, agent in self._air_agent.items():
            norm_action = self.task.normalize_action(self, agent_id, unpacked_action[agent_id])
            self.agent_action_per_tick[agent_id] = norm_action

        self.task.step(self)
        for k in range(self.tick_per_action):
            for agent_id, agent in self._air_agent.items():
                norm_action = self.agent_action_per_tick[agent_id]
                self.task.adaptor.send_action(norm_action, agent.port)
                agent.observe()
                if self.task.truncated:
                    break

        obs = self.get_obs()
        self.current_step += 1
        rewards = {}
        for agent_id, agent in self.agents.items():
            reward, info = self.task.get_reward(self, agent_id, info)
            rewards[agent_id] = [reward]

        return self._pack(obs), self._pack(rewards), self._pack(dones), info

    def close(self):
        self.task.adaptor.close()

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'mycombatv1':
            self.task = MyCombatTaskV1(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")
