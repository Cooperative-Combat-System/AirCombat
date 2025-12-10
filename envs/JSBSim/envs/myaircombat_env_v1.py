from typing import Dict, Any, Tuple
import numpy as np
from .env_base import BaseEnv
from ..core.my_agent_v1 import MyAirAgentSimV1
from ..tasks import MyCombatTaskV1
import random
from collections import defaultdict

def generate_initial_state():
    my_initial_state = np.array([0, 0, 100, 0, 0, 0, 10, 0, 0, 0, 0, 0])
    enemy_initial_state = np.array([100, random.uniform(-20,20), 100, 0, 0, 3.14, -10, 0, 0, 0, 0, 0])
    return my_initial_state, enemy_initial_state

class AirCombatEnvV1(BaseEnv):
    def __init__(self, config_name: str):
        super().__init__(config_name)
        self.current_step = 0
        self._side_flag = False
        self._step_logs = defaultdict(list)  # 每步收集各 agent 的奖励分量
        self._episode_logs = defaultdict(list)  # 也可以顺手存一份做 episode 统计

    def log_step_components(self, agent_id: str, comp: dict):
        """在每一步被 reward 函数调用，记录分量"""
        self._step_logs[agent_id].append(comp)
        self._episode_logs[agent_id].append(comp)

    def pop_step_aggregate(self):
        """runner 每步取平均后清空步级缓存"""
        if not self._step_logs:
            return None
        # 聚合为所有 agent 的均值
        keys = set().union(*[c.keys() for v in self._step_logs.values() for c in v]) if self._step_logs else set()
        out = {}
        for k in keys:
            vals = []
            for ag, lst in self._step_logs.items():
                for c in lst:
                    if k in c:
                        vals.append(c[k])
            if vals:
                out[k] = float(np.mean(vals))
        self._step_logs.clear()
        return out

    def pop_episode_aggregate(self):
        """episode 结束时统计每个分量的均值/和（你选其一，这里给均值）并清空"""
        if not self._episode_logs:
            return None
        keys = set().union(*[c.keys() for v in self._episode_logs.values() for c in v]) if self._episode_logs else set()
        out = {}
        for k in keys:
            vals = []
            for ag, lst in self._episode_logs.items():
                for c in lst:
                    if k in c:
                        vals.append(c[k])
            if vals:
                out[f"ep_mean/{k}"] = float(np.mean(vals))
                out[f"ep_sum/{k}"] = float(np.sum(vals))
        self._episode_logs.clear()
        return out

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
        self._step_logs.clear()
        self._episode_logs.clear()
        self._side_flag = not self._side_flag
        self.task.adaptor.reconnect()
        self.reload_agent()
        self.task.reset(self)
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
        self.current_step += 1
        info = {"current_step": self.current_step}

        dones = {}
        for agent_id, agent in self.agents.items():
            done, info = self.task.get_termination(self, agent_id, info)
            dones[agent_id] = [done]

        if any(v[0] for v in dones.values()):
            for k in dones:
                dones[k] = [True]
            self.task.truncated = True
        # 解包动作
        unpacked_action = self._unpack(action)
        for agent_id, agent in self._air_agent.items():
            norm_action = self.task.normalize_action(self, agent_id, unpacked_action[agent_id])
            self.task.adaptor.send_action(norm_action, agent.port)

        self.task.step(self)
        for agent in self.agents.values():
            agent.observe()

        obs = self.get_obs()

        rewards = {}
        for agent_id, agent in self.agents.items():
            reward, info = self.task.get_reward(self, agent_id, info)
            rewards[agent_id] = [reward]

        step_comp = self.pop_episode_aggregate()
        if step_comp:
            for k, v in step_comp.items():
                info[f"rc/{k}"] = float(v)

        if all(v[0] for v in dones.values()):
            ep_comp = self.pop_episode_aggregate()
            if ep_comp:
                for k, v in ep_comp.items():
                    info[f"erc/{k}"] = float(v)
        return self._pack(obs), self._pack(rewards), self._pack(dones), info

    def close(self):
        self.task.adaptor.close()

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'mycombatv1':
            self.task = MyCombatTaskV1(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")
